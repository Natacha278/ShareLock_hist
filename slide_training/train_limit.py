#!/usr/bin/env python
"""
LIMIT-style training (Locked Image Multiple Instance pretraining; "Beyond Sparse Captions", NeurIPS 2026,
CONCH-Z) of a text projection on TCGA slide-report pairs, with frozen UNI2-h patch features, plus
patch-level zero-shot evaluation (BACH/CRC/MHIST/BreakHis) and ARCH retrieval every epoch.

What follows the paper (Eq. 1-4, Sec. 4):
  * image side frozen, nothing trainable on it; only the text side is trained (here: an MLP on frozen
    BioLORD report features, mapping into the UNI2-h space, like ShareLock project="text")
  * parameter-free report-conditioned attention over the raw patch embeddings:
        a_i(r) = softmax_i( v_i . r / tau_attn )            raw dot product, single head, tau_attn fixed
        h_W(r) = sum_i a_i(r) v_i
  * bags of K = N + 1 slide-report pairs from the SAME ORGAN (hard negatives; N = 4 by default)
  * every slide in the bag scored against every report of the bag: s_ij = cos(h_{W_i}(r_j), r_j) / tau_nce
  * similarity-weighted InfoNCE, slide -> report direction only, averaged over the K slides:
        L_i = -log( e^{s_ii} / (e^{s_ii} + sum_{j!=i} w_ij e^{s_ij}) ),   w_ij = 1 - (cos(r_i, r_j) + 1) / 2
    with NO gradient through w_ij; tau_nce = 0.07 fixed
  * AdamW, lr 1e-5, weight decay 1e-5, cosine annealing, short schedule (3 epochs), batch = 1 bag,
    best epoch on a held-out validation split
What differs (unavoidable here): text side is an MLP on frozen BioLORD features (paper: full CONCH v1.5
text tower), UNI2-h 256 px patches (paper: CONCH v1.5 448 px), ~6.5k TCGA PathText reports (paper: 335k
LLM-filtered clinical reports). Unspecified in the paper, chosen here: tau_attn (default 1.0, see the
attention diagnostics printed every --log-every steps), cosine similarity in s_ij, all patches per slide.
tau_attn must match the scale of the embeddings (raw dot product): with large-norm features (UNI2-h) a
small fixed value makes the initial attention lock onto random patches and the model memorises, a large
one gives near-uniform attention (= mean pooling, clearly worse in the paper's ablation). 'auto' searches
the tau_attn for which the initial attention spreads over --attn-target-eff (default 100) effective patches
per slide, matching the paper's finding that top-100 pooling performs like LIMIT. (A multiplier on the
pooled score std, used before, was miscalibrated on real slides: it mixes between-slide offsets into the
spread and ignores the number of patches; 13k-17k patches per TCGA slide.)

Ablation switches (paper Table 4 analogues): --negatives random, --no-weighting, --loss symmetric,
--attn-score cos.

Patch-level zero-shot and ARCH retrieval are evaluated before training (epoch 0 = the initial model,
e.g. the Quilt projection) and after every epoch; they are monitoring only. Model selection uses the
TCGA validation split (--select val_recall, default) or the last epoch -- never the benchmarks.

Outputs in <runs-dir>/<run-name>_MMDD-HHMM/:
  config.json, train_log.csv (loss, val retrieval, zero-shot balanced accuracies, ARCH R@k per epoch),
  model_best.pt / model_last.pt (full model), model_best_projection.pt / model_last_projection.pt
  (train_projection format -> eval_projection.py), zeroshot/epoch<e>/ (confusion matrices, prompts),
  results.json; one row appended to <runs-dir>/summary_limit.csv.

Example (start from the patch-trained Quilt projection):
  python train_limit.py \
    --features-dir $SLURM_TMPDIR/bags --txt-feats .../tcga_reports_BioLORD-2023.pt \
    --init-txt-from .../runs/<quilt run>/model_best.pt \
    --test-feats .../features/uni2h --txt-model .../weights/BioLORD-2023 \
    --arch-feats .../arch_pubmed.pt .../arch_books.pt \
    --runs-dir .../runs_limit --run-name limit_quiltinit
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import csv
import json
import math
import platform
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from projection import MLP, ProjectionModel
from tcga_bags import TCGABags, collate_bags
from train_projection import SUMMARY_COLS, arch_retrieval, load_text_encoder, zero_shot
from train_slide_text import evaluate, unique_dir

# TCGA project -> organ (several projects share an organ: those are the hardest negatives)
TCGA_ORGAN = {
    "ACC": "adrenal", "PCPG": "adrenal", "BLCA": "bladder", "BRCA": "breast", "CESC": "cervix",
    "CHOL": "bile_duct", "COAD": "colorectal", "READ": "colorectal", "DLBC": "lymph_node",
    "ESCA": "esophagus", "GBM": "brain", "LGG": "brain", "HNSC": "head_neck", "KICH": "kidney",
    "KIRC": "kidney", "KIRP": "kidney", "LIHC": "liver", "LUAD": "lung", "LUSC": "lung",
    "MESO": "pleura", "OV": "ovary", "PAAD": "pancreas", "PRAD": "prostate", "SARC": "soft_tissue",
    "SKCM": "skin", "STAD": "stomach", "TGCT": "testis", "THCA": "thyroid", "THYM": "thymus",
    "UCEC": "uterus", "UCS": "uterus", "UVM": "eye",
}


def organ_of(project):
    p = str(project).replace("TCGA-", "").upper()
    return TCGA_ORGAN.get(p, p or "unknown")


# ----------------------------------------------------------------------------- data
class BagSampler(Sampler):
    """Yields, per step, the indices of `bags_per_step` bags of exactly K slides.
    negatives="organ": every bag from one organ (LIMIT); "random": bags drawn across all organs.
    Each epoch reshuffles; slides left over (organ size not a multiple of K) differ from epoch to epoch."""

    def __init__(self, groups, k, bags_per_step=1, negatives="organ", seed=0):
        self.k, self.bps, self.negatives, self.seed, self.epoch = k, bags_per_step, negatives, seed, 0
        self.groups = {g: idx for g, idx in groups.items() if len(idx) >= k} if negatives == "organ" else \
            {"all": [i for idx in groups.values() for i in idx]}
        self.skipped = {g: len(idx) for g, idx in groups.items() if negatives == "organ" and len(idx) < k}

    def _bags(self):
        rng = random.Random(self.seed + self.epoch)
        bags = []
        for idx in self.groups.values():
            idx = idx[:]
            rng.shuffle(idx)
            bags += [idx[s:s + self.k] for s in range(0, len(idx) - self.k + 1, self.k)]
        rng.shuffle(bags)
        return bags

    def __iter__(self):
        bags = self._bags()
        self.epoch += 1
        for s in range(0, len(bags) - self.bps + 1, self.bps):
            yield [i for b in bags[s:s + self.bps] for i in b]

    def __len__(self):
        return sum(len(idx) // self.k for idx in self.groups.values()) // self.bps


# ----------------------------------------------------------------------------- model
class LimitModel(nn.Module):
    def __init__(self, img_dim, txt_dim, txt_depth=2, txt_width=1024, txt_norm=True, txt_dropout=0.0,
                 attn_temperature=1.0, nce_temperature=0.07, attn_score="dot"):
        super().__init__()
        self.txt_proj = MLP(txt_dim, img_dim, depth=txt_depth, width=txt_width, norm=txt_norm, dropout=txt_dropout)
        self.attn_temperature, self.nce_temperature, self.attn_score = attn_temperature, nce_temperature, attn_score

    @property
    def temperature(self):
        return self.nce_temperature

    def encode_text(self, txt):
        """r = text embedding in the patch space (unnormalised: the attention uses the raw dot product)."""
        return self.txt_proj(txt)

    def _scores(self, r, v):
        if self.attn_score == "cos":
            r, v = F.normalize(r, dim=-1), F.normalize(v, dim=-1)
        elif self.attn_score == "unitq":          # unit-norm report, raw patches: tau_attn alone sets the sharpness
            r = F.normalize(r, dim=-1)
        return r, v

    def attention(self, r, patches, mask):
        """r [T, D], patches [S, N, D], mask [S, N] -> attention [T, S, N] (softmax over each slide's patches)."""
        if not torch.is_autocast_enabled(patches.device.type):
            patches = patches.to(r.dtype)
        q, k = self._scores(r, patches)
        a = torch.einsum("td,snd->tsn", q, k.to(q.dtype)) / self.attn_temperature
        a = a.masked_fill(~mask[None], float("-inf"))
        return a.float().softmax(-1)

    def slide_embeddings(self, r, patches, mask):
        """Normalised report-conditioned slide embeddings h_W(r) [T, S, D] (same API as SlideTextModel)."""
        a = self.attention(r, patches, mask)
        if not torch.is_autocast_enabled(patches.device.type):
            patches = patches.to(a.dtype)
        h = torch.einsum("tsn,snd->tsd", a.to(patches.dtype), patches)
        return F.normalize(h.float(), dim=-1)

    def bag_logits(self, r, patches, mask):
        """r [B, K, D], patches [B, K, N, D], mask [B, K, N] -> s [B, K_slide, K_report] = cos(h_{W_i}(r_j), r_j) / tau."""
        if not torch.is_autocast_enabled(patches.device.type):
            patches = patches.to(r.dtype)
        q, k = self._scores(r, patches)
        a = torch.einsum("bjd,bind->bjin", q, k.to(q.dtype)) / self.attn_temperature        # [B, report, slide, N]
        a = a.masked_fill(~mask[:, None], float("-inf")).float().softmax(-1)
        h = torch.einsum("bjin,bind->bijd", a.to(patches.dtype), patches)                  # [B, slide, report, D]
        return torch.einsum("bijd,bjd->bij", F.normalize(h.float(), dim=-1),
                            F.normalize(r.float(), dim=-1)) / self.nce_temperature, a

    def projection_checkpoint(self, img_dim, txt_dim, cfg):
        sd = {k: v.detach().cpu() for k, v in self.state_dict().items() if k.startswith("txt_proj.")}
        sd["logit_scale"] = torch.tensor(math.log(1.0 / self.nce_temperature))
        pcfg = dict(img_dim=img_dim, txt_dim=txt_dim, project="text", embed_dim=None, depth=cfg["txt_depth"],
                    width=cfg["txt_width"], norm=cfg["txt_norm"], dropout=cfg["txt_dropout"],
                    init_temperature=self.nce_temperature, learn_temperature=True)
        return sd, pcfg


def limit_loss(s, r, weighting=True, direction="slide2report"):
    """s [B, K, K] (slide i, report j), r [B, K, D] report embeddings. Weighted InfoNCE (Eq. 3-4)."""
    B, K, _ = s.shape
    target = torch.arange(K, device=s.device).repeat(B)
    eye = torch.eye(K, dtype=torch.bool, device=s.device)
    logits = s
    if weighting:
        with torch.no_grad():                                                   # no gradient through w_ij
            z = F.normalize(r.float(), dim=-1)
            w = 1.0 - (z @ z.transpose(1, 2) + 1.0) / 2.0                      # [B, K, K]
            logw = torch.log(w.clamp_min(1e-8)).masked_fill(eye, 0.0)          # positive term unweighted
        logits = s + logw                                                       # w e^s = e^(s + log w)
    loss = F.cross_entropy(logits.reshape(B * K, K), target)
    if direction == "symmetric":
        lt = s.transpose(1, 2) + (logw.transpose(1, 2) if weighting else 0)
        loss = 0.5 * (loss + F.cross_entropy(lt.reshape(B * K, K), target))
    return loss


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def negative_weights(r):
    """Mean / min of the LIMIT negative weights w_ij = 1 - (cos(r_i, r_j) + 1) / 2 over the off-diagonal pairs."""
    z = F.normalize(r.float(), dim=-1)
    w = 1.0 - (z @ z.transpose(1, 2) + 1.0) / 2.0
    off = w[:, ~torch.eye(w.shape[1], dtype=torch.bool, device=w.device)]
    return float(off.mean()), float(off.min())


@torch.no_grad()
def attention_stats(a, mask):
    """a [B, report, slide, N] -> effective number of patches (exp entropy) and top-1 weight, own slide."""
    K = a.shape[1]
    own = a[:, torch.arange(K), torch.arange(K)]                                    # [B, K, N]
    m = mask.float()
    ent = -(own.clamp_min(1e-12).log() * own * m).sum(-1)
    return float(ent.mean().exp()), float(own.max(-1).values.mean()), float(m.sum(-1).mean())   # geometric mean


@torch.no_grad()
def calibration_scores(model, dl, K, device, n_batches=8):
    """Raw report.patch scores of the initial model, one vector per (report, own slide) pair (real patches only)."""
    out = []
    for b, (patches, mask, txt, _) in enumerate(dl):
        B = patches.shape[0] // K
        patches = patches.to(device).view(B, K, *patches.shape[1:])
        mask = mask.to(device).view(B, K, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):   # same precision as training
            if device != "cuda":
                patches = patches.float()
            r = model.encode_text(txt.to(device).view(B, K, -1).float())
            q, k = model._scores(r, patches)
            sc = torch.einsum("bkd,bknd->bkn", q, k.to(q.dtype)).float()
        out += [sc[i, j][mask[i, j]] for i in range(B) for j in range(K)]
        if b + 1 >= n_batches:
            break
    return out


def effective_patches(scores, tau, summary=False):
    """Geometric mean over (report, slide) pairs of exp(entropy(softmax(scores / tau))), the effective number of
    attended patches. Geometric, not arithmetic: the per-slide counts are very skewed (a few near-uniform slides
    would otherwise dominate, e.g. mean 100 while the median slide attends to 1-4 patches)."""
    eff = []
    for sc in scores:
        a = (sc.float() / tau).softmax(-1)
        eff.append(torch.exp(-(a * a.clamp_min(1e-30).log()).sum()))
    eff = torch.stack(eff)
    g = float(eff.log().mean().exp())
    if summary:
        return g, float(eff.min()), float(eff.median()), float(eff.max())
    return g


def tau_for_target(scores, target, iters=60):
    """Bisection (in log tau) for the tau_attn giving `target` effective patches (geometric mean over slides)."""
    within = float(torch.stack([sc.std() for sc in scores]).mean())          # within-slide spread
    lo, hi = math.log(within * 1e-3), math.log(within * 1e3)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if effective_patches(scores, math.exp(mid)) > target:
            hi = mid                                                          # too soft -> smaller tau
        else:
            lo = mid
    return math.exp(0.5 * (lo + hi)), within


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--features-dir", required=True, help="fp16 .npy bags (stage_bags.py) or TRIDENT .h5")
    p.add_argument("--txt-feats", required=True, help="tcga_reports_<model>.pt (extract_report_features.py)")
    p.add_argument("--txt-view", choices=["full", "sentmean"], default="full")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-patches", type=int, default=0, help="random patches per slide (0 = all, as in LIMIT)")
    # bags / loss (LIMIT defaults)
    p.add_argument("--n-negatives", type=int, default=4, help="N negatives per bag, K = N + 1 (paper Sec. 4.5: 4)")
    p.add_argument("--bags-per-step", type=int, default=1, help="bags per optimisation step (paper: batch size 1)")
    p.add_argument("--negatives", choices=["organ", "random"], default="organ", help="same-organ (LIMIT) or random bags")
    p.add_argument("--no-weighting", action="store_true", help="plain InfoNCE (ablation)")
    p.add_argument("--loss", choices=["slide2report", "symmetric"], default="slide2report")
    p.add_argument("--attn-temperature", default="auto",
                   help="tau_attn, fixed during training (value not given in the paper). 'auto' (default): chosen "
                        "on a few training bags before training, see --attn-target-eff; or a number")
    p.add_argument("--attn-target-eff", type=float, default=100.0,
                   help="auto: tau_attn such that the initial attention spreads over this many effective patches "
                        "(exp of the entropy; geometric mean over the calibration slides). Default 100: in the paper, "
                        "top-100 pooling matches LIMIT (66.7 vs 66.6). 0 = use --attn-auto-scale instead")
    p.add_argument("--attn-auto-scale", type=float, default=1.0,
                   help="only with --attn-target-eff 0: tau_attn = this x the mean within-slide std of the scores")
    p.add_argument("--nce-temperature", type=float, default=0.07)
    p.add_argument("--attn-score", choices=["dot", "unitq", "cos"], default="dot",
                   help="dot = raw v.r (paper); unitq = unit-norm r . raw v (the MLP cannot sharpen the attention "
                        "by growing its output norm); cos = both normalised")
    # text model
    p.add_argument("--init-txt-from", default=None,
                   help="train_projection-format checkpoint (project=text), e.g. Quilt model_best.pt; its "
                        "depth/width/norm override the --txt-* flags")
    p.add_argument("--txt-depth", type=int, default=2)
    p.add_argument("--txt-width", type=int, default=1024)
    p.add_argument("--no-txt-norm", action="store_true")
    p.add_argument("--txt-dropout", type=float, default=0.0)
    # optimisation (LIMIT defaults)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=0.0, help="0 = off (not in the paper)")
    p.add_argument("--select", choices=["val_recall", "last"], default="val_recall")
    # patch-level monitoring
    p.add_argument("--test-feats", default=None, help="--out-dir of extract_uni2h_test_data.py (zero-shot each epoch)")
    p.add_argument("--txt-model", default=None, help="local sentence-transformers folder (needed with --test-feats)")
    p.add_argument("--test-modes", default="resize,tile")
    p.add_argument("--arch-feats", nargs="*", default=[])
    p.add_argument("--no-eval-epoch0", action="store_true", help="skip the evaluation of the initial model")
    # misc
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--runs-dir", default="runs_limit")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    if args.test_feats and not args.txt_model:
        raise SystemExit("--test-feats needs --txt-model (to encode the zero-shot prompts)")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible to PyTorch; --allow-cpu to force")
    print(f"device: {device} ({torch.cuda.get_device_name() if device == 'cuda' else 'CPU'})", flush=True)
    name = args.run_name or "limit"
    run_dir = unique_dir(Path(args.runs_dir) / f"{name}_{datetime.now():%m%d-%H%M}")
    print(f"run dir: {run_dir}", flush=True)

    # ------------------------------------------------------------ data
    txt_all = torch.load(args.txt_feats, map_location="cpu")
    kw = dict(txt_view=args.txt_view, val_frac=args.val_frac, test_frac=args.test_frac)
    train = TCGABags(args.features_dir, txt_all, "train", max_patches=args.max_patches or None, **kw)
    val = TCGABags(args.features_dir, txt_all, "val", **kw)
    test = TCGABags(args.features_dir, txt_all, "test", **kw)
    del txt_all
    img_dim, txt_dim = train.img_dim, train.txt_dim
    groups = defaultdict(list)
    for i, proj in enumerate(train.projects):
        groups[organ_of(proj)].append(i)
    K = args.n_negatives + 1
    sampler = BagSampler(groups, K, args.bags_per_step, args.negatives, args.seed)
    if sampler.skipped:
        print(f"organs with fewer than K={K} training slides (not used): {sampler.skipped}")
    print(f"slides  train {len(train):,}  val {len(val):,}  test {len(test):,}  |  organs "
          f"{len(groups)}  ({', '.join(f'{g} {len(v)}' for g, v in sorted(groups.items(), key=lambda x: -len(x[1])))})",
          flush=True)
    dl = DataLoader(train, batch_sampler=sampler, collate_fn=collate_bags, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=args.num_workers > 0,
                    prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    # ------------------------------------------------------------ model
    auto_tau = str(args.attn_temperature).lower() == "auto"
    cfg = dict(img_dim=img_dim, txt_dim=txt_dim, txt_depth=args.txt_depth, txt_width=args.txt_width,
               txt_norm=not args.no_txt_norm, txt_dropout=args.txt_dropout,
               attn_temperature=1.0 if auto_tau else float(args.attn_temperature),
               nce_temperature=args.nce_temperature,
               attn_score=args.attn_score)
    init = None
    if args.init_txt_from:
        init = torch.load(args.init_txt_from, map_location="cpu")
        c = init["model_cfg"]
        if c.get("project") != "text" or (c["img_dim"], c["txt_dim"]) != (img_dim, txt_dim):
            raise SystemExit(f"--init-txt-from must be a project=text checkpoint with dims {img_dim}/{txt_dim}")
        if init.get("txt_model") and train.txt_model and init["txt_model"] != train.txt_model:
            raise SystemExit(f"--init-txt-from was trained with {init['txt_model']}, reports use {train.txt_model}")
        cfg.update(txt_depth=c["depth"], txt_width=c["width"], txt_norm=c.get("norm", True),
                   txt_dropout=c.get("dropout", 0.0))
    model = LimitModel(**cfg).to(device)
    if init is not None:
        model.load_state_dict({k: v for k, v in init["state_dict"].items() if k.startswith("txt_proj.")})
        print(f"text MLP initialised from {args.init_txt_from} (depth {cfg['txt_depth']}, width {cfg['txt_width']})")
    if auto_tau:
        model.eval()
        with torch.no_grad():
            cal = calibration_scores(model, dl, K, device)
            n_mean = float(np.mean([len(c) for c in cal]))
            if args.attn_target_eff > 0:
                tau, within = tau_for_target(cal, args.attn_target_eff)
                how = f"target {args.attn_target_eff:.0f} effective patches"
            else:
                within = float(torch.stack([c.std() for c in cal]).mean())
                tau, how = within * args.attn_auto_scale, f"{args.attn_auto_scale} x within-slide std"
            g, lo_, med_, hi_ = effective_patches(cal, tau, summary=True)
        cfg["attn_temperature"] = model.attn_temperature = tau
        print(f"tau_attn auto ({how}): within-slide score std {within:.3f} -> tau_attn {tau:.4f}\n"
              f"  initial attention on {len(cal)} calibration slides ({n_mean:.0f} patches on average): effective "
              f"patches geometric mean {g:.0f}, min {lo_:.0f}, median {med_:.0f}, max {hi_:.0f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(sampler)
    total = steps_per_epoch * args.epochs
    n_params = sum(q.numel() for q in model.parameters())

    config = {"args": vars(args), "model": cfg, "n_trainable_params": n_params, "K": K,
              "data": {"n_train": len(train), "n_val": len(val), "n_test": len(test), "txt_model": train.txt_model,
                       "organs": {g: len(v) for g, v in groups.items()}, "steps_per_epoch": steps_per_epoch},
              "env": {"torch": torch.__version__, "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"params {n_params:,}  K {K} ({args.negatives} negatives)  steps/epoch {steps_per_epoch}  "
          f"lr {args.lr}  tau_attn {cfg['attn_temperature']:.4g} ({args.attn_score})  tau_nce {args.nce_temperature}  "
          f"loss {args.loss}{'' if args.no_weighting else ' weighted'}", flush=True)

    # ------------------------------------------------------------ patch-level monitoring
    encoder = load_text_encoder(args.txt_model, device) if args.test_feats else None
    modes = [m.strip() for m in args.test_modes.split(",") if m.strip()]
    txt_model_name = train.txt_model

    def projection():
        sd, pcfg = model.projection_checkpoint(img_dim, txt_dim, cfg)
        pm = ProjectionModel(**pcfg).to(device).eval()
        pm.load_state_dict(sd)
        return pm, sd, pcfg

    def patch_eval(epoch):
        out = {}
        if not (encoder or args.arch_feats):
            return out
        pm, _, _ = projection()
        with torch.no_grad():
            if encoder:
                d = run_dir / "zeroshot" / f"epoch{epoch}"
                d.mkdir(parents=True, exist_ok=True)
                prompts = {}
                zs = zero_shot(pm, encoder, args.test_feats, modes, device, d, prompts)
                (d / "prompts.json").write_text(json.dumps(prompts, indent=1))
                out.update({f"zs_{k}": v["balanced_acc"] for k, v in zs.items()})
            if args.arch_feats:
                retr = arch_retrieval(pm, args.arch_feats, txt_model_name, device)
                out.update({f"{n}_{k}": v for n, r in retr.items() for k, v in r.items() if "R@" in k})
        if out:
            zs_vals = [v for k, v in out.items() if k.startswith("zs_")]
            if zs_vals:
                out["zs_mean_bal_acc"] = float(np.mean(zs_vals))
        return out

    def save(path, epoch, metrics):
        torch.save({"state_dict": model.state_dict(), "model_cfg": cfg, "epoch": epoch, "val": metrics,
                    "txt_model": train.txt_model, "txt_view": args.txt_view, "source": "train_limit.py"}, path)
        _, sd, pcfg = projection()
        torch.save({"state_dict": sd, "model_cfg": pcfg, "epoch": epoch, "val": metrics, "txt_model": train.txt_model,
                    "pairs": None, "source": "train_limit.py"}, path.with_name(path.stem + "_projection.pt"))

    log_rows = []

    def write_log(row):
        """Rewrite the whole CSV each epoch, with the union of all columns seen so far."""
        log_rows.append(row)
        fields = []
        for r_ in log_rows:
            fields += [k for k in r_ if k not in fields]
        with open(run_dir / "train_log.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(log_rows)

    def summarise(prefix, row):
        keys = ["t2i_R@1", "i2t_R@1", "mean_recall", "zs_mean_bal_acc"]
        return prefix + "  " + "  ".join(f"{k} {row[k]:.4f}" for k in keys if k in row)

    # ------------------------------------------------------------ epoch 0 = initial model
    best, best_epoch = -1.0, 0
    if not args.no_eval_epoch0:
        print("\nepoch 0 (initial model):", flush=True)
        row = {"epoch": 0, **evaluate(model, val, device, args.num_workers), **patch_eval(0)}
        write_log(row)
        print(summarise("epoch   0", row), flush=True)
        if args.select == "val_recall":
            best = row.get("mean_recall", -1.0)
            save(run_dir / "model_best.pt", 0, row)

    # ------------------------------------------------------------ train
    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, wmeans, t_data, t_ep, t_last = [], [], 0.0, time.time(), time.time()
        for patches, mask, txt, _ in dl:
            t_data += time.time() - t_last
            B = patches.shape[0] // K
            patches = patches.to(device, non_blocking=True).view(B, K, *patches.shape[1:])
            mask = mask.to(device, non_blocking=True).view(B, K, -1)
            txt = txt.to(device, non_blocking=True).view(B, K, -1)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.warmup_steps, args.lr)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                r = model.encode_text(txt)
                s, a = model.bag_logits(r, patches, mask)
                loss = limit_loss(s.float(), r, weighting=not args.no_weighting, direction=args.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())
            wmean, wmin = negative_weights(r)
            wmeans.append(wmean)
            if step == 0 and not args.no_weighting and wmean < 0.1:
                print(f"  WARNING: mean negative weight {wmean:.3f}: the report embeddings in a bag are nearly "
                      f"collinear (cos ~ {1 - 2 * wmean:.2f}), so the weighting almost removes the negatives and "
                      f"there is little to learn. Compare with --no-weighting.", flush=True)
            step += 1
            t_last = time.time()
            i_ep = len(losses)
            if i_ep == 1 or (args.log_every and i_ep % args.log_every == 0):
                eff, top1, n_p = attention_stats(a, mask)
                el = t_last - t_ep
                print(f"  epoch {epoch} step {i_ep}/{steps_per_epoch}  loss {np.mean(losses[-args.log_every:]):.4f}  "
                      f"attn: {eff:.0f} eff. patches of {n_p:.0f}, top-1 {top1:.3f}  neg w {np.mean(wmeans[-args.log_every:]):.3f}  "
                      f"{el / i_ep:.2f}s/step ({100 * t_data / max(el, 1e-9):.0f}% data)  "
                      f"ETA {(steps_per_epoch - i_ep) * el / i_ep / 60:.0f} min", flush=True)

        row = {"epoch": epoch, "loss": float(np.mean(losses)), "neg_weight_mean": float(np.mean(wmeans)),
               "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(time.time() - t_ep), "data_wait_s": round(t_data),
               **evaluate(model, val, device, args.num_workers), **patch_eval(epoch)}
        write_log(row)
        print(summarise(f"epoch {epoch:3d}  loss {row['loss']:.4f}  time {row['epoch_s']}s", row), flush=True)
        if args.select == "last" or row.get("mean_recall", -1.0) > best:
            best, best_epoch = row.get("mean_recall", -1.0), epoch
            save(run_dir / "model_best.pt", epoch, row)
    save(run_dir / "model_last.pt", args.epochs, row)

    # ------------------------------------------------------------ test (selected checkpoint)
    ck = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    test_metrics = evaluate(model, test, device, args.num_workers)
    final_patch = {}
    if encoder or args.arch_feats:
        print("\npatch-level evaluation of the selected checkpoint:", flush=True)
        final_patch = patch_eval("best")
    print(f"selected epoch {best_epoch} ({args.select})\ntest: "
          + "  ".join(f"{k} {v:.4f}" for k, v in test_metrics.items()), flush=True)
    (run_dir / "results.json").write_text(json.dumps(
        {"selected_epoch": best_epoch, "select": args.select, "val": ck["val"], "test": test_metrics,
         "patch_eval": final_patch, "train_time_s": round(time.time() - t0)}, indent=2, default=float))

    summary = {"run": run_dir.name, "init_txt_from": Path(args.init_txt_from).parent.name if args.init_txt_from else "",
               "negatives": args.negatives, "K": K, "weighted": not args.no_weighting, "loss": args.loss,
               "attn_score": args.attn_score, "tau_attn": round(cfg["attn_temperature"], 5), "lr": args.lr,
               "epochs": args.epochs, "bags_per_step": args.bags_per_step, "seed": args.seed,
               "selected_epoch": best_epoch, **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
               **{k: round(v, 4) for k, v in final_patch.items()}}
    patch_cols = ["zs_" + c[:-len("_bal_acc")] for c in SUMMARY_COLS if c.endswith("_bal_acc")]
    patch_cols += ["zs_mean_bal_acc"] + [c for c in SUMMARY_COLS if c.startswith("arch_")]
    fields = [k for k in summary if not k.startswith(("zs_", "arch_"))] + patch_cols   # fixed columns across runs
    path = Path(args.runs_dir) / "summary_limit.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(summary)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
