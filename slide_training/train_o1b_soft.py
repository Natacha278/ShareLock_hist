#!/usr/bin/env python
"""
Objective 1-B (MLP version) with SOFT witness assignment: same EM as train_o1b_em.py, but the E-step gives
every candidate patch a weight w in [0, 1] (its estimated probability of being a true witness of the
sentence) instead of a hard witness set, and the M-step uses those weights.

Everything else is shared with train_o1b_em.py (patch cache, sentences, concepts, decoy sampling, batches,
Quilt prior, L2-SP, monitoring, model selection, outputs), so runs of the two scripts compare directly.

E-step (current MLP, no gradient), per training sentence t of slide i, with --n-decoys other-organ decoy
slides (default 10: the local estimate needs more decoy scores than the hard rule):
  --assign soft (default): local FDR from the decoys.
      F(k)    = #{decoy scores >= k-th target score} x N / N_d   (expected chance matches in the top k)
      lfdr(k) = [F(k + h) - F(k - h)] / 2h                       (slope of F, h = --lfdr-window)
      w(k)    = clip(1 - lfdr(k), 0, 1), made non-increasing in rank
    Strong witnesses (far above every decoy) get w ~ 1; where target and decoy scores mix, w -> 0; a
    sentence scoring like the decoys everywhere gets w ~ 0 for all its patches (soft NULL). sum(w) =
    expected number of true witnesses (the soft counterpart of the hard k*).
  --assign posterior (ablation): softmax over the slide's patches + a NULL option,
      q(p | t) = softmax{ s(t, p) / tau,  NULL = logsumexp(decoy scores / tau) + log(N / N_d) }
    (the decoys' total mass, rescaled to the target slide, is the mass of chance matches); w = q / max q,
    sentence weight = 1 - q(NULL); tau = --posterior-temperature (default: the InfoNCE temperature).
  Candidates kept: w > --min-weight, at most --max-candidates per sentence.
  Sentence weight s: --sent-weight max (default, = max w) | noisy-or (1 - prod(1 - w)) | none (1);
  sentences with s < --min-sent-weight (0.5) are soft-NULL for this round (not trained on); in simulations a
  pure-noise sentence reaches max w >= 0.5 in ~1-5% of cases (median 0.07), so lower thresholds let noise in.
  For comparison, the hard FDR rule of train_o1b_em.py (--fdr) is also evaluated on the same scores and logged.

M-step (one pair per sentence per pass, batches as train_o1b_em.py):
  * the training patch of a sentence is drawn with probability proportional to w (not uniformly);
  * weighted multi-positive InfoNCE, both directions:
        loss_a = s_a * [ log sum_b exp(L_ab)  -  log sum_b M_ab exp(L_ab) ]
    with M_ab = weight of patch b for sentence a's concept (--positives concept; a's own candidates with
    --positives witness; identity with --positives single), 1 on the diagonal, 0 for non-candidates;
    --weighting adds LIMIT-style weights on the pairs with M_ab = 0 (denominator only).
  All weights are computed in the E-step, without gradient.

Monitoring as train_o1b_em.py, plus soft statistics: soft-NULL rate, mean sentence weight, expected
witnesses (sum w), candidates per sentence, and the hard-rule NULL rate / k* on the same E-step.

Outputs in <runs-dir>/<run-name>_<MMDD-HHMM>/ as train_o1b_em.py; estep_round<r>.csv has one row per training
sentence (candidates, sum w, sentence weight, max w, hard k*, best score, text); witnesses_round<r>.pt holds
the candidates with their weights; one row is appended to <runs-dir>/summary_o1b_soft.csv.

Example:
  python train_o1b_soft.py --features-dir $SLURM_TMPDIR/bags --txt-feats .../tcga_reports_BioLORD-2023.pt \
    --pairs .../quilt_pairs_t05_biolord.pt --init-txt-from .../runs/<quilt run>/model_best.pt \
    --test-feats .../features/uni2h --txt-model .../weights/BioLORD-2023 --runs-dir .../runs_o1b --run-name soft
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import csv
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_slide_zeroshot import flat as flat_slide_zs
from eval_slide_zeroshot import slide_zero_shot
from train_o1b_em import (DecoySampler, build_cache, build_concepts, encode_sentences, fdr_select,
                          make_epoch_batches, write_concepts)

import pathlib, sys  # noqa: E401  (shared modules live in ../utils)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from projection import ProjectionModel, clip_loss
from quilt_pairs import QuiltPairs
from tcga_bags import TCGABags, split_of
from train_limit import organ_of, set_seed
from train_limit_mix import quilt_batches
from train_projection import (append_summary, arch_retrieval, load_text_encoder, param_groups, patch_summary_cols,
                              retrieval_metrics, zero_shot, zs_flat)
from train_slide_text import unique_dir


# ============================================================================ soft E-step
def lfdr_weights(st, sd, window=8, plus_one=False):
    """st [n, N] target scores, sd [n, Nd] decoy scores ->
    order [n, N] (target patches by decreasing score), w [n, N] (weight of the patch at each rank,
    non-increasing), F [n, N] (expected number of chance matches among the top k, k = 1..N)."""
    n, N = st.shape
    Nd = sd.shape[1]
    st_sorted, order = st.sort(dim=1, descending=True)
    sd_asc = sd.sort(dim=1).values.contiguous()
    n_dec = Nd - torch.searchsorted(sd_asc, st_sorted.contiguous(), right=False)    # decoys >= each threshold
    Fk = (n_dec.float() + float(plus_one)) * (N / Nd)
    Fp = torch.cat([torch.zeros(n, 1, device=st.device), Fk], 1)                     # F(0) = 0
    k = torch.arange(1, N + 1, device=st.device)
    lo, hi = (k - window).clamp(min=0), (k + window).clamp(max=N)
    lfdr = (Fp[:, hi] - Fp[:, lo]) / (hi - lo).float()
    w = torch.cummin((1.0 - lfdr).clamp(0.0, 1.0), dim=1).values
    return order, w, Fk


def posterior_weights(st, sd, tau):
    """Softmax over the slide's patches + NULL (decoy-calibrated) -> order, w (= q / max q, by rank), q_null."""
    N, Nd = st.shape[1], sd.shape[1]
    null = torch.logsumexp(sd / tau, 1) + math.log(N / Nd)
    q = torch.softmax(torch.cat([st / tau, null[:, None]], 1), 1)
    q_sorted, order = q[:, :N].sort(dim=1, descending=True)
    w = q_sorted / q_sorted[:, :1].clamp_min(1e-12)
    return order, w, q[:, N]


@torch.no_grad()
def soft_e_step(zt, cache, counts, slide_sents, slides, sampler, args, device, tau):
    """Per-sentence soft weights. Returns per-sentence stats and the candidate table (sent, slide, patch, w)."""
    stats = {k: [] for k in ("sent", "n_cand", "sum_w", "max_w", "s", "k_fdr", "own_max", "auc")}
    cand = {"sent": [], "slide": [], "patch": [], "w": []}
    for i in slides:
        a, b = slide_sents[i]
        if b <= a or counts[i] == 0:
            continue
        dec = [d for d in sampler(i) if counts[d] > 0]
        T_ = zt[a:b].float()
        st = T_ @ cache[i, :counts[i]].to(device).float().t()
        sents = torch.arange(a, b)
        own_max = st.max(1).values
        stats["sent"].append(sents)
        stats["own_max"].append(own_max.cpu())
        if not dec:                                                      # no decoy: cannot calibrate -> soft NULL
            z = torch.zeros(len(T_))
            for k in ("n_cand", "k_fdr"):
                stats[k].append(z.long())
            for k in ("sum_w", "max_w", "s"):
                stats[k].append(z)
            stats["auc"].append(torch.full((len(T_),), float("nan")))
            continue
        sd = T_ @ torch.cat([cache[d, :counts[d]] for d in dec]).to(device).float().t()
        dmax = torch.stack([x.max(1).values for x in sd.split([int(counts[d]) for d in dec], 1)], 1)
        stats["auc"].append(((own_max[:, None] > dmax).float().mean(1)
                             + 0.5 * (own_max[:, None] == dmax).float().mean(1)).cpu())
        _, _, k_fdr, _ = fdr_select(st, sd, args.fdr, args.max_candidates, args.fdr_plus_one, "fdr")
        stats["k_fdr"].append(k_fdr.cpu())
        if args.assign == "soft":
            order, w, _ = lfdr_weights(st, sd, args.lfdr_window, args.fdr_plus_one)
        else:
            order, w, q_null = posterior_weights(st, sd, tau)
        keep = (w > args.min_weight) & (torch.arange(w.shape[1], device=device)[None] < args.max_candidates)
        wk = w * keep
        if args.assign == "posterior":
            s = 1.0 - q_null
        elif args.sent_weight == "noisy-or":
            s = 1.0 - torch.prod(1.0 - wk, 1)
        else:
            s = wk.max(1).values
        if args.assign == "soft" and args.sent_weight == "none":
            s = (wk.max(1).values > 0).float()
        stats["n_cand"].append(keep.sum(1).cpu())
        stats["sum_w"].append(wk.sum(1).cpu())
        stats["max_w"].append(wk.max(1).values.cpu())
        stats["s"].append(s.cpu())
        r, c = keep.nonzero(as_tuple=True)
        if len(r):
            cand["sent"].append(sents[r.cpu()])
            cand["slide"].append(torch.full((len(r),), i, dtype=torch.long))
            cand["patch"].append(order[r, c].cpu())
            cand["w"].append(w[r, c].cpu())
    out = {k: torch.cat(v) if v else torch.empty(0) for k, v in stats.items()}
    out["cand"] = {k: torch.cat(v) if v else torch.empty(0, dtype=torch.float if k == "w" else torch.long)
                   for k, v in cand.items()}
    return out


# ============================================================================ M-step
def weight_mask(sent_b, key_b, concept, ck_keys, ck_w, keyspace, mode):
    """M[a, b] = weight of patch b for sentence a's concept (0 if not a candidate); diagonal = 1."""
    B = len(sent_b)
    eye = torch.eye(B, device=key_b.device)
    if mode == "single" or len(ck_keys) == 0:
        return eye
    q = concept[sent_b][:, None] * keyspace + key_b[None]
    idx = torch.searchsorted(ck_keys, q).clamp(max=len(ck_keys) - 1)
    M = torch.where(ck_keys[idx] == q, ck_w[idx], torch.zeros((), device=key_b.device))
    return torch.maximum(M, eye)


def soft_pos_nce(logits, M, row_w, logw=None):
    """sum_a row_w[a] * [logsumexp_b(L_ab (+ log w_ab on non-positives)) - logsumexp_b(L_ab + log M_ab)] / B."""
    pos = M > 0
    den = logits if logw is None else logits + logw.masked_fill(pos, 0.0)
    num = logits + torch.log(M)                                     # -inf where M = 0 (no gradient through M)
    return (row_w * (torch.logsumexp(den, 1) - torch.logsumexp(num, 1))).sum() / len(row_w)


# ============================================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--features-dir", required=True, help="fp16 .npy bags (stage_bags.py) or TRIDENT .h5")
    p.add_argument("--txt-feats", required=True, help="tcga_reports_<model>.pt with sent_features / sent_offsets")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--cache-patches", type=int, default=1024, help="patches per slide kept for the E/M-steps")
    p.add_argument("--cache-device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--resample-cache", action="store_true", help="new random patch subset at every round")
    p.add_argument("--min-sent-chars", type=int, default=20, help="drop shorter sentences")
    # E-step (soft)
    p.add_argument("--assign", choices=["soft", "posterior"], default="soft",
                   help="soft = local-FDR weights from the decoys; posterior = softmax over patches + NULL (ablation)")
    p.add_argument("--decoys", choices=["other-organ", "same-organ", "random"], default="other-organ")
    p.add_argument("--n-decoys", type=int, default=10, help="decoy slides per sentence's slide")
    p.add_argument("--lfdr-window", type=int, default=8, help="h: ranks on each side for the slope of F (soft)")
    p.add_argument("--posterior-temperature", type=float, default=None,
                   help="tau of --assign posterior (default: the InfoNCE temperature)")
    p.add_argument("--fdr-plus-one", action="store_true", help="conservative (#decoys + 1) counts")
    p.add_argument("--min-weight", type=float, default=0.05, help="candidates need w > this")
    p.add_argument("--max-candidates", type=int, default=64, help="max candidate patches per sentence")
    p.add_argument("--sent-weight", choices=["max", "noisy-or", "none"], default="max",
                   help="sentence weight s (posterior: always 1 - q(NULL)); none = 1 for every sentence with a candidate")
    p.add_argument("--min-sent-weight", type=float, default=0.5,
                   help="sentences with s below are soft-NULL (0.5: pure-noise sentences reach it in ~1-5%% of cases)")
    p.add_argument("--fdr", type=float, default=0.1, help="q of the hard rule, only logged for comparison")
    # M-step
    p.add_argument("--rounds", type=int, default=4, help="EM rounds (one E-step each)")
    p.add_argument("--m-epochs", type=int, default=2, help="passes over the sentences per M-step")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--sents-per-slide", type=int, default=4, help="max sentences of one slide per batch")
    p.add_argument("--batch-organ", choices=["mixed", "same"], default="mixed",
                   help="same = M-step batches drawn within one organ (hard, same-organ negatives)")
    p.add_argument("--same-organ-frac", type=float, default=1.0,
                   help="with --batch-organ same: fraction of the pairs put in same-organ batches (rest: mixed)")
    p.add_argument("--positives", choices=["concept", "witness", "single"], default="concept")
    p.add_argument("--concepts", choices=["embed", "exact", "none"], default="embed")
    p.add_argument("--concept-threshold", type=float, default=0.95, help="cosine of frozen sentence embeddings")
    p.add_argument("--weighting", action="store_true", help="LIMIT-style weights on the non-positive pairs")
    p.add_argument("--temperature", type=float, default=None, help="InfoNCE temperature (default: init checkpoint's)")
    # Quilt prior / regularisation
    p.add_argument("--pairs", default=None, help="Quilt pairs file; required if --quilt-weight > 0")
    p.add_argument("--quilt-weight", type=float, default=1.0)
    p.add_argument("--quilt-batch-size", type=int, default=1024)
    p.add_argument("--l2sp", type=float, default=0.0, help="weight of ||theta - theta_init||^2")
    # text model
    p.add_argument("--init-txt-from", default=None, help="train_projection-format project=text checkpoint (Quilt run)")
    p.add_argument("--txt-depth", type=int, default=2)
    p.add_argument("--txt-width", type=int, default=1024)
    p.add_argument("--no-txt-norm", action="store_true")
    # optimisation
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--select", choices=["combined", "val_auc", "quilt_val", "last"], default="combined")
    # monitoring
    p.add_argument("--test-feats", default=None)
    p.add_argument("--txt-model", default=None)
    p.add_argument("--test-modes", default="resizecrop,resize,tile")
    p.add_argument("--arch-feats", nargs="*", default=[])
    p.add_argument("--no-slide-zs", action="store_true")
    p.add_argument("--slide-zs-ks", default="1,10,50,100,0")
    p.add_argument("--pancancer-min", type=int, default=10, help="min slides of a project in pan-cancer slide zero-shot")
    # misc
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--runs-dir", default="runs_o1b")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    if args.test_feats and not args.txt_model:
        raise SystemExit("--test-feats needs --txt-model")
    if args.quilt_weight > 0 and not args.pairs:
        raise SystemExit("--quilt-weight > 0 needs --pairs (or --quilt-weight 0)")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible to PyTorch; --allow-cpu to force")
    cache_device = args.cache_device if device == "cuda" else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name() if device == 'cuda' else 'CPU'})", flush=True)
    run_dir = unique_dir(Path(args.runs_dir) / f"{args.run_name or 'o1b_soft'}_{datetime.now():%m%d-%H%M}")
    print(f"run dir: {run_dir}", flush=True)

    # ------------------------------------------------------------ slides + sentences
    txt_all = torch.load(args.txt_feats, map_location="cpu")
    for k in ("sent_features", "sent_offsets", "sent_texts"):
        if k not in txt_all:
            raise SystemExit(f"{args.txt_feats} has no '{k}': re-run extract_report_features.py")
    bags = TCGABags(args.features_dir, txt_all, "all", val_frac=args.val_frac, test_frac=args.test_frac)
    if len(bags) == 0:
        raise SystemExit("no slide has both a feature file and a report")
    S = len(bags)
    split = np.array([split_of(b, args.val_frac, args.test_frac) for b in bags.barcodes])
    idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
    organs = [organ_of(pj) for pj in bags.projects]
    row_of = {s: i for i, s in enumerate(txt_all["slide_ids"])}
    off = txt_all["sent_offsets"]
    sf, stx = txt_all["sent_features"], txt_all["sent_texts"]
    keep_rows, slide_sents, sent_slide, n_short = [], [], [], 0
    for i, sid in enumerate(bags.slide_ids):
        r = row_of[sid]
        start = len(keep_rows)
        for j in range(int(off[r]), int(off[r + 1])):
            if len(stx[j].strip()) >= args.min_sent_chars:
                keep_rows.append(j)
            else:
                n_short += 1
        slide_sents.append((start, len(keep_rows)))
        sent_slide += [i] * (len(keep_rows) - start)
    sent_feat_cpu = sf[torch.tensor(keep_rows, dtype=torch.long)].float()
    sent_texts = [stx[j] for j in keep_rows]
    sent_slide = np.array(sent_slide, dtype=np.int64)
    sent_split = split[sent_slide] if len(sent_slide) else np.array([])
    txt_model_name = txt_all.get("model")
    del txt_all, sf
    sent_feat = sent_feat_cpu.to(device)
    img_dim, txt_dim = bags.img_dim, sent_feat.shape[1]
    n_sent = {s: int((sent_split == s).sum()) for s in ("train", "val", "test")}
    print(f"slides  train {len(idx['train']):,}  val {len(idx['val']):,}  test {len(idx['test']):,}  |  sentences "
          f"train {n_sent['train']:,}  val {n_sent['val']:,}  ({n_short:,} shorter than {args.min_sent_chars} chars "
          f"dropped)  {len(sent_texts) / S:.1f} per slide", flush=True)

    # ------------------------------------------------------------ concepts (training sentences)
    concept = torch.arange(len(sent_texts))
    if args.positives == "concept" and args.concepts != "none":
        t0 = time.time()
        tr = np.flatnonzero(sent_split == "train")
        c_tr = build_concepts([sent_texts[j] for j in tr], sent_feat_cpu[tr], args.concepts,
                              args.concept_threshold, device)
        concept = torch.arange(len(sent_texts)) + int(c_tr.max()) + 1 if len(tr) else concept   # non-train: own id
        concept[torch.from_numpy(tr)] = c_tr
        sizes, top = write_concepts(run_dir / "concepts.csv", c_tr, [sent_texts[j] for j in tr])
        print(f"concepts ({args.concepts}, thr {args.concept_threshold}): {len(sizes):,} concepts for "
              f"{len(tr):,} training sentences in {time.time() - t0:.0f}s; {int((sizes > 1).sum()):,} with >1 "
              f"sentence, covering {sizes[sizes > 1].sum() / max(1, sizes.sum()):.1%} of sentences; largest:", flush=True)
        for n_, t_ in top:
            print(f"    {n_:6d}  {t_[:110]}")
    concept_dev = concept.to(device)

    # ------------------------------------------------------------ patch cache
    t0 = time.time()
    est_gb = S * args.cache_patches * img_dim * 2 / 1e9
    print(f"building patch cache: {S:,} slides x {args.cache_patches} patches ({est_gb:.1f} GB, {cache_device})",
          flush=True)
    cache, counts = build_cache(bags, args.cache_patches, args.seed, 0, cache_device, args.num_workers)
    counts_l = counts.tolist()
    print(f"  done in {time.time() - t0:.0f}s; patches per slide: median {int(np.median(counts_l))}, "
          f"min {min(counts_l)}", flush=True)
    keyspace = S * args.cache_patches

    # ------------------------------------------------------------ model
    cfg = dict(img_dim=img_dim, txt_dim=txt_dim, project="text", embed_dim=None, depth=args.txt_depth,
               width=args.txt_width, norm=not args.no_txt_norm, dropout=0.0, init_temperature=0.07,
               learn_temperature=True)
    init = None
    if args.init_txt_from:
        init = torch.load(args.init_txt_from, map_location="cpu")
        c = init["model_cfg"]
        if c.get("project") != "text" or (c["img_dim"], c["txt_dim"]) != (img_dim, txt_dim):
            raise SystemExit(f"--init-txt-from must be a project=text checkpoint with dims {img_dim}/{txt_dim}")
        if init.get("txt_model") and txt_model_name and init["txt_model"] != txt_model_name:
            raise SystemExit(f"--init-txt-from was trained with {init['txt_model']}, reports use {txt_model_name}")
        cfg.update({k: c[k] for k in ("depth", "width", "norm", "dropout") if k in c})
    else:
        print("WARNING: no --init-txt-from: the first E-step uses a random MLP, its weights are noise")
    T = args.temperature
    if T is None:
        T = 1.0 / float(init["state_dict"]["logit_scale"].exp()) if init is not None and \
            "logit_scale" in init["state_dict"] else 0.07
    tau = args.posterior_temperature or T
    cfg["init_temperature"] = T
    model = ProjectionModel(**cfg).to(device)
    if init is not None:
        model.load_state_dict(init["state_dict"])
        print(f"text MLP initialised from {args.init_txt_from} (depth {cfg['depth']}, width {cfg['width']})")
    with torch.no_grad():
        model.logit_scale.fill_(math.log(1.0 / T))
    model.logit_scale.requires_grad_(False)                                  # fixed temperature
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    theta0 = [q.detach().clone() for q in model.txt_proj.parameters()] if args.l2sp > 0 else None

    quilt = quilt_val = q_stream = None
    if args.quilt_weight > 0:
        quilt = QuiltPairs(args.pairs, device=device)
        if quilt.dims != (img_dim, txt_dim):
            raise SystemExit(f"Quilt pairs dims {quilt.dims} != TCGA dims {(img_dim, txt_dim)}")
        if quilt.meta.get("txt_model") and txt_model_name and quilt.meta["txt_model"] != txt_model_name:
            raise SystemExit(f"Quilt captions encoded with {quilt.meta['txt_model']}, reports with {txt_model_name}")
        quilt_val = quilt.retrieval_set("val") if len(quilt.images["val"]) else None
        args.quilt_batch_size = min(args.quilt_batch_size, len(quilt.images["train"]))
        q_stream = quilt_batches(quilt, args.quilt_batch_size)

    config = {"args": vars(args), "model_cfg": cfg, "temperature": T, "posterior_tau": tau,
              "data": {"n_slides": {k: len(v) for k, v in idx.items()}, "n_sentences": n_sent,
                       "txt_model": txt_model_name, "cache_gb": est_gb},
              "env": {"torch": torch.__version__, "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    e_desc = (f"soft (local FDR, window {args.lfdr_window})" if args.assign == "soft"
              else f"posterior (tau {tau:.4f})")
    print(f"E-step {e_desc}: {args.n_decoys} {args.decoys} decoys, w > {args.min_weight}, <= {args.max_candidates} "
          f"candidates, sentence weight {args.sent_weight if args.assign == 'soft' else '1 - q(NULL)'} "
          f"(soft NULL < {args.min_sent_weight})  |  positives {args.positives}  weighting {args.weighting}  "
          f"T {T:.4f}  lr {args.lr}  batch {args.batch_size} ({args.batch_organ} organ)  |  Quilt weight "
          f"{args.quilt_weight}", flush=True)

    # ------------------------------------------------------------ monitoring
    encoder = load_text_encoder(args.txt_model, device) if args.txt_model else None
    modes = [m.strip() for m in args.test_modes.split(",") if m.strip()]
    zs_ks = tuple(int(k) for k in args.slide_zs_ks.split(","))
    val_samplers = {m: DecoySampler(idx["val"], organs, m, args.n_decoys, args.seed + 1)
                    for m in ("other-organ", "same-organ")}

    def val_stats():
        zt = encode_sentences(model, sent_feat)
        out = {}
        for m, smp in val_samplers.items():
            smp.rng = np.random.default_rng(args.seed + 1)                     # same decoys every round
            e = soft_e_step(zt, cache, counts_l, slide_sents, idx["val"], smp, args, device, tau)
            tag = "other" if m == "other-organ" else "same"
            out[f"val_auc_{tag}"] = float(np.nanmean(e["auc"].numpy())) if len(e["auc"]) else float("nan")
            if m == args.decoys and len(e["s"]):
                live = e["s"] >= args.min_sent_weight
                out["val_null_rate"] = float((~live).float().mean())          # soft NULL
                out["val_sent_weight_mean"] = float(e["s"][live].mean()) if live.any() else 0.0
                out["val_expected_witnesses"] = float(e["sum_w"][live].mean()) if live.any() else 0.0
                out["val_hard_null_rate"] = float((e["k_fdr"] == 0).float().mean())
        return out

    def patch_eval(tag):
        out = {}
        model.eval()
        with torch.no_grad():
            if quilt_val is not None:
                out.update({f"quilt_val_{k}": v for k, v in retrieval_metrics(model, *quilt_val).items()})
            if encoder is not None and args.test_feats:
                d = run_dir / "zeroshot" / f"round{tag}"
                d.mkdir(parents=True, exist_ok=True)
                prompts = {}
                zs = zero_shot(model, encoder, args.test_feats, modes, device, d, prompts)
                (d / "prompts.json").write_text(json.dumps(prompts, indent=1))
                out.update(zs_flat(zs))
            if args.arch_feats:
                retr = arch_retrieval(model, args.arch_feats, txt_model_name, device)
                out.update({f"{n}_{k}": v for n, r in retr.items() for k, v in r.items() if "R@" in k})
        return out

    def slide_zs(split_name, tag):
        if encoder is None or args.no_slide_zs or len(idx[split_name]) == 0:
            return {}
        d = run_dir / "slide_zs" / f"{split_name}_round{tag}"
        d.mkdir(parents=True, exist_ok=True)
        slides = ((int(i), cache[i, :counts_l[i]]) for i in idx[split_name] if counts_l[i] > 0)
        res = slide_zero_shot(model, encoder, slides, bags.projects, device, zs_ks, args.pancancer_min, out_dir=d)
        (d / "results.json").write_text(json.dumps(res, indent=2))
        return {k.replace("slide_zs_", f"slide_zs_{split_name}_"): v for k, v in flat_slide_zs(res).items()}

    def score(row):
        a, q = row.get("val_auc_same", -1.0), row.get("quilt_val_mean_recall", -1.0)
        if args.select == "combined" and quilt_val is None:
            return a
        return {"val_auc": a, "quilt_val": q, "combined": 0.5 * (a + q), "last": 0.0}[args.select]

    def save(path, rnd, metrics):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "model_cfg": cfg,
                    "epoch": rnd, "val": metrics, "txt_model": txt_model_name, "pairs": args.pairs,
                    "source": "train_o1b_soft.py"}, path)

    log_rows = []

    def write_log(row):
        log_rows.append(row)
        fields = []
        for r_ in log_rows:
            fields += [k for k in r_ if k not in fields]
        with open(run_dir / "train_log.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(log_rows)

    def summarise(prefix, row):
        names = [("val_auc_other", "val AUC other"), ("val_auc_same", "same"), ("val_null_rate", "val soft NULL"),
                 ("quilt_val_mean_recall", "quilt val R"), ("zs_mean_bal_acc", "patch zs"),
                 ("slide_zs_val_mean_top10", "slide zs (top10)")]
        return prefix + "  " + "  ".join(f"{n} {row[k]:.4f}" for k, n in names if k in row)

    def evaluate_round(rnd):
        return {**val_stats(), **patch_eval(rnd), **slide_zs("val", rnd)}

    # ------------------------------------------------------------ round 0
    print("\nround 0 (initial model):", flush=True)
    row = {"round": 0, **evaluate_round(0)}
    write_log(row)
    print(summarise("round   0", row), flush=True)
    best, best_round = score(row), 0
    if args.select != "last":
        save(run_dir / "model_best.pt", 0, row)

    # ------------------------------------------------------------ EM
    train_sampler = DecoySampler(idx["train"], organs, args.decoys, args.n_decoys, args.seed)
    organ_ids = {o: k for k, o in enumerate(sorted(set(organs)))}
    slide_organ_np = np.array([organ_ids[o] for o in organs], dtype=np.int64)
    slide_organ = torch.from_numpy(slide_organ_np).to(device)
    same_frac = args.same_organ_frac if args.batch_organ == "same" else 0.0
    t_start = time.time()
    for rnd in range(1, args.rounds + 1):
        if args.resample_cache:
            t0 = time.time()
            cache, counts = build_cache(bags, args.cache_patches, args.seed, rnd, cache_device, args.num_workers,
                                        out=cache, counts=counts)
            counts_l = counts.tolist()
            print(f"  patch cache resampled in {time.time() - t0:.0f}s", flush=True)
        # ---------------- E-step
        t0 = time.time()
        zt = encode_sentences(model, sent_feat)
        e = soft_e_step(zt, cache, counts_l, slide_sents, idx["train"], train_sampler, args, device, tau)
        s_all, cand = e["s"], e["cand"]
        live = s_all >= args.min_sent_weight
        torch.save({"cand_sent": cand["sent"], "cand_slide": cand["slide"], "cand_patch": cand["patch"],
                    "cand_w": cand["w"], "sent": e["sent"], "sent_weight": s_all, "sum_w": e["sum_w"],
                    "n_cand": e["n_cand"], "k_fdr": e["k_fdr"], "own_max": e["own_max"], "assign": args.assign,
                    "cache_round": rnd if args.resample_cache else 0, "slide_ids": bags.slide_ids},
                   run_dir / f"witnesses_round{rnd}.pt")
        with open(run_dir / f"estep_round{rnd}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sentence", "slide_id", "project", "n_candidates", "sum_w", "sent_weight", "max_w",
                        "hard_k_fdr", "best_score", "auc_decoys", "text"])
            for j, nc, sw, sv, mw, kf, om, au in zip(e["sent"].tolist(), e["n_cand"].tolist(), e["sum_w"].tolist(),
                                                     s_all.tolist(), e["max_w"].tolist(), e["k_fdr"].tolist(),
                                                     e["own_max"].tolist(), e["auc"].tolist()):
                si = sent_slide[j]
                w.writerow([j, bags.slide_ids[si], bags.projects[si], nc, round(sw, 3), round(sv, 3), round(mw, 3),
                            kf, round(om, 4), round(au, 3), sent_texts[j]])
        k_hard = e["k_fdr"]
        e_row = {"soft_null_rate": float((~live).float().mean()),
                 "sent_weight_mean": float(s_all[live].mean()) if live.any() else 0.0,
                 "expected_witnesses_mean": float(e["sum_w"][live].mean()) if live.any() else 0.0,
                 "candidates_mean": float(e["n_cand"][live].float().mean()) if live.any() else 0.0,
                 "hard_null_rate": float((k_hard == 0).float().mean()),
                 "hard_witnesses_mean": float(k_hard[k_hard > 0].clamp(max=args.max_candidates).float().mean())
                 if (k_hard > 0).any() else 0.0,
                 "n_train_sentences": int(live.sum()), "n_candidates": int(len(cand["sent"])),
                 "estep_s": round(time.time() - t0)}
        print(f"\nround {rnd} E-step: {len(s_all):,} sentences, soft NULL {e_row['soft_null_rate']:.1%} "
              f"(hard rule: {e_row['hard_null_rate']:.1%}), sentence weight {e_row['sent_weight_mean']:.2f}, "
              f"expected witnesses {e_row['expected_witnesses_mean']:.1f} (hard k* {e_row['hard_witnesses_mean']:.1f}), "
              f"candidates {e_row['candidates_mean']:.1f}  ({e_row['estep_s']}s)", flush=True)
        order = torch.argsort(e["sum_w"], descending=True)
        for lab, js in (("most witnesses", order[:3]), ("soft NULL", order[~live[order]][:3])):
            for j in js.tolist():
                print(f"    [{lab}] sum w {e['sum_w'][j]:5.1f}  s {s_all[j]:.2f}  "
                      f"{bags.projects[sent_slide[e['sent'][j]]]:10s} {sent_texts[e['sent'][j]][:90]}")
        train_sents = e["sent"][live].numpy()
        if len(train_sents) < args.batch_size:
            print(f"  only {len(train_sents)} sentences above the weight threshold (< batch size {args.batch_size}): "
                  f"stopping")
            write_log({"round": rnd, **e_row})
            break

        # candidate tables on the GPU (sorted by sentence)
        cs, csl, cp, cw = (cand[k].to(device) for k in ("sent", "slide", "patch", "w"))
        o = torch.argsort(cs, stable=True)
        cs, csl, cp, cw = cs[o], csl[o], cp[o], cw[o]
        c_cnt = torch.bincount(cs, minlength=len(sent_texts))
        c_off = torch.zeros(len(sent_texts) + 1, dtype=torch.long, device=device)
        c_off[1:] = torch.cumsum(c_cnt, 0)
        cum = torch.cumsum(cw.double(), 0)
        cum_pad = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), cum])
        sent_w = torch.zeros(len(sent_texts), device=device)
        sent_w[e["sent"].to(device)] = s_all.to(device)
        keys = concept_dev[cs] * keyspace + csl * args.cache_patches + cp
        ck_keys, inv = torch.unique(keys, return_inverse=True)                      # sorted
        ck_w = torch.zeros(len(ck_keys), device=device).scatter_reduce(0, inv, cw, "amax", include_self=False)

        # ---------------- M-step
        model.train()
        losses, qls, regs, pos_mass, pos_frac, roww, same_pairs, t0 = [], [], [], [], [], [], [], time.time()
        for ep in range(args.m_epochs):
            batches, _ = make_epoch_batches(train_sents, sent_slide, slide_organ_np, args.sents_per_slide,
                                            args.batch_size, rng, same_frac)
            for sb in batches:
                sb = torch.from_numpy(sb).to(device)
                base, total = cum_pad[c_off[sb]], cum_pad[c_off[sb] + c_cnt[sb]] - cum_pad[c_off[sb]]
                u = base + torch.rand(len(sb), device=device, dtype=torch.float64) * total
                pick = torch.searchsorted(cum, u, right=True)                   # patch drawn proportionally to w
                pick = torch.minimum(torch.maximum(pick, c_off[sb]), c_off[sb] + c_cnt[sb] - 1)
                slide_b, patch_b = csl[pick], cp[pick]
                img = cache[slide_b.to(cache.device), patch_b.to(cache.device)].to(device).float()
                key_b = slide_b * args.cache_patches + patch_b
                M = weight_mask(sb, key_b, concept_dev, ck_keys, ck_w, keyspace,
                                "single" if args.positives == "single" else "concept")
                rw = sent_w[sb]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    z_t = F.normalize(model.txt_proj(sent_feat[sb]).float(), dim=-1)
                    logits = (z_t @ img.t()) / T
                    logw = None
                    if args.weighting:
                        with torch.no_grad():
                            logw = torch.log((1.0 - (z_t @ z_t.t() + 1.0) / 2.0).clamp_min(1e-6))
                    loss = 0.5 * (soft_pos_nce(logits, M, rw, logw) +
                                  soft_pos_nce(logits.t(), M.t(), rw, None if logw is None else logw.t()))
                    l_q = torch.zeros((), device=device)
                    if q_stream is not None:
                        q_img, q_txt = next(q_stream)
                        l_q = clip_loss(F.normalize(q_img.float(), dim=-1),
                                        F.normalize(model.txt_proj(q_txt).float(), dim=-1), 1.0 / T)
                    total_loss = loss + args.quilt_weight * l_q
                reg = torch.zeros((), device=device)
                if theta0 is not None:
                    reg = sum(((q - q0) ** 2).sum() for q, q0 in zip(model.txt_proj.parameters(), theta0))
                    total_loss = total_loss + args.l2sp * reg
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(loss.item())
                qls.append(l_q.item())
                regs.append(float(reg))
                B = len(sb)
                pos_mass.append(float(M.sum(1).mean()))
                pos_frac.append(float(((M > 0).sum() - B) / max(1, B * B - B)))
                roww.append(float(rw.mean()))
                ob = slide_organ[slide_b]
                same_pairs.append(float(((ob[:, None] == ob[None]).sum() - B) / max(1, B * B - B)))
            print(f"  M-step epoch {ep + 1}/{args.m_epochs}: {len(losses)} steps  loss {np.mean(losses):.4f}  "
                  f"quilt {np.mean(qls):.4f}  positive mass/row {np.mean(pos_mass):.2f}  "
                  f"off-diag positives {np.mean(pos_frac):.2%}  sentence weight {np.mean(roww):.2f}  "
                  f"same-organ pairs {np.mean(same_pairs):.1%}  ({time.time() - t0:.0f}s)", flush=True)

        row = {"round": rnd, **e_row, "loss": float(np.mean(losses)) if losses else float("nan"),
               "loss_quilt": float(np.mean(qls)) if qls else float("nan"),
               "l2sp": float(np.mean(regs)) if regs else 0.0,
               "positive_mass_per_row": float(np.mean(pos_mass)) if pos_mass else float("nan"),
               "offdiag_positive_frac": float(np.mean(pos_frac)) if pos_frac else float("nan"),
               "batch_sent_weight_mean": float(np.mean(roww)) if roww else float("nan"),
               "same_organ_pair_frac": float(np.mean(same_pairs)) if same_pairs else float("nan"),
               "m_steps": len(losses), "mstep_s": round(time.time() - t0), **evaluate_round(rnd)}
        write_log(row)
        print(summarise(f"round {rnd:3d}  loss {row['loss']:.4f}  quilt {row['loss_quilt']:.4f}", row), flush=True)
        if args.select == "last" or score(row) > best:
            best, best_round = score(row), rnd
            save(run_dir / "model_best.pt", rnd, row)
    save(run_dir / "model_last.pt", log_rows[-1]["round"], log_rows[-1])

    # ------------------------------------------------------------ selected checkpoint: final evaluation
    ck = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    print(f"\nselected round {best_round} ({args.select}); final evaluation:", flush=True)
    final = {**patch_eval("best"), **slide_zs("test", "best")}
    (run_dir / "results.json").write_text(json.dumps(
        {"selected_round": best_round, "select": args.select, "val": ck["val"], "final": final,
         "train_time_s": round(time.time() - t_start)}, indent=2, default=float))

    summary = {"run": run_dir.name,
               "init_txt_from": Path(args.init_txt_from).parent.name if args.init_txt_from else "",
               "assign": args.assign, "decoys": args.decoys, "n_decoys": args.n_decoys,
               "lfdr_window": args.lfdr_window if args.assign == "soft" else "",
               "posterior_tau": round(tau, 4) if args.assign == "posterior" else "",
               "min_weight": args.min_weight, "max_candidates": args.max_candidates,
               "sent_weight": args.sent_weight if args.assign == "soft" else "1-qnull",
               "min_sent_weight": args.min_sent_weight, "positives": args.positives,
               "concepts": args.concepts if args.positives == "concept" else "", "weighting": args.weighting,
               "batch_organ": args.batch_organ if args.batch_organ == "mixed" else f"same{args.same_organ_frac:g}",
               "quilt_weight": args.quilt_weight, "l2sp": args.l2sp, "lr": args.lr, "rounds": args.rounds,
               "m_epochs": args.m_epochs, "seed": args.seed, "select": args.select, "selected_round": best_round,
               **{k: round(ck["val"][k], 4) for k in ("val_auc_other", "val_auc_same", "val_null_rate",
                                                      "val_hard_null_rate", "quilt_val_mean_recall")
                  if k in ck["val"]},
               **{k: round(v, 4) for k, v in final.items()}}
    patch_cols = patch_summary_cols()
    slide_cols = [f"slide_zs_test_{t}_{'mean' if k == 0 else f'top{k}'}" for t in ("lung", "kidney", "brain",
                                                                                    "pancancer", "mean") for k in zs_ks]
    base = [k for k in summary if not k.startswith(("zs_", "arch_", "slide_zs_", "quilt_val_i", "quilt_val_t"))]
    append_summary(Path(args.runs_dir) / "summary_o1b_soft.csv", summary, base + patch_cols + slide_cols)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
