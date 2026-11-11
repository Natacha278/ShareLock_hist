#!/usr/bin/env python
"""
Simplified Objective 1-B (MLP version): EM alignment of report SENTENCES with their witness PATCHES,
with a decoy-calibrated (target-decoy FDR) E-step and a NULL option, and an MLP M-step.

Only the text MLP (BioLORD -> UNI2-h space, ShareLock-style, project="text") is trained; UNI2-h patch
features stay frozen and L2-normalised, so every checkpoint is a plain train_projection-format file
(eval_projection.py / eval_slide_zeroshot.py work on it directly).

Setup (once)
  * patch cache: a fixed random subset of --cache-patches patches per slide (train/val/test), L2-normalised,
    fp16, on the GPU (~28 GB for 9k slides x 1024 patches; --cache-device cpu if short on GPU memory).
    Witness ids index this cache. --resample-cache draws a new subset at every round.
  * sentences: sent_features / sent_offsets from extract_report_features.py; sentences shorter than
    --min-sent-chars are dropped.
  * concepts: equivalent sentences across reports are grouped (--concepts embed: leader clustering of the
    frozen sentence embeddings at cosine > --concept-threshold; exact: identical text after lower-casing and
    removing punctuation; none). Saved to concepts.csv (largest first) so the threshold can be checked.

Round r = 1..--rounds
  E-step (current MLP, no gradient), for every training sentence t of slide i:
    target scores  s(t, p) for the cached patches p of slide i
    decoy scores   s(t, p') for the patches of --n-decoys slides of OTHER organs (--decoys), where the
                   sentence should have no true witness
    estimated FDR of the top-k target patches = (#decoy scores >= k-th target score) * N_target/N_decoy / k
    witnesses = the largest top-k with FDR <= --fdr, capped at --max-witnesses; k = 0 -> NULL (the sentence
    is not visually grounded in this slide, e.g. clinical history, margins, lymph-node counts)
    (--select-witnesses topk: top --max-witnesses patches for every sentence, no decoys, no NULL = ablation)
  M-step: --m-epochs passes of CLIP-style training on pseudo-pairs (sentence, one random witness), with
    false-negative handling:
      (a) multi-positive InfoNCE (MIL-NCE / SupCon style): in a batch, patch b is a positive of sentence a
          whenever b is a witness of a  (--positives witness), or of any sentence of a's concept
          (--positives concept, default; includes (b)); --positives single = plain CLIP (ablation)
      (c) --weighting: LIMIT-style down-weighting of the remaining negatives, w = 1 - (cos(t_a, t_b) + 1) / 2
          on the current text embeddings, no gradient
      (d) batches: one pair per sentence, at most --sents-per-slide sentences of a slide per batch
      --loss sigmoid: SigLIP-style loss with positives (the mask), trusted negatives (pairs from different
          organs; --sigmoid-negatives all = every non-positive) and IGNORED same-organ unknown pairs
    plus the Quilt prior: + --quilt-weight * CLIP loss on Quilt-1M pairs (as train_limit_mix.py), and an
    optional L2-SP penalty --l2sp * ||theta - theta_init||^2.
    The InfoNCE temperature is fixed (the init checkpoint's, as for the Quilt loss).

Monitoring at round 0 (initial model) and after every round:
  E-step statistics (NULL rate, witnesses per sentence, estimated FDR), batch statistics (positives per row),
  validation sentences: NULL rate with the same rule, and AUC own-slide vs decoy-slide (best-patch score)
  against other-organ and same-organ decoys; Quilt val retrieval; patch zero-shot (BACH/CRC/MHIST/
  BreakHis) + ARCH; slide-level zero-shot on validation slides (eval_slide_zeroshot.py; on the cache).
  Model selection (--select): combined = mean(val AUC same-organ, Quilt val mean recall) -- validation
  splits only, never the benchmarks; or val_auc / quilt_val / last.

Outputs in <runs-dir>/<run-name>_<MMDD-HHMM>/:
  config.json, train_log.csv, concepts.csv, estep_round<r>.csv (one row per training sentence: text,
  witnesses, FDR, best score), witnesses_round<r>.pt, model_best.pt / model_last.pt (train_projection format),
  results.json (incl. slide zero-shot on the TEST slides); one row appended to <runs-dir>/summary_o1b.csv.

Example (start from the Quilt projection):
  python train_o1b_em.py --features-dir $SLURM_TMPDIR/bags --txt-feats .../tcga_reports_BioLORD-2023.pt \
    --pairs .../quilt_pairs_t05_biolord.pt --init-txt-from .../runs/<quilt run>/model_best.pt \
    --test-feats .../features/uni2h --txt-model .../weights/BioLORD-2023 --arch-feats ... \
    --runs-dir .../runs_o1b --run-name o1b
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import csv
import json
import math
import platform
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_slide_zeroshot import flat as flat_slide_zs
from eval_slide_zeroshot import slide_zero_shot
from projection import ProjectionModel, clip_loss
from quilt_pairs import QuiltPairs
from tcga_bags import TCGABags, split_of
from train_limit import organ_of, set_seed
from train_limit_mix import quilt_batches
from train_projection import SUMMARY_COLS, arch_retrieval, load_text_encoder, param_groups, retrieval_metrics, zero_shot
from train_slide_text import unique_dir


# ============================================================================ patch cache
class FixedSubset(Dataset):
    """Patches of slide i: a random subset of n patches, reproducible from (seed, round, i)."""

    def __init__(self, bags, n, seed, round_=0):
        self.bags, self.n, self.seed, self.round = bags, n, seed, round_
        self.bags_dim = bags.img_dim

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        x = self.bags.read(i, mmap=True)
        if len(x) > self.n:
            rng = np.random.default_rng([self.seed, self.round, i])
            x = x[np.sort(rng.choice(len(x), self.n, replace=False))]
        x = np.array(x, dtype=np.float32)
        if x.ndim != 2 or len(x) == 0:
            x = np.zeros((0, self.bags_dim), dtype=np.float32)
        return torch.from_numpy(x), i


def build_cache(bags, n, seed, round_, device, num_workers, out=None, counts=None):
    """-> cache [S, n, D] fp16 (L2-normalised rows, zero padding), counts [S] (long, CPU)."""
    ds = FixedSubset(bags, n, seed, round_)
    S, D = len(bags), ds.bags_dim
    if out is None:
        out = torch.zeros(S, n, D, dtype=torch.float16, device=device)
        if device == "cpu" and torch.cuda.is_available():
            out = out.pin_memory()
        counts = torch.zeros(S, dtype=torch.long)
    else:
        out.zero_()
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers, collate_fn=lambda b: b[0])
    t0 = time.time()
    for k, (x, i) in enumerate(dl, 1):
        counts[i] = len(x)
        if len(x):
            out[i, :len(x)] = F.normalize(x.to(out.device), dim=-1).half()
        if k % 1000 == 0:
            print(f"  cache: {k}/{S} slides ({time.time() - t0:.0f}s)", flush=True)
    return out, counts


# ============================================================================ concepts
def normalise_text(t):
    return " ".join(re.sub(r"[^\w\s]", " ", t.lower()).split())


@torch.no_grad()
def build_concepts(texts, feats, mode, thr, device, chunk=4096):
    """Concept id per sentence. feats [S, D] frozen sentence embeddings."""
    if mode == "none":
        return torch.arange(len(texts))
    norm = [normalise_text(t) for t in texts]
    uniq, first, inverse = {}, [], []
    for j, t in enumerate(norm):
        if t not in uniq:
            uniq[t] = len(uniq)
            first.append(j)
        inverse.append(uniq[t])
    inverse = torch.tensor(inverse)
    if mode == "exact":
        return inverse
    U = len(first)
    counts = torch.bincount(inverse, minlength=U)
    order = torch.argsort(counts, descending=True, stable=True)              # frequent texts become leaders
    X = F.normalize(feats[torch.tensor(first)][order].float().to(device), dim=-1)
    concept_of_sorted = torch.full((U,), -1, dtype=torch.long)
    L = X[:0]
    for s in range(0, U, chunk):
        c = X[s:s + chunk]
        assign = torch.full((len(c),), -1, dtype=torch.long)
        if len(L):
            best, arg = (c @ L.t()).max(1)
            hit = (best > thr).cpu()
            assign[hit] = arg.cpu()[hit]
        rest = (assign < 0).nonzero().squeeze(1)
        if len(rest):
            R = c[rest.to(device)]
            Sm = (R @ R.t()).cpu().numpy()
            new = []
            for j in range(len(rest)):
                if new:
                    sims = Sm[j, new]
                    b = int(sims.argmax())
                    if sims[b] > thr:
                        assign[rest[j]] = len(L) + b
                        continue
                assign[rest[j]] = len(L) + len(new)
                new.append(j)
            L = torch.cat([L, R[torch.tensor(new, device=device)]])
        concept_of_sorted[s:s + len(c)] = assign
    concept_of_unique = torch.empty(U, dtype=torch.long)
    concept_of_unique[order] = concept_of_sorted
    return concept_of_unique[inverse]


def write_concepts(path, concept, texts, top=2000):
    by = {}
    for j, c in enumerate(concept.tolist()):
        by.setdefault(c, []).append(j)
    rows = sorted(by.items(), key=lambda kv: -len(kv[1]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept", "n_sentences", "n_unique_texts", "most_frequent_text", "other_examples"])
        for c, js in rows[:top]:
            cnt = Counter(normalise_text(texts[j]) for j in js)
            common = [t for t, _ in cnt.most_common(4)]
            w.writerow([c, len(js), len(cnt), common[0], " || ".join(common[1:])])
    sizes = np.array([len(js) for _, js in rows])
    return sizes, [(len(js), texts[js[0]]) for _, js in rows[:5]]


# ============================================================================ E-step
class DecoySampler:
    def __init__(self, pool, organs, mode, n, seed):
        self.mode, self.n, self.rng = mode, n, np.random.default_rng(seed)
        self.pool = np.asarray(pool)
        org = np.array([organs[i] for i in self.pool])
        self.same = {o: self.pool[org == o] for o in set(org)}
        self.other = {o: self.pool[org != o] for o in set(org)}
        self.organs = organs

    def __call__(self, i):
        o = self.organs[i]
        if self.mode == "same-organ":
            cand = self.same.get(o, self.pool)
        elif self.mode == "other-organ":
            cand = self.other.get(o, self.pool)
        else:
            cand = self.pool
        cand = cand[cand != i]
        if len(cand) == 0:
            return np.empty(0, dtype=np.int64)
        return self.rng.choice(cand, min(self.n, len(cand)), replace=False)


def fdr_select(st, sd, q, m, plus_one=False, mode="fdr"):
    """st [n, N] target scores, sd [n, Nd] decoy scores ->
    order [n, N] (target patches by decreasing score), n_sel [n], k_fdr [n] (largest k with FDR <= q), fdr [n]
    (estimated FDR of the selected set; NaN when NULL)."""
    n, N = st.shape
    Nd = sd.shape[1]
    st_sorted, order = st.sort(dim=1, descending=True)
    sd_asc = sd.sort(dim=1).values.contiguous()
    n_dec = Nd - torch.searchsorted(sd_asc, st_sorted.contiguous(), right=False)    # decoys >= each threshold
    k = torch.arange(1, N + 1, device=st.device, dtype=torch.float32)
    fdr = (n_dec.float() + float(plus_one)) * (N / Nd) / k                        # [n, N]
    ok = fdr <= q
    last = N - ok.flip(1).float().argmax(1)                                       # largest k with FDR <= q
    k_fdr = torch.where(ok.any(1), last, torch.zeros_like(last))
    n_sel = k_fdr.clamp(max=m) if mode == "fdr" else torch.full_like(k_fdr, min(m, N))
    idx = (n_sel - 1).clamp(min=0)
    fdr_sel = fdr.gather(1, idx[:, None]).squeeze(1)
    fdr_sel = torch.where(n_sel > 0, fdr_sel, torch.full_like(fdr_sel, float("nan")))
    return order, n_sel, k_fdr, fdr_sel


@torch.no_grad()
def e_step(zt, cache, counts, slide_sents, slides, sampler, args, device, select=True):
    """zt [S_sent, D] normalised sentence embeddings (current MLP). Returns per-sentence stats and witnesses."""
    stats = {k: [] for k in ("sent", "n_sel", "k_fdr", "fdr", "own_max", "auc")}
    wit = {"sent": [], "slide": [], "patch": []}
    for i in slides:
        a, b = slide_sents[i]
        if b <= a or counts[i] == 0:
            continue
        dec = [d for d in sampler(i) if counts[d] > 0]
        T = zt[a:b].float()
        P = cache[i, :counts[i]].to(device).float()
        st = T @ P.t()
        sents = torch.arange(a, b)
        own_max = st.max(1).values
        if dec:
            Pd = torch.cat([cache[d, :counts[d]] for d in dec]).to(device).float()
            sd = T @ Pd.t()
            dmax = torch.stack([x.max(1).values for x in sd.split([int(counts[d]) for d in dec], 1)], 1)
            auc = (own_max[:, None] > dmax).float().mean(1) + 0.5 * (own_max[:, None] == dmax).float().mean(1)
        else:
            sd, auc = None, torch.full_like(own_max, float("nan"))
        stats["sent"].append(sents)
        stats["own_max"].append(own_max.cpu())
        stats["auc"].append(auc.cpu())
        if not select:
            continue
        if sd is None:
            if args.select_witnesses == "fdr":                                # no decoy -> cannot calibrate -> NULL
                stats["n_sel"].append(torch.zeros(len(T), dtype=torch.long))
                stats["k_fdr"].append(torch.zeros(len(T), dtype=torch.long))
                stats["fdr"].append(torch.full((len(T),), float("nan")))
                continue
            sd = torch.full((len(T), 1), -2.0, device=device)                 # topk: decoys unused
        order, n_sel, k_fdr, fdr = fdr_select(st, sd, args.fdr, args.max_witnesses, args.fdr_plus_one,
                                              args.select_witnesses)
        stats["n_sel"].append(n_sel.cpu())
        stats["k_fdr"].append(k_fdr.cpu())
        stats["fdr"].append(fdr.cpu())
        keep = torch.arange(order.shape[1], device=device)[None] < n_sel[:, None]
        r, c = keep.nonzero(as_tuple=True)
        if len(r):
            wit["sent"].append(sents[r.cpu()])
            wit["slide"].append(torch.full((len(r),), i, dtype=torch.long))
            wit["patch"].append(order[r, c].cpu())
    out = {k: torch.cat(v) if v else torch.empty(0) for k, v in stats.items()}
    out["wit"] = {k: torch.cat(v) if v else torch.empty(0, dtype=torch.long) for k, v in wit.items()}
    return out


@torch.no_grad()
def encode_sentences(model, sent_feat, chunk=16384):
    model.eval()
    return torch.cat([model.encode_text(sent_feat[s:s + chunk]) for s in range(0, len(sent_feat), chunk)])


# ============================================================================ M-step
def make_batches(sents, sent_slide, cap, batch_size, rng):
    """One pair per sentence; sentences of a slide are spread over 'layers' of at most `cap`, so that a batch
    holds at most `cap` sentences of the same slide (except possibly across a layer boundary)."""
    sents = np.asarray(sents)
    if len(sents) == 0:
        return []
    sents = sents[rng.permutation(len(sents))]
    slide = sent_slide[sents]
    o = np.argsort(slide, kind="stable")
    s_sorted = slide[o]
    start = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    rank_sorted = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    rank = np.empty(len(o), dtype=np.int64)
    rank[o] = rank_sorted
    layer = rank // cap
    _, inv = np.unique(slide.astype(np.int64) * 100_000 + layer, return_inverse=True)
    rkey = rng.random(inv.max() + 1)[inv]
    order = np.lexsort((rank, rkey, layer))
    s = sents[order]
    return [s[k:k + batch_size] for k in range(0, len(s) - batch_size + 1, batch_size)]


def positive_mask(sent_b, key_b, concept, cw_keys, keyspace, mode):
    """M[a, b] = patch b is a witness of sentence a (witness) / of a's concept (concept); diagonal always."""
    B = len(sent_b)
    eye = torch.eye(B, dtype=torch.bool, device=key_b.device)
    if mode == "single" or len(cw_keys) == 0:
        return eye
    q = concept[sent_b][:, None] * keyspace + key_b[None]
    idx = torch.searchsorted(cw_keys, q).clamp(max=len(cw_keys) - 1)
    return (cw_keys[idx] == q) | eye


def multi_pos_nce(logits, pos, logw=None):
    """-log(sum_pos e^l / sum_all w e^l) per row; positives unweighted."""
    den = logits if logw is None else logits + logw.masked_fill(pos, 0.0)
    num = logits.masked_fill(~pos, float("-inf"))
    return (torch.logsumexp(den, 1) - torch.logsumexp(num, 1)).mean()


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
    # E-step
    p.add_argument("--select-witnesses", choices=["fdr", "topk"], default="fdr")
    p.add_argument("--decoys", choices=["other-organ", "same-organ", "random"], default="other-organ")
    p.add_argument("--n-decoys", type=int, default=3, help="decoy slides per sentence's slide")
    p.add_argument("--fdr", type=float, default=0.1, help="target FDR q of the witness sets")
    p.add_argument("--fdr-plus-one", action="store_true", help="conservative (#decoys + 1) FDR estimate")
    p.add_argument("--max-witnesses", type=int, default=32, help="cap m on witnesses per sentence")
    # M-step
    p.add_argument("--rounds", type=int, default=4, help="EM rounds (one E-step each)")
    p.add_argument("--m-epochs", type=int, default=2, help="passes over the pseudo-pairs per M-step")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--sents-per-slide", type=int, default=4, help="max sentences of one slide per batch")
    p.add_argument("--loss", choices=["infonce", "sigmoid"], default="infonce")
    p.add_argument("--positives", choices=["concept", "witness", "single"], default="concept")
    p.add_argument("--concepts", choices=["embed", "exact", "none"], default="embed")
    p.add_argument("--concept-threshold", type=float, default=0.95, help="cosine of frozen sentence embeddings")
    p.add_argument("--weighting", action="store_true", help="LIMIT-style weights on the remaining negatives")
    p.add_argument("--sigmoid-negatives", choices=["other-organ", "all"], default="other-organ")
    p.add_argument("--temperature", type=float, default=None, help="InfoNCE temperature (default: init checkpoint's)")
    # Quilt prior / regularisation
    p.add_argument("--pairs", default=None, help="Quilt pairs file (build_quilt_pairs.py); required if --quilt-weight > 0")
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
    p.add_argument("--test-modes", default="resize,tile")
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
    run_dir = unique_dir(Path(args.runs_dir) / f"{args.run_name or 'o1b'}_{datetime.now():%m%d-%H%M}")
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
    keep_rows_t = torch.tensor(keep_rows, dtype=torch.long)
    sent_feat_cpu = sf[keep_rows_t].float()
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
        print("WARNING: no --init-txt-from: the first E-step uses a random MLP, its witnesses are noise")
    T = args.temperature
    if T is None:
        T = 1.0 / float(init["state_dict"]["logit_scale"].exp()) if init is not None and \
            "logit_scale" in init["state_dict"] else 0.07
    cfg["init_temperature"] = T
    model = ProjectionModel(**cfg).to(device)
    if init is not None:
        model.load_state_dict(init["state_dict"])
        print(f"text MLP initialised from {args.init_txt_from} (depth {cfg['depth']}, width {cfg['width']})")
    with torch.no_grad():
        model.logit_scale.fill_(math.log(1.0 / T))
    model.logit_scale.requires_grad_(False)                                  # fixed temperature
    sig = None
    if args.loss == "sigmoid":                                               # SigLIP init: t = 10, b = -10
        sig = torch.nn.ParameterDict({"log_t": torch.nn.Parameter(torch.tensor(math.log(10.0), device=device)),
                                      "b": torch.nn.Parameter(torch.tensor(-10.0, device=device))})
    groups = param_groups(model, args.weight_decay)
    if sig is not None:
        groups.append({"params": list(sig.parameters()), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr)
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

    config = {"args": vars(args), "model_cfg": cfg, "temperature": T,
              "data": {"n_slides": {k: len(v) for k, v in idx.items()}, "n_sentences": n_sent,
                       "txt_model": txt_model_name, "cache_gb": est_gb},
              "env": {"torch": torch.__version__, "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"loss {args.loss}  positives {args.positives}  weighting {args.weighting}  T {T:.4f}  lr {args.lr}  "
          f"batch {args.batch_size} (<= {args.sents_per_slide}/slide)  |  E-step {args.select_witnesses}: q {args.fdr}, "
          f"{args.n_decoys} {args.decoys} decoys, cap {args.max_witnesses}  |  Quilt weight {args.quilt_weight}",
          flush=True)

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
            e = e_step(zt, cache, counts_l, slide_sents, idx["val"], smp, args, device, select=m == args.decoys)
            tag = "other" if m == "other-organ" else "same"
            out[f"val_auc_{tag}"] = float(np.nanmean(e["auc"].numpy())) if len(e["auc"]) else float("nan")
            if m == args.decoys and len(e["n_sel"]):
                out["val_null_rate"] = float((e["n_sel"] == 0).float().mean())
                nz = e["n_sel"][e["n_sel"] > 0].float()
                out["val_witnesses_mean"] = float(nz.mean()) if len(nz) else 0.0
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
                out.update({f"zs_{k}": v["balanced_acc"] for k, v in zs.items()})
            if args.arch_feats:
                retr = arch_retrieval(model, args.arch_feats, txt_model_name, device)
                out.update({f"{n}_{k}": v for n, r in retr.items() for k, v in r.items() if "R@" in k})
        zs_vals = [v for k, v in out.items() if k.startswith("zs_")]
        if zs_vals:
            out["zs_mean_bal_acc"] = float(np.mean(zs_vals))
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
                    "sigmoid": {k: float(v) for k, v in sig.items()} if sig is not None else None,
                    "source": "train_o1b_em.py"}, path)

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
        names = [("val_auc_other", "val AUC other"), ("val_auc_same", "same"), ("val_null_rate", "val NULL"),
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
    slide_organ = torch.tensor([organ_ids[o] for o in organs], device=device)
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
        e = e_step(zt, cache, counts_l, slide_sents, idx["train"], train_sampler, args, device)
        n_sel = e["n_sel"]
        null_rate = float((n_sel == 0).float().mean())
        nz = n_sel[n_sel > 0].float()
        fdr_vals = e["fdr"][n_sel > 0]
        wit = e["wit"]
        torch.save({"wit_sent": wit["sent"], "wit_slide": wit["slide"], "wit_patch": wit["patch"],
                    "sent": e["sent"], "n_sel": n_sel, "k_fdr": e["k_fdr"], "fdr": e["fdr"],
                    "own_max": e["own_max"], "cache_round": rnd if args.resample_cache else 0,
                    "slide_ids": bags.slide_ids}, run_dir / f"witnesses_round{rnd}.pt")
        with open(run_dir / f"estep_round{rnd}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sentence", "slide_id", "project", "n_witnesses", "k_fdr", "fdr_est", "best_score", "auc_decoys",
                        "text"])
            for j, ns, kf, fd, om, au in zip(e["sent"].tolist(), n_sel.tolist(), e["k_fdr"].tolist(),
                                             e["fdr"].tolist(), e["own_max"].tolist(), e["auc"].tolist()):
                si = sent_slide[j]
                w.writerow([j, bags.slide_ids[si], bags.projects[si], ns, kf,
                            "" if math.isnan(fd) else round(fd, 4), round(om, 4), round(au, 3), sent_texts[j]])
        e_row = {"null_rate": null_rate, "witnesses_mean": float(nz.mean()) if len(nz) else 0.0,
                 "witnesses_median": float(nz.median()) if len(nz) else 0.0,
                 "capped_frac": float((e["k_fdr"] > args.max_witnesses).float().mean()),
                 "fdr_est_mean": float(fdr_vals.mean()) if len(fdr_vals) else float("nan"),
                 "n_pseudo_pairs_sentences": int(len(nz)), "n_witness_patches": int(len(wit["sent"])),
                 "estep_s": round(time.time() - t0)}
        print(f"\nround {rnd} E-step: {len(n_sel):,} sentences, NULL {null_rate:.1%}, witnesses/sentence mean "
              f"{e_row['witnesses_mean']:.1f} median {e_row['witnesses_median']:.0f} (capped {e_row['capped_frac']:.1%}), "
              f"est. FDR {e_row['fdr_est_mean']:.3f}  ({e_row['estep_s']}s)", flush=True)
        order = torch.argsort(n_sel, descending=True)
        for lab, js in (("most witnesses", order[:3]), ("NULL", order[n_sel[order] == 0][:3])):
            for j in js.tolist():
                print(f"    [{lab}] {n_sel[j]:2d}  {bags.projects[sent_slide[e['sent'][j]]]:10s} "
                      f"{sent_texts[e['sent'][j]][:100]}")
        if len(nz) < args.batch_size:
            print(f"  only {len(nz)} sentences have witnesses (< batch size {args.batch_size}): stopping")
            write_log({"round": rnd, **e_row})
            break

        # witness tables on the GPU
        ws, wsl, wp = wit["sent"].to(device), wit["slide"].to(device), wit["patch"].to(device)
        o = torch.argsort(ws, stable=True)
        ws, wsl, wp = ws[o], wsl[o], wp[o]
        w_cnt = torch.bincount(ws, minlength=len(sent_texts))
        w_off = torch.zeros(len(sent_texts) + 1, dtype=torch.long, device=device)
        w_off[1:] = torch.cumsum(w_cnt, 0)
        cw_keys = torch.unique(concept_dev[ws] * keyspace + wsl * args.cache_patches + wp)    # sorted
        has_wit = (w_cnt > 0).nonzero().squeeze(1).cpu().numpy()

        # ---------------- M-step
        model.train()
        losses, qls, regs, pos_rows, pos_frac, t0 = [], [], [], [], [], time.time()
        for ep in range(args.m_epochs):
            for sb in make_batches(has_wit, sent_slide, args.sents_per_slide, args.batch_size, rng):
                sb = torch.from_numpy(sb).to(device)
                pick = w_off[sb] + (torch.rand(len(sb), device=device) * w_cnt[sb]).long()
                slide_b, patch_b = wsl[pick], wp[pick]
                img = cache[slide_b.to(cache.device), patch_b.to(cache.device)].to(device).float()
                key_b = slide_b * args.cache_patches + patch_b
                M = positive_mask(sb, key_b, concept_dev, cw_keys,
                                  keyspace, "single" if args.positives == "single" else "concept")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    z_t = F.normalize(model.txt_proj(sent_feat[sb]).float(), dim=-1)
                    sim = z_t @ img.t()
                    if args.loss == "infonce":
                        logits = sim / T
                        logw = None
                        if args.weighting:
                            with torch.no_grad():
                                logw = torch.log((1.0 - (z_t @ z_t.t() + 1.0) / 2.0).clamp_min(1e-6))
                        loss = 0.5 * (multi_pos_nce(logits, M, logw) +
                                      multi_pos_nce(logits.t(), M.t(), None if logw is None else logw.t()))
                    else:
                        logits = sim * sig["log_t"].exp() + sig["b"]
                        y = M.float() * 2 - 1
                        valid = M.clone()
                        if args.sigmoid_negatives == "all":
                            valid[:] = True
                        else:
                            ob = slide_organ[slide_b]
                            valid |= ob[:, None] != ob[None]
                        loss = -(F.logsigmoid(y * logits) * valid).sum() / len(sb)
                    l_q = torch.zeros((), device=device)
                    if q_stream is not None:
                        q_img, q_txt = next(q_stream)
                        l_q = clip_loss(F.normalize(q_img.float(), dim=-1),
                                        F.normalize(model.txt_proj(q_txt).float(), dim=-1), 1.0 / T)
                    total = loss + args.quilt_weight * l_q
                reg = torch.zeros((), device=device)
                if theta0 is not None:
                    reg = sum(((q - q0) ** 2).sum() for q, q0 in zip(model.txt_proj.parameters(), theta0))
                    total = total + args.l2sp * reg
                opt.zero_grad(set_to_none=True)
                total.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(loss.item())
                qls.append(l_q.item())
                regs.append(float(reg))
                B = len(sb)
                pos_rows.append(float(M.sum(1).float().mean()))
                pos_frac.append(float((M.sum() - B) / max(1, B * B - B)))
            print(f"  M-step epoch {ep + 1}/{args.m_epochs}: {len(losses)} steps  loss {np.mean(losses):.4f}  "
                  f"quilt {np.mean(qls):.4f}  positives/row {np.mean(pos_rows):.2f}  "
                  f"off-diag positives {np.mean(pos_frac):.2%}  ({time.time() - t0:.0f}s)", flush=True)

        row = {"round": rnd, **e_row, "loss": float(np.mean(losses)) if losses else float("nan"),
               "loss_quilt": float(np.mean(qls)) if qls else float("nan"),
               "l2sp": float(np.mean(regs)) if regs else 0.0,
               "positives_per_row": float(np.mean(pos_rows)) if pos_rows else float("nan"),
               "offdiag_positive_frac": float(np.mean(pos_frac)) if pos_frac else float("nan"),
               "m_steps": len(losses), "mstep_s": round(time.time() - t0),
               **({k: float(v) for k, v in sig.items()} if sig is not None else {}),
               **evaluate_round(rnd)}
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
               "select_witnesses": args.select_witnesses, "decoys": args.decoys, "n_decoys": args.n_decoys,
               "fdr": args.fdr, "max_witnesses": args.max_witnesses, "loss": args.loss, "positives": args.positives,
               "concepts": args.concepts if args.positives == "concept" else "", "weighting": args.weighting,
               "quilt_weight": args.quilt_weight, "l2sp": args.l2sp, "lr": args.lr, "rounds": args.rounds,
               "m_epochs": args.m_epochs, "seed": args.seed, "select": args.select, "selected_round": best_round,
               **{k: round(ck["val"][k], 4) for k in ("val_auc_other", "val_auc_same", "val_null_rate",
                                                      "quilt_val_mean_recall") if k in ck["val"]},
               **{k: round(v, 4) for k, v in final.items()}}
    patch_cols = ["zs_" + c[:-len("_bal_acc")] for c in SUMMARY_COLS if c.endswith("_bal_acc")]
    patch_cols += ["zs_mean_bal_acc"] + [c for c in SUMMARY_COLS if c.startswith("arch_")]
    slide_cols = [f"slide_zs_test_{t}_{'mean' if k == 0 else f'top{k}'}" for t in ("lung", "kidney", "brain",
                                                                                    "pancancer", "mean") for k in zs_ks]
    base = [k for k in summary if not k.startswith(("zs_", "arch_", "slide_zs_", "quilt_val_i", "quilt_val_t"))]
    path = Path(args.runs_dir) / "summary_o1b.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=base + patch_cols + slide_cols, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(summary)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
