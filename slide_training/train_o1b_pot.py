#!/usr/bin/env python
"""
Objective 1-B (MLP version) with a PARTIAL OPTIMAL TRANSPORT E-step: for every slide, the report's sentences
and the slide's patches are matched JOINTLY by an entropic partial transport plan, with NULL on both sides.
The M-step (weighted pairs) and everything else are shared with train_o1b_soft.py / train_o1b_em.py (patch
cache, sentences, concepts, decoy sampling, batches, Quilt prior, monitoring, model selection, outputs).

E-step for one slide (current MLP, no gradient). Sentences t_1..t_n, cached patches x_1..x_N (normalised),
cost C_aj = 1 - cos(t_a, x_j). Masses in patch units:
    sentence budget   mu  (--ot-sentence-mass, max witnesses per sentence)
    patch capacity    c   (--ot-patch-capacity: 1 = a patch explains one sentence in total, sentences compete)
    maximise  sum_aj (lambda - C_aj) P_aj + eps H(P)   s.t.  sum_j P_aj <= mu,  sum_a P_aj <= c,  P >= 0
  A pair is worth matching only if C_aj < lambda (lambda = price of NULL); eps = --ot-eps (soft plan).
  Solved as a balanced problem with a NULL row and a NULL column (dummy-point construction):
        cost  [ C      lambda ]      row marginals [mu .. mu, N c]   column marginals [c .. c, n mu]
              [ 0      0      ]
  by log-domain Sinkhorn (--ot-iters), batched over slides (padded) on the GPU.
  Sentences of the same concept within a report (e.g. "invasive ductal carcinoma" / "IDC, grade 3") would
  compete for the same patches; by default they are merged into one row before solving and share its plan
  (--no-ot-merge to disable).

Price of NULL lambda, calibrated with the decoys (--ot-lambda-scope):
  hard (default): one lambda per sentence row from the per-sentence top-k decoy rule of train_o1b_em.py:
    k* = largest k with FDR(top-k) <= --fdr against --n-decoys other-organ decoy slides, lambda_a = cost of the
    k*-th best patch + eps (k* = 0: nothing matched). The joint plan then redistributes this budget with
    competition between sentences, patch capacities and entropic softness.
  sentence: one lambda per row from the plan masses: for each value of a grid over the observed cost range
    (--ot-grid points between the 0.05% and 50% cost quantiles of --ot-calib-slides slides), solve against the
    slide's patches and against the decoys (capacity rescaled to c N / N_d); a row passes if it matches
    >= --ot-min-mass patches with mass_decoy / mass_target <= --fdr; lambda_a = the largest passing value.
  global: one lambda per round for all slides, pooled over the calibration slides (grid then bisection).
  Synthetic results (round 3, ground truth): hard -> NULL non-visual 76 % / generic tissue 53 % / tumour 3 %,
  weighted precision 90 %; sentence -> 93 % / 91 % / 25 % (precision 95 %, too conservative: the mass-based
  FDR penalises generic findings whose witnesses also exist in other-organ decoys, and co-referent sentences
  lose to each other); global -> nothing matched (pooled FDR >= 0.41 at every lambda, because generic findings
  match decoys as well as their own slide).
  The FDR is not monotone near the smallest costs (almost no mass, ratio of entropic leaks), hence the grid
  instead of a plain bisection from 0.

From the plan P to the M-step (as train_o1b_soft.py):
  weight w_aj = min(1, P_aj) (one patch-unit = full witness, whatever the capacity c); candidates w > --min-weight, <= --max-candidates per sentence;
  sentence weight s = max_j w (--sent-weight max) or min(1, sum_j w) (mass); s < --min-sent-weight -> soft NULL;
  M-step: patch drawn proportionally to w, loss scaled by s, weighted multi-positive InfoNCE + Quilt loss.

Logged per round, besides the usual metrics: lambda, calibration FDR, matched mass per slide, Sinkhorn
marginal error, soft-NULL rate, expected witnesses (sum w), shared-patch fraction (candidate patches given to
>= 2 sentence rows of the same slide), and the hard FDR rule of train_o1b_em.py on the same model.

Outputs as train_o1b_soft.py (estep_round<r>.csv, witnesses_round<r>.pt with candidates and weights,
train_log.csv, model_best.pt ...); one row is appended to <runs-dir>/summary_o1b_pot.csv.
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
from train_o1b_em import (DecoySampler, build_cache, build_concepts, e_step, encode_sentences, fdr_select,
                          make_epoch_batches, write_concepts)
from train_o1b_soft import soft_pos_nce, weight_mask

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


# ============================================================================ partial OT
def sinkhorn_partial(C, row_mask, col_mask, mu, cap, lam, eps, iters):
    """Batched entropic partial OT with a NULL row and column.
    C [G, n, N] costs; row_mask [G, n], col_mask [G, N] (padding = False); cap [G] patch capacity per problem.
    Returns P [G, n, N] (padding rows / columns ~ 0) and the max row-marginal error (columns are exact)."""
    G, n, N = C.shape
    Ct = C.new_zeros(G, n + 1, N + 1)
    Ct[:, :n, :N] = C
    Ct[:, :n, N] = lam                                              # sentence mass sent to NULL pays lambda
    rm, cm = row_mask.float(), col_mask.float()
    a = torch.cat([mu * rm, (cap * cm.sum(1))[:, None]], 1)         # rows: sentences + NULL row
    b = torch.cat([cap[:, None] * cm, (mu * rm.sum(1))[:, None]], 1)  # columns: patches + NULL column
    loga, logb = torch.log(a.clamp_min(1e-30)), torch.log(b.clamp_min(1e-30))
    Ce = Ct / eps
    f = torch.zeros(G, n + 1, device=C.device)
    g = torch.zeros(G, N + 1, device=C.device)
    for _ in range(iters):
        f = loga - torch.logsumexp(g[:, None, :] - Ce, 2)
        g = logb - torch.logsumexp(f[:, :, None] - Ce, 1)
    P = torch.exp(f[:, :, None] + g[:, None, :] - Ce)
    err = ((P.sum(2) - a).abs() / a.clamp_min(1e-6) * (a > 1e-12)).amax()
    return P[:, :n, :N], float(err)


def slide_rows(zt, rng_, concept, merge):
    """Rows of the OT problem for a slide's sentences zt[a:b]: one per sentence, or one per concept (merged,
    mean embedding renormalised). Returns (rows [r, D], inv [n] row of each sentence)."""
    a, b = rng_
    if not merge:
        return zt[a:b], torch.arange(b - a, device=zt.device)
    uc, inv = torch.unique(concept[a:b], return_inverse=True)
    E = torch.zeros(len(uc), zt.shape[1], device=zt.device).index_add_(0, inv, zt[a:b].float())
    return F.normalize(E, dim=-1), inv


def pad_stack(tensors, dim_len=None):
    L = dim_len or max(len(t) for t in tensors)
    out = tensors[0].new_zeros(len(tensors), L, tensors[0].shape[1])
    mask = torch.zeros(len(tensors), L, dtype=torch.bool, device=tensors[0].device)
    for k, t in enumerate(tensors):
        out[k, :len(t)] = t
        mask[k, :len(t)] = True
    return out, mask


@torch.no_grad()
def solve_slides(rows, pats, caps, lam, args, device):
    """rows / pats: lists of [r_i, D] / [N_i, D] tensors; caps: list of floats; lam: float (one price of NULL for
    every row) or list of [r_i] tensors (one per row). -> list of plans [r_i, N_i], max marginal error."""
    R, rmask = pad_stack(rows)
    X, cmask = pad_stack(pats)
    if not isinstance(lam, float):
        L = torch.zeros(len(rows), R.shape[1], device=device)
        for k, l in enumerate(lam):
            L[k, :len(l)] = l
        lam = L
    dt = torch.float16 if X.is_cuda else torch.float32               # fp16 matmul on GPU (patches are fp16)
    C = 1.0 - (R.to(dt) @ X.to(dt).transpose(1, 2)).float()
    P, err = sinkhorn_partial(C, rmask, cmask, float(args.ot_sentence_mass),
                              torch.tensor(caps, device=device, dtype=torch.float32), lam, args.ot_eps, args.ot_iters)
    return [P[k, :len(rows[k]), :len(pats[k])] for k in range(len(rows))], err


def chunks(seq, size):
    for s in range(0, len(seq), size):
        yield seq[s:s + size]


@torch.no_grad()
def calibrate_lambda(zt, cache, counts, slide_sents, slides, sampler, concept, args, device):
    """Largest lambda with plan-level FDR = mass(decoys) / mass(target) <= --fdr, pooled over `slides`."""
    prob = []
    for i in slides:
        a, b = slide_sents[i]
        if b <= a or counts[i] == 0:
            continue
        dec = [d for d in sampler(i) if counts[d] > 0]
        if not dec:
            continue
        rows, _ = slide_rows(zt, (a, b), concept, args.ot_merge)
        prob.append((rows, i, dec))                                    # patches gathered per batch (memory)
    if not prob:
        return 0.0, float("nan"), 0.0, np.array([0.0])

    def masses(lam):
        mt = md = 0.0
        c = args.ot_patch_capacity
        for ch in chunks(prob, args.ot_batch):
            tgt = [cache[i, :counts[i]].to(device) for _, i, _ in ch]
            dcy = [torch.cat([cache[d, :counts[d]] for d in dec]).to(device) for _, _, dec in ch]
            Pt, _ = solve_slides([p[0] for p in ch], tgt, [c] * len(ch), lam, args, device)
            Pd, _ = solve_slides([p[0] for p in ch], dcy, [c * len(t) / len(x) for t, x in zip(tgt, dcy)],
                                 lam, args, device)
            mt += sum(float(x.sum()) for x in Pt)
            md += sum(float(x.sum()) for x in Pd)
        return mt, md

    # grid over the observed cost range (a plain bisection on [0, 2] is not valid: below the smallest costs the
    # matched mass is ~0 and the ratio is meaningless), then bisection between the last passing grid point and
    # the next one
    sample = torch.cat([(1.0 - p[0].float() @ cache[p[1], :counts[p[1]]].to(device).float().t()).flatten()
                        for p in prob[:200]])
    q_lo, q_hi = torch.quantile(sample[torch.randperm(len(sample))[:1_000_000]],
                                torch.tensor([0.0005, 0.5], device=sample.device)).tolist()
    min_mass = args.ot_min_mass * len(prob)

    def ok(lam):
        mt, md = masses(lam)
        return (mt >= min_mass and md / mt <= args.fdr), mt, md

    grid = np.linspace(q_lo, q_hi, args.ot_grid)
    if args.ot_lambda_scope != "global":                            # per-row lambdas are calibrated in the E-step
        return None, float("nan"), float("nan"), grid
    res = [ok(float(l)) for l in grid]
    passing = [k for k, r in enumerate(res) if r[0]]
    if not passing:
        print(f"  WARNING: no lambda in [{q_lo:.3f}, {q_hi:.3f}] reaches plan FDR <= {args.fdr} with >= "
              f"{args.ot_min_mass} matched patches/slide; using the smallest (almost nothing matched)")
        return float(grid[0]), float("nan"), res[0][1] / len(prob), grid
    k = passing[-1]
    lo, best = float(grid[k]), res[k][1:]
    if k + 1 < len(grid):
        hi = float(grid[k + 1])
        for _ in range(args.ot_bisect):
            mid = 0.5 * (lo + hi)
            good, mt, md = ok(mid)
            if good:
                lo, best = mid, (mt, md)
            else:
                hi = mid
    mt, md = best
    return lo, md / mt, mt / len(prob), grid


@torch.no_grad()
def row_lambdas(rows, tgt, dcy, grid, args, device):
    """Per-row price of NULL: for each grid value, solve against the slide's patches and against its decoys
    (capacity rescaled); a row passes if it matches >= --ot-min-mass patches with row FDR = mass_decoy / mass_target
    <= --fdr. lambda_a = the largest passing grid value (rows that never pass get the smallest grid value - 1,
    i.e. nothing matched). Returns (list of [r_i] lambda tensors, list of [r_i] row FDR at the chosen value)."""
    c = args.ot_patch_capacity
    caps_d = [c * len(t) / len(x) for t, x in zip(tgt, dcy)]
    lam = [torch.full((len(r),), float(grid[0]) - 1.0, device=device) for r in rows]
    fdr = [torch.full((len(r),), float("nan"), device=device) for r in rows]
    for g in grid:
        Pt, _ = solve_slides(rows, tgt, [c] * len(rows), float(g), args, device)
        Pd, _ = solve_slides(rows, dcy, caps_d, float(g), args, device)
        for k in range(len(rows)):
            mt, md = Pt[k].sum(1), Pd[k].sum(1)
            f = md / mt.clamp_min(1e-12)
            good = (mt >= args.ot_min_mass) & (f <= args.fdr)
            lam[k] = torch.where(good, torch.full_like(lam[k], float(g)), lam[k])
            fdr[k] = torch.where(good, f, fdr[k])
    return lam, fdr


@torch.no_grad()
def hard_row_lambdas(rows, tgt, dcy, args, device):
    """Per-row price of NULL from the hard top-k decoy rule of train_o1b_em.py: k* = largest k with
    FDR(top-k) <= --fdr; lambda_a = cost of the k*-th best patch + eps (that patch has a positive gain), so the plan
    matches about the hard witness set, then competition / capacity / entropy redistribute it. k* = 0: nothing."""
    lam, fdr = [], []
    for r, t, d in zip(rows, tgt, dcy):
        st, sd = r.float() @ t.float().t(), r.float() @ d.float().t()
        order, _, k_fdr, f = fdr_select(st, sd, args.fdr, args.max_candidates, False, "fdr")
        kk = k_fdr.clamp(min=1, max=args.max_candidates) - 1
        s_k = st.gather(1, order.gather(1, kk[:, None])).squeeze(1)
        lam.append(torch.where(k_fdr > 0, 1.0 - s_k + args.ot_eps, torch.full_like(s_k, -1.0)))
        fdr.append(f)
    return lam, fdr


@torch.no_grad()
def pot_e_step(zt, cache, counts, slide_sents, slides, concept, lam, args, device, sampler=None, grid=None):
    """Plans for every slide -> per-sentence stats and candidate table (sent, slide, patch, w).
    lam: global price of NULL (--ot-lambda-scope global), or None with sampler + grid for per-row lambdas."""
    stats = {k: [] for k in ("sent", "n_cand", "sum_w", "max_w", "s", "own_max", "lam", "row_fdr")}
    cand = {"sent": [], "slide": [], "patch": [], "w": []}
    shared, used, errs = 0, 0, []
    todo = [i for i in slides if slide_sents[i][1] > slide_sents[i][0] and counts[i] > 0]
    c = args.ot_patch_capacity
    for ch in chunks(todo, args.ot_batch):
        rows, invs, pats = [], [], []
        for i in ch:
            r, inv = slide_rows(zt, slide_sents[i], concept, args.ot_merge)
            rows.append(r)
            invs.append(inv)
            pats.append(cache[i, :counts[i]].to(device))
        if lam is None:
            dcy = []
            for k, i in enumerate(ch):
                dec = [d for d in sampler(i) if counts[d] > 0]
                dcy.append(torch.cat([cache[d, :counts[d]] for d in dec]).to(device) if dec else pats[k][:1] * 0)
            if args.ot_lambda_scope == "hard":
                lam_rows, fdr_rows = hard_row_lambdas(rows, pats, dcy, args, device)
            else:
                lam_rows, fdr_rows = row_lambdas(rows, pats, dcy, grid, args, device)
        else:
            lam_rows = [torch.full((len(r),), lam, device=device) for r in rows]
            fdr_rows = [torch.full((len(r),), float("nan"), device=device) for r in rows]
        plans, err = solve_slides(rows, pats, [c] * len(ch), lam_rows, args, device)
        errs.append(err)
        for i, P, inv, X, r, lr_, fr_ in zip(ch, plans, invs, pats, rows, lam_rows, fdr_rows):
            a, b = slide_sents[i]
            W = P.clamp(max=1.0)                  # [rows, N]: one patch-unit to a row = full witness, whatever c
            hit = W > args.min_weight
            shared += int((hit.sum(0) >= 2).sum())
            used += int(hit.any(0).sum())
            Ws = W[inv]                                                    # per sentence (merged rows copied)
            vals, order = Ws.sort(dim=1, descending=True)
            keep = (vals > args.min_weight) & (torch.arange(vals.shape[1], device=device)[None] < args.max_candidates)
            wk = vals * keep
            s = wk.max(1).values if args.sent_weight == "max" else wk.sum(1).clamp(max=1.0)
            sents = torch.arange(a, b)
            stats["sent"].append(sents)
            stats["n_cand"].append(keep.sum(1).cpu())
            stats["sum_w"].append(wk.sum(1).cpu())
            stats["max_w"].append(wk.max(1).values.cpu())
            stats["s"].append(s.cpu())
            stats["own_max"].append((zt[a:b].float() @ X.float().t()).max(1).values.cpu())
            stats["lam"].append(lr_[inv].cpu())
            stats["row_fdr"].append(fr_[inv].cpu())
            rr, cc = keep.nonzero(as_tuple=True)
            if len(rr):
                cand["sent"].append(sents[rr.cpu()])
                cand["slide"].append(torch.full((len(rr),), i, dtype=torch.long))
                cand["patch"].append(order[rr, cc].cpu())
                cand["w"].append(vals[rr, cc].cpu())
    out = {k: torch.cat(v) if v else torch.empty(0) for k, v in stats.items()}
    out["cand"] = {k: torch.cat(v) if v else torch.empty(0, dtype=torch.float if k == "w" else torch.long)
                   for k, v in cand.items()}
    out["shared_frac"] = shared / max(1, used)
    out["sinkhorn_err"] = max(errs) if errs else float("nan")
    return out


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
    # E-step (partial OT)
    p.add_argument("--ot-eps", type=float, default=0.02, help="entropic regularisation (cosine units)")
    p.add_argument("--ot-sentence-mass", type=float, default=32.0, help="mu: max patches' worth per sentence row")
    p.add_argument("--ot-patch-capacity", type=float, default=1.0, help="c: total mass a patch can receive")
    p.add_argument("--no-ot-merge", dest="ot_merge", action="store_false",
                   help="do not merge same-concept sentences of a report into one row")
    p.add_argument("--ot-iters", type=int, default=300, help="Sinkhorn iterations")
    p.add_argument("--ot-batch", type=int, default=64, help="slides solved together")
    p.add_argument("--ot-calib-slides", type=int, default=1000, help="training slides used to calibrate lambda")
    p.add_argument("--ot-lambda-scope", choices=["hard", "sentence", "global"], default="hard",
                   help="sentence: per-row plan-mass FDR; hard: per-row price from the hard top-k decoy rule; "
                        "global: one price per round")
    p.add_argument("--ot-grid", type=int, default=16,
                   help="lambda grid points between the 0.05%% and 50%% quantiles of the costs")
    p.add_argument("--ot-bisect", type=int, default=6, help="bisection steps after the grid")
    p.add_argument("--ot-min-mass", type=float, default=0.5,
                   help="a lambda must match >= this many patches (per row, or per calibration slide) to be accepted")
    p.add_argument("--fdr", type=float, default=0.1, help="plan-level FDR target of the lambda calibration")
    p.add_argument("--decoys", choices=["other-organ", "same-organ", "random"], default="other-organ")
    p.add_argument("--n-decoys", type=int, default=3, help="decoy slides per slide (calibration, AUC, hard rule)")
    p.add_argument("--min-weight", type=float, default=0.05, help="candidates need w > this")
    p.add_argument("--max-candidates", type=int, default=64, help="max candidate patches per sentence")
    p.add_argument("--sent-weight", choices=["max", "mass"], default="max",
                   help="sentence weight s: max w, or min(1, sum w)")
    p.add_argument("--min-sent-weight", type=float, default=0.5, help="sentences with s below are soft-NULL")
    # M-step
    p.add_argument("--rounds", type=int, default=4, help="EM rounds (one E-step each)")
    p.add_argument("--m-epochs", type=int, default=2, help="passes over the sentences per M-step")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--sents-per-slide", type=int, default=4, help="max sentences of one slide per batch")
    p.add_argument("--batch-organ", choices=["mixed", "same"], default="mixed")
    p.add_argument("--same-organ-frac", type=float, default=1.0)
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
    # fields used by train_o1b_em.e_step (hard rule for comparison / AUC)
    args.select_witnesses, args.max_witnesses, args.fdr_plus_one = "fdr", args.max_candidates, False

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible to PyTorch; --allow-cpu to force")
    cache_device = args.cache_device if device == "cuda" else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name() if device == 'cuda' else 'CPU'})", flush=True)
    run_dir = unique_dir(Path(args.runs_dir) / f"{args.run_name or 'o1b_pot'}_{datetime.now():%m%d-%H%M}")
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

    # ------------------------------------------------------------ concepts (training sentences; also used to merge OT rows)
    concept = torch.arange(len(sent_texts))
    if args.concepts != "none" and (args.positives == "concept" or args.ot_merge):
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
    mask_concept = concept_dev if args.positives == "concept" else torch.arange(len(sent_texts), device=device)

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
        print("WARNING: no --init-txt-from: the first E-step uses a random MLP, its plans are noise")
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

    config = {"args": vars(args), "model_cfg": cfg, "temperature": T,
              "data": {"n_slides": {k: len(v) for k, v in idx.items()}, "n_sentences": n_sent,
                       "txt_model": txt_model_name, "cache_gb": est_gb},
              "env": {"torch": torch.__version__, "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"E-step partial OT: eps {args.ot_eps}, sentence mass {args.ot_sentence_mass:g}, patch capacity "
          f"{args.ot_patch_capacity:g}, merge concepts {args.ot_merge}, lambda at plan FDR {args.fdr} "
          f"({args.ot_calib_slides} calib slides, {args.n_decoys} {args.decoys} decoys)  |  w > {args.min_weight}, "
          f"soft NULL < {args.min_sent_weight} ({args.sent_weight})  |  positives {args.positives}  weighting "
          f"{args.weighting}  T {T:.4f}  lr {args.lr}  batch {args.batch_size} ({args.batch_organ} organ)  |  "
          f"Quilt weight {args.quilt_weight}", flush=True)

    # ------------------------------------------------------------ monitoring
    encoder = load_text_encoder(args.txt_model, device) if args.txt_model else None
    modes = [m.strip() for m in args.test_modes.split(",") if m.strip()]
    zs_ks = tuple(int(k) for k in args.slide_zs_ks.split(","))
    val_samplers = {m: DecoySampler(idx["val"], organs, m, args.n_decoys, args.seed + 1)
                    for m in ("other-organ", "same-organ")}
    calib_rng = np.random.default_rng(args.seed + 2)
    calib = np.sort(calib_rng.choice(idx["train"], min(args.ot_calib_slides, len(idx["train"])), replace=False))
    calib_sampler = DecoySampler(idx["train"], organs, args.decoys, args.n_decoys, args.seed + 3)
    state = {"lam": None}

    def calibrate(zt):
        t0 = time.time()
        calib_sampler.rng = np.random.default_rng(args.seed + 3)                 # same decoys every round
        lam, fdr_c, mass, grid = calibrate_lambda(zt, cache, counts_l, slide_sents, calib, calib_sampler,
                                                  concept_dev, args, device)
        state.update(lam=lam, calib_fdr=fdr_c, calib_mass=mass, grid=grid, calib_s=round(time.time() - t0))
        if lam is None:
            print(f"  lambda grid {grid[0]:.3f}..{grid[-1]:.3f} (per-row calibration in the E-step)", flush=True)
        else:
            print(f"  lambda {lam:.4f} (cosine > {1 - lam:.4f}), plan FDR {fdr_c:.3f}, matched mass {mass:.1f} "
                  f"patches/slide  ({state['calib_s']}s)", flush=True)

    def val_stats():
        zt = encode_sentences(model, sent_feat)
        calibrate(zt)
        out = {} if state["lam"] is None else {"lambda": state["lam"], "calib_fdr": state["calib_fdr"],
                                               "calib_mass_per_slide": state["calib_mass"]}
        for m, smp in val_samplers.items():
            smp.rng = np.random.default_rng(args.seed + 1)                     # same decoys every round
            e = e_step(zt, cache, counts_l, slide_sents, idx["val"], smp, args, device, select=m == args.decoys)
            tag = "other" if m == "other-organ" else "same"
            out[f"val_auc_{tag}"] = float(np.nanmean(e["auc"].numpy())) if len(e["auc"]) else float("nan")
            if m == args.decoys and len(e["n_sel"]):
                out["val_hard_null_rate"] = float((e["n_sel"] == 0).float().mean())
        smp = val_samplers[args.decoys] if args.decoys in val_samplers else val_samplers["other-organ"]
        smp.rng = np.random.default_rng(args.seed + 4)
        ev = pot_e_step(zt, cache, counts_l, slide_sents, idx["val"], concept_dev, state["lam"], args, device,
                        smp, state["grid"])
        if len(ev["s"]):
            live = ev["s"] >= args.min_sent_weight
            out["val_null_rate"] = float((~live).float().mean())
            out["val_expected_witnesses"] = float(ev["sum_w"][live].mean()) if live.any() else 0.0
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
                    "source": "train_o1b_pot.py"}, path)

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
    hard_sampler = DecoySampler(idx["train"], organs, args.decoys, args.n_decoys, args.seed)
    row_sampler = DecoySampler(idx["train"], organs, args.decoys, args.n_decoys, args.seed + 5)
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
            calibrate(encode_sentences(model, sent_feat))                     # lambda for the new patch subset
        # ---------------- E-step (lambda from the evaluation of the current model)
        t0 = time.time()
        zt = encode_sentences(model, sent_feat)
        e = pot_e_step(zt, cache, counts_l, slide_sents, idx["train"], concept_dev, state["lam"], args, device,
                       row_sampler, state["grid"])
        h = e_step(zt, cache, counts_l, slide_sents, idx["train"], hard_sampler, args, device)   # hard rule + AUC
        hard_k = dict(zip(h["sent"].tolist(), h["k_fdr"].tolist()))
        auc = dict(zip(h["sent"].tolist(), h["auc"].tolist()))
        s_all, cand = e["s"], e["cand"]
        live = s_all >= args.min_sent_weight
        k_hard = torch.tensor([hard_k.get(j, 0) for j in e["sent"].tolist()])
        torch.save({"cand_sent": cand["sent"], "cand_slide": cand["slide"], "cand_patch": cand["patch"],
                    "cand_w": cand["w"], "sent": e["sent"], "sent_weight": s_all, "sum_w": e["sum_w"],
                    "n_cand": e["n_cand"], "k_fdr": k_hard, "own_max": e["own_max"], "lambda": state["lam"],
                    "row_lambda": e["lam"], "row_fdr": e["row_fdr"],
                    "cache_round": rnd if args.resample_cache else 0, "slide_ids": bags.slide_ids},
                   run_dir / f"witnesses_round{rnd}.pt")
        with open(run_dir / f"estep_round{rnd}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sentence", "slide_id", "project", "n_candidates", "sum_w", "sent_weight", "max_w",
                        "hard_k_fdr", "best_score", "auc_decoys", "text"])
            for j, nc, sw, sv, mw, kf, om in zip(e["sent"].tolist(), e["n_cand"].tolist(), e["sum_w"].tolist(),
                                                 s_all.tolist(), e["max_w"].tolist(), k_hard.tolist(),
                                                 e["own_max"].tolist()):
                si = sent_slide[j]
                au = auc.get(j, float("nan"))
                w.writerow([j, bags.slide_ids[si], bags.projects[si], nc, round(sw, 3), round(sv, 3), round(mw, 3),
                            kf, round(om, 4), "" if math.isnan(au) else round(au, 3), sent_texts[j]])
        lam_s = e["lam"][e["lam"] > max(float(state["grid"][0]) - 0.5, -0.5)] if state["lam"] is None else e["lam"]
        e_row = {"lambda_used": state["lam"] if state["lam"] is not None else float(lam_s.mean()) if len(lam_s) else float("nan"),
                 "row_lambda_set_frac": float(len(lam_s) / max(1, len(e["lam"]))),
                 "row_fdr_mean": float(np.nanmean(e["row_fdr"].numpy())) if len(e["row_fdr"]) else float("nan"),
                 "soft_null_rate": float((~live).float().mean()),
                 "sent_weight_mean": float(s_all[live].mean()) if live.any() else 0.0,
                 "expected_witnesses_mean": float(e["sum_w"][live].mean()) if live.any() else 0.0,
                 "candidates_mean": float(e["n_cand"][live].float().mean()) if live.any() else 0.0,
                 "shared_patch_frac": e["shared_frac"], "sinkhorn_err": e["sinkhorn_err"],
                 "hard_null_rate": float((k_hard == 0).float().mean()),
                 "hard_witnesses_mean": float(k_hard[k_hard > 0].float().mean()) if (k_hard > 0).any() else 0.0,
                 "n_train_sentences": int(live.sum()), "n_candidates": int(len(cand["sent"])),
                 "estep_s": round(time.time() - t0)}
        lam_desc = (f"lambda {state['lam']:.4f}" if state["lam"] is not None else
                    f"per-row lambda: {e_row['row_lambda_set_frac']:.0%} of rows matched, mean {e_row['lambda_used']:.3f}, "
                    f"row FDR {e_row['row_fdr_mean']:.3f}")
        print(f"\nround {rnd} E-step ({lam_desc}): {len(s_all):,} sentences, soft NULL "
              f"{e_row['soft_null_rate']:.1%} (hard rule: {e_row['hard_null_rate']:.1%}), sentence weight "
              f"{e_row['sent_weight_mean']:.2f}, expected witnesses {e_row['expected_witnesses_mean']:.1f} (hard k* "
              f"{e_row['hard_witnesses_mean']:.1f}), shared patches {e_row['shared_patch_frac']:.1%}, Sinkhorn err "
              f"{e_row['sinkhorn_err']:.1e}  ({e_row['estep_s']}s)", flush=True)
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
        keys = mask_concept[cs] * keyspace + csl * args.cache_patches + cp
        ck_keys, inv = torch.unique(keys, return_inverse=True)                      # sorted
        ck_w = torch.zeros(len(ck_keys), device=device).scatter_reduce(0, inv, cw, "amax", include_self=False)

        # ---------------- M-step (as train_o1b_soft.py)
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
                M = weight_mask(sb, key_b, mask_concept, ck_keys, ck_w, keyspace,
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
               "ot_eps": args.ot_eps, "ot_sentence_mass": args.ot_sentence_mass,
               "ot_patch_capacity": args.ot_patch_capacity, "ot_merge": args.ot_merge,
               "ot_lambda_scope": args.ot_lambda_scope, "fdr": args.fdr,
               "decoys": args.decoys, "n_decoys": args.n_decoys, "min_weight": args.min_weight,
               "max_candidates": args.max_candidates, "sent_weight": args.sent_weight,
               "min_sent_weight": args.min_sent_weight, "positives": args.positives,
               "concepts": args.concepts, "weighting": args.weighting,
               "batch_organ": args.batch_organ if args.batch_organ == "mixed" else f"same{args.same_organ_frac:g}",
               "quilt_weight": args.quilt_weight, "l2sp": args.l2sp, "lr": args.lr, "rounds": args.rounds,
               "m_epochs": args.m_epochs, "seed": args.seed, "select": args.select, "selected_round": best_round,
               **{k: round(ck["val"][k], 4) for k in ("lambda", "val_auc_other", "val_auc_same", "val_null_rate",
                                                      "val_hard_null_rate", "quilt_val_mean_recall")
                  if k in ck["val"]},
               **{k: round(v, 4) for k, v in final.items()}}
    patch_cols = patch_summary_cols()
    slide_cols = [f"slide_zs_test_{t}_{'mean' if k == 0 else f'top{k}'}" for t in ("lung", "kidney", "brain",
                                                                                    "pancancer", "mean") for k in zs_ks]
    base = [k for k in summary if not k.startswith(("zs_", "arch_", "slide_zs_", "quilt_val_i", "quilt_val_t"))]
    append_summary(Path(args.runs_dir) / "summary_o1b_pot.csv", summary, base + patch_cols + slide_cols)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
