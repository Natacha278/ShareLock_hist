#!/usr/bin/env python
"""
LIMIT-style slide-report training (train_limit.py) MIXED with the original Quilt-1M patch-caption objective,
to keep the text projection anchored to patch-level semantics while it learns from TCGA reports.

Why: with LIMIT alone, TCGA retrieval improved but patch-level zero-shot dropped, and the CRC/BACH confusion
matrices showed the report language redefining tissue words ("lymph" -> lymph node status, "normal" ->
uninvolved margins, stroma -> tumour stroma; hubness correction did not recover CRC). Quilt captions use
these words in their tissue sense, so training on both objectives at once counters that drift.

Each optimisation step:
    loss = loss_limit(TCGA bags)  +  --quilt-weight * clip_loss(Quilt batch)
  * loss_limit: exactly train_limit.py (same-organ bags, report-conditioned attention over frozen UNI2-h
    patches, similarity-weighted InfoNCE slide -> report, fixed tau_nce / tau_attn, auto tau_attn)
  * clip_loss: exactly train_projection.py's symmetric InfoNCE on Quilt pairs (frozen image features,
    normalised; text through the same MLP), one random caption per image, temperature fixed to the init
    checkpoint's (or --quilt-temperature)
The text MLP is shared: it is the only trainable part (plus nothing else -- temperatures are fixed).

Monitoring each epoch (and epoch 0 = initial model): TCGA val retrieval (attention scoring), Quilt val
retrieval (held-out Quilt captions), patch zero-shot (BACH/CRC/MHIST/BreakHis) and ARCH. The weighted LIMIT
loss is not comparable over time (its weights depend on the current embeddings), so the unweighted loss is
logged too.

Model selection: --select combined (default) = mean of TCGA val mean recall and Quilt val mean recall
(both validation splits, never the benchmarks); or tcga_val / quilt_val / last.

Example (start from the Quilt projection):
  python train_limit_mix.py \
    --features-dir $SLURM_TMPDIR/bags --txt-feats .../tcga_reports_BioLORD-2023.pt \
    --pairs .../quilt_pairs_t05_biolord.pt --init-txt-from .../runs/<quilt run>/model_best.pt \
    --test-feats .../features/uni2h --txt-model .../weights/BioLORD-2023 --arch-feats ... \
    --quilt-weight 1.0 --runs-dir .../runs_limit --run-name limitmix_w1
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pathlib, sys  # noqa: E401  (shared modules live in ../utils)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from projection import ProjectionModel, clip_loss
from quilt_pairs import QuiltPairs
from tcga_bags import TCGABags, collate_bags
from train_limit import (BagSampler, LimitModel, attention_stats, calibration_scores, effective_patches,
                         limit_loss, lr_at, negative_weights, organ_of, set_seed, tau_for_target)
from train_projection import SUMMARY_COLS, append_summary, patch_summary_cols, zs_flat, arch_retrieval, load_text_encoder, retrieval_metrics, zero_shot
from train_slide_text import evaluate, unique_dir


def quilt_batches(data, batch_size):
    """Endless stream of Quilt training batches (one random caption per image), reshuffled every pass."""
    while True:
        for img, txt in data.epoch(batch_size, "train", by="image"):
            yield img, txt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # TCGA (as train_limit.py)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--txt-feats", required=True)
    p.add_argument("--txt-view", choices=["full", "sentmean"], default="full")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-patches", type=int, default=0)
    p.add_argument("--n-negatives", type=int, default=4)
    p.add_argument("--bags-per-step", type=int, default=1)
    p.add_argument("--negatives", choices=["organ", "random"], default="organ")
    p.add_argument("--no-weighting", action="store_true")
    p.add_argument("--loss", choices=["slide2report", "symmetric"], default="slide2report")
    p.add_argument("--attn-temperature", default="auto")
    p.add_argument("--attn-target-eff", type=float, default=100.0)
    p.add_argument("--attn-auto-scale", type=float, default=1.0)
    p.add_argument("--nce-temperature", type=float, default=0.07)
    p.add_argument("--attn-score", choices=["dot", "unitq", "cos"], default="dot")
    # Quilt
    p.add_argument("--pairs", required=True, help="Quilt pairs file from build_quilt_pairs.py (as train_projection.py)")
    p.add_argument("--quilt-weight", type=float, default=1.0, help="weight of the Quilt CLIP loss (0 = train_limit.py)")
    p.add_argument("--quilt-batch-size", type=int, default=1024)
    p.add_argument("--quilt-temperature", type=float, default=None,
                   help="fixed CLIP temperature for the Quilt loss (default: the init checkpoint's, else 0.07)")
    # text model
    p.add_argument("--init-txt-from", default=None)
    p.add_argument("--txt-depth", type=int, default=2)
    p.add_argument("--txt-width", type=int, default=1024)
    p.add_argument("--no-txt-norm", action="store_true")
    p.add_argument("--txt-dropout", type=float, default=0.0)
    # optimisation
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--select", choices=["combined", "tcga_val", "quilt_val", "last"], default="combined")
    # monitoring
    p.add_argument("--test-feats", default=None)
    p.add_argument("--txt-model", default=None)
    p.add_argument("--test-modes", default="resizecrop,resize,tile")
    p.add_argument("--arch-feats", nargs="*", default=[])
    p.add_argument("--no-eval-epoch0", action="store_true")
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
        raise SystemExit("--test-feats needs --txt-model")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible to PyTorch; --allow-cpu to force")
    print(f"device: {device} ({torch.cuda.get_device_name() if device == 'cuda' else 'CPU'})", flush=True)
    run_dir = unique_dir(Path(args.runs_dir) / f"{args.run_name or 'limitmix'}_{datetime.now():%m%d-%H%M}")
    print(f"run dir: {run_dir}", flush=True)

    # ------------------------------------------------------------ data
    txt_all = torch.load(args.txt_feats, map_location="cpu")
    kw = dict(txt_view=args.txt_view, val_frac=args.val_frac, test_frac=args.test_frac)
    train = TCGABags(args.features_dir, txt_all, "train", max_patches=args.max_patches or None, **kw)
    val = TCGABags(args.features_dir, txt_all, "val", **kw)
    test = TCGABags(args.features_dir, txt_all, "test", **kw)
    del txt_all
    img_dim, txt_dim = train.img_dim, train.txt_dim

    quilt = QuiltPairs(args.pairs, device=device)
    q_img_dim, q_txt_dim = quilt.dims
    if (q_img_dim, q_txt_dim) != (img_dim, txt_dim):
        raise SystemExit(f"Quilt pairs dims {q_img_dim}/{q_txt_dim} != TCGA dims {img_dim}/{txt_dim}")
    if quilt.meta.get("txt_model") and train.txt_model and quilt.meta["txt_model"] != train.txt_model:
        raise SystemExit(f"Quilt captions encoded with {quilt.meta['txt_model']}, reports with {train.txt_model}")
    quilt_val = quilt.retrieval_set("val") if len(quilt.images["val"]) else None
    n_qtrain = len(quilt.images["train"])
    if args.quilt_batch_size > n_qtrain:            # incomplete batches are dropped: the stream would yield nothing
        print(f"--quilt-batch-size {args.quilt_batch_size} > {n_qtrain} Quilt training images: using {n_qtrain}")
        args.quilt_batch_size = n_qtrain

    groups = defaultdict(list)
    for i, proj in enumerate(train.projects):
        groups[organ_of(proj)].append(i)
    K = args.n_negatives + 1
    sampler = BagSampler(groups, K, args.bags_per_step, args.negatives, args.seed)
    if sampler.skipped:
        print(f"organs with fewer than K={K} training slides (not used): {sampler.skipped}")
    print(f"TCGA slides  train {len(train):,}  val {len(val):,}  test {len(test):,}  ({len(groups)} organs)  |  "
          f"Quilt images train {len(quilt.images['train']):,}  val {len(quilt.images['val']):,}", flush=True)
    dl = DataLoader(train, batch_sampler=sampler, collate_fn=collate_bags, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=args.num_workers > 0,
                    prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    # ------------------------------------------------------------ model
    auto_tau = str(args.attn_temperature).lower() == "auto"
    cfg = dict(img_dim=img_dim, txt_dim=txt_dim, txt_depth=args.txt_depth, txt_width=args.txt_width,
               txt_norm=not args.no_txt_norm, txt_dropout=args.txt_dropout,
               attn_temperature=1.0 if auto_tau else float(args.attn_temperature),
               nce_temperature=args.nce_temperature, attn_score=args.attn_score)
    init, quilt_T = None, args.quilt_temperature
    if args.init_txt_from:
        init = torch.load(args.init_txt_from, map_location="cpu")
        c = init["model_cfg"]
        if c.get("project") != "text" or (c["img_dim"], c["txt_dim"]) != (img_dim, txt_dim):
            raise SystemExit(f"--init-txt-from must be a project=text checkpoint with dims {img_dim}/{txt_dim}")
        if init.get("txt_model") and train.txt_model and init["txt_model"] != train.txt_model:
            raise SystemExit(f"--init-txt-from was trained with {init['txt_model']}, reports use {train.txt_model}")
        cfg.update(txt_depth=c["depth"], txt_width=c["width"], txt_norm=c.get("norm", True),
                   txt_dropout=c.get("dropout", 0.0))
        if quilt_T is None and "logit_scale" in init["state_dict"]:
            quilt_T = 1.0 / float(init["state_dict"]["logit_scale"].exp())
    quilt_T = quilt_T or 0.07
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
    q_stream = quilt_batches(quilt, args.quilt_batch_size)
    config = {"args": vars(args), "model": cfg, "quilt_temperature": quilt_T, "K": K,
              "n_trainable_params": sum(q.numel() for q in model.parameters()),
              "data": {"n_train": len(train), "n_val": len(val), "n_test": len(test), "txt_model": train.txt_model,
                       "n_quilt_train": len(quilt.images["train"]), "n_quilt_val": len(quilt.images["val"]),
                       "organs": {g_: len(v) for g_, v in groups.items()}, "steps_per_epoch": steps_per_epoch},
              "env": {"torch": torch.__version__, "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"K {K} ({args.negatives} negatives)  steps/epoch {steps_per_epoch}  lr {args.lr}  "
          f"tau_attn {cfg['attn_temperature']:.4g} ({args.attn_score})  tau_nce {args.nce_temperature}  |  "
          f"Quilt: weight {args.quilt_weight}, batch {args.quilt_batch_size}, T {quilt_T:.4f}", flush=True)

    # ------------------------------------------------------------ monitoring
    encoder = load_text_encoder(args.txt_model, device) if args.test_feats else None
    modes = [m.strip() for m in args.test_modes.split(",") if m.strip()]

    def projection():
        sd, pcfg = model.projection_checkpoint(img_dim, txt_dim, cfg)
        sd["logit_scale"] = torch.tensor(math.log(1.0 / quilt_T))
        pcfg["init_temperature"] = quilt_T
        pm = ProjectionModel(**pcfg).to(device).eval()
        pm.load_state_dict(sd)
        return pm, sd, pcfg

    def quilt_val_metrics():
        if quilt_val is None:
            return {}
        pm, _, _ = projection()
        with torch.no_grad():
            r = retrieval_metrics(pm, *quilt_val)
        return {f"quilt_val_{k}": v for k, v in r.items()}

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
                out.update(zs_flat(zs))
            if args.arch_feats:
                retr = arch_retrieval(pm, args.arch_feats, train.txt_model, device)
                out.update({f"{n}_{k}": v for n, r in retr.items() for k, v in r.items() if "R@" in k})
        return out

    def score(row):
        t, q = row.get("mean_recall", -1.0), row.get("quilt_val_mean_recall", -1.0)
        return {"tcga_val": t, "quilt_val": q, "combined": 0.5 * (t + q), "last": 0.0}[args.select]

    def save(path, epoch, metrics):
        torch.save({"state_dict": model.state_dict(), "model_cfg": cfg, "epoch": epoch, "val": metrics,
                    "txt_model": train.txt_model, "quilt_temperature": quilt_T, "source": "train_limit_mix.py"}, path)
        _, sd, pcfg = projection()
        torch.save({"state_dict": sd, "model_cfg": pcfg, "epoch": epoch, "val": metrics, "txt_model": train.txt_model,
                    "pairs": args.pairs, "source": "train_limit_mix.py"}, path.with_name(path.stem + "_projection.pt"))

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
        keys = ["mean_recall", "quilt_val_mean_recall", "zs_mean_bal_acc"]
        names = {"mean_recall": "tcga_val_recall", "quilt_val_mean_recall": "quilt_val_recall",
                 "zs_mean_bal_acc": "zs_mean_bal_acc"}
        return prefix + "  " + "  ".join(f"{names[k]} {row[k]:.4f}" for k in keys if k in row)

    # ------------------------------------------------------------ epoch 0
    best, best_epoch = -float("inf"), 0
    if not args.no_eval_epoch0:
        print("\nepoch 0 (initial model):", flush=True)
        row = {"epoch": 0, **evaluate(model, val, device, args.num_workers), **quilt_val_metrics(), **patch_eval(0)}
        write_log(row)
        print(summarise("epoch   0", row), flush=True)
        if args.select != "last":
            best = score(row)
            save(run_dir / "model_best.pt", 0, row)

    # ------------------------------------------------------------ train
    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        lim, lim_u, qls, wmeans, t_data, t_ep, t_last = [], [], [], [], 0.0, time.time(), time.time()
        for patches, mask, txt, _ in dl:
            t_data += time.time() - t_last
            B = patches.shape[0] // K
            patches = patches.to(device, non_blocking=True).view(B, K, *patches.shape[1:])
            mask = mask.to(device, non_blocking=True).view(B, K, -1)
            txt = txt.to(device, non_blocking=True).view(B, K, -1)
            q_img, q_txt = next(q_stream)
            for g_ in opt.param_groups:
                g_["lr"] = lr_at(step, total, args.warmup_steps, args.lr)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                r = model.encode_text(txt)
                s, a = model.bag_logits(r, patches, mask)
                l_lim = limit_loss(s.float(), r, weighting=not args.no_weighting, direction=args.loss)
                if args.quilt_weight > 0:
                    z_t = F.normalize(model.encode_text(q_txt).float(), dim=-1)
                    l_q = clip_loss(F.normalize(q_img.float(), dim=-1), z_t, 1.0 / quilt_T)
                else:
                    l_q = torch.zeros((), device=device)
                loss = l_lim + args.quilt_weight * l_q
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                lim_u.append(limit_loss(s.float(), r, weighting=False, direction=args.loss).item())
            lim.append(l_lim.item())
            qls.append(l_q.item())
            wmeans.append(negative_weights(r)[0])
            step += 1
            t_last = time.time()
            i_ep = len(lim)
            if i_ep == 1 or (args.log_every and i_ep % args.log_every == 0):
                eff, top1, n_p = attention_stats(a, mask)
                el = t_last - t_ep
                n = args.log_every
                print(f"  epoch {epoch} step {i_ep}/{steps_per_epoch}  limit {np.mean(lim[-n:]):.4f} "
                      f"(unweighted {np.mean(lim_u[-n:]):.4f})  quilt {np.mean(qls[-n:]):.4f}  "
                      f"attn: {eff:.0f} eff. patches of {n_p:.0f}  neg w {np.mean(wmeans[-n:]):.3f}  "
                      f"{el / i_ep:.2f}s/step ({100 * t_data / max(el, 1e-9):.0f}% data)  "
                      f"ETA {(steps_per_epoch - i_ep) * el / i_ep / 60:.0f} min", flush=True)

        row = {"epoch": epoch, "loss_limit": float(np.mean(lim)), "loss_limit_unweighted": float(np.mean(lim_u)),
               "loss_quilt": float(np.mean(qls)), "neg_weight_mean": float(np.mean(wmeans)),
               "lr": opt.param_groups[0]["lr"], "epoch_s": round(time.time() - t_ep), "data_wait_s": round(t_data),
               **evaluate(model, val, device, args.num_workers), **quilt_val_metrics(), **patch_eval(epoch)}
        write_log(row)
        print(summarise(f"epoch {epoch:3d}  limit {row['loss_limit']:.4f} (unw. {row['loss_limit_unweighted']:.4f})  "
                        f"quilt {row['loss_quilt']:.4f}  time {row['epoch_s']}s", row), flush=True)
        if args.select == "last" or score(row) > best:
            best, best_epoch = score(row), epoch
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
               "quilt_weight": args.quilt_weight, "quilt_batch_size": args.quilt_batch_size,
               "quilt_T": round(quilt_T, 5), "negatives": args.negatives, "K": K, "weighted": not args.no_weighting,
               "loss": args.loss, "attn_score": args.attn_score, "tau_attn": round(cfg["attn_temperature"], 5),
               "lr": args.lr, "epochs": args.epochs, "seed": args.seed, "select": args.select,
               "selected_epoch": best_epoch, "quilt_val_mean_recall": round(ck["val"].get("quilt_val_mean_recall", -1), 4),
               **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
               **{k: round(v, 4) for k, v in final_patch.items()}}
    patch_cols = patch_summary_cols()
    fields = [k for k in summary if not k.startswith(("zs_", "arch_"))] + patch_cols
    path = Path(args.runs_dir) / "summary_limit_mix.csv"
    append_summary(path, summary, fields)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
