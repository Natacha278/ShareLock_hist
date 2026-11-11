#!/usr/bin/env python
"""
Train report-guided slide embeddings (cross-attention: report = query, patches = keys/values) with a
CLIP InfoNCE loss between report embeddings and slide embeddings, on TCGA slide-report pairs.

Inputs: TRIDENT UNI2-h patch features (<slide_id>.h5, no pooling) + report features from
extract_report_features.py. Split by patient (same rule as build_tcga_pairs.py).

Everything for one run goes to <runs-dir>/<run-name>_<timestamp>/:
  config.json, train_log.csv, model_best.pt (best val mean recall), model_last.pt, results.json
and one line is appended to <runs-dir>/summary_slide_text.csv.

Example:
  python train_slide_text.py \
    --features-dir $SCRATCH/tcga_trident/20x_256px_0px_overlap/features_uni_v2 \
    --txt-feats /project/.../tcga_report_text/tcga_reports_BioLORD-2023.pt \
    --batch-size 32 --max-patches 8192 --epochs 30 --run-name xattn_full
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from slide_text_model import SlideTextModel, pairwise_clip_loss
from tcga_bags import TCGABags, collate_bags


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(model, weight_decay, attn_temp_lr_mult=1.0):
    """No weight decay on biases/LayerNorm/temperatures/queries; the attention temperature gets its own lr
    multiplier (one scalar: with the base lr it can barely move in a run, so attention would stay ~uniform)."""
    decay, no_decay, attn_t = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "attn_log_scale":
            attn_t.append(p)
        elif p.ndim < 2 or "logit_scale" in name or "queries" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay, "lr_mult": 1.0},
              {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}]
    if attn_t:
        groups.append({"params": attn_t, "weight_decay": 0.0, "lr_mult": attn_temp_lr_mult})
    return groups


def recall_from_logits(logits, ks=(1, 5, 10)):
    """logits [T, S], text i matches slide i. Ties count against."""
    out = {}
    for name, s in (("t2i", logits), ("i2t", logits.t())):
        pos = s.diagonal()[:, None]
        rank = (s >= pos).sum(1) - 1
        for k in ks:
            out[f"{name}_R@{k}"] = (rank < k).float().mean().item()
    out["mean_recall"] = float(np.mean(list(out.values())))
    return out


@torch.no_grad()
def evaluate(model, ds, device, num_workers, txt_chunk=1024, amp=torch.bfloat16):
    """Full pairwise retrieval on a split: every report vs every slide (all patches of each slide)."""
    model.eval()
    if len(ds) < 2:
        return {}
    max_p, ds.max_patches = ds.max_patches, None                  # evaluate on full bags
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_bags)
    txt = ds.txt.to(device)
    logits = torch.empty(len(ds), len(ds))
    with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
        t_all = model.encode_text(txt)
        for patches, mask, _, idx in dl:
            patches, mask = patches.to(device, non_blocking=True), mask.to(device)
            for a in range(0, len(ds), txt_chunk):
                t = t_all[a:a + txt_chunk]
                s = model.slide_embeddings(t, patches, mask)            # [T or 1, 1, dim]
                z_t = torch.nn.functional.normalize(t, dim=-1)
                logits[a:a + txt_chunk, idx[0]] = torch.einsum("td,tsd->ts", z_t.float(),
                                                               s.float().expand(len(t), -1, -1))[:, 0].cpu()
    ds.max_patches = max_p
    return recall_from_logits(logits)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--txt-feats", required=True)
    p.add_argument("--txt-view", choices=["full", "sentmean"], default="full")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-patches", type=int, default=8192,
                   help="random patches per slide during training (0 = all; GPU memory grows with it)")
    # model
    p.add_argument("--image", choices=["locked", "learned"], default="locked",
                   help="locked (default, LIMIT-like): patch features untouched, parameter-free attention in the "
                        "patch space, only the text side is trained; learned: trainable patch MLP + multi-head attention")
    p.add_argument("--query", choices=["report", "learned"], default="report")
    p.add_argument("--n-queries", type=int, default=1, help="only for --query learned")
    p.add_argument("--txt-depth", type=int, default=2)
    p.add_argument("--txt-width", type=int, default=1024)
    p.add_argument("--no-txt-norm", action="store_true", help="no LayerNorm in the text MLP")
    p.add_argument("--txt-dropout", type=float, default=0.0)
    p.add_argument("--init-txt-from", default=None,
                   help="train_projection.py checkpoint (project=text, e.g. the Quilt model_best.pt): initialise "
                        "the text MLP (and temperature) from it; its depth/width/norm override the flags. --image locked only")
    p.add_argument("--init-attn-temperature", type=float, default=1.0,
                   help="--image locked: initial softmax temperature over patches (soft start; sharper inits (<=0.1) "
                        "lock onto random patches and memorise in tests)")
    p.add_argument("--fixed-attn-temperature", action="store_true")
    p.add_argument("--attn-temp-lr-mult", type=float, default=100.0,
                   help="--image locked: lr multiplier for the attention temperature (log-scale parameter)")
    p.add_argument("--dim", type=int, default=512, help="--image learned only")
    p.add_argument("--heads", type=int, default=8, help="--image learned only")
    p.add_argument("--patch-depth", type=int, default=1, help="--image learned only")
    p.add_argument("--patch-width", type=int, default=1024, help="--image learned only")
    p.add_argument("--attn-dropout", type=float, default=0.0, help="--image learned only")
    p.add_argument("--init-temperature", type=float, default=0.07)
    p.add_argument("--fixed-temperature", action="store_true")
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs_slide_text")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.run_name or f"xattn_{args.image}img_{args.query}q_{args.txt_view}"
    run_dir = Path(args.runs_dir) / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    print(f"run dir: {run_dir}", flush=True)

    # ------------------------------------------------------------ data
    txt_all = torch.load(args.txt_feats, map_location="cpu")
    kw = dict(txt_view=args.txt_view, val_frac=args.val_frac, test_frac=args.test_frac)
    train = TCGABags(args.features_dir, txt_all, "train", max_patches=args.max_patches or None, **kw)
    val = TCGABags(args.features_dir, txt_all, "val", **kw)
    test = TCGABags(args.features_dir, txt_all, "test", **kw)
    del txt_all
    if len(train) < args.batch_size:
        raise SystemExit(f"only {len(train)} training slides for batch size {args.batch_size}")
    with __import__("h5py").File(os.path.join(args.features_dir, train.slide_ids[0] + ".h5"), "r") as f:
        img_dim = f["features"].shape[1]
    print(f"slides  train {len(train):,}  val {len(val):,}  test {len(test):,}  |  "
          f"img dim {img_dim}  txt dim {train.txt_dim} ({train.txt_model}, {args.txt_view})", flush=True)
    dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_bags,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    # ------------------------------------------------------------ model
    model_cfg = dict(img_dim=img_dim, txt_dim=train.txt_dim, image=args.image, dim=args.dim, heads=args.heads,
                     txt_depth=args.txt_depth, txt_width=args.txt_width, txt_norm=not args.no_txt_norm,
                     txt_dropout=args.txt_dropout, patch_depth=args.patch_depth, patch_width=args.patch_width,
                     attn_dropout=args.attn_dropout, query=args.query, n_queries=args.n_queries,
                     init_attn_temperature=args.init_attn_temperature,
                     learn_attn_temperature=not args.fixed_attn_temperature,
                     init_temperature=args.init_temperature, learn_temperature=not args.fixed_temperature)
    init = None
    if args.init_txt_from:
        if args.image != "locked":
            raise SystemExit("--init-txt-from needs --image locked (the projection maps text into the patch space)")
        init = torch.load(args.init_txt_from, map_location="cpu")
        c = init["model_cfg"]
        if c.get("project") != "text" or (c["img_dim"], c["txt_dim"]) != (img_dim, train.txt_dim):
            raise SystemExit(f"--init-txt-from must be a project=text checkpoint with dims {img_dim}/{train.txt_dim}, "
                             f"got project={c.get('project')} dims {c['img_dim']}/{c['txt_dim']}")
        if init.get("txt_model") and train.txt_model and init["txt_model"] != train.txt_model:
            raise SystemExit(f"--init-txt-from was trained with {init['txt_model']}, reports use {train.txt_model}")
        model_cfg.update(txt_depth=c["depth"], txt_width=c["width"], txt_norm=c.get("norm", True),
                         txt_dropout=c.get("dropout", 0.0))
    model = SlideTextModel(**model_cfg).to(device)
    if init is not None:
        sd = {k: v for k, v in init["state_dict"].items() if k.startswith("txt_proj.") or k == "logit_scale"}
        missing = model.load_state_dict(sd, strict=False).missing_keys
        assert not [k for k in missing if k.startswith("txt_proj.")], missing
        print(f"text MLP initialised from {args.init_txt_from} (depth {model_cfg['txt_depth']}, "
              f"width {model_cfg['txt_width']}, T {model.temperature:.4f})", flush=True)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay, args.attn_temp_lr_mult), lr=args.lr,
                            betas=(0.9, 0.98))
    steps_per_epoch = len(dl)
    total, warmup = steps_per_epoch * args.epochs, int(steps_per_epoch * args.warmup_epochs)

    config = {"args": vars(args), "model": model_cfg, "n_trainable_params": n_params,
              "data": {"n_train": len(train), "n_val": len(val), "n_test": len(test), "txt_model": train.txt_model,
                       "steps_per_epoch": steps_per_epoch},
              "env": {"torch": torch.__version__, "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"params {n_params:,}  steps/epoch {steps_per_epoch}", flush=True)

    def save(path, epoch, metrics):
        torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch, "val": metrics,
                    "txt_model": train.txt_model, "txt_view": args.txt_view}, path)
        if args.image == "locked":     # same text MLP as a train_projection.py checkpoint (patch-level zero-shot)
            sd, cfg = model.projection_checkpoint(img_dim, train.txt_dim, model_cfg["txt_depth"],
                                                  model_cfg["txt_width"], model_cfg["txt_norm"],
                                                  model_cfg["txt_dropout"])
            torch.save({"state_dict": sd, "model_cfg": cfg, "epoch": epoch, "val": metrics,
                        "txt_model": train.txt_model, "pairs": None, "source": "train_slide_text.py"},
                       path.with_name(path.stem + "_projection.pt"))

    # ------------------------------------------------------------ train
    log_f, log = open(run_dir / "train_log.csv", "w", newline=""), None
    best, best_epoch, step, t0 = -1.0, -1, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, t_data, t_ep = [], 0.0, time.time()
        t_last = time.time()
        for patches, mask, txt, _ in dl:
            t_data += time.time() - t_last
            patches = patches.to(device, non_blocking=True)
            mask, txt = mask.to(device, non_blocking=True), txt.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, args.lr) * g["lr_mult"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = pairwise_clip_loss(model(patches, mask, txt).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())
            step += 1
            t_last = time.time()

        metrics = evaluate(model, val, device, args.num_workers)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "temperature": model.temperature, "attn_temperature": model.attn_temperature,
               "lr": opt.param_groups[0]["lr"], "epoch_s": round(time.time() - t_ep),
               "data_wait_s": round(t_data), **metrics}
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_f.flush()
        print(f"epoch {epoch:3d}  loss {row['loss']:.4f}  T {row['temperature']:.4f}  "
              + (f"T_attn {row['attn_temperature']:.4f}  " if row['attn_temperature'] is not None else "")
              + f"time {row['epoch_s']}s (data wait {row['data_wait_s']}s)  "
              + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items() if k in ("t2i_R@1", "i2t_R@1", "mean_recall")),
              flush=True)
        score = metrics.get("mean_recall", -row["loss"])
        if score > best:
            best, best_epoch = score, epoch
            save(run_dir / "model_best.pt", epoch, metrics)
    save(run_dir / "model_last.pt", args.epochs, metrics)
    log_f.close()

    # ------------------------------------------------------------ test (best checkpoint)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device)["state_dict"])
    test_metrics = evaluate(model, test, device, args.num_workers)
    print(f"best epoch {best_epoch} (val mean recall {best:.4f})\ntest: "
          + "  ".join(f"{k} {v:.4f}" for k, v in test_metrics.items()), flush=True)
    results = {"best_epoch": best_epoch, "val": torch.load(run_dir / "model_best.pt")["val"], "test": test_metrics,
               "train_time_s": round(time.time() - t0)}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    summary = {"run": run_dir.name, "image": args.image, "query": args.query, "txt_view": args.txt_view, "dim": args.dim,
               "heads": args.heads, "max_patches": args.max_patches, "batch_size": args.batch_size,
               "lr": args.lr, "epochs": args.epochs, "seed": args.seed, "best_epoch": best_epoch,
               "val_mean_recall": round(best, 4), **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()}}
    path = Path(args.runs_dir) / "summary_slide_text.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
