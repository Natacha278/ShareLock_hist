#!/usr/bin/env python
"""
Train a projection (ShareLock-style) on precomputed Quilt-1M features, then evaluate zero-shot
classification on BACH, CRC-VAL-HE-7K, MHIST and BreakHis (benign/malignant and 8 subtypes).

Everything for one run goes to <runs-dir>/<run-name>/:
  config.json          all arguments + data / environment info
  model_best.pt        projection weights (best epoch on Quilt validation retrieval) + model config
  model_last.pt        weights after the last epoch
  train_log.csv        per-epoch loss, temperature, learning rate, validation retrieval
  results.json         zero-shot metrics per dataset / image mode, ARCH retrieval, validation retrieval
  confusion_<dataset>_<mode>.csv
  prompts.json         the prompts used for every class
and a one-line summary is appended to <runs-dir>/summary.csv, to compare runs.

Model selection uses ONLY the Quilt validation split; the benchmarks are evaluated once at the end.

Example:
  python train_projection.py \
    --pairs /project/.../quilt_pairs_t05_biolord.pt \
    --test-feats /project/.../features/uni2h \
    --txt-model /project/rrg-josedolz/natgill/weights/BioLORD-2023 \
    --project text --depth 4 --width 2048 --epochs 30 --run-name biolord_text_d4_w2048
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
import torch.nn.functional as F

from projection import ProjectionModel, clip_loss
from quilt_pairs import QuiltPairs
from zeroshot_prompts import prompts_for

# task -> (feature folder, label column in metadata.csv; None = labels stored in the feature file)
TASKS = {
    "bach": ("bach", None),
    "crc": ("crc", None),
    "mhist": ("mhist", None),
    "breakhis": ("breakhis", None),                     # benign vs malignant
    "breakhis_subtype": ("breakhis", "subtype"),        # 8 tumour subtypes
}
SUMMARY_COLS = [f"{t}/{m}_bal_acc" for t, ms in
                [("bach", ["resize", "tile"]), ("crc", ["native"]), ("mhist", ["native"]),
                 ("breakhis", ["resize", "tile"]), ("breakhis_subtype", ["resize", "tile"])] for m in ms]
SUMMARY_COLS += [f"arch_{s}_{k}" for s in ("pubmed", "books")
                 for k in ("i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10")]


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model, weight_decay):
    """No weight decay on biases, LayerNorm and the temperature."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "logit_scale" in name else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def retrieval_metrics(model, img, txt, gt, ks=(1, 5, 10)):
    """Recall@k for image->text and text->image; an image may have several correct captions."""
    model.eval()
    sim = model.encode_image(img) @ model.encode_text(txt).t()            # [I, T]
    out = {}
    for name, s, g in (("i2t", sim, gt), ("t2i", sim.t(), gt.t())):
        best_pos = s.masked_fill(~g, -float("inf")).max(1).values          # score of best correct match
        rank = ((s >= best_pos[:, None]) & ~g).sum(1)                       # wrong items ranked above (ties count against)
        for k in ks:
            out[f"{name}_R@{k}"] = (rank < k).float().mean().item()
    out["mean_recall"] = float(np.mean([v for k, v in out.items()]))
    return out


@torch.no_grad()
def arch_retrieval(model, files, txt_model_name, device):
    """Cross-modal retrieval on ARCH feature files from extract_arch_features.py."""
    out = {}
    for f in files:
        d = torch.load(f, map_location="cpu")
        name = f"arch_{d.get('subset', Path(f).stem.replace('arch_', ''))}"
        if d.get("txt_model") and d["txt_model"] != txt_model_name:
            raise SystemExit(f"{f} was encoded with {d['txt_model']}, but this run uses {txt_model_name}")
        img = d["img_features"].float().to(device)
        txt = d["txt_features"].float().to(device)
        gt = torch.zeros(len(img), len(txt), dtype=torch.bool, device=device)
        gt[torch.arange(len(img), device=device), d["img_to_text"].to(device)] = True
        r = retrieval_metrics(model, img, txt, gt)
        r.update(n_images=len(img), n_texts=len(txt))
        out[name] = r
        print(f"  {name:16s} images {len(img):5d} texts {len(txt):5d}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in r.items() if "R@" in k))
    return out


def load_text_encoder(path, device):
    """Same settings as extract_text_features.py (fp16 on GPU, no normalisation)."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(path, device=device)
    if device == "cuda":
        m.half()
    return m


def encode_prompts(encoder, prompts, batch_size=256):
    texts = [" ".join(p.split()) for p in prompts]
    return encoder.encode(texts, batch_size=batch_size, convert_to_tensor=True,
                          normalize_embeddings=False, show_progress_bar=False).float()


def test_files(test_dir, dataset, modes):
    d = Path(test_dir) / dataset
    files = {p.stem.replace("uni2h_", ""): p for p in sorted(d.glob("uni2h_*.pt"))}
    return {m: f for m, f in files.items() if m == "native" or m in modes}


def eval_split(dataset, meta):
    """Which images to evaluate: MHIST uses its official test partition, others all images."""
    if dataset == "mhist" and meta is not None and "split" in meta.columns:
        return torch.tensor((meta["split"].astype(str).str.lower() == "test").values)
    return None


def balanced_acc(y, pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, pred))


@torch.no_grad()
def zero_shot(model, encoder, test_dir, modes, device, run_dir, prompt_cache):
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score
    model.eval()
    results = {}
    for ds, (folder, label_col) in TASKS.items():
        files = test_files(test_dir, folder, modes)
        if not files:
            print(f"  [{ds}] no feature files in {Path(test_dir) / folder}, skipping")
            continue
        meta_csv = Path(test_dir) / folder / "metadata.csv"
        meta = pd.read_csv(meta_csv) if meta_csv.exists() else None
        if label_col and (meta is None or label_col not in meta.columns):
            print(f"  [{ds}] no '{label_col}' column in {meta_csv}, skipping")
            continue
        prompts = prompts_for(ds)
        labels = list(prompts)
        prompt_cache[ds] = prompts
        # class embeddings: encode -> project -> normalise every prompt, average per class, renormalise
        cls = []
        for lab in labels:
            z = model.encode_text(encode_prompts(encoder, prompts[lab]).to(device))
            cls.append(F.normalize(z.mean(0), dim=-1))
        cls = torch.stack(cls)                                              # [C, D]

        for mode, f in files.items():
            d = torch.load(f, map_location="cpu")
            feats = d["features"].float()
            y_str = np.array(meta[label_col].astype(str) if label_col else d["labels"])
            if meta is not None and len(meta) != len(feats):
                raise SystemExit(f"[{ds}] {meta_csv} has {len(meta)} rows but {f.name} has {len(feats)}")
            mag = meta["magnification"].values if meta is not None and "magnification" in meta else None
            keep = eval_split(ds, meta)
            if keep is not None:
                feats, y_str = feats[keep], y_str[keep.numpy()]
                mag = mag[keep.numpy()] if mag is not None else None
            unknown = set(y_str) - set(labels)
            if unknown:
                raise SystemExit(f"[{ds}] labels {unknown} have no prompts in zeroshot_prompts.py")
            y = np.array([labels.index(l) for l in y_str])
            pred = (model.encode_image(feats.to(device)) @ cls.t()).argmax(1).cpu().numpy()

            r = {
                "n": int(len(y)),
                "balanced_acc": balanced_acc(y, pred),
                "acc": float((pred == y).mean()),
                "weighted_f1": float(f1_score(y, pred, average="weighted")),
                "per_class_recall": {labels[c]: float((pred[y == c] == c).mean()) for c in range(len(labels))
                                     if (y == c).any()},
            }
            if mag is not None:                      # BreakHis: 40X / 100X / 200X / 400X
                r["balanced_acc_by_magnification"] = {
                    f"{m}X": balanced_acc(y[mag == m], pred[mag == m]) for m in sorted(set(mag))}
            results[f"{ds}/{mode}"] = r
            cm = confusion_matrix(y, pred, labels=range(len(labels)))
            with open(run_dir / f"confusion_{ds}_{mode}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["true\\pred"] + labels)
                for lab, row in zip(labels, cm):
                    w.writerow([lab] + row.tolist())
            print(f"  {ds:16s} {mode:7s} n={r['n']:5d}  bal_acc {r['balanced_acc']:.4f}  "
                  f"acc {r['acc']:.4f}  wF1 {r['weighted_f1']:.4f}"
                  + ("  by mag " + " ".join(f"{k}:{v:.3f}" for k, v in r["balanced_acc_by_magnification"].items())
                     if "balanced_acc_by_magnification" in r else ""))
    return results


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--pairs", required=True, help="file from build_quilt_pairs.py")
    p.add_argument("--test-feats", required=True, help="--out-dir of extract_uni2h_test_data.py")
    p.add_argument("--txt-model", required=True,
                   help="local sentence-transformers folder of the text encoder used for --pairs")
    p.add_argument("--test-modes", default="resize,tile", help="image modes for BACH/BreakHis")
    p.add_argument("--arch-feats", nargs="*", default=[],
                   help="ARCH feature files from extract_arch_features.py (cross-modal retrieval)")
    # model
    p.add_argument("--project", choices=["text", "image", "both"], default="text")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--embed-dim", type=int, default=None, help="only for --project both")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-norm", action="store_true", help="no LayerNorm in the MLP")
    p.add_argument("--init-temperature", type=float, default=0.07)
    p.add_argument("--fixed-temperature", action="store_true")
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--sample-by", choices=["image", "pair"], default="image")
    p.add_argument("--seed", type=int, default=0)
    # output
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    for f in args.arch_feats + [args.pairs]:          # fail now, not after training
        if not Path(f).exists():
            raise SystemExit(f"file not found: {f}")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.run_name or f"{args.project}_d{args.depth}_w{args.width}"
    run_dir = Path(args.runs_dir) / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    print(f"run dir: {run_dir}")

    # ------------------------------------------------------------ data + model
    data = QuiltPairs(args.pairs, device=device)
    img_dim, txt_dim = data.dims
    txt_model_name = Path(args.txt_model.rstrip("/")).name
    if data.meta.get("txt_model") and data.meta["txt_model"] != txt_model_name:
        raise SystemExit(f"--txt-model {txt_model_name} differs from the encoder used for --pairs "
                         f"({data.meta['txt_model']}); the prompts must use the same encoder")

    model_cfg = dict(img_dim=img_dim, txt_dim=txt_dim, project=args.project, embed_dim=args.embed_dim,
                     depth=args.depth, width=args.width, norm=not args.no_norm, dropout=args.dropout,
                     init_temperature=args.init_temperature, learn_temperature=not args.fixed_temperature)
    model = ProjectionModel(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.98))

    steps_per_epoch = len(data.images["train"] if args.sample_by == "image" else data.pairs["train"]) \
        // args.batch_size
    total, warmup = steps_per_epoch * args.epochs, int(steps_per_epoch * args.warmup_epochs)
    val_set = data.retrieval_set("val") if len(data.images["val"]) else None

    config = {
        "args": vars(args), "model": model_cfg, "n_trainable_params": n_params,
        "data": {**data.meta, "n_train_images": len(data.images["train"]),
                 "n_val_images": len(data.images["val"]), "steps_per_epoch": steps_per_epoch},
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"train images {len(data.images['train']):,}  val images {len(data.images['val']):,}  "
          f"dims img {img_dim} txt {txt_dim}  params {n_params:,}  steps/epoch {steps_per_epoch}")

    def save(path, epoch, val):
        torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch,
                    "val": val, "txt_model": txt_model_name, "pairs": os.path.abspath(args.pairs)}, path)

    # ------------------------------------------------------------ train
    log_f = open(run_dir / "train_log.csv", "w", newline="")
    log = None
    best, best_epoch, step = -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for img, txt in data.epoch(args.batch_size, "train", by=args.sample_by):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, args.lr)
            loss = clip_loss(*model(img, txt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1

        val = retrieval_metrics(model, *val_set) if val_set else {}
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "temperature": model.temperature,
               "lr": opt.param_groups[0]["lr"], "time_s": round(time.time() - t0), **val}
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_f.flush()
        print(f"epoch {epoch:3d}  loss {row['loss']:.4f}  T {row['temperature']:.4f}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in val.items() if k in ("i2t_R@1", "t2i_R@1", "mean_recall")))

        score = val.get("mean_recall", -row["loss"])
        if score > best:
            best, best_epoch = score, epoch
            save(run_dir / "model_best.pt", epoch, val)
    save(run_dir / "model_last.pt", args.epochs, val)
    log_f.close()
    print(f"best epoch {best_epoch} (val mean recall {best:.4f})")

    # ------------------------------------------------------------ zero-shot on best checkpoint
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device)["state_dict"])
    del data
    torch.cuda.empty_cache()
    print("\nzero-shot evaluation (best checkpoint):")
    encoder = load_text_encoder(args.txt_model, device)
    prompt_cache = {}
    modes = [m.strip() for m in args.test_modes.split(",") if m.strip()]
    zs = zero_shot(model, encoder, args.test_feats, modes, device, run_dir, prompt_cache)
    (run_dir / "prompts.json").write_text(json.dumps(prompt_cache, indent=1))

    retr = {}
    if args.arch_feats:
        print("\ncross-modal retrieval (best checkpoint):")
        retr = arch_retrieval(model, args.arch_feats, txt_model_name, device)

    results = {"best_epoch": best_epoch, "val_retrieval": torch.load(run_dir / "model_best.pt")["val"],
               "zero_shot": zs, "retrieval": retr}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    # ------------------------------------------------------------ one-line summary across runs
    summary = {"run": run_dir.name, "txt_model": txt_model_name, "project": args.project,
               "depth": args.depth, "width": args.width, "epochs": args.epochs, "lr": args.lr,
               "batch_size": args.batch_size, "seed": args.seed, "best_epoch": best_epoch,
               "val_mean_recall": round(best, 4),
               **{f"{k}_bal_acc": round(v["balanced_acc"], 4) for k, v in zs.items()},
               **{f"{n}_{k}": round(v, 4) for n, r in retr.items() for k, v in r.items() if "R@" in k}}
    fields = list(summary)[:11] + SUMMARY_COLS    # fixed columns, so rows of all runs line up
    summ_path = Path(args.runs_dir) / "summary.csv"
    write_header = not summ_path.exists()
    with open(summ_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", restval="")
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print(f"\nsaved run to {run_dir}  (summary appended to {summ_path})")


if __name__ == "__main__":
    main()