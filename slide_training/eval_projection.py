#!/usr/bin/env python
"""
Evaluate a trained text projection on the patch-level benchmarks, without training:
zero-shot classification (BACH, CRC-VAL-HE-7K, MHIST, BreakHis benign/malignant and subtypes) and
ARCH cross-modal retrieval, with exactly the functions train_projection.py uses at the end of a run.

Works with any checkpoint in train_projection.py format:
  * runs_slide_text/<run>/model_best_projection.pt   (text MLP from train_slide_text.py, --image locked)
  * runs/<quilt run>/model_best.pt                    (Quilt-trained projections, to compare like for like)

Results go to <out-dir> (default: <checkpoint folder>/eval_<timestamp>/):
  results.json, confusion_<dataset>_<mode>.csv, prompts.json, config.json
and one line is appended to <runs-dir>/summary_eval.csv with the same columns as summary.csv.

Example:
  python eval_projection.py \
    --ckpt /project/.../runs_slide_data/xattn_locked_noinit_<ts>/model_best_projection.pt \
    --test-feats /project/.../features/uni2h \
    --txt-model /project/rrg-josedolz/natgill/weights/BioLORD-2023 \
    --arch-feats /project/.../arch_pubmed.pt /project/.../arch_books.pt
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from projection import ProjectionModel
from train_slide_text import unique_dir
from train_projection import (HUB_METHODS, SUMMARY_COLS, arch_retrieval, load_text_encoder,
                              zero_shot)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="train_projection-format checkpoint (state_dict + model_cfg)")
    p.add_argument("--test-feats", required=True, help="--out-dir of extract_uni2h_test_data.py")
    p.add_argument("--txt-model", required=True, help="local sentence-transformers folder (same encoder as training)")
    p.add_argument("--test-modes", default="resize,tile", help="image modes for BACH/BreakHis")
    p.add_argument("--arch-feats", nargs="*", default=[], help="ARCH feature files from extract_arch_features.py")
    p.add_argument("--hub-correction", choices=HUB_METHODS, default="none",
                   help="zero-shot hubness correction; reference = each test set itself (transductive)")
    p.add_argument("--hub-k", type=int, default=100)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--runs-dir", default=None, help="where summary_eval.csv goes (default: parent of the run folder)")
    p.add_argument("--name", default=None, help="label for the summary row (default: run folder name)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu")
    if "model_cfg" not in ck or "state_dict" not in ck or "image" in ck.get("model_cfg", {}):
        raise SystemExit(f"{ckpt_path} is not a train_projection-format checkpoint "
                         f"(for train_slide_text.py runs use model_best_projection.pt, not model_best.pt)")
    cfg = ck["model_cfg"]
    model = ProjectionModel(**cfg).to(device).eval()
    model.load_state_dict(ck["state_dict"])

    txt_model_name = Path(args.txt_model.rstrip("/")).name
    if ck.get("txt_model") and ck["txt_model"] != txt_model_name:
        raise SystemExit(f"checkpoint was trained with {ck['txt_model']}, --txt-model is {txt_model_name}")

    run_name = args.name or ckpt_path.parent.name
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = unique_dir(ckpt_path.parent / f"eval_{datetime.now():%m%d-%H%M}")
    (out / "config.json").write_text(json.dumps({"args": vars(args), "model_cfg": cfg,
                                                 "ckpt_epoch": ck.get("epoch"), "source": ck.get("source")},
                                                indent=2, default=str))
    print(f"checkpoint: {ckpt_path}  (epoch {ck.get('epoch')}, {cfg.get('project')} projection, "
          f"depth {cfg.get('depth')}, width {cfg.get('width')})\nresults -> {out}", flush=True)

    hub = None
    if args.hub_correction != "none":
        hub = {"method": args.hub_correction, "k": args.hub_k, "ref": None}

    print("\nzero-shot evaluation:")
    encoder = load_text_encoder(args.txt_model, device)
    prompt_cache = {}
    modes = [m.strip() for m in args.test_modes.split(",") if m.strip()]
    with torch.no_grad():
        zs = zero_shot(model, encoder, args.test_feats, modes, device, out, prompt_cache, hub=hub)
    (out / "prompts.json").write_text(json.dumps(prompt_cache, indent=1))

    retr = {}
    if args.arch_feats:
        print("\ncross-modal retrieval:")
        retr = arch_retrieval(model, args.arch_feats, txt_model_name, device)

    (out / "results.json").write_text(json.dumps(
        {"ckpt": str(ckpt_path), "zero_shot": zs, "retrieval": retr,
         "hub_correction": {"method": args.hub_correction, "k": args.hub_k, "ref": "test"}}, indent=2))

    summary = {"run": run_name, "ckpt": ckpt_path.name, "txt_model": txt_model_name,
               "depth": cfg.get("depth"), "width": cfg.get("width"), "hub": args.hub_correction,
               **{f"{k}_bal_acc": round(v["balanced_acc"], 4) for k, v in zs.items()},
               **{f"{n}_{k}": round(v, 4) for n, r in retr.items() for k, v in r.items() if "R@" in k}}
    fields = list(summary)[:6] + SUMMARY_COLS
    runs_dir = Path(args.runs_dir) if args.runs_dir else ckpt_path.parent.parent
    path = runs_dir / "summary_eval.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(summary)
    print(f"\nsaved results to {out}  (summary row appended to {path})")


if __name__ == "__main__":
    main()
