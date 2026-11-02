#!/usr/bin/env python
"""
Precompute text features for the Quilt-1M captions with a sentence-transformers encoder
(default: BioLORD-2023, loaded from a local folder; compute nodes have no internet).

Each distinct caption string is encoded once; every CSV row points to its caption's feature.

Output: <out-dir>/<model-name>.pt, a torch.save'd dict with
    features      HalfTensor  [U, D]  one embedding per unique caption (as returned by the model)
    texts         list[str]   [U]     the unique (whitespace-normalised) caption strings
    row_to_text   LongTensor  [N]     CSV row i -> index into features/texts (-1 = empty caption)
    image_paths   list[str]   [N]     image filename of each CSV row (join key with the image features)
    n_truncated   int                 number of unique captions longer than the model's max_seq_length
    model, max_seq_length, caption_col, csv

Typical use afterwards:
    d = torch.load("BioLORD-2023.pt")
    ok = d["row_to_text"] >= 0
    pair_text_feats = d["features"][d["row_to_text"][ok]]     # [n_pairs, D], aligned with CSV rows
"""
import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"(?<!\w)@\w+")


def normalise(text, clean_social):
    if not isinstance(text, str):
        return ""
    if clean_social:  # URLs and @handles (Twitter-derived captions); keep hashtag words, drop '#'
        text = URL_RE.sub(" ", text)
        text = HANDLE_RE.sub(" ", text)
        text = text.replace("#", "")
    return " ".join(text.split())


def find_image_col(df):
    for c in ("image_path", "image", "filename", "Filename", "image_name"):
        if c in df.columns:
            return c
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Quilt-1M metadata CSV (e.g. quilt_1M_lookup.csv)")
    p.add_argument("--model-path", required=True, help="local sentence-transformers model folder")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--caption-col", default="caption")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="override the model's max_seq_length (default: keep the model's own)")
    p.add_argument("--clean-social", action="store_true",
                   help="strip URLs/@handles/'#' from captions before encoding")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (for testing)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    model_name = Path(args.model_path.rstrip("/")).name
    out = Path(args.out_dir) / f"{model_name}.pt"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ captions
    df = pd.read_csv(args.csv, low_memory=False, nrows=args.limit)
    if args.caption_col not in df.columns:
        raise SystemExit(f"column '{args.caption_col}' not in CSV; columns: {list(df.columns)}")
    img_col = find_image_col(df)
    caps = [normalise(t, args.clean_social) for t in df[args.caption_col]]

    texts, index, row_to_text = [], {}, []
    for c in caps:
        if not c:
            row_to_text.append(-1)
            continue
        j = index.get(c)
        if j is None:
            j = index[c] = len(texts)
            texts.append(c)
        row_to_text.append(j)
    n_empty = sum(r < 0 for r in row_to_text)
    print(f"rows: {len(df):,}   empty captions: {n_empty:,}   unique captions: {len(texts):,}", flush=True)

    # ------------------------------------------------------------------ model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible, running on CPU (slow)")
    model = SentenceTransformer(args.model_path, device=device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    if device == "cuda":
        model.half()
    max_len = model.max_seq_length
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    print(f"model: {model_name}   dim: {get_dim()}   "
          f"max_seq_length: {max_len}   device: {device}", flush=True)

    # how many captions will be cut by truncation (counted on a sample if very large)
    tok = model.tokenizer
    sample = texts if len(texts) <= 200_000 else texts[::max(1, len(texts) // 200_000)]
    lens = [len(x) for x in tok(sample, add_special_tokens=True, truncation=False)["input_ids"]]
    frac = sum(l > max_len for l in lens) / max(1, len(lens))
    n_trunc = round(frac * len(texts))
    s = pd.Series(lens)
    print(f"caption length (tokens): median {s.median():.0f}, p95 {s.quantile(.95):.0f}, "
          f"max {s.max()}   -> ~{n_trunc:,} captions ({frac:.2%}) longer than {max_len}", flush=True)

    # ------------------------------------------------------------------ encode
    t0 = time.time()
    feats = model.encode(texts, batch_size=args.batch_size, convert_to_tensor=True,
                         normalize_embeddings=False, show_progress_bar=True)
    feats = feats.half().cpu()
    print(f"encoded {tuple(feats.shape)} in {time.time() - t0:.0f}s", flush=True)

    torch.save({
        "features": feats,
        "texts": texts,
        "row_to_text": torch.tensor(row_to_text, dtype=torch.long),
        "image_paths": df[img_col].astype(str).tolist() if img_col else None,
        "n_truncated": n_trunc,
        "model": model_name,
        "max_seq_length": max_len,
        "caption_col": args.caption_col,
        "clean_social": args.clean_social,
        "csv": os.path.abspath(args.csv),
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()