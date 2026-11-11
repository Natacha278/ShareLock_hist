#!/usr/bin/env python
"""
Precompute text features for TCGA slide-level reports (PathText / MI-Gen captions paired with
DX1 slides by pair_pathtext_slides.py) with a sentence-transformers encoder (default BioLORD-2023,
loaded from a local folder; compute nodes have no internet).

Encoder settings are identical to extract_text_features.py (fp16 on GPU, no normalisation), so the
features live in the same space as the Quilt-1M caption features used to train the projection.

Reports are much longer than Quilt captions and can exceed the encoder's max_seq_length, so two
views are stored:
  * full report  : the whole report encoded at once (truncated at max_seq_length, as the model does)
  * sentences    : each sentence encoded separately (no truncation in practice), plus their mean
                   -> closer to the short-caption regime the projection was trained on

Output: <out-dir>/tcga_reports_<model-name>.pt, a torch.save'd dict with
    slide_ids        list[str]   [N]   join key with the TRIDENT feature files (<slide_id>.h5)
    barcodes, projects, texts           list[str] [N]
    features         HalfTensor  [N, D]  full-report embedding (row i <-> slide_ids[i])
    features_sentmean HalfTensor [N, D]  mean of the report's sentence embeddings
    sent_features    HalfTensor  [S, D]  one embedding per sentence
    sent_texts       list[str]   [S]
    sent_offsets     LongTensor  [N+1]   sentences of report i = sent_features[sent_offsets[i]:sent_offsets[i+1]]
    truncated        BoolTensor  [N]     full report longer than max_seq_length
    n_tokens         LongTensor  [N]     report length in tokens
    model, max_seq_length, csv

Typical use:
    d = torch.load("tcga_reports_BioLORD-2023.pt")
    txt = d["features"]                          # or d["features_sentmean"]
    row = {s: i for i, s in enumerate(d["slide_ids"])}
"""
import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch

# split after . ! ? ; followed by whitespace and an upper-case letter / digit (keeps "2.5 cm", "e.g. the")
SENT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(])")


def normalise(text):
    return " ".join(text.split()) if isinstance(text, str) else ""


def split_sentences(text, min_chars):
    sents = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    # glue fragments that are too short to mean anything onto the previous sentence
    out = []
    for s in sents:
        if out and len(s) < min_chars:
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out or ([text] if text else [])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="pairs.csv from pair_pathtext_slides.py (slide_id, text, ...)")
    p.add_argument("--model-path", required=True, help="local sentence-transformers model folder")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--key-col", default="slide_id")
    p.add_argument("--features-dir", default=None,
                   help="optional: TRIDENT features_<encoder> folder; keep only slides that have a <slide_id>.h5")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="override the model's max_seq_length (default: keep the model's own)")
    p.add_argument("--min-sentence-chars", type=int, default=15)
    p.add_argument("--limit", type=int, default=None, help="only the first N reports (for testing)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    model_name = Path(args.model_path.rstrip("/")).name
    out = Path(args.out_dir) / f"tcga_reports_{model_name}.pt"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ reports
    df = pd.read_csv(args.pairs, nrows=args.limit)
    for c in (args.text_col, args.key_col):
        if c not in df.columns:
            raise SystemExit(f"column '{c}' not in {args.pairs}; columns: {list(df.columns)}")
    df["_text"] = df[args.text_col].map(normalise)
    n0 = len(df)
    df = df[df["_text"] != ""]
    if args.features_dir:
        have = {f[:-3] for f in os.listdir(args.features_dir) if f.endswith(".h5")}
        df = df[df[args.key_col].isin(have)]
    if df[args.key_col].duplicated().any():
        raise SystemExit(f"duplicate {args.key_col} values in {args.pairs}")
    df = df.reset_index(drop=True)
    print(f"reports: {n0:,} in csv -> {len(df):,} kept"
          + (f" (with features in {args.features_dir})" if args.features_dir else ""), flush=True)

    texts = df["_text"].tolist()
    sents, offsets = [], [0]
    for t in texts:
        s = split_sentences(t, args.min_sentence_chars)
        sents.extend(s)
        offsets.append(len(sents))
    print(f"sentences: {len(sents):,} ({len(sents) / max(1, len(texts)):.1f} per report)", flush=True)

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
    print(f"model: {model_name}   dim: {get_dim()}   max_seq_length: {max_len}   device: {device}", flush=True)

    tok = model.tokenizer
    n_tokens = [len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    s_len = [len(x) for x in tok(sents, add_special_tokens=True, truncation=False)["input_ids"]]
    truncated = [n > max_len for n in n_tokens]
    s = pd.Series(n_tokens)
    print(f"report length (tokens): median {s.median():.0f}, p95 {s.quantile(.95):.0f}, max {s.max()}   "
          f"-> {sum(truncated):,} reports ({sum(truncated) / len(texts):.1%}) longer than {max_len}", flush=True)
    print(f"sentence length (tokens): median {pd.Series(s_len).median():.0f}, max {max(s_len)}   "
          f"-> {sum(l > max_len for l in s_len):,} sentences longer than {max_len}", flush=True)

    # ------------------------------------------------------------------ encode
    enc = dict(batch_size=args.batch_size, convert_to_tensor=True, normalize_embeddings=False,
               show_progress_bar=True)
    t0 = time.time()
    feats = model.encode(texts, **enc).half().cpu()
    sent_feats = model.encode(sents, **enc).half().cpu()
    off = torch.tensor(offsets, dtype=torch.long)
    # mean over each report's sentences (fp32 accumulation)
    owner = torch.repeat_interleave(torch.arange(len(texts)), off[1:] - off[:-1])
    sent_mean = torch.zeros(len(texts), sent_feats.shape[1]).index_add_(0, owner, sent_feats.float())
    sent_mean = (sent_mean / (off[1:] - off[:-1]).clamp(min=1).unsqueeze(1)).half()
    print(f"encoded reports {tuple(feats.shape)} and sentences {tuple(sent_feats.shape)} "
          f"in {time.time() - t0:.0f}s", flush=True)

    col = lambda c: df[c].astype(str).tolist() if c in df.columns else None
    torch.save({
        "slide_ids": col(args.key_col),
        "barcodes": col("barcode"),
        "projects": col("project"),
        "texts": texts,
        "features": feats,
        "features_sentmean": sent_mean,
        "sent_features": sent_feats,
        "sent_texts": sents,
        "sent_offsets": off,
        "truncated": torch.tensor(truncated),
        "n_tokens": torch.tensor(n_tokens, dtype=torch.long),
        "model": model_name,
        "max_seq_length": max_len,
        "csv": os.path.abspath(args.pairs),
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()