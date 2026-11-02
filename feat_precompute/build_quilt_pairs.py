#!/usr/bin/env python
"""
Build an image-text paired dataset from precomputed Quilt-1M features.

Inputs
  --kept-csv   list of kept images (column 'Filename'), e.g. quilt_clean_images_t05.csv
  --img-feats  UNI2-h feature files for Quilt (glob), one per zip part:
               .npz with 'feats' + 'names'   (extract_uni2h.py)   or
               .pt  with 'features' + 'paths'/'names'
  --txt-feats  text feature file from extract_text_features.py (e.g. BioLORD-2023.pt)

Output (torch.save'd dict, only kept images and the captions they use):
  img_feats   HalfTensor [I, D_img]     img_names  list[str] [I]
  txt_feats   HalfTensor [T, D_txt]     texts      list[str] [T]
  pair_img    LongTensor [P]            pair_txt   LongTensor [P]   (pair p = (image pair_img[p], caption pair_txt[p]))
  img_group   list[str]  [I]            source group (YouTube video / tweet / single image), for leakage-free splits
  img_is_val  BoolTensor [I]            validation images (whole groups, deterministic hash split)
  meta        dict                      counts and settings

Pairs are unique (image, caption) combinations; repeated CSV rows are merged.
"""
import argparse
import glob
import os
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TWEET_RE = re.compile(r"^(\d{15,20})_\d+$")


def source_group(name):
    """Group images that come from the same source, so validation is not leaked by near-duplicates."""
    stem = Path(name).stem
    if "_image_" in stem:                       # YouTube frame: <video_id>_image_<uuid>
        return "yt:" + stem.split("_image_")[0]
    m = TWEET_RE.match(stem)                    # Twitter: <tweet_id>_<k>
    if m:
        return "tw:" + m.group(1)
    return "img:" + stem                        # PubMed / LAION: one image per group


def load_image_features(pattern):
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    if not files:
        raise SystemExit(f"no image feature files match {pattern}")
    feats, names = [], []
    for f in files:
        if f.endswith(".npz"):
            d = np.load(f, allow_pickle=False)
            feats.append(torch.from_numpy(d["feats"]))
            names += [str(n) for n in d["names"]]
        else:
            d = torch.load(f, map_location="cpu")
            feats.append(d["features"])
            names += list(d.get("paths") or d["names"])
        print(f"  {os.path.basename(f)}: {feats[-1].shape[0]:,} images")
    return torch.cat(feats).half(), [os.path.basename(n) for n in names]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kept-csv", required=True)
    p.add_argument("--img-feats", required=True, help="glob, e.g. '.../feats/uni2h/uni2h_part_*.npz'")
    p.add_argument("--txt-feats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="fraction of source groups held out for validation (0 to disable)")
    args = p.parse_args()

    # ------------------------------------------------------------ kept images
    kept = pd.read_csv(args.kept_csv)["Filename"].map(os.path.basename)
    kept_set = set(kept)
    print(f"kept images in CSV          : {len(kept_set):,}")

    # ------------------------------------------------------------ image features
    print("loading image features:")
    all_img, all_names = load_image_features(args.img_feats)
    pos = {n: i for i, n in enumerate(all_names)}
    if len(pos) != len(all_names):
        print(f"  warning: {len(all_names) - len(pos):,} duplicate image names across parts (last one kept)")
    have = [n for n in sorted(kept_set) if n in pos]
    print(f"kept images with a feature  : {len(have):,}  (missing: {len(kept_set) - len(have):,})")

    # ------------------------------------------------------------ text features
    t = torch.load(args.txt_feats, map_location="cpu")
    if t.get("image_paths") is None:
        raise SystemExit("text feature file has no image_paths; re-run extract_text_features.py")
    row_img = [os.path.basename(x) for x in t["image_paths"]]
    row_txt = t["row_to_text"].tolist()

    img_index = {n: i for i, n in enumerate(have)}
    pairs = set()
    for name, j in zip(row_img, row_txt):
        i = img_index.get(name)
        if i is not None and j >= 0:
            pairs.add((i, j))
    pairs = sorted(pairs)                       # sorted by image -> captions of an image are contiguous
    pair_img = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    old_txt = torch.tensor([b for _, b in pairs], dtype=torch.long)

    with_caption = torch.zeros(len(have), dtype=torch.bool)
    with_caption[pair_img] = True
    if (~with_caption).any():                   # drop kept images that have no caption at all
        keep_img = with_caption.nonzero().squeeze(1)
        remap = torch.full((len(have),), -1, dtype=torch.long)
        remap[keep_img] = torch.arange(len(keep_img))
        pair_img = remap[pair_img]
        have = [have[i] for i in keep_img.tolist()]
        print(f"kept images without caption : {(~with_caption).sum().item():,} (dropped)")

    used_txt, pair_txt = torch.unique(old_txt, return_inverse=True)   # keep only captions in use
    img_feats = all_img[torch.tensor([pos[n] for n in have])]
    txt_feats = t["features"][used_txt].half()
    texts = [t["texts"][j] for j in used_txt.tolist()]

    # ------------------------------------------------------------ split by source group
    groups = [source_group(n) for n in have]
    val = torch.tensor([(zlib.crc32(g.encode()) % 10_000) < args.val_frac * 10_000 for g in groups])

    caps_per_img = torch.bincount(pair_img, minlength=len(have))
    n_src = pd.Series([g.split(":")[0] for g in groups]).value_counts().to_dict()
    meta = {
        "kept_csv": os.path.abspath(args.kept_csv),
        "img_feats": args.img_feats, "txt_feats": os.path.abspath(args.txt_feats),
        "txt_model": t.get("model"), "n_images": len(have), "n_texts": len(texts), "n_pairs": len(pair_img),
        "n_val_images": int(val.sum()), "val_frac": args.val_frac, "images_per_source": n_src,
    }

    print(f"\nimages                      : {len(have):,}   {n_src}")
    print(f"unique captions             : {len(texts):,}")
    print(f"pairs                       : {len(pair_img):,}")
    print(f"captions per image          : mean {caps_per_img.float().mean():.2f}, "
          f"median {caps_per_img.median().item()}, max {caps_per_img.max().item()}")
    print(f"validation images / pairs   : {int(val.sum()):,} / {int(val[pair_img].sum()):,}")
    print(f"feature dims                : image {img_feats.shape[1]}, text {txt_feats.shape[1]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "img_feats": img_feats, "img_names": have,
        "txt_feats": txt_feats, "texts": texts,
        "pair_img": pair_img, "pair_txt": pair_txt,
        "img_group": groups, "img_is_val": val, "meta": meta,
    }, args.out)
    print(f"\nsaved {args.out}  ({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()