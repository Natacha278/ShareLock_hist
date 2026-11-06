#!/usr/bin/env python
"""
Precompute UNI2-h image features and text features (BioLORD-2023 or any sentence-transformers
model) for an ARCH subset (PubMed or books), for cross-modal retrieval evaluation.

Expected layout (standard ARCH release):
  <arch-root>/captions.json   {"0": {"uuid": ..., "letter": ..., "caption": ...}, ...}
  <arch-root>/images/<uuid>.<jpg|png|...>

Output: <out-dir>/arch_<subset>.pt, a torch.save'd dict with
  img_features  FloatTensor [N, 1536]   UNI2-h, whole image resized to 224x224
  txt_features  HalfTensor  [T, D]      one per unique caption (same settings as extract_text_features.py)
  img_to_text   LongTensor  [N]         image i is described by caption img_to_text[i]
  texts, uuids, letters, paths, img_model, txt_model, n_truncated
and <out-dir>/arch_<subset>_metadata.csv (one row per image).

Several images can share one caption (figure panels), so a caption can have several correct images.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # UNI2-h normalisation


def build_uni2h(weights_path):
    import timm
    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
        "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU, "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model.eval()


def read_arch(root):
    root = Path(root)
    cap_file = root / "captions.json"
    if not cap_file.exists():
        sys.exit(f"{cap_file} not found; --arch-root should be the folder holding captions.json and images/")
    caps = json.loads(cap_file.read_text())
    entries = list(caps.values()) if isinstance(caps, dict) else list(caps)
    img_dir = root / "images" if (root / "images").is_dir() else root
    by_stem = {p.stem: p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith("._")}

    rows, missing = [], 0
    for e in entries:
        uuid = str(e.get("uuid", ""))
        p = by_stem.get(uuid)
        if p is None:
            missing += 1
            continue
        rows.append({"uuid": uuid, "letter": e.get("letter", ""), "caption": e.get("caption", ""),
                     "path": str(p)})
    if missing:
        print(f"warning: {missing} captions.json entries have no image file in {img_dir}")
    if not rows:
        sys.exit("no images matched; check the folder layout")
    return pd.DataFrame(rows)


class Images(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i


@torch.inference_mode()
def image_features(model, paths, device, batch_size, workers):
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    dl = DataLoader(Images(paths), batch_size=batch_size, num_workers=workers, pin_memory=device == "cuda")
    feats = torch.empty(len(paths), 1536)
    t0 = time.time()
    for b, (x, idx) in enumerate(dl):
        with torch.autocast(device, dtype=amp, enabled=device == "cuda"):
            feats[idx] = model(x.to(device, non_blocking=True)).float().cpu()
        if b % 10 == 0:
            print(f"  images {min((b + 1) * batch_size, len(paths))}/{len(paths)}  ({time.time() - t0:.0f}s)", flush=True)
    return feats


def text_features(model_path, texts, device, batch_size):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_path, device=device)
    if device == "cuda":
        m.half()
    lens = [len(x) for x in m.tokenizer(texts, truncation=False)["input_ids"]]
    n_trunc = sum(l > m.max_seq_length for l in lens)
    s = pd.Series(lens)
    print(f"  caption tokens: median {s.median():.0f}, p95 {s.quantile(.95):.0f}, max {s.max()}  -> "
          f"{n_trunc} of {len(texts)} longer than max_seq_length {m.max_seq_length}", flush=True)
    f = m.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=False,
                 show_progress_bar=False)
    return f.half().cpu(), n_trunc, m.max_seq_length


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch-root", required=True, help="folder with captions.json and images/")
    p.add_argument("--subset", default="pubmed", help="name used in the output file (pubmed / books)")
    p.add_argument("--uni2h-weights", required=True, help="UNI2-h pytorch_model.bin")
    p.add_argument("--txt-model", required=True, help="local sentence-transformers folder (e.g. BioLORD-2023)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir) / f"arch_{args.subset}.pt"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists (use --overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = read_arch(args.arch_root)
    meta["caption"] = meta["caption"].fillna("").map(lambda t: " ".join(str(t).split()))
    meta = meta[meta["caption"] != ""].reset_index(drop=True)
    texts = list(dict.fromkeys(meta["caption"]))           # unique, in order of appearance
    tix = {t: i for i, t in enumerate(texts)}
    img_to_text = torch.tensor([tix[c] for c in meta["caption"]], dtype=torch.long)
    per_caption = torch.bincount(img_to_text)
    print(f"images {len(meta):,}   unique captions {len(texts):,}   "
          f"captions with >1 image {(per_caption > 1).sum().item():,}", flush=True)

    print("UNI2-h image features:")
    model = build_uni2h(args.uni2h_weights).to(device)
    img = image_features(model, meta["path"].tolist(), device, args.batch_size, args.num_workers)
    del model
    torch.cuda.empty_cache()

    print(f"text features ({Path(args.txt_model.rstrip('/')).name}):")
    txt, n_trunc, max_len = text_features(args.txt_model, texts, device, batch_size=128)

    rel = [os.path.relpath(p, args.arch_root) for p in meta["path"]]
    torch.save({
        "img_features": img, "txt_features": txt, "img_to_text": img_to_text, "texts": texts,
        "uuids": meta["uuid"].tolist(), "letters": meta["letter"].astype(str).tolist(), "paths": rel,
        "img_model": "UNI2-h", "txt_model": Path(args.txt_model.rstrip("/")).name,
        "n_truncated": n_trunc, "max_seq_length": max_len, "subset": args.subset,
    }, out)
    meta.assign(path=rel, text_index=img_to_text.numpy()).to_csv(
        Path(args.out_dir) / f"arch_{args.subset}_metadata.csv", index=False)
    print(f"saved {out}  images {tuple(img.shape)}  texts {tuple(txt.shape)}")


if __name__ == "__main__":
    main()
