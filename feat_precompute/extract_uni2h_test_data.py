#!/usr/bin/env python
"""
Precompute UNI2-h (MahmoodLab/UNI2-h) features for BACH, CRC-VAL-HE-7K, MHIST and BreakHis.

Outputs, per dataset, in <out-dir>/<dataset>/:
  metadata.csv          one row per image: path (relative to dataset root), label, extra fields
  uni2h_<mode>.pt       torch.save'd dict with
      features        FloatTensor [N, 1536]  one embedding per image (row i <-> metadata row i)
      labels          list[str]
      paths           list[str]
      tile_features   FloatTensor [sum T_i, 1536]   (tile mode only)
      tile_counts     LongTensor  [N]               (tile mode only; torch.split(tile_features, tile_counts.tolist()))

Modes (only relevant for BACH 2048x1536 and BreakHis 700x460; CRC and MHIST are already 224x224
and get a single file, uni2h_native.pt):
  resize  the whole image resized to 224x224 (no crop: all tissue kept, aspect ratio not preserved)
  tile    non-overlapping --tile-size tiles at native resolution (centred grid), each resized to 224
          if needed, embedded, then mean-pooled into one image embedding
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
INPUT_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # UNI2-h normalisation (ImageNet)
NATIVE_224 = {"crc", "mhist"}


# ----------------------------------------------------------------------------- model
def build_model(weights_path):
    import timm  # imported here so the indexing code can be used without timm

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval()


# ----------------------------------------------------------------------------- indexing
def _images(root):
    return sorted(
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in IMG_EXT and not p.name.startswith("._") and p.is_file()
    )


def index_bach(root):
    photos = [p for p in Path(root).rglob("Photos") if p.is_dir()]
    if not photos:
        sys.exit(f"[bach] no 'Photos' folder found under {root}")
    rows = []
    for f in _images(photos[0]):
        rows.append({"path": str(f), "label": f.parent.name})
    return pd.DataFrame(rows)


def index_crc(root):
    classes = {"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"}
    rows = [
        {"path": str(f), "label": f.parent.name, "split": f.parent.parent.name}
        for f in _images(root) if f.parent.name in classes
    ]
    return pd.DataFrame(rows)


def index_mhist(root):
    csvs = list(Path(root).rglob("annotations.csv"))
    if not csvs:
        sys.exit(f"[mhist] annotations.csv not found under {root}")
    ann = pd.read_csv(csvs[0])

    def col(*keys):
        for c in ann.columns:
            if all(k in c.lower() for k in keys):
                return c
        sys.exit(f"[mhist] no column matching {keys} in {list(ann.columns)}")

    name_c, label_c, part_c = col("image"), col("majority"), col("partition")
    ssa_c = next((c for c in ann.columns if "annotator" in c.lower()), None)
    by_name = {p.name: p for p in _images(root)}
    rows, missing = [], 0
    for _, r in ann.iterrows():
        p = by_name.get(r[name_c])
        if p is None:
            missing += 1
            continue
        row = {"path": str(p), "label": r[label_c], "split": r[part_c]}
        if ssa_c:
            row["n_annotators_ssa"] = r[ssa_c]
        rows.append(row)
    if missing:
        print(f"[mhist] warning: {missing} images listed in annotations.csv were not found")
    return pd.DataFrame(rows)


BREAKHIS_SUBTYPES = {
    "A": "adenosis", "F": "fibroadenoma", "PT": "phyllodes_tumor", "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma", "LC": "lobular_carcinoma", "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}
BREAKHIS_RE = re.compile(r"^SOB_([BM])_([A-Z]+)-(\d+)-([0-9A-Za-z]+)-(\d+)-(\d+)\.png$")


def index_breakhis(root):
    rows = []
    for f in _images(root):
        m = BREAKHIS_RE.match(f.name)
        if not m:
            print(f"[breakhis] skipping unrecognised file name: {f.name}")
            continue
        bm, sub, year, pid, mag, _ = m.groups()
        rows.append({
            "path": str(f),
            "label": "benign" if bm == "B" else "malignant",
            "subtype": BREAKHIS_SUBTYPES.get(sub, sub),
            "magnification": int(mag),
            "patient_id": f"{year}-{pid}",
        })
    return pd.DataFrame(rows)


INDEXERS = {"bach": index_bach, "crc": index_crc, "mhist": index_mhist, "breakhis": index_breakhis}


# ----------------------------------------------------------------------------- data
def make_tiles(img, tile_size):
    w, h = img.size
    if w < tile_size or h < tile_size:
        return [img]
    nx, ny = w // tile_size, h // tile_size
    x0, y0 = (w - nx * tile_size) // 2, (h - ny * tile_size) // 2
    return [
        img.crop((x0 + i * tile_size, y0 + j * tile_size,
                  x0 + (i + 1) * tile_size, y0 + (j + 1) * tile_size))
        for j in range(ny) for i in range(nx)
    ]


class ImageDataset(Dataset):
    def __init__(self, paths, mode, tile_size):
        self.paths, self.mode, self.tile_size = list(paths), mode, tile_size
        self.tf = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE),
                              interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        crops = make_tiles(img, self.tile_size) if self.mode == "tile" else [img]
        return torch.stack([self.tf(c) for c in crops]), i


def collate(batch):
    tiles = torch.cat([b[0] for b in batch])
    counts = [b[0].shape[0] for b in batch]
    idx = [b[1] for b in batch]
    return tiles, counts, idx


# ----------------------------------------------------------------------------- extraction
@torch.inference_mode()
def extract(model, paths, mode, args, device, amp_dtype):
    ds = ImageDataset(paths, mode, args.tile_size)
    images_per_batch = args.batch_size if mode != "tile" else max(1, args.batch_size // 16)
    loader = DataLoader(ds, batch_size=images_per_batch, shuffle=False, collate_fn=collate,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    n = len(ds)
    feats = torch.empty(n, 1536)
    tile_feats = [None] * n
    t0 = time.time()
    for b, (tiles, counts, idx) in enumerate(loader):
        tiles = tiles.to(device, non_blocking=True)
        outs = []
        for s in range(0, tiles.shape[0], args.batch_size):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outs.append(model(tiles[s:s + args.batch_size]).float().cpu())
        out = torch.cat(outs)
        for i, f in zip(idx, torch.split(out, counts)):
            feats[i] = f.mean(0)
            if mode == "tile":
                tile_feats[i] = f
        if b % 20 == 0:
            done = min((b + 1) * images_per_batch, n)
            print(f"    {done}/{n} images  ({time.time() - t0:.0f}s)", flush=True)
    result = {"features": feats}
    if mode == "tile":
        result["tile_counts"] = torch.tensor([t.shape[0] for t in tile_feats])
        result["tile_features"] = torch.cat(tile_feats)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to UNI2-h pytorch_model.bin")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bach-root")
    p.add_argument("--crc-root")
    p.add_argument("--mhist-root")
    p.add_argument("--breakhis-root")
    p.add_argument("--modes", default="resize,tile",
                   help="comma-separated modes for BACH/BreakHis: resize, tile (default both)")
    p.add_argument("--tile-size", type=int, default=224, help="tile size in native pixels (tile mode)")
    p.add_argument("--batch-size", type=int, default=128, help="tiles per forward pass")
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    roots = {k: getattr(args, f"{k}_root") for k in INDEXERS if getattr(args, f"{k}_root")}
    if not roots:
        sys.exit("give at least one --<dataset>-root")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    assert all(m in {"resize", "tile"} for m in modes), modes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU visible, running on CPU (this will be very slow)")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    print(f"device={device}  amp={amp_dtype}  workers={args.num_workers}")

    model = build_model(args.weights).to(device)

    for name, root in roots.items():
        print(f"\n=== {name}  ({root})")
        meta = INDEXERS[name](root)
        if meta.empty:
            print(f"[{name}] no images found, skipping")
            continue
        print(f"    {len(meta)} images, labels: {meta['label'].value_counts().to_dict()}")
        out = Path(args.out_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        rel = [os.path.relpath(p, root) for p in meta["path"]]
        meta.assign(path=rel).to_csv(out / "metadata.csv", index=False)

        for mode in (["native"] if name in NATIVE_224 else modes):
            f = out / f"uni2h_{mode}.pt"
            if f.exists() and not args.overwrite:
                print(f"    {f.name} exists, skipping (use --overwrite)")
                continue
            print(f"  mode={mode}")
            res = extract(model, meta["path"].tolist(), "resize" if mode == "native" else mode,
                          args, device, amp_dtype)
            res.update(labels=meta["label"].astype(str).tolist(), paths=rel, mode=mode,
                       tile_size=args.tile_size if mode == "tile" else None, model="UNI2-h")
            torch.save(res, f)
            print(f"    saved {f}  features {tuple(res['features'].shape)}")


if __name__ == "__main__":
    main()