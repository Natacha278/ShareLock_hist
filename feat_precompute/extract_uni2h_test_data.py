#!/usr/bin/env python
"""
Precompute UNI2-h (MahmoodLab/UNI2-h) features for BACH, CRC-VAL-HE-7K, MHIST, BreakHis and
BRACS (RoI set).

Outputs, per dataset, in <out-dir>/<dataset>/:
  metadata.csv          one row per image: path (relative to dataset root), label, extra fields
  uni2h_<mode>.pt       torch.save'd dict with
      features        FloatTensor [N, 1536]  one embedding per image (row i <-> metadata row i)
      labels          list[str]
      paths           list[str]
      tile_features   FloatTensor [sum T_i, 1536]   (tile mode only)
      tile_counts     LongTensor  [N]               (tile mode only; torch.split(tile_features, tile_counts.tolist()))

Modes (only relevant for BACH 2048x1536, BreakHis 700x460 and BRACS RoIs of variable size; CRC and
MHIST are already 224x224 and get a single file, uni2h_native.pt):
  resizecrop  the usual benchmark rule (eva ResizeAndCrop, THUNDER / timm eval transform for UNI2-h):
              shorter side resized to 224 keeping the aspect ratio, then a 224x224 centre crop.
              Use this one to compare with published linear-probe / zero-shot numbers.
  resize      the whole image resized to 224x224 (no crop: all tissue kept, aspect ratio not preserved)
  tile        non-overlapping --tile-size tiles at native resolution (centred grid), each resized to
              224 if needed, embedded, then mean-pooled into one image embedding. An image smaller
              than one tile along a side is padded with white to the tile size first (keeps the scale).
For 224x224 images (CRC, MHIST) all three are the identity, hence the single uni2h_native.pt.

BRACS RoIs are scanned at 40x (0.25 um/px), so they are tiled with --bracs-tile-size (default 448):
448 px at 40x resized to 224 is 0.5 um/px, the 20x scale UNI2-h was mostly trained at.
metadata.csv for BRACS has: label (7-class subtype: N, PB, UDH, FEA, ADH, DCIS, IC),
group (3-class: benign, atypical, malignant), split (train/val/test), wsi_id, width, height.
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


BRACS_GROUPS = {
    "N": "benign", "PB": "benign", "UDH": "benign",
    "FEA": "atypical", "ADH": "atypical",
    "DCIS": "malignant", "IC": "malignant",
}
BRACS_RE = re.compile(r"^BRACS_(\d+)_([A-Za-z]+)_(\d+)\.png$")
BRACS_SPLITS = {"train": "train", "training": "train",
                "val": "val", "validation": "val",
                "test": "test", "testing": "test"}


def index_bracs(root):
    """BRACS RoI set. Works with the official FTP layout (.../train/0_N/BRACS_1238_N_3.png, or the
    older Group_BT/Type_N layout): the subtype comes from the file name (folder name as fallback),
    the split from the nearest train/val/test ancestor. 'previous_versions' is ignored."""
    rows, skipped_old, unknown = [], 0, []
    for f in _images(root):
        parts = [q.lower() for q in f.relative_to(root).parts[:-1]]
        if any("previous" in q for q in parts):
            skipped_old += 1
            continue
        m = BRACS_RE.match(f.name)
        if m:
            wsi, sub = f"BRACS_{m.group(1)}", m.group(2).upper()
        else:  # fall back to a class folder such as "0_N" or "Type_N"
            wsi, sub = None, f.parent.name.split("_")[-1].upper()
        if sub not in BRACS_GROUPS:
            unknown.append(f.name)
            continue
        split = next((BRACS_SPLITS[q] for q in reversed(parts) if q in BRACS_SPLITS), "unknown")
        with Image.open(f) as im:  # reads the header only
            w, h = im.size
        rows.append({"path": str(f), "label": sub, "group": BRACS_GROUPS[sub], "split": split,
                     "wsi_id": wsi, "width": w, "height": h})
    if skipped_old:
        print(f"[bracs] ignored {skipped_old} images under previous_versions")
    if unknown:
        print(f"[bracs] skipped {len(unknown)} images with no recognised subtype, e.g. {unknown[:3]}")
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"[bracs] splits: {df['split'].value_counts().to_dict()}  "
              f"median size {int(df['width'].median())}x{int(df['height'].median())}  "
              f"max {df['width'].max()}x{df['height'].max()}")
    return df


INDEXERS = {"bach": index_bach, "crc": index_crc, "mhist": index_mhist, "breakhis": index_breakhis,
            "bracs": index_bracs}


# ----------------------------------------------------------------------------- data
def make_tiles(img, tile_size):
    w, h = img.size
    if w < tile_size or h < tile_size:
        # pad the short side(s) with white so tiles keep the native scale instead of being stretched
        pw, ph = max(w, tile_size), max(h, tile_size)
        canvas = Image.new("RGB", (pw, ph), (255, 255, 255))
        canvas.paste(img, ((pw - w) // 2, (ph - h) // 2))
        img, (w, h) = canvas, (pw, ph)
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
        bicubic = transforms.InterpolationMode.BICUBIC
        if mode == "resizecrop":   # int size = shorter side -> 224, aspect ratio kept
            geometry = [transforms.Resize(INPUT_SIZE, interpolation=bicubic, antialias=True),
                        transforms.CenterCrop(INPUT_SIZE)]
        else:                      # resize / tile / native: (squash) to 224x224
            geometry = [transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=bicubic, antialias=True)]
        self.tf = transforms.Compose(geometry + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

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
def extract(model, paths, mode, args, device, amp_dtype, tile_size):
    ds = ImageDataset(paths, mode, tile_size)
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
        c = result["tile_counts"].float()
        print(f"    tiles per image: mean {c.mean():.1f}  median {c.median():.0f}  max {c.max():.0f}  "
              f"total {int(c.sum())}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to UNI2-h pytorch_model.bin")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bach-root")
    p.add_argument("--crc-root")
    p.add_argument("--mhist-root")
    p.add_argument("--breakhis-root")
    p.add_argument("--bracs-root", help="BRACS 'Region of Interest Set' / BRACS_RoI folder")
    p.add_argument("--modes", default="resizecrop,resize,tile",
                   help="comma-separated modes for BACH/BreakHis/BRACS: resizecrop (benchmark rule), "
                        "resize, tile (default all three)")
    p.add_argument("--tile-size", type=int, default=224, help="tile size in native pixels (tile mode)")
    p.add_argument("--bracs-tile-size", type=int, default=448,
                   help="tile size for BRACS (40x scans; 448 -> 224 gives the 20x scale)")
    p.add_argument("--batch-size", type=int, default=128, help="tiles per forward pass")
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    roots = {k: getattr(args, f"{k}_root") for k in INDEXERS if getattr(args, f"{k}_root")}
    if not roots:
        sys.exit("give at least one --<dataset>-root")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    assert all(m in {"resizecrop", "resize", "tile"} for m in modes), modes

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
            tile_size = args.bracs_tile_size if name == "bracs" else args.tile_size
            print(f"  mode={mode}" + (f"  tile_size={tile_size}" if mode == "tile" else ""))
            res = extract(model, meta["path"].tolist(), "resize" if mode == "native" else mode,
                          args, device, amp_dtype, tile_size)
            res.update(labels=meta["label"].astype(str).tolist(), paths=rel, mode=mode,
                       tile_size=tile_size if mode == "tile" else None, model="UNI2-h")
            torch.save(res, f)
            print(f"    saved {f}  features {tuple(res['features'].shape)}")


if __name__ == "__main__":
    main()