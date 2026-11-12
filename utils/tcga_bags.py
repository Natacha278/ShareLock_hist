"""
TCGA slide-report dataset that keeps the full bag of patch features (no pooling).

    from tcga_bags import TCGABags, collate_bags
    ds = TCGABags(features_dir, "tcga_reports_BioLORD-2023.pt", split="train", max_patches=8192)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=8, collate_fn=collate_bags)
    for patches, mask, txt, idx in dl:   # patches [B, Nmax, 1536] fp16, mask [B, Nmax] bool, txt [B, D]
        ...

* slides = those with BOTH a feature file and a report. features_dir may hold
    - TRIDENT <slide_id>.h5 (dataset 'features' [N, 1536]), or
    - <slide_id>.npy fp16 copies made by stage_bags.py (preferred when both exist): memory-mapped, only the
      sampled patches are read -> fast on node-local disk ($SLURM_TMPDIR)
* split by patient (crc32 of the TCGA barcode), the same rule as build_tcga_pairs.py
* max_patches: random subset of patches per slide (re-drawn every time the slide is loaded) to bound
  GPU memory; None = all patches (use for evaluation)
* txt_view: 'full' (whole report) or 'sentmean' (mean of sentence embeddings), from extract_report_features.py
"""
import os
import zlib

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def split_of(barcode, val_frac=0.1, test_frac=0.1):
    u = (zlib.crc32(barcode.encode()) % 10_000) / 10_000
    return "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"


class TCGABags(Dataset):
    def __init__(self, features_dir, txt_feats, split="train", txt_view="full", max_patches=None,
                 val_frac=0.1, test_frac=0.1, min_patches=1):
        t = torch.load(txt_feats, map_location="cpu") if isinstance(txt_feats, str) else txt_feats
        files = os.listdir(features_dir)
        self.ext = {s[:-4]: ".npy" for s in files if s.endswith(".npy")}
        for s in files:
            if s.endswith(".h5"):
                self.ext.setdefault(s[:-3], ".h5")
        have = set(self.ext)
        key = {"full": "features", "sentmean": "features_sentmean"}[txt_view]
        barcodes = t.get("barcodes") or [s[:12] for s in t["slide_ids"]]
        rows = [i for i, s in enumerate(t["slide_ids"])
                if s in have and (split == "all" or split_of(barcodes[i], val_frac, test_frac) == split)]
        self.features_dir, self.max_patches, self.min_patches = features_dir, max_patches, min_patches
        self.slide_ids = [t["slide_ids"][i] for i in rows]
        self.barcodes = [barcodes[i] for i in rows]
        self.projects = [t["projects"][i] for i in rows] if t.get("projects") else [""] * len(rows)
        self.texts = [t["texts"][i] for i in rows]
        self.txt = t[key][torch.tensor(rows, dtype=torch.long)].float() if rows else torch.empty(0)
        self.txt_model, self.txt_view, self.split = t.get("model"), txt_view, split

    def __len__(self):
        return len(self.slide_ids)

    @property
    def txt_dim(self):
        return self.txt.shape[1]

    @property
    def img_dim(self):
        x = self.read(0, mmap=True)
        return x.shape[1]

    def read(self, i, mmap=False):
        s = self.slide_ids[i]
        path = os.path.join(self.features_dir, s + self.ext[s])
        if self.ext[s] == ".npy":
            return np.load(path, mmap_mode="r")
        with h5py.File(path, "r") as f:
            return f["features"][:]

    def load_patches(self, i):
        x = self.read(i)
        if self.max_patches and len(x) > self.max_patches:
            x = x[np.sort(np.random.choice(len(x), self.max_patches, replace=False))]   # .npy: reads only these rows
        else:
            x = np.array(x)                                  # materialise (memory-mapped .npy is read-only)
        if len(x) < self.min_patches:                 # empty bag: one zero patch, keeps shapes valid
            x = np.zeros((1, x.shape[1] if x.ndim == 2 else 1536), dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).half()

    def __getitem__(self, i):
        return self.load_patches(i), self.txt[i], i


def collate_bags(batch):
    patches, txt, idx = zip(*batch)
    n = max(p.shape[0] for p in patches)
    out = torch.zeros(len(patches), n, patches[0].shape[1], dtype=patches[0].dtype)
    mask = torch.zeros(len(patches), n, dtype=torch.bool)
    for b, p in enumerate(patches):
        out[b, :len(p)] = p
        mask[b, :len(p)] = True
    return out, mask, torch.stack(txt), torch.tensor(idx)
