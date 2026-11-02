"""
Extract UNI2-h image features for one Quilt-1M zip part.

Usage:
    python extract_uni2h.py --zip images_part_1_resized.zip \
        --ckpt /path/to/uni2-h/pytorch_model.bin \
        --out feats/uni2h_part_1.npz

Output .npz contains:
    feats : (N, 1536) float16  -- CLS embedding from UNI2-h
    names : (N,)      str      -- image filename inside the zip (join key with the CSV)
"""
import argparse
import io
import os
import zipfile

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXT = (".jpg", ".jpeg", ".png")


def build_uni2h(ckpt_path):
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
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
    model = timm.create_model(pretrained=False, **timm_kwargs)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval()


# Same preprocessing as the UNI2-h model card (ImageNet normalization, 224px).
# Quilt images are 512x512, so Resize(224) keeps the whole image; CenterCrop is then a no-op.
TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class ZipImages(Dataset):
    def __init__(self, zip_path, transform):
        self.zip_path = zip_path
        self.transform = transform
        with zipfile.ZipFile(zip_path) as z:
            self.names = sorted(n for n in z.namelist() if n.lower().endswith(IMG_EXT))
        self.zf = None  # opened lazily so each DataLoader worker has its own handle

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path)
        name = self.names[i]
        try:
            img = Image.open(io.BytesIO(self.zf.read(name))).convert("RGB")
            ok = True
        except Exception:
            img = Image.new("RGB", (224, 224))
            ok = False
        return self.transform(img), name, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = p.parse_args()

    device = "cuda"
    model = build_uni2h(args.ckpt).to(device)

    ds = ZipImages(args.zip, TRANSFORM)
    print(f"{len(ds)} images in {args.zip}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4)

    feats, names, bad = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step, (x, n, ok) in enumerate(dl):
            f = model(x.to(device, non_blocking=True))
            feats.append(f.float().cpu().numpy().astype(np.float16))
            names.extend(n)
            bad.extend([nm for nm, o in zip(n, ok.tolist()) if not o])
            if step % 50 == 0:
                print(f"batch {step}/{len(dl)}", flush=True)

    feats = np.concatenate(feats)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, feats=feats, names=np.array(names))
    print(f"saved {feats.shape} to {args.out}; {len(bad)} unreadable images", flush=True)
    if bad:
        with open(args.out.replace(".npz", "_bad.txt"), "w") as fh:
            fh.write("\n".join(bad))


if __name__ == "__main__":
    main()