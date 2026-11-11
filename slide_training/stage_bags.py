#!/usr/bin/env python
"""
Copy slide patch features to fast storage as fp16 .npy files (one per slide), for train_slide_text.py.

Reading ~90 MB of float32 per slide from Lustre ($SCRATCH) every epoch makes training I/O-bound. This
script reads each slide once, with many parallel readers, and writes <slide_id>.npy (fp16, half the size)
to --dst. TCGABags then memory-maps those files and reads only the sampled patches.

Two typical uses:
  1) once, persistent fp16 copy on scratch (CPU job; later jobs copy plain files, which is faster):
       python stage_bags.py --src $SCRATCH/tcga_trident/.../features_uni_v2 --dst $SCRATCH/tcga_bags_fp16 \
                            --txt-feats tcga_reports_BioLORD-2023.pt --workers 32
  2) at the start of every training job, to the node-local disk (wiped when the job ends):
       python stage_bags.py --src $SCRATCH/tcga_bags_fp16 --dst $SLURM_TMPDIR/bags --workers 16 \
                            --txt-feats tcga_reports_BioLORD-2023.pt
     then train with --features-dir $SLURM_TMPDIR/bags

--src may hold TRIDENT .h5 files (converted to fp16) or .npy files from a previous run (copied as is).
Only slides that have a report in --txt-feats are staged (if given). Already staged files are skipped,
so an interrupted run can simply be restarted.
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def stage_one(src_dir, dst_dir, name):
    import numpy as np
    out = os.path.join(dst_dir, name + ".npy")
    if os.path.exists(out):
        return name, 0, "skipped"
    tmp = out + ".part.npy"
    npy = os.path.join(src_dir, name + ".npy")
    if os.path.exists(npy):
        shutil.copyfile(npy, tmp)
    else:
        import h5py
        with h5py.File(os.path.join(src_dir, name + ".h5"), "r") as f:
            x = f["features"][:]
        np.save(tmp, x.astype(np.float16))
    os.replace(tmp, out)
    return name, os.path.getsize(out), "ok"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="folder with <slide_id>.h5 (TRIDENT) or <slide_id>.npy")
    p.add_argument("--dst", required=True)
    p.add_argument("--txt-feats", default=None, help="only stage slides that have a report in this file")
    p.add_argument("--workers", type=int, default=16, help="parallel readers (Lustre likes many)")
    args = p.parse_args()

    if not args.dst or args.dst.startswith("/bags"):
        sys.exit("--dst looks wrong (is $SLURM_TMPDIR set in this shell?)")
    os.makedirs(args.dst, exist_ok=True)
    names = {}
    for f in os.listdir(args.src):
        if f.endswith(".npy") and not f.endswith(".part.npy"):
            names[f[:-4]] = 1
        elif f.endswith(".h5"):
            names.setdefault(f[:-3], 1)
    if args.txt_feats:
        import torch
        keep = set(torch.load(args.txt_feats, map_location="cpu")["slide_ids"])
        names = {n: 1 for n in names if n in keep}
    names = sorted(names)
    free = shutil.disk_usage(args.dst).free
    print(f"{len(names):,} slides: {args.src} -> {args.dst}  ({free / 1e9:,.0f} GB free there)", flush=True)

    t0, done, nbytes, fails = time.time(), 0, 0, []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(stage_one, args.src, args.dst, n): n for n in names}
        for fut in as_completed(futs):
            try:
                _, b, _ = fut.result()
                nbytes += b
            except Exception as e:          # keep going; report at the end
                fails.append(f"{futs[fut]}\t{e}")
            done += 1
            if done % 250 == 0 or done == len(names):
                el = time.time() - t0
                print(f"  {done:,}/{len(names):,}  {nbytes / 1e9:,.0f} GB written  {nbytes / 1e9 / max(el, 1e-9):.2f} GB/s  "
                      f"ETA {(len(names) - done) * el / done / 60:.0f} min", flush=True)
    if fails:
        with open(os.path.join(args.dst, "stage_failures.txt"), "w") as f:
            f.write("\n".join(fails) + "\n")
        print(f"{len(fails)} slides failed, see {args.dst}/stage_failures.txt")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
