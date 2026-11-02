"""
Filter Quilt-1M images using the QuiltCleaner impurity predictions
(Aubreville et al., MIDL 2024; https://github.com/DeepMicroscopy/QuiltCleaner)
and count what remains.

Inputs (from the QuiltCleaner repo):
  predictions_quiltcleaner.csv.bz2   classifier probabilities for 99% of the images
  {train,val,test}_annotations.csv   manual labels for the other 1%
Optional:
  the Quilt-1M metadata CSV (e.g. quilt_1M_lookup.csv) to also count image-caption pairs

Usage:
  git clone https://github.com/DeepMicroscopy/QuiltCleaner.git
  python filter_quilt_impurities.py --qc-dir QuiltCleaner \
      --quilt-csv ~/scratch/quilt1m/quilt_1M_lookup.csv \
      --threshold 0.5 --out quilt_clean_images.csv

  # only drop some categories (e.g. keep images with arrows or control buttons):
  python filter_quilt_impurities.py --qc-dir QuiltCleaner \
      --categories "Persons/Photos" "Image Perspective/Quality" "Multi-Panel Image"
"""
import argparse
import os

import pandas as pd

CATEGORIES = [
    "Persons/Photos",
    "Desktop/Windows/SlideViewer",
    "Text/Logo in Image",
    "Arrow/Annotation",
    "Image Perspective/Quality",
    "Additional (On-Slide) Overview",
    "Additional Control Elements",
    "Multi-Panel Image",
]


def load_scores(qc_dir):
    """One row per image: Filename + a 0..1 score per category.
    Predictions give probabilities; manual annotations give 0/1."""
    pred = pd.read_csv(os.path.join(qc_dir, "predictions_quiltcleaner.csv.bz2"), index_col=0)
    pred = pred[["Filename"] + CATEGORIES]
    pred["source"] = "predicted"

    ann = pd.concat(
        [pd.read_csv(os.path.join(qc_dir, f"{s}_annotations.csv")) for s in ("train", "val", "test")]
    )
    ann["Filename"] = ann["Image"].str.replace("images/", "", regex=False)
    # The annotation files also contain extra TCGA crops added by the authors; not part of Quilt-1M.
    ann = ann[~ann["Filename"].str.startswith("TCGA")]
    ann[CATEGORIES] = ann[CATEGORIES].astype(float)
    ann = ann[["Filename"] + CATEGORIES]
    ann["source"] = "manual"

    return pd.concat([pred, ann], ignore_index=True).drop_duplicates("Filename")


def find_image_col(df):
    for c in ("image_path", "image", "filename", "Filename", "image_name"):
        if c in df.columns:
            return c
    raise ValueError(f"No image filename column found in Quilt CSV; columns are {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-dir", required=True, help="path to the cloned QuiltCleaner repo")
    ap.add_argument("--quilt-csv", help="Quilt-1M metadata CSV, to count image-caption pairs")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="an image is impure if any selected category score >= threshold")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES,
                    help="which impurity categories to filter on (default: all 8)")
    ap.add_argument("--out", help="write the list of kept images (with scores) to this CSV")
    args = ap.parse_args()

    scores = load_scores(args.qc_dir)
    max_score = scores[args.categories].max(axis=1)
    keep = max_score < args.threshold

    print(f"Images with QuiltCleaner scores : {len(scores):,} "
          f"({(scores.source == 'predicted').sum():,} predicted, {(scores.source == 'manual').sum():,} manual)")
    print(f"Filtering on {len(args.categories)} categories, threshold {args.threshold}")
    print(f"Images kept                     : {keep.sum():,} ({keep.mean():.1%})")
    print(f"Images removed                  : {(~keep).sum():,}")

    print("\nShare of images flagged per category:")
    for c in args.categories:
        print(f"  {c:32s} {(scores[c] >= args.threshold).mean():6.1%}")

    print("\nImages kept at other thresholds:")
    for t in (0.3, 0.5, 0.7, 0.9):
        print(f"  {t:.1f}: {(max_score < t).sum():,}")

    kept = scores[keep]

    if args.quilt_csv:
        quilt = pd.read_csv(args.quilt_csv, low_memory=False)
        col = find_image_col(quilt)
        names = quilt[col].astype(str).map(os.path.basename)
        in_qc = names.isin(scores["Filename"])
        pairs_kept = names.isin(kept["Filename"])
        print(f"\nQuilt CSV rows (image-caption pairs): {len(quilt):,}")
        print(f"  rows whose image has a score    : {in_qc.sum():,}")
        print(f"  pairs kept                      : {pairs_kept.sum():,}")
        print(f"  unique images among kept pairs  : {names[pairs_kept].nunique():,}")
        for t in (0.3, 0.5, 0.7, 0.9):
            ok = set(scores.loc[max_score < t, "Filename"])
            print(f"  pairs kept at {t:.1f}: {names.isin(ok).sum():,}")

    if args.out:
        kept.to_csv(args.out, index=False)
        print(f"\nWrote {len(kept):,} kept images to {args.out}")


if __name__ == "__main__":
    main()