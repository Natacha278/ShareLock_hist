#!/usr/bin/env python
"""
Side-by-side table of the video-mean centering runs of fit_procrustes.py (--group-center ...) and
their global-centering references, read from <runs-dir>/*/config.json + results.json.

Only plain runs are compared: no --hub-correction, no --max-images, --test-center quilt, so the
group centering is the only difference. For every method (same tag without the _gc... suffix) the
most recent global-centering run is the reference, and d_* columns give the change against it.

  python summarize_group_runs.py --runs-dir $ROOT/ShareLock_hist/runs --out group_center_table.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def load_runs(runs_dir):
    rows = []
    for d in sorted(Path(runs_dir).iterdir()):
        cf, rf = d / "config.json", d / "results.json"
        if not (cf.exists() and rf.exists()):
            continue
        cfg, res = json.loads(cf.read_text()), json.loads(rf.read_text())
        a, model = cfg.get("args", {}), cfg.get("model", {})
        if model.get("method") not in ("orthogonal", "linear", "cca"):
            continue
        if a.get("hub_correction", "none") != "none" or a.get("max_images") or a.get("test_center", "quilt") != "quilt":
            continue
        tag = model.get("tag", "")
        gc = model.get("group_center", "none")
        grp = res.get("fit", {}).get("group", {})
        zs = {k: v["balanced_acc"] for k, v in res.get("zero_shot", {}).items()}
        val = res.get("val_retrieval", {})
        rows.append({
            "run": d.name, "start": cfg.get("env", {}).get("start", ""),
            "base": tag.split("_gc")[0], "group_center": gc,
            "kappa": a.get("group_kappa") if gc != "none" else None,
            "sides": a.get("group_sides") if gc != "none" else None,
            "between_img": grp.get("between_var_frac_img"), "between_txt": grp.get("between_var_frac_txt"),
            "frac_imgs_centred": grp.get("frac_train_images_centred"),
            "val_mean_recall": val.get("mean_recall"), "val_foscttm_i2t": val.get("foscttm_i2t"),
            **{f"arch_{k}": v.get("mean_recall") for k, v in res.get("retrieval", {}).items()},
            **zs, "zs_mean": sum(zs.values()) / len(zs) if zs else None,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", default=None, help="optional CSV path")
    args = p.parse_args()

    df = load_runs(args.runs_dir)
    if df.empty:
        raise SystemExit(f"no closed-form runs in {args.runs_dir}")
    if (df.group_center != "none").sum() == 0:
        print("note: no --group-center runs yet; showing the references only")
    ref = df[df.group_center == "none"].sort_values("start").groupby("base").tail(1).set_index("base")
    metrics = [c for c in ("val_mean_recall", "zs_mean") if c in df] + \
              [c for c in df.columns if "/" in c or c.startswith("arch_")]
    for c in metrics:
        df[f"d_{c}"] = df[c] - df.base.map(ref[c]) if c in ref else None
    df["_order"] = df.group_center.map({"none": 0, "video": 1, "random": 2})
    df = df.sort_values(["base", "_order", "sides", "kappa"], na_position="first").drop(columns="_order")
    keep = ["base", "group_center", "kappa", "sides", "between_img", "between_txt", "val_mean_recall",
            "d_val_mean_recall", "zs_mean", "d_zs_mean"]
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(df[[c for c in keep if c in df]].to_string(index=False))
        tasks = [c for c in df.columns if "/" in c and not c.startswith("d_")]
        if tasks:
            print("\nchange in balanced accuracy vs global centering, per benchmark:")
            print(df[df.group_center != "none"][["base", "group_center", "kappa", "sides"] +
                                                 [f"d_{c}" for c in tasks]].to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()