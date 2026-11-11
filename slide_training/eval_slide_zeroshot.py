#!/usr/bin/env python
"""
Slide-level zero-shot classification of TCGA slides with a text projection (MI-Zero style), no training.

For every slide: cosine between each (frozen, L2-normalised) UNI2-h patch and each class embedding, then
the slide score of a class = mean of its top-k patch scores (k = 1, 10, 50, 100 by default; 0 = mean over
all patches). Predicted class = argmax. Class embedding = prompts (templates x class descriptions,
zeroshot_prompts.TEMPLATES) -> text encoder -> projection -> normalise -> mean -> renormalise, exactly
as the patch-level zero-shot in train_projection.py.

Tasks (labels = TCGA project of the slide; a task is skipped if it has fewer than 2 classes present):
  lung       LUAD / LUSC
  kidney     KIRC / KIRP / KICH
  brain      GBM / LGG
  pancancer  every TCGA project present with >= --pancancer-min slides
             (COAD+READ merged into colorectal, they are not distinguishable morphologically)

Use the TEST split (default): it is split by patient with the same rule as training (tcga_bags.split_of),
so those slides and their reports were never seen during TCGA training. Reports name the diagnosis, so
evaluating on training slides would be optimistic.

Works with any train_projection-format checkpoint (project="text"): Quilt runs (model_best.pt),
train_slide_text / train_limit / train_limit_mix (*_projection.pt), train_o1b_em.py (model_best.pt).

Results go to <out-dir> (default <checkpoint folder>/slide_zs_<timestamp>/): results.json,
confusion_<task>_top<k>.csv, prompts.json; one row is appended to <runs-dir>/summary_slide_zs.csv.

Example:
  python eval_slide_zeroshot.py --ckpt .../runs_o1b/<run>/model_best.pt \
    --features-dir $SLURM_TMPDIR/bags --txt-feats .../tcga_reports_BioLORD-2023.pt \
    --txt-model .../weights/BioLORD-2023
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from zeroshot_prompts import TEMPLATES

# TCGA project -> class descriptions (synonyms; each combined with every template)
PROJECT_CLASSES = {
    "ACC": ["adrenocortical carcinoma"],
    "BLCA": ["urothelial carcinoma of the bladder", "bladder urothelial carcinoma"],
    "BRCA": ["invasive breast carcinoma", "breast cancer"],
    "CESC": ["cervical squamous cell carcinoma", "cervical cancer"],
    "CHOL": ["cholangiocarcinoma", "bile duct carcinoma"],
    "COLORECTAL": ["colorectal adenocarcinoma", "colon adenocarcinoma"],
    "DLBC": ["diffuse large B-cell lymphoma"],
    "ESCA": ["esophageal carcinoma", "esophageal adenocarcinoma or squamous cell carcinoma"],
    "GBM": ["glioblastoma", "glioblastoma multiforme"],
    "HNSC": ["head and neck squamous cell carcinoma"],
    "KICH": ["chromophobe renal cell carcinoma"],
    "KIRC": ["clear cell renal cell carcinoma"],
    "KIRP": ["papillary renal cell carcinoma"],
    "LGG": ["lower grade glioma", "low-grade glioma", "diffuse astrocytoma or oligodendroglioma"],
    "LIHC": ["hepatocellular carcinoma", "liver cancer"],
    "LUAD": ["lung adenocarcinoma", "adenocarcinoma of the lung"],
    "LUSC": ["lung squamous cell carcinoma", "squamous cell carcinoma of the lung"],
    "MESO": ["mesothelioma", "malignant pleural mesothelioma"],
    "OV": ["ovarian serous carcinoma", "high-grade serous ovarian carcinoma"],
    "PAAD": ["pancreatic adenocarcinoma", "pancreatic ductal adenocarcinoma"],
    "PCPG": ["pheochromocytoma or paraganglioma"],
    "PRAD": ["prostate adenocarcinoma"],
    "SARC": ["sarcoma", "soft tissue sarcoma"],
    "SKCM": ["cutaneous melanoma", "skin melanoma"],
    "STAD": ["gastric adenocarcinoma", "stomach adenocarcinoma"],
    "TGCT": ["testicular germ cell tumor"],
    "THCA": ["papillary thyroid carcinoma", "thyroid carcinoma"],
    "THYM": ["thymoma"],
    "UCEC": ["uterine endometrial carcinoma", "endometrioid carcinoma of the uterus"],
    "UCS": ["uterine carcinosarcoma"],
    "UVM": ["uveal melanoma"],
}
SUBTYPE_TASKS = {"lung": ["LUAD", "LUSC"], "kidney": ["KIRC", "KIRP", "KICH"], "brain": ["GBM", "LGG"]}
DEFAULT_KS = (1, 10, 50, 100, 0)


def label_of(project):
    p = str(project).replace("TCGA-", "").upper()
    return "COLORECTAL" if p in ("COAD", "READ") else p


def build_tasks(projects, pancancer_min=10, tasks=("lung", "kidney", "brain", "pancancer")):
    """{task: [labels]} restricted to labels present in `projects` (TCGA projects of the evaluated slides)."""
    counts = {}
    for p in projects:
        counts[label_of(p)] = counts.get(label_of(p), 0) + 1
    out = {}
    for t in tasks:
        if t == "pancancer":
            labs = sorted(l for l, n in counts.items() if n >= pancancer_min and l in PROJECT_CLASSES)
        else:
            labs = [l for l in SUBTYPE_TASKS[t] if counts.get(l, 0) > 0]
        if len(labs) >= 2:
            out[t] = labs
    return out


@torch.no_grad()
def class_embeddings(model, encoder, tasks, device, prompt_cache=None):
    """One embedding per label used by any task -> (labels list, E [L, D] normalised)."""
    from train_projection import encode_prompts
    labels = sorted({l for labs in tasks.values() for l in labs})
    E = []
    for lab in labels:
        prompts = [t.format(d) for d in PROJECT_CLASSES[lab] for t in TEMPLATES]
        if prompt_cache is not None:
            prompt_cache[lab] = prompts
        z = model.encode_text(encode_prompts(encoder, prompts).to(device))
        E.append(F.normalize(z.mean(0), dim=-1))
    return labels, torch.stack(E)


@torch.no_grad()
def pooled_scores(patches, E, ks=DEFAULT_KS):
    """patches [N, D] (any dtype, unnormalised ok), E [L, D] -> [len(ks), L] top-k mean scores (k=0: all)."""
    z = F.normalize(patches.float(), dim=-1)
    s = z @ E.float().t()                                           # [N, L]
    kmax = min(max([k for k in ks if k > 0] or [1]), len(s))
    top = s.topk(kmax, dim=0).values                                # [kmax, L], descending
    return torch.stack([s.mean(0) if k == 0 else top[:min(k, kmax)].mean(0) for k in ks])


def k_name(k):
    return "mean" if k == 0 else f"top{k}"


@torch.no_grad()
def slide_zero_shot(model, encoder, slides, projects, device, ks=DEFAULT_KS, pancancer_min=10,
                    out_dir=None, prompt_cache=None, tasks=("lung", "kidney", "brain", "pancancer"), verbose=True):
    """slides: iterable of (index, patches [N, D] tensor) with projects[index] = its TCGA project.
    Returns {"<task>/<topk>": {"balanced_acc", "acc", "n", "per_class_recall"}} (+ "mean_<topk>")."""
    from sklearn.metrics import confusion_matrix
    from train_projection import balanced_acc
    model.eval()
    # tasks are defined by the slides actually evaluated, so embed every known class first
    all_labels = {"all": sorted(PROJECT_CLASSES)}
    labels, E = class_embeddings(model, encoder, all_labels, device, prompt_cache)
    col = {l: i for i, l in enumerate(labels)}
    pooled, y_lab = [], []
    for i, patches in slides:
        lab = label_of(projects[i])
        if lab not in col or len(patches) == 0:
            continue
        pooled.append(pooled_scores(patches.to(device), E, ks).cpu())
        y_lab.append(lab)
    task_labels = build_tasks(y_lab, pancancer_min, tasks)
    if not task_labels:
        if verbose:
            print("  slide zero-shot: no task has >= 2 classes among these slides, skipping")
        return {}
    if prompt_cache is not None:
        used = {l for labs in task_labels.values() for l in labs}
        for l in list(prompt_cache):
            if l not in used:
                del prompt_cache[l]
    P = torch.stack(pooled)                                         # [S, len(ks), L]
    y_lab = np.array(y_lab)
    results = {}
    for task, labs in task_labels.items():
        keep = np.isin(y_lab, labs)
        if keep.sum() == 0:
            continue
        y = np.array([labs.index(l) for l in y_lab[keep]])
        cols = [col[l] for l in labs]
        for ki, k in enumerate(ks):
            pred = P[torch.from_numpy(keep)][:, ki, cols].argmax(1).numpy()
            r = {"n": int(keep.sum()), "balanced_acc": balanced_acc(y, pred), "acc": float((pred == y).mean()),
                 "per_class_recall": {labs[c]: float((pred[y == c] == c).mean()) for c in range(len(labs))
                                      if (y == c).any()},
                 "n_per_class": {labs[c]: int((y == c).sum()) for c in range(len(labs))}}
            results[f"{task}/{k_name(k)}"] = r
            if out_dir is not None:
                cm = confusion_matrix(y, pred, labels=range(len(labs)))
                with open(Path(out_dir) / f"confusion_{task}_{k_name(k)}.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["true\\pred"] + labs)
                    for lab, row in zip(labs, cm):
                        w.writerow([lab] + row.tolist())
        if verbose:
            print(f"  slide-zs {task:10s} n={int(keep.sum()):5d} ({len(labs)} classes)  "
                  + "  ".join(f"{k_name(k)} {results[f'{task}/{k_name(k)}']['balanced_acc']:.4f}" for k in ks),
                  flush=True)
    for k in ks:
        v = [r["balanced_acc"] for n, r in results.items() if n.endswith("/" + k_name(k))]
        if v:
            results[f"mean_{k_name(k)}"] = float(np.mean(v))
    return results


def flat(results):
    """{'slide_zs_<task>_<topk>': bal_acc, 'slide_zs_mean_<topk>': ...} for logs / summaries."""
    out = {}
    for n, r in results.items():
        out["slide_zs_" + n.replace("/", "_")] = r["balanced_acc"] if isinstance(r, dict) else r
    return out


def main():
    from torch.utils.data import DataLoader

    from projection import ProjectionModel
    from tcga_bags import TCGABags, collate_bags
    from train_projection import load_text_encoder
    from train_slide_text import unique_dir

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="train_projection-format checkpoint (project='text')")
    p.add_argument("--features-dir", required=True, help="fp16 .npy bags (stage_bags.py) or TRIDENT .h5 folder")
    p.add_argument("--txt-feats", required=True, help="tcga_reports_<model>.pt (slide ids, projects, split keys)")
    p.add_argument("--txt-model", required=True, help="local sentence-transformers folder (same encoder as training)")
    p.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--tasks", default="lung,kidney,brain,pancancer")
    p.add_argument("--ks", default="1,10,50,100,0", help="top-k pooling values (0 = mean over all patches)")
    p.add_argument("--pancancer-min", type=int, default=10, help="min slides of a project to enter pan-cancer")
    p.add_argument("--max-patches", type=int, default=0, help="random patch subset per slide (0 = all)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--runs-dir", default=None, help="where summary_slide_zs.csv goes (default: parent of the run folder)")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu")
    if "model_cfg" not in ck or "image" in ck["model_cfg"] or ck["model_cfg"].get("project") != "text":
        raise SystemExit(f"{ckpt_path} is not a project='text' train_projection-format checkpoint "
                         f"(for train_slide_text / train_limit runs use the *_projection.pt file)")
    model = ProjectionModel(**ck["model_cfg"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    txt_model_name = Path(args.txt_model.rstrip("/")).name
    if ck.get("txt_model") and ck["txt_model"] != txt_model_name:
        raise SystemExit(f"checkpoint was trained with {ck['txt_model']}, --txt-model is {txt_model_name}")

    ks = tuple(int(k) for k in args.ks.split(","))
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    ds = TCGABags(args.features_dir, args.txt_feats, args.split, max_patches=args.max_patches or None,
                  val_frac=args.val_frac, test_frac=args.test_frac)
    if ds.txt_model and ds.txt_model != txt_model_name:
        print(f"note: reports encoded with {ds.txt_model}; only slide ids / projects are used here")
    print(f"checkpoint {ckpt_path} (epoch {ck.get('epoch')})\n{args.split} split: {len(ds):,} slides", flush=True)

    out = Path(args.out_dir) if args.out_dir else unique_dir(ckpt_path.parent / f"slide_zs_{datetime.now():%m%d-%H%M}")
    out.mkdir(parents=True, exist_ok=True)
    keep = [i for i in range(len(ds)) if label_of(ds.projects[i]) in PROJECT_CLASSES]
    sub = torch.utils.data.Subset(ds, keep)
    dl = DataLoader(sub, batch_size=1, num_workers=args.num_workers, collate_fn=collate_bags)
    slides = ((int(idx[0]), patches[0][mask[0]]) for patches, mask, _, idx in dl)

    encoder = load_text_encoder(args.txt_model, device)
    prompts = {}
    res = slide_zero_shot(model, encoder, slides, ds.projects, device, ks, args.pancancer_min, out, prompts, tasks)
    (out / "prompts.json").write_text(json.dumps(prompts, indent=1))
    (out / "results.json").write_text(json.dumps({"ckpt": str(ckpt_path), "split": args.split, "ks": ks,
                                                  "results": res}, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    row = {"run": args.name or ckpt_path.parent.name, "ckpt": ckpt_path.name, "split": args.split,
           **{k: round(v, 4) for k, v in flat(res).items()}}
    runs_dir = Path(args.runs_dir) if args.runs_dir else ckpt_path.parent.parent
    path = runs_dir / "summary_slide_zs.csv"
    cols = ["run", "ckpt", "split"] + [f"slide_zs_{t}_{k_name(k)}" for t in ("lung", "kidney", "brain", "pancancer", "mean")
                                       for k in ks]
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nsaved results to {out}  (summary row appended to {path})")


if __name__ == "__main__":
    main()
