#!/usr/bin/env python
"""
Closed-form text -> image alignment on precomputed Quilt-1M features (no training loop), then the
same evaluation as train_projection.py: zero-shot on BACH / CRC / MHIST / BreakHis, ARCH retrieval,
Quilt validation retrieval. Results go to <runs-dir>/<run-name>_<time>/ and one row is appended to
<runs-dir>/summary.csv, so they line up with the MLP projection runs.

Methods (X = text features, Y = image features, both preprocessed as below):
  orthogonal  W = polar(X^T D Y) = U V^T with X^T D Y = U S V^T              (orthogonal Procrustes)
              W is 768 x 1536 with orthonormal rows (Stiefel), so it is an isometry of the text
              space into the UNI2-h space. This is the "orthogonal" baseline of the
              Wasserstein-Procrustes paper (Schnaus et al. 2026), solved here with all known pairs.
  linear      W = (X^T D X + lam I)^-1 X^T D Y                               (least squares / ridge)
--direction i2t maps images into the text space instead (ShareLock project="image" analogue):
  orthogonal  W_i2t = W_t2i^T (1536 x 768, orthonormal columns: a projection of UNI2-h onto 768 dims)
  linear      W_i2t = (Y^T D Y + lam I)^-1 Y^T D X  (a different regression, not the transpose)
D is a diagonal pair weighting: --weighting image gives every image the same total weight
(1 / n_captions per pair, like sampling by="image" in train_projection.py); pair weighs pairs equally.

  cca         whitening + orthogonal map + re-weighting + de-whitening + truncation, the multi-step
              framework of Artetxe, Labaka & Agirre (AAAI 2018). Both sides are mapped into a shared
              k-dim space:  Wx = Cx^(-a/2), Wy = Cy^(-a/2),  Wx Cxy Wy = U S V^T,
              text -> x Wx U_k S_k^(b/2) Dw,  image -> y Wy V_k S_k^(b/2) Dw
              --whiten a: 0 = no whitening, 1 = full whitening (then S are the canonical correlations,
              i.e. CCA); --shrink: covariance shrinkage towards a scaled identity; --cca-dim k: keep the
              top-k directions; --reweight b: weight directions by their correlation; --dewhiten: Dw
              restores the geometry of the text (src) or image (trg) space inside the shared space.
              a=0, b=0, k=768, no de-whitening ranks texts/classes for every image exactly like
              orthogonal Procrustes (same zero-shot predictions); a=1 is CCA.

Preprocessing (as in the paper, Sec. B): each side is L2-normalised, the mean of its own TRAINING
features is subtracted, and rows are L2-normalised again. The Quilt means are stored with the map and
reused for prompts and test images (--test-center self centres each test set on its own mean instead:
transductive, no labels used).

Video-mean centering (group_centering.py): --group-center video centres every TRAINING pair on the
(shrunk, --group-kappa) mean of its source group (YouTube video / tweet) instead of the global mean, so
the map is fitted on within-video contrasts. --group-center random is the same-size random-group control.
Test-time preprocessing is unchanged (global Quilt mean).

Only the Quilt train split (img_is_val == False) is used to fit; the val split gives retrieval metrics,
FOSCTTM and a linear CKA between the two spaces (the paper's predictor of alignability).

Example:
  python utils/fit_procrustes.py \
    --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
    --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
    --arch-feats $ROOT/data/patch_datasets/uni_2h_feat/arch/arch_pubmed.pt \
    --txt-model $ROOT/weights/BioLORD-2023 \
    --runs-dir $ROOT/ShareLock_hist/runs \
    --method orthogonal

  CCA sweep:   --method cca --whiten 1.0 --cca-dim 128 [--reweight 1] [--dewhiten trg]
  Hubness:     --hub-correction csls --hub-k 100 --hub-ref quilt   (any method; see train_projection.hub_penalty)
  Video mean:  --group-center video --group-kappa 4                 (any method; see group_centering.py)
"""
import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from group_centering import add_group_args, fit_group_centers, group_cfg, group_tag
from quilt_pairs import QuiltPairs
from train_projection import (HUB_METHODS, SUMMARY_COLS, append_summary, zs_summary, arch_retrieval, load_text_encoder,
                              quilt_reference, retrieval_metrics, set_seed, zero_shot)


# ----------------------------------------------------------------------------- model
class LinearAlignment(nn.Module):
    """Linear maps with per-side centring; same interface as ProjectionModel.

    A [txt_dim, k] maps text, B [img_dim, k] maps images; None leaves that side in its own space.
      t2i (orthogonal / linear): A = W [768, 1536], B = None  -> text lands in the UNI2-h space
      i2t (orthogonal / linear): A = None, B = W [1536, 768]  -> images land in the BioLORD space
      cca:                       A [768, k], B [1536, k]      -> both land in a shared k-dim space
    """

    def __init__(self, A, B, mu_txt, mu_img):
        super().__init__()
        self.register_buffer("A", None if A is None else A.float())
        self.register_buffer("B", None if B is None else B.float())
        self.register_buffer("mu_txt", mu_txt.float())
        self.register_buffer("mu_img", mu_img.float())
        self.test_center = "quilt"         # "self": centre a whole test set on its own mean (zero-shot only)

    def prep_txt(self, x):
        return F.normalize(F.normalize(x.float(), dim=-1) - self.mu_txt, dim=-1)

    def prep_img(self, y):
        y = F.normalize(y.float(), dim=-1)
        mu = y.mean(0, keepdim=True) if self.test_center == "self" else self.mu_img
        return F.normalize(y - mu, dim=-1)

    def encode_text(self, x):
        x = self.prep_txt(x)
        return x if self.A is None else F.normalize(x @ self.A, dim=-1)

    def encode_image(self, y):
        y = self.prep_img(y)
        return y if self.B is None else F.normalize(y @ self.B, dim=-1)


# ----------------------------------------------------------------------------- fitting
def pair_weights(data, pairs, weighting):
    """Weight of each training pair, summing to 1."""
    if weighting == "pair":
        w = torch.ones(len(pairs), device=pairs.device, dtype=torch.float64)
    else:                                   # every image weighs the same, split over its captions
        w = 1.0 / data.n_caps[data.pair_img[pairs]].double()
    return w / w.sum()


def sym_power(C, p, eps=1e-10):
    """C^p for a symmetric PSD matrix (eigenvalues floored at eps * largest)."""
    lam, Q = torch.linalg.eigh(C)
    lam = lam.clamp(min=eps * lam.max())
    return (Q * lam ** p) @ Q.t()


def shrink_cov(C, s):
    """Ledoit-Wolf-style shrinkage towards a scaled identity: (1 - s) C + s tr(C)/d I."""
    d = C.shape[0]
    return (1 - s) * C + s * torch.trace(C) / d * torch.eye(d, dtype=C.dtype, device=C.device)


@torch.no_grad()
def fit(data, pairs, args, chunk=65536):
    """Closed-form maps from the training pairs (moments accumulated in float64, chunk by chunk).

    Returns A (text map or None), B (image map or None), the two means and diagnostics.
    Orthogonal: polar(Y^T D X) = (polar(X^T D Y))^T, so i2t is the transpose of the t2i solution
    (a projection of UNI2-h onto a 768-d subspace). Linear: i2t is its own regression of X on Y.
    With --group-center, pass 2 centres every pair on its (shrunk) source-group mean instead of the
    global mean; the returned means, used for prompts and test images, are still the global ones.
    """
    method, direction = args.method, args.direction
    dev = data.img.device
    w = pair_weights(data, pairs, args.weighting)
    dt, di = data.txt.shape[1], data.img.shape[1]

    def chunks():
        for s in range(0, len(pairs), chunk):
            p = pairs[s:s + chunk]
            yield (p, F.normalize(data.txt[data.pair_txt[p]], dim=-1).double(),
                   F.normalize(data.img[data.pair_img[p]], dim=-1).double(), w[s:s + chunk, None])

    # pass 1: weighted means of the (normalised) training features
    mu_t = torch.zeros(dt, dtype=torch.float64, device=dev)
    mu_i = torch.zeros(di, dtype=torch.float64, device=dev)
    if not args.no_center:
        for _, x, y, ww in chunks():
            mu_t += (ww * x).sum(0)
            mu_i += (ww * y).sum(0)

    # source-group centres (None without --group-center): replace mu_t / mu_i pair by pair in pass 2
    gc = fit_group_centers(data, pairs, w, mu_t, mu_i, args, chunk=chunk, seed=args.seed)

    # pass 2: cross-covariance X^T D Y, plus the covariances needed by linear / cca
    need_x = method == "cca" or (method == "linear" and direction == "t2i")
    need_y = method == "cca" or (method == "linear" and direction == "i2t")
    C = torch.zeros(dt, di, dtype=torch.float64, device=dev)
    Cx = torch.zeros(dt, dt, dtype=torch.float64, device=dev) if need_x else None
    Cy = torch.zeros(di, di, dtype=torch.float64, device=dev) if need_y else None
    for p, x, y, ww in chunks():
        ct = mu_t if gc is None else gc.centers("t", p)
        ci = mu_i if gc is None else gc.centers("i", p)
        x = F.normalize(x - ct, dim=-1)
        y = F.normalize(y - ci, dim=-1)
        C += (ww * x).t() @ y
        if need_x:
            Cx += (ww * x).t() @ x
        if need_y:
            Cy += (ww * y).t() @ y

    diag = {"n_train_pairs": int(len(pairs)), "weighting": args.weighting}
    if gc is not None:
        diag["group"] = gc.info
    A = B = None
    if method in ("orthogonal", "linear"):
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)       # U [dt,dt], S [dt], Vh [dt,di]
        diag.update(cross_cov_singular_values_top10=S[:10].tolist(),
                    procrustes_objective_tr_S=float(S.sum()))      # = weighted mean cos(xW, y) for orthogonal
        if method == "orthogonal":
            W = U @ Vh                                             # [dt, di], orthonormal rows
            A, B = (W, None) if direction == "t2i" else (None, W.t())
        else:
            G, rhs = (Cx, C) if direction == "t2i" else (Cy, C.t())
            lam = args.ridge * torch.trace(G) / G.shape[0]         # ridge relative to the mean eigenvalue
            W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=dev), rhs)
            A, B = (W, None) if direction == "t2i" else (None, W)
    else:                                                          # cca / multi-step framework
        Cx, Cy = shrink_cov(Cx, args.shrink), shrink_cov(Cy, args.shrink)
        Wx, Wy = sym_power(Cx, -args.whiten / 2), sym_power(Cy, -args.whiten / 2)
        U, S, Vh = torch.linalg.svd(Wx @ C @ Wy, full_matrices=False)
        k = min(args.cca_dim or len(S), len(S))
        Uk, Vk, Sk = U[:, :k], Vh[:k].t(), S[:k]
        A, B = Wx @ Uk, Wy @ Vk                                    # whitening + orthogonal map
        R = torch.diag(Sk.clamp(min=0) ** (args.reweight / 2))     # re-weighting, split over both sides
        A, B = A @ R, B @ R
        if args.dewhiten != "none":                                # de-whitening in the shared space
            Dw = Uk.t() @ sym_power(Cx, args.whiten / 2) @ Uk if args.dewhiten == "src" \
                else Vk.t() @ sym_power(Cy, args.whiten / 2) @ Vk
            A, B = A @ Dw, B @ Dw
        diag.update(singular_values_top10=S[:10].tolist(),         # canonical correlations when whiten = 1
                    singular_values_kept_sum=float(Sk.sum()), cca_dim=k)
    f = lambda M: None if M is None else M.float()
    return f(A), f(B), mu_t.float(), mu_i.float(), diag


# ----------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def foscttm(model, img, txt, gt):
    """Fraction Of Samples Closer Than the True Match (0 = perfect, 0.5 = random), both directions."""
    sim = model.encode_image(img) @ model.encode_text(txt).t()
    out = {}
    for name, s, g in (("i2t", sim, gt), ("t2i", sim.t(), gt.t())):
        best = s.masked_fill(~g, -float("inf")).max(1).values[:, None]
        wrong = (~g).sum(1).clamp(min=1)
        closer = (((s > best) & ~g).sum(1) + 0.5 * ((s == best) & ~g).sum(1)) / wrong
        out[f"foscttm_{name}"] = closer.mean().item()
    return out


@torch.no_grad()
def linear_cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return float((b.t() @ a).norm() ** 2 / ((a.t() @ a).norm() * (b.t() @ b).norm()))


@torch.no_grad()
def val_diagnostics(model, data, seed, n_cka=10_000):
    """Matched-pair cosine and CKA on val images with one random caption each."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    imgs = data.images["val"]
    imgs = imgs[torch.randperm(len(imgs), generator=g)[:n_cka].to(imgs.device)]
    txt = data.txt[data.random_caption(imgs)]
    x, y = model.prep_txt(txt), model.prep_img(data.img[imgs])          # raw (centred) spaces, for CKA
    z_txt, z_img = model.encode_text(txt), model.encode_image(data.img[imgs])   # shared space, either direction
    return {"val_matched_cos": float((z_txt * z_img).sum(1).mean()),
            "val_cka_before": linear_cka(x, y),                    # alignability of the raw spaces
            "n_cka": int(len(imgs))}


# ----------------------------------------------------------------------------- main
def method_tag(args):
    if args.method == "cca":
        t = (f"cca_a{args.whiten:g}_k{args.cca_dim or 'all'}"
             + (f"_rw{args.reweight:g}" if args.reweight else "")
             + (f"_dw{args.dewhiten}" if args.dewhiten != "none" else ""))
    else:
        t = f"{args.method}_{args.direction}"
    return t + group_tag(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="file from build_quilt_pairs.py")
    p.add_argument("--test-feats", required=True, help="--out-dir of extract_uni2h_test_data.py")
    p.add_argument("--txt-model", required=True, help="text encoder used for --pairs (for the prompts)")
    p.add_argument("--test-modes", default="resizecrop,resize,tile")
    p.add_argument("--arch-feats", nargs="*", default=[])
    p.add_argument("--method", choices=["orthogonal", "linear", "cca"], default="orthogonal")
    p.add_argument("--direction", choices=["t2i", "i2t"], default="t2i",
                   help="t2i: text -> UNI2-h space; i2t: image -> BioLORD space")
    p.add_argument("--ridge", type=float, default=1e-3, help="linear only, relative to the mean eigenvalue")
    p.add_argument("--whiten", type=float, default=1.0, help="cca: whitening strength a in [0, 1]")
    p.add_argument("--shrink", type=float, default=0.01, help="cca: covariance shrinkage in [0, 1]")
    p.add_argument("--cca-dim", type=int, default=None, help="cca: keep top-k directions (default all 768)")
    p.add_argument("--reweight", type=float, default=0.0, help="cca: weight directions by S^b (0 = off)")
    p.add_argument("--dewhiten", choices=["none", "src", "trg"], default="none",
                   help="cca: restore text (src) or image (trg) geometry in the shared space")
    p.add_argument("--hub-correction", choices=HUB_METHODS, default="none",
                   help="zero-shot only: subtract a per-class hubness penalty (train_projection.hub_penalty)")
    p.add_argument("--hub-k", type=int, default=100,
                   help="neighbours for --hub-correction csls (large k: few classes, many images)")
    p.add_argument("--hub-ref", choices=["quilt", "test"], default="quilt",
                   help="reference images: Quilt train images, or each test set itself (transductive)")
    p.add_argument("--weighting", choices=["image", "pair"], default="image")
    p.add_argument("--no-center", action="store_true", help="skip mean subtraction (ablation)")
    p.add_argument("--test-center", choices=["quilt", "self"], default="quilt",
                   help="centre test images on the Quilt image mean or on each test set's own mean")
    p.add_argument("--max-images", type=int, default=None,
                   help="fit on a random subset of N training images (few-pair sweeps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-name", default=None)
    add_group_args(p)
    args = p.parse_args()

    for f in args.arch_feats + [args.pairs]:
        if not Path(f).exists():
            raise SystemExit(f"file not found: {f}")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = method_tag(args)
    name = args.run_name or tag + (f"_n{args.max_images}" if args.max_images else "") \
        + (f"_hub{args.hub_correction}" if args.hub_correction != "none" else "")
    run_dir = Path(args.runs_dir) / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    print(f"run dir: {run_dir}")

    data = QuiltPairs(args.pairs, device=device)
    txt_model_name = Path(args.txt_model.rstrip("/")).name
    if data.meta.get("txt_model") and data.meta["txt_model"] != txt_model_name:
        raise SystemExit(f"--txt-model {txt_model_name} differs from the encoder used for --pairs "
                         f"({data.meta['txt_model']})")

    # ------------------------------------------------------------ training pairs
    train_imgs = data.images["train"]
    if args.max_images and args.max_images < len(train_imgs):
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        train_imgs = train_imgs[torch.randperm(len(train_imgs), generator=g)[:args.max_images].to(device)]
    pairs = torch.isin(data.pair_img, train_imgs).nonzero().squeeze(1)
    print(f"fitting {tag} on {len(train_imgs):,} images / {len(pairs):,} pairs  "
          f"(dims txt {data.dims[1]} -> img {data.dims[0]})")

    t0 = time.time()
    A, B, mu_t, mu_i, diag = fit(data, pairs, args)
    model = LinearAlignment(A, B, mu_t, mu_i).to(device).eval()
    diag["fit_time_s"] = round(time.time() - t0, 2)
    if args.method == "orthogonal":                                # sanity: W W^T = I
        Wr = A if A is not None else B.t()                         # orthonormal rows [768, 1536]
        diag["orthogonality_error"] = float((Wr @ Wr.t() - torch.eye(Wr.shape[0], device=Wr.device)).abs().max())
    print(f"fitted in {diag['fit_time_s']}s   "
          + ("  ".join(f"{k} {v:.4f}" for k, v in diag.items() if k in ("procrustes_objective_tr_S",
                                                                        "singular_values_kept_sum"))))

    # ------------------------------------------------------------ Quilt validation
    val, val_set = {}, None
    if len(data.images["val"]):
        val_set = data.retrieval_set("val")
        val = {**retrieval_metrics(model, *val_set), **foscttm(model, *val_set),
               **val_diagnostics(model, data, args.seed)}
        print("quilt val  " + "  ".join(f"{k} {v:.4f}" for k, v in val.items() if isinstance(v, float)))

    cfg = {"method": args.method, "tag": tag, "img_dim": data.dims[0], "txt_dim": data.dims[1],
           "center": not args.no_center, "weighting": args.weighting, "direction": args.direction,
           **({k: getattr(args, k) for k in ("whiten", "shrink", "cca_dim", "reweight", "dewhiten")}
              if args.method == "cca" else {}),
           **group_cfg(args)}
    cpu = lambda M: None if M is None else M.cpu()
    torch.save({"A": cpu(A), "B": cpu(B), "mu_txt": mu_t.cpu(), "mu_img": mu_i.cpu(), "cfg": cfg,
                "txt_model": txt_model_name, "pairs": os.path.abspath(args.pairs), "val": val},
               run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "args": vars(args), "model": cfg, "fit": diag,
        "data": {**data.meta, "n_fit_images": len(train_imgs), "n_val_images": len(data.images["val"])},
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()},
    }, indent=2, default=str))

    # ------------------------------------------------------------ zero-shot + ARCH
    hub = None
    if args.hub_correction != "none" and args.hub_ref == "quilt" and args.test_center == "self":
        print("warning: --hub-ref quilt encodes the reference images with the Quilt mean, while "
              "--test-center self centres test images on their own mean; prefer --hub-ref test")
    if args.hub_correction != "none":
        ref = quilt_reference(model, data, seed=args.seed) if args.hub_ref == "quilt" else None
        hub = {"method": args.hub_correction, "k": args.hub_k, "ref": ref}
    del data, val_set
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\nzero-shot evaluation:")
    encoder = load_text_encoder(args.txt_model, device)
    prompt_cache = {}
    modes = [m.strip() for m in args.test_modes.split(",") if m.strip()]
    model.test_center = args.test_center
    zs = zero_shot(model, encoder, args.test_feats, modes, device, run_dir, prompt_cache, hub=hub)
    model.test_center = "quilt"
    (run_dir / "prompts.json").write_text(json.dumps(prompt_cache, indent=1))

    retr = {}
    if args.arch_feats:
        print("\ncross-modal retrieval:")
        retr = arch_retrieval(model, args.arch_feats, txt_model_name, device)

    (run_dir / "results.json").write_text(json.dumps(
        {"fit": diag, "val_retrieval": val, "zero_shot": zs, "retrieval": retr,
         "hub_correction": {"method": args.hub_correction, "k": args.hub_k, "ref": args.hub_ref}}, indent=2))

    # ------------------------------------------------------------ same summary.csv as train_projection.py
    summary = {"run": run_dir.name, "txt_model": txt_model_name, "project": tag + (f"+hub_{args.hub_correction}_{args.hub_ref}" if args.hub_correction != "none" else ""),
               "depth": 0, "width": 0, "epochs": 0, "lr": "", "batch_size": "", "seed": args.seed,
               "best_epoch": "", "val_mean_recall": round(val.get("mean_recall", float("nan")), 4),
               **zs_summary(zs),
               **{f"{n}_{k}": round(v, 4) for n, r in retr.items() for k, v in r.items() if "R@" in k}}
    fields = list(summary)[:11] + SUMMARY_COLS
    summ_path = Path(args.runs_dir) / "summary.csv"
    append_summary(summ_path, summary, fields)
    print(f"\nsaved run to {run_dir}  (summary appended to {summ_path})")


if __name__ == "__main__":
    main()