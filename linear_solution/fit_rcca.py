#!/usr/bin/env python
"""
Randomized kernel CCA (RCCA; Lopez-Paz, Sra, Smola, Ghahramani & Schoelkopf, ICML 2014) between precomputed
Quilt-1M text and image features, then the same evaluation as train_projection.py / fit_procrustes.py /
fit_krr.py (zero-shot on BACH / CRC / MHIST / BreakHis, ARCH retrieval, Quilt val, summary.csv row).

Kernel CCA (Bach & Jordan 2002; Hardoon et al. 2004) finds nonlinear directions f(text), g(image) of maximal
correlation. With random features it is linear CCA on lifted features:
    phi(x) = [random features of x, x]        for each side chosen by --lift (text | image | both | none)
    Cxx, Cyy, Cxy                             weighted, centred moments of the (lifted) features
    Wx = (Cxx + r tr(Cxx)/Dx I)^-1/2          regularised whitening on a lifted side (r from --reg)
    Wy = shrink(Cyy, s)^-1/2                  shrinkage whitening on a linear side (s = --shrink, as fit_procrustes cca)
    Wx Cxy Wy = U S V^T,  keep the top k      S = canonical correlations
    text  -> normalise((phi(x) - mean) Wx U_k S_k^(b/2)),   image -> normalise((psi(y) - mean) Wy V_k S_k^(b/2))
Regularisation is mandatory on a lifted side: in a rich feature space any two samples can be made perfectly
correlated, so unregularised KCCA returns canonical correlations of 1 that carry no information
(Hardoon et al. 2004). Both r and k are selected on Quilt val mean recall over the --reg x --cca-dim grid
(benchmarks are never used for selection); every grid point costs one diagonal rescaling and one SVD.

--lift text (default) keeps the UNI2-h space frozen in the sense that images only get a linear map, so test
images go through the same simple preprocessing as Quilt images. --lift image / both also lift UNI2-h.
--lift none is plain CCA and reproduces fit_procrustes.py --method cca --whiten 1 (same --shrink).

Random features (as fit_krr.py): rff = Gaussian kernel, bandwidth --sigma x median distance between the
side's prepared features; arccos = arc-cosine kernel (random ReLU units). --input-dim / --img-input-dim
compute the random features on the top-k principal directions only: with a rotation-invariant kernel in
dimension d, the degree-l part of a map needs n >~ d^l samples (Ghorbani et al. 2021), so on ~1e6 pairs a
nonlinear map is only learnable on a few dozen directions.

Example:
  python fit_rcca.py --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
    --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
    --arch-feats $ROOT/data/patch_datasets/uni_2h_feat/arch/arch_pubmed.pt \
    --txt-model $ROOT/weights/BioLORD-2023 --runs-dir $ROOT/ShareLock_hist/runs \
    --lift text --features rff --n-features 8192 --input-dim 64 \
    --reg 1e-3,1e-2,1e-1 --cca-dim 64,128,256,512
"""
import argparse
import csv
import json
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fit_krr import feature_map, median_distance

import pathlib, sys  # noqa: E401  (shared modules live in ../utils)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from fit_procrustes import foscttm, pair_weights, val_diagnostics
from quilt_pairs import QuiltPairs
from train_projection import (HUB_METHODS, SUMMARY_COLS, append_summary, zs_summary, arch_retrieval, load_text_encoder,
                              quilt_reference, retrieval_metrics, set_seed, zero_shot)

SIDES = ("t", "i")                                  # text, image


# ----------------------------------------------------------------------------- model
class RCCAAlignment(nn.Module):
    """Both sides mapped into a shared k-dim canonical space; same interface as ProjectionModel.

    lift[s] is None (linear side) or {"omega", "bias", "proj"} (random features on side s).
    """

    def __init__(self, mu_txt, mu_img, lift, mean, A, features="rff", with_linear=True, chunk=8192):
        super().__init__()
        self.register_buffer("mu_txt", mu_txt.float())
        self.register_buffer("mu_img", mu_img.float())
        self.lifted = {s: lift[s] is not None for s in SIDES}
        for s in SIDES:
            self.register_buffer(f"{s}_mean", mean[s].float())
            self.register_buffer(f"{s}_A", A[s].float())
            for k in ("omega", "bias", "proj"):
                v = lift[s][k] if lift[s] is not None else None
                self.register_buffer(f"{s}_{k}", None if v is None else v.float())
        self.features, self.with_linear, self.chunk = features, with_linear, chunk
        self.test_center = "quilt"         # "self": centre a whole test set on its own mean (zero-shot only)

    def prep_txt(self, x):
        return F.normalize(F.normalize(x.float(), dim=-1) - self.mu_txt, dim=-1)

    def prep_img(self, y):
        y = F.normalize(y.float(), dim=-1)
        mu = y.mean(0, keepdim=True) if self.test_center == "self" else self.mu_img
        return F.normalize(y - mu, dim=-1)

    def lift(self, s, z):
        if not self.lifted[s]:
            return z
        return feature_map(z, getattr(self, f"{s}_omega"), getattr(self, f"{s}_bias"), self.features,
                           self.with_linear, getattr(self, f"{s}_proj"))

    def _encode(self, s, z):
        out = []
        for a in range(0, len(z), self.chunk):
            h = self.lift(s, z[a:a + self.chunk]) - getattr(self, f"{s}_mean")
            out.append(F.normalize(h @ getattr(self, f"{s}_A"), dim=-1))
        return torch.cat(out)

    def encode_text(self, x):
        return self._encode("t", self.prep_txt(x))

    def encode_image(self, y):
        return self._encode("i", self.prep_img(y))


# ----------------------------------------------------------------------------- fitting
@torch.no_grad()
def make_lift(sample, args, side, g, dev):
    """Random-feature parameters for one side, from a sample of its prepared features."""
    input_dim = args.input_dim if side == "t" else args.img_input_dim
    proj, din = None, sample.shape[1]
    if input_dim and input_dim < din:                                       # top-k principal directions
        sc = sample.double() - sample.double().mean(0)
        _, evecs = torch.linalg.eigh(sc.t() @ sc / len(sc))
        proj = evecs[:, -input_dim:].flip(1).float()
        din, sample = input_dim, sample @ proj
    D = args.n_features if side == "t" else (args.img_n_features or args.n_features)
    sigma = None
    if args.features == "rff":
        sigma = args.sigma * median_distance(sample, seed=args.seed)
        omega = torch.randn(din, D, generator=g) / sigma
        bias = torch.rand(D, generator=g) * 2 * math.pi
    else:
        omega = torch.randn(din, D, generator=g)
        bias = torch.zeros(D)
    return {"omega": omega.to(dev), "bias": bias.to(dev), "proj": proj, "sigma": sigma, "n_features": D}


@torch.no_grad()
def fit_moments(data, pairs, args):
    """Preprocessing means, lifts, and the centred weighted moments Cxx, Cyy, Cxy (float64)."""
    dev = data.img.device
    w = pair_weights(data, pairs, args.weighting)
    dt, di = data.txt.shape[1], data.img.shape[1]
    norm = lambda z: F.normalize(z.float(), dim=-1)
    prep = lambda z, mu: F.normalize(z - mu, dim=-1)

    def chunks():
        for s in range(0, len(pairs), args.chunk):
            p = pairs[s:s + args.chunk]
            yield norm(data.txt[data.pair_txt[p]]), norm(data.img[data.pair_img[p]]), w[s:s + args.chunk, None]

    # pass 1: preprocessing means (same as fit_procrustes / fit_krr)
    mu = {"t": torch.zeros(dt, dtype=torch.float64, device=dev), "i": torch.zeros(di, dtype=torch.float64, device=dev)}
    if not args.no_center:
        for x, y, ww in chunks():
            mu["t"] += (ww * x.double()).sum(0)
            mu["i"] += (ww * y.double()).sum(0)
    mu = {s: v.float() for s, v in mu.items()}

    # random features for the lifted sides (seeded on CPU, so independent of the device)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    idx = pairs[torch.randperm(len(pairs), generator=g)[:50_000].to(pairs.device)]
    lifted = {"t": args.lift in ("text", "both"), "i": args.lift in ("image", "both")}
    lift = {"t": None, "i": None}
    if lifted["t"]:
        lift["t"] = make_lift(prep(norm(data.txt[data.pair_txt[idx]]), mu["t"]), args, "t", g, dev)
    if lifted["i"]:
        lift["i"] = make_lift(prep(norm(data.img[data.pair_img[idx]]), mu["i"]), args, "i", g, dev)

    def phi(s, z):
        L = lift[s]
        return z if L is None else feature_map(z, L["omega"], L["bias"], args.features, args.with_linear, L["proj"])

    # pass 2: raw weighted moments, centred analytically afterwards
    S = {}
    for x, y, ww in chunks():
        fx = phi("t", prep(x, mu["t"])).double()
        fy = phi("i", prep(y, mu["i"])).double()
        if not S:
            S = {"xx": torch.zeros(fx.shape[1], fx.shape[1], dtype=torch.float64, device=dev),
                 "yy": torch.zeros(fy.shape[1], fy.shape[1], dtype=torch.float64, device=dev),
                 "xy": torch.zeros(fx.shape[1], fy.shape[1], dtype=torch.float64, device=dev),
                 "mx": torch.zeros(fx.shape[1], dtype=torch.float64, device=dev),
                 "my": torch.zeros(fy.shape[1], dtype=torch.float64, device=dev)}
        wx, wy = ww * fx, ww * fy
        S["xx"] += wx.t() @ fx
        S["yy"] += wy.t() @ fy
        S["xy"] += wx.t() @ fy
        S["mx"] += wx.sum(0)
        S["my"] += wy.sum(0)
    C = {"xx": S["xx"] - torch.outer(S["mx"], S["mx"]),
         "yy": S["yy"] - torch.outer(S["my"], S["my"]),
         "xy": S["xy"] - torch.outer(S["mx"], S["my"])}
    return dict(mu=mu, lift=lift, lifted=lifted, C=C, mean={"t": S["mx"], "i": S["my"]})


@torch.no_grad()
def whitening_scales(lam, lifted, reg, shrink, eps=1e-12):
    """Diagonal of C^-1/2 in C's eigenbasis: ridge r on a lifted side, fit_procrustes shrinkage on a linear one.

    The shrinkage (1-s) C + s m I is used as C + s/(1-s) m I: same directions and same normalised
    embeddings, but the canonical correlations stay <= 1 (scaling C down by (1-s) would inflate them).
    """
    lam = lam.clamp(min=0)
    m = lam.mean()
    reg_lam = lam + (reg if lifted else shrink / (1 - shrink)) * m
    return reg_lam.clamp(min=eps * lam.max()).rsqrt()


@torch.no_grad()
def top_svd(M, k):
    """Top-k SVD; randomized (Halko et al. 2011) when both sides are large."""
    if min(M.shape) <= 4096:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k].t()
    U, S, V = torch.svd_lowrank(M, q=min(k + 64, min(M.shape)), niter=6)
    return U[:, :k], S[:k], V[:, :k]


@torch.no_grad()
def cca_path(m, args, regs, ks):
    """Yields (reg, k, A_text, A_image, S) for every grid point; one eigendecomposition per side in total."""
    lam_x, Qx = torch.linalg.eigh(m["C"]["xx"])
    lam_y, Qy = torch.linalg.eigh(m["C"]["yy"])
    K = Qx.t() @ m["C"]["xy"] @ Qy                                           # cross-covariance in the eigenbases
    kmax = min(max(ks), *K.shape)
    for r in regs:
        dx = whitening_scales(lam_x, m["lifted"]["t"], r, args.shrink)
        dy = whitening_scales(lam_y, m["lifted"]["i"], r, args.shrink)
        U, S, V = top_svd(dx[:, None] * K * dy[None], kmax)                  # = SVD of Wx Cxy Wy, rotated
        Ax, Ay = Qx @ (dx[:, None] * U), Qy @ (dy[:, None] * V)
        for k in ks:
            k = min(k, kmax)
            R = S[:k].clamp(min=0) ** (args.reweight / 2)
            yield r, k, (Ax[:, :k] * R).float(), (Ay[:, :k] * R).float(), S


# ----------------------------------------------------------------------------- main
def run_tag(args, reg=None, k=None):
    t = f"rcca_{args.lift}"
    if args.lift != "none":
        t += f"_{args.features}_D{args.n_features}"
        if args.lift in ("image", "both") and args.img_n_features:
            t += f"-{args.img_n_features}"
        if args.input_dim and args.lift in ("text", "both"):
            t += f"_in{args.input_dim}"
        if args.img_input_dim and args.lift in ("image", "both"):
            t += f"_imgin{args.img_input_dim}"
        if args.features == "rff" and args.sigma != 1.0:
            t += f"_s{args.sigma:g}"
        if not args.with_linear:
            t += "_nolin"
        if reg is not None:
            t += f"_r{reg:g}"
    if k is not None:
        t += f"_k{k}"
    if args.reweight:
        t += f"_rw{args.reweight:g}"
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="file from build_quilt_pairs.py")
    p.add_argument("--test-feats", required=True, help="--out-dir of extract_uni2h_test_data.py")
    p.add_argument("--txt-model", required=True, help="text encoder used for --pairs (for the prompts)")
    p.add_argument("--test-modes", default="resizecrop,resize,tile")
    p.add_argument("--arch-feats", nargs="*", default=[])
    # model
    p.add_argument("--lift", choices=["text", "image", "both", "none"], default="text",
                   help="which side(s) get random features; the other side stays linear")
    p.add_argument("--features", choices=["rff", "arccos"], default="rff")
    p.add_argument("--n-features", type=int, default=8192, help="random features on the text side")
    p.add_argument("--img-n-features", type=int, default=None, help="random features on the image side (default: same)")
    p.add_argument("--input-dim", type=int, default=None, help="text: random features on the top-k PCs only")
    p.add_argument("--img-input-dim", type=int, default=None, help="image: random features on the top-k PCs only")
    p.add_argument("--sigma", type=float, default=1.0, help="rff bandwidth, multiple of the median heuristic")
    p.add_argument("--no-linear", dest="with_linear", action="store_false",
                   help="drop the linear features from the lifted sides (by default RCCA contains linear CCA)")
    p.add_argument("--reg", default="1e-3,1e-2,1e-1",
                   help="comma-separated ridge values for the lifted sides, relative to the mean eigenvalue")
    p.add_argument("--shrink", type=float, default=0.01, help="covariance shrinkage of a linear side (as fit_procrustes)")
    p.add_argument("--cca-dim", default="64,128,256,512", help="comma-separated numbers of canonical directions")
    p.add_argument("--reweight", type=float, default=0.0, help="weight directions by S^b (0 = off)")
    # data / preprocessing (as fit_procrustes.py)
    p.add_argument("--weighting", choices=["image", "pair"], default="image")
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--test-center", choices=["quilt", "self"], default="quilt")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--chunk", type=int, default=8192, help="pairs per chunk")
    # zero-shot hubness correction
    p.add_argument("--hub-correction", choices=HUB_METHODS, default="none")
    p.add_argument("--hub-k", type=int, default=100)
    p.add_argument("--hub-ref", choices=["quilt", "test"], default="quilt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    regs = [float(r) for r in args.reg.split(",") if r.strip()] if args.lift != "none" else [0.0]
    ks = sorted({int(k) for k in args.cca_dim.split(",") if k.strip()})

    for f in args.arch_feats + [args.pairs]:
        if not Path(f).exists():
            raise SystemExit(f"file not found: {f}")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.run_name or run_tag(args) + (f"_n{args.max_images}" if args.max_images else "") \
        + (f"_hub{args.hub_correction}" if args.hub_correction != "none" else "")
    run_dir = Path(args.runs_dir) / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    print(f"run dir: {run_dir}")

    data = QuiltPairs(args.pairs, device=device)
    txt_model_name = Path(args.txt_model.rstrip("/")).name
    if data.meta.get("txt_model") and data.meta["txt_model"] != txt_model_name:
        raise SystemExit(f"--txt-model {txt_model_name} differs from the encoder used for --pairs "
                         f"({data.meta['txt_model']})")

    train_imgs = data.images["train"]
    if args.max_images and args.max_images < len(train_imgs):
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        train_imgs = train_imgs[torch.randperm(len(train_imgs), generator=g)[:args.max_images].to(device)]
    pairs = torch.isin(data.pair_img, train_imgs).nonzero().squeeze(1)
    print(f"fitting {run_tag(args)} on {len(train_imgs):,} images / {len(pairs):,} pairs")

    # ------------------------------------------------------------ moments + (reg, k) grid
    t0 = time.time()
    m = fit_moments(data, pairs, args)
    t_mom = time.time() - t0
    dims = {"text": m["C"]["xx"].shape[0], "image": m["C"]["yy"].shape[0]}
    print(f"moments in {t_mom:.1f}s   feature dims {dims}"
          + "".join(f"   sigma_{s} {m['lift'][s]['sigma']:.4f}" for s in SIDES
                    if m["lift"][s] is not None and m["lift"][s]["sigma"]))

    val_set = data.retrieval_set("val") if len(data.images["val"]) else None
    if val_set is None and len(regs) * len(ks) > 1:
        raise SystemExit("no Quilt validation split: pass a single --reg and --cca-dim value")

    def make_model(At, Ai):
        return RCCAAlignment(m["mu"]["t"], m["mu"]["i"], m["lift"], m["mean"], {"t": At, "i": Ai},
                             args.features, args.with_linear).to(device).eval()

    path, best = [], None
    for r, k, At, Ai, S in cca_path(m, args, regs, ks):
        model = make_model(At, Ai)
        score = retrieval_metrics(model, *val_set) if val_set else {"mean_recall": float("nan")}
        path.append({"reg": r, "k": k, "canon_corr_top": float(S[0]), "canon_corr_k": float(S[k - 1]), **score})
        print(f"  reg {r:<7g} k {k:4d}   corr[0] {float(S[0]):.3f} corr[k-1] {float(S[k - 1]):.3f}   "
              f"val mean_recall {score['mean_recall']:.4f}   i2t_R@1 {score.get('i2t_R@1', float('nan')):.4f}")
        if best is None or score["mean_recall"] > best[0]:
            best = (score["mean_recall"], r, k, At, Ai, S[:10].tolist())
    _, reg, k, At, Ai, canon = best
    model = make_model(At, Ai)
    fit_info = {"moments_time_s": round(t_mom, 1), "fit_time_s": round(time.time() - t0, 1),
                "feature_dims": dims, "chosen_reg": reg, "chosen_k": k,
                "canonical_correlations_top10": canon, "grid": path, "n_train_pairs": int(len(pairs)),
                "sigma": {s: (m["lift"][s] or {}).get("sigma") for s in SIDES}}
    tag = run_tag(args, reg if args.lift != "none" else None, k)
    print(f"chosen reg {reg:g}  k {k}  ({fit_info['fit_time_s']}s total)")

    # ------------------------------------------------------------ Quilt validation (chosen model)
    val = {}
    if val_set is not None:
        val = {**retrieval_metrics(model, *val_set), **foscttm(model, *val_set),
               **val_diagnostics(model, data, args.seed)}
        print("quilt val  " + "  ".join(f"{a} {v:.4f}" for a, v in val.items() if isinstance(v, float)))

    cfg = {"method": "rcca", "tag": tag, "lift": args.lift, "features": args.features,
           "n_features": args.n_features, "img_n_features": args.img_n_features, "input_dim": args.input_dim,
           "img_input_dim": args.img_input_dim, "with_linear": args.with_linear, "reg": reg, "k": k,
           "shrink": args.shrink, "reweight": args.reweight, "center": not args.no_center,
           "weighting": args.weighting}
    torch.save({"state": {a: v.cpu() for a, v in model.state_dict().items()}, "cfg": cfg,
                "txt_model": txt_model_name, "pairs": os.path.abspath(args.pairs), "val": val},
               run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "args": vars(args), "model": cfg, "fit": fit_info,
        "data": {**data.meta, "n_fit_images": len(train_imgs), "n_val_images": len(data.images["val"])},
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start": datetime.now().isoformat()},
    }, indent=2, default=str))

    # ------------------------------------------------------------ zero-shot + ARCH
    hub = None
    if args.hub_correction != "none":
        ref = quilt_reference(model, data, seed=args.seed) if args.hub_ref == "quilt" else None
        hub = {"method": args.hub_correction, "k": args.hub_k, "ref": ref}
    del data, val_set, m
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\nzero-shot evaluation:")
    encoder = load_text_encoder(args.txt_model, device)
    prompt_cache = {}
    modes = [s.strip() for s in args.test_modes.split(",") if s.strip()]
    model.test_center = args.test_center
    zs = zero_shot(model, encoder, args.test_feats, modes, device, run_dir, prompt_cache, hub=hub)
    model.test_center = "quilt"
    (run_dir / "prompts.json").write_text(json.dumps(prompt_cache, indent=1))

    retr = {}
    if args.arch_feats:
        print("\ncross-modal retrieval:")
        retr = arch_retrieval(model, args.arch_feats, txt_model_name, device)

    (run_dir / "results.json").write_text(json.dumps(
        {"fit": fit_info, "val_retrieval": val, "zero_shot": zs, "retrieval": retr,
         "hub_correction": {"method": args.hub_correction, "k": args.hub_k, "ref": args.hub_ref}}, indent=2))

    # ------------------------------------------------------------ same summary.csv as the other scripts
    summary = {"run": run_dir.name, "txt_model": txt_model_name,
               "project": tag + (f"+hub_{args.hub_correction}_{args.hub_ref}" if args.hub_correction != "none" else ""),
               "depth": 0, "width": args.n_features if args.lift != "none" else 0, "epochs": 0, "lr": "",
               "batch_size": "", "seed": args.seed, "best_epoch": "",
               "val_mean_recall": round(val.get("mean_recall", float("nan")), 4),
               **zs_summary(zs),
               **{f"{n}_{a}": round(v, 4) for n, r in retr.items() for a, v in r.items() if "R@" in a}}
    fields = list(summary)[:11] + SUMMARY_COLS
    summ_path = Path(args.runs_dir) / "summary.csv"
    append_summary(summ_path, summary, fields)
    print(f"\nsaved run to {run_dir}  (summary appended to {summ_path})")


if __name__ == "__main__":
    main()
