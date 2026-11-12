#!/usr/bin/env python
"""
Nonlinear closed-form text -> image map: kernel ridge regression (KRR) with random features, fitted
on precomputed Quilt-1M features, then the same evaluation as train_projection.py / fit_procrustes.py
(zero-shot on BACH / CRC / MHIST / BreakHis, ARCH retrieval, Quilt val retrieval, summary.csv row).

Model (text side only; UNI2-h image features stay frozen, as in ShareLock):
    x   = prep(text)                                  L2-normalise, subtract Quilt text mean, renormalise
    phi = [ feature map of x  (, x) ]                 D random features (+ the linear features, default)
    y^  = (phi - phi_mean) W  (T)  + y_mean           ridge solution in feature space, optional recolouring
    text embedding = normalise(y^),  image embedding = prep(image)

Feature maps (both approximate a kernel on the unit sphere; Rahimi & Recht 2007, Cho & Saul 2009):
    rff     sqrt(2/D) cos(x Omega + b),  Omega ~ N(0, 1/sigma^2),  b ~ U[0, 2 pi]     Gaussian kernel
            sigma = --sigma x median pairwise distance between training captions (median heuristic)
    arccos  sqrt(2/D) relu(x Omega),     Omega ~ N(0, 1)                             arc-cosine kernel
            (= the last layer of a very wide one-hidden-layer ReLU MLP with a frozen random first layer)
About sqrt(n) log n features reach the statistical rate of exact KRR (Rudi & Rosasco 2017).

High-dimension caveat: with a rotation-invariant kernel in dimension d, learning the degree-l part of the
map needs n >~ d^l samples (Ghorbani, Mei, Misiakiewicz & Montanari 2021). For d = 768 and ~1e6 Quilt
pairs that is at most quadratic corrections. --input-dim k computes the random features on the top-k
principal directions of the text features only (the linear block keeps all 768 dims), so that a few
dozen dimensions get a genuinely nonlinear map.

Closed form: with weighted, centred moments G = Cov(phi) [D, D] and C = Cov(phi, y) [D, 1536],
    W(lam) = (G + lam I)^-1 C,   lam = ridge * tr(G) / D
G is eigendecomposed once, so every value of --ridge costs one matrix product; the value with the best
Quilt-val mean recall is kept (benchmarks are never used for selection).

--recolor maps the predictions' covariance P onto the true image covariance Cy with the optimal-transport
map between Gaussians, T = P^-1/2 (P^1/2 Cy P^1/2)^1/2 P^-1/2. Regression predicts E[image | text], which is
shrunk towards the mean and creates hubs (Dinu et al. 2015); recolouring undoes the shrinkage.

Example:
  python fit_krr.py --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
    --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
    --arch-feats $ROOT/data/patch_datasets/uni_2h_feat/arch/arch_pubmed.pt \
    --txt-model $ROOT/weights/BioLORD-2023 --runs-dir $ROOT/ShareLock_hist/runs \
    --features rff --n-features 8192 --ridge 1e-4,1e-3,1e-2,1e-1
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

import pathlib, sys  # noqa: E401  (shared modules live in ../utils)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from fit_procrustes import foscttm, pair_weights, shrink_cov, sym_power, val_diagnostics
from quilt_pairs import QuiltPairs
from train_projection import (HUB_METHODS, SUMMARY_COLS, append_summary, zs_summary, arch_retrieval, load_text_encoder,
                              quilt_reference, retrieval_metrics, set_seed, zero_shot)


# ----------------------------------------------------------------------------- model
class KRRAlignment(nn.Module):
    """Random-feature ridge map text -> UNI2-h; same interface as ProjectionModel."""

    def __init__(self, mu_txt, mu_img, omega, bias, W, phi_mean, y_mean, T=None, features="rff",
                 with_linear=True, proj=None, chunk=8192):
        super().__init__()
        for k, v in dict(mu_txt=mu_txt, mu_img=mu_img, omega=omega, bias=bias, W=W,
                         phi_mean=phi_mean, y_mean=y_mean, T=T, proj=proj).items():
            self.register_buffer(k, None if v is None else v.float())
        self.features, self.with_linear, self.chunk = features, with_linear, chunk
        self.test_center = "quilt"         # "self": centre a whole test set on its own mean (zero-shot only)

    def prep_txt(self, x):
        return F.normalize(F.normalize(x.float(), dim=-1) - self.mu_txt, dim=-1)

    def prep_img(self, y):
        y = F.normalize(y.float(), dim=-1)
        mu = y.mean(0, keepdim=True) if self.test_center == "self" else self.mu_img
        return F.normalize(y - mu, dim=-1)

    def phi(self, x):
        """Feature map of PREPARED text features x [B, 768] -> [B, D (+768)]."""
        return feature_map(x, self.omega, self.bias, self.features, self.with_linear, self.proj)

    def predict(self, x):
        """Predicted (unnormalised) image embedding of raw text features."""
        out = []
        for s in range(0, len(x), self.chunk):
            h = (self.phi(self.prep_txt(x[s:s + self.chunk])) - self.phi_mean) @ self.W
            if self.T is not None:
                h = h @ self.T
            out.append(h + self.y_mean)
        return torch.cat(out)

    def encode_text(self, x):
        return F.normalize(self.predict(x), dim=-1)

    def encode_image(self, y):
        return self.prep_img(y)


def feature_map(x, omega, bias, features, with_linear, proj=None):
    """[random features of (x proj), x]; proj [768, k] restricts the nonlinear part to k directions."""
    D = omega.shape[1]
    z = (x if proj is None else x @ proj) @ omega
    if features == "rff":
        f = math.sqrt(2.0 / D) * torch.cos(z + bias)
    elif features == "arccos":
        f = math.sqrt(2.0 / D) * torch.relu(z)
    else:
        raise ValueError(features)
    return torch.cat([f, x], dim=1) if with_linear else f


# ----------------------------------------------------------------------------- fitting
@torch.no_grad()
def median_distance(x, n=4096, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = x[torch.randperm(len(x), generator=g)[:n].to(x.device)]
    d = torch.cdist(x, x)
    return float(d[~torch.eye(len(x), dtype=torch.bool, device=x.device)].median())


@torch.no_grad()
def fit_moments(data, pairs, args):
    """Means, random-feature parameters and the centred weighted moments G, C, Cy (float64)."""
    dev = data.img.device
    w = pair_weights(data, pairs, args.weighting)
    dt, di = data.txt.shape[1], data.img.shape[1]
    norm = lambda z: F.normalize(z.float(), dim=-1)

    def chunks():
        for s in range(0, len(pairs), args.chunk):
            p = pairs[s:s + args.chunk]
            yield norm(data.txt[data.pair_txt[p]]), norm(data.img[data.pair_img[p]]), w[s:s + args.chunk, None]

    # pass 1: preprocessing means (same as fit_procrustes)
    mu_t = torch.zeros(dt, dtype=torch.float64, device=dev)
    mu_i = torch.zeros(di, dtype=torch.float64, device=dev)
    if not args.no_center:
        for x, y, ww in chunks():
            mu_t += (ww * x.double()).sum(0)
            mu_i += (ww * y.double()).sum(0)
    mu_t, mu_i = mu_t.float(), mu_i.float()
    prep = lambda z, mu: F.normalize(z - mu, dim=-1)

    # random features (seeded, generated on CPU so they do not depend on the device)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    D = args.n_features
    idx = pairs[torch.randperm(len(pairs), generator=g)[:50_000].to(pairs.device)]
    sample = prep(norm(data.txt[data.pair_txt[idx]]), mu_t)                # prepared captions
    proj, din = None, dt
    if args.input_dim and args.input_dim < dt:                              # top-k text principal directions
        sc = sample.double() - sample.double().mean(0)
        evals, evecs = torch.linalg.eigh(sc.t() @ sc / len(sc))
        proj = evecs[:, -args.input_dim:].flip(1).float()                   # [768, k]
        din = args.input_dim
        sample = sample @ proj
    sigma = None
    if args.features == "rff":
        sigma = args.sigma * median_distance(sample, seed=args.seed)
        omega = torch.randn(din, D, generator=g) / sigma
        bias = torch.rand(D, generator=g) * 2 * math.pi
    else:
        omega = torch.randn(din, D, generator=g)
        bias = torch.zeros(D)
    omega, bias = omega.to(dev), bias.to(dev)

    # pass 2: raw weighted moments of phi and y, centred analytically afterwards
    Dt = D + (dt if args.with_linear else 0)
    S_pp = torch.zeros(Dt, Dt, dtype=torch.float64, device=dev)
    S_py = torch.zeros(Dt, di, dtype=torch.float64, device=dev)
    S_yy = torch.zeros(di, di, dtype=torch.float64, device=dev) if args.recolor else None
    m_p = torch.zeros(Dt, dtype=torch.float64, device=dev)
    m_y = torch.zeros(di, dtype=torch.float64, device=dev)
    for x, y, ww in chunks():
        f = feature_map(prep(x, mu_t), omega, bias, args.features, args.with_linear, proj).double()
        y = prep(y, mu_i).double()
        fw = ww * f
        S_pp += fw.t() @ f
        S_py += fw.t() @ y
        m_p += fw.sum(0)
        m_y += (ww * y).sum(0)
        if S_yy is not None:
            S_yy += (ww * y).t() @ y
    G = S_pp - torch.outer(m_p, m_p)
    C = S_py - torch.outer(m_p, m_y)
    Cy = None if S_yy is None else S_yy - torch.outer(m_y, m_y)
    return dict(mu_t=mu_t, mu_i=mu_i, omega=omega, bias=bias, sigma=sigma, G=G, C=C, Cy=Cy,
                phi_mean=m_p, y_mean=m_y, proj=proj)


@torch.no_grad()
def ridge_solutions(G, C, ridges):
    """W(lam) for every relative ridge value, from one eigendecomposition of G."""
    lam, Q = torch.linalg.eigh(G)
    lam = lam.clamp(min=0)
    QtC = Q.t() @ C
    scale = lam.sum() / len(lam)                                   # tr(G) / D
    for r in ridges:
        yield r, Q @ (QtC / (lam + r * scale)[:, None])


@torch.no_grad()
def recolor_map(W, G, Cy, shrink):
    """Gaussian optimal-transport map from Cov(predictions) = W^T G W to the image covariance Cy."""
    P = shrink_cov(W.t() @ G @ W, shrink)
    Ph, Pmh = sym_power(P, 0.5), sym_power(P, -0.5)
    return Pmh @ sym_power(Ph @ Cy @ Ph, 0.5) @ Pmh


# ----------------------------------------------------------------------------- main
def run_tag(args, ridge=None):
    t = f"krr_{args.features}_D{args.n_features}"
    if args.input_dim:
        t += f"_in{args.input_dim}"
    if args.features == "rff" and args.sigma != 1.0:
        t += f"_s{args.sigma:g}"
    if not args.with_linear:
        t += "_nolin"
    if ridge is not None:
        t += f"_r{ridge:g}"
    if args.recolor:
        t += "_recolor"
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="file from build_quilt_pairs.py")
    p.add_argument("--test-feats", required=True, help="--out-dir of extract_uni2h_test_data.py")
    p.add_argument("--txt-model", required=True, help="text encoder used for --pairs (for the prompts)")
    p.add_argument("--test-modes", default="resizecrop,resize,tile")
    p.add_argument("--arch-feats", nargs="*", default=[])
    # model
    p.add_argument("--features", choices=["rff", "arccos"], default="rff")
    p.add_argument("--n-features", type=int, default=8192, help="number of random features D")
    p.add_argument("--input-dim", type=int, default=None,
                   help="random features on the top-k text principal directions only (default: all 768)")
    p.add_argument("--sigma", type=float, default=1.0, help="rff bandwidth, as a multiple of the median heuristic")
    p.add_argument("--no-linear", dest="with_linear", action="store_false",
                   help="drop the linear features (by default phi = [random features, x], so KRR contains ridge)")
    p.add_argument("--ridge", default="1e-4,1e-3,1e-2,1e-1",
                   help="comma-separated ridge values relative to tr(G)/D; the best on Quilt val is kept")
    p.add_argument("--recolor", action="store_true", help="match the predictions' covariance to the images'")
    p.add_argument("--recolor-shrink", type=float, default=0.01)
    # data / preprocessing (as fit_procrustes.py)
    p.add_argument("--weighting", choices=["image", "pair"], default="image")
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--test-center", choices=["quilt", "self"], default="quilt")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--chunk", type=int, default=8192, help="pairs per chunk (memory: chunk x D x 8 bytes)")
    # zero-shot hubness correction
    p.add_argument("--hub-correction", choices=HUB_METHODS, default="none")
    p.add_argument("--hub-k", type=int, default=100)
    p.add_argument("--hub-ref", choices=["quilt", "test"], default="quilt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    ridges = [float(r) for r in args.ridge.split(",") if r.strip()]

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

    # ------------------------------------------------------------ moments + ridge path
    t0 = time.time()
    m = fit_moments(data, pairs, args)
    t_mom = time.time() - t0
    print(f"moments in {t_mom:.1f}s   feature dim {m['G'].shape[0]}"
          + (f"   sigma {m['sigma']:.4f}" if m["sigma"] else ""))

    val_set = data.retrieval_set("val") if len(data.images["val"]) else None
    if val_set is None and len(ridges) > 1:
        raise SystemExit("no Quilt validation split: pass a single --ridge value")

    def make_model(W, T=None):
        return KRRAlignment(m["mu_t"], m["mu_i"], m["omega"], m["bias"], W, m["phi_mean"], m["y_mean"], T,
                            args.features, args.with_linear, m["proj"]).to(device).eval()

    path, best = [], None
    for r, W in ridge_solutions(m["G"], m["C"], ridges):
        T = recolor_map(W, m["G"], m["Cy"], args.recolor_shrink) if args.recolor else None
        model = make_model(W, T)
        score = retrieval_metrics(model, *val_set) if val_set else {"mean_recall": float("nan")}
        path.append({"ridge": r, **score})
        print(f"  ridge {r:g}   val mean_recall {score['mean_recall']:.4f}   "
              f"i2t_R@1 {score.get('i2t_R@1', float('nan')):.4f}   t2i_R@1 {score.get('t2i_R@1', float('nan')):.4f}")
        if best is None or score["mean_recall"] > best[0]:
            best = (score["mean_recall"], r, W, T)
    _, ridge, W, T = best
    model = make_model(W, T)
    fit_info = {"moments_time_s": round(t_mom, 1), "fit_time_s": round(time.time() - t0, 1),
                "feature_dim": int(m["G"].shape[0]), "sigma": m["sigma"], "chosen_ridge": ridge,
                "ridge_path": path, "n_train_pairs": int(len(pairs))}
    tag = run_tag(args, ridge)
    print(f"chosen ridge {ridge:g}  ({fit_info['fit_time_s']}s total)")

    # ------------------------------------------------------------ Quilt validation (chosen model)
    val = {}
    if val_set is not None:
        val = {**retrieval_metrics(model, *val_set), **foscttm(model, *val_set),
               **val_diagnostics(model, data, args.seed)}
        print("quilt val  " + "  ".join(f"{k} {v:.4f}" for k, v in val.items() if isinstance(v, float)))

    cfg = {"method": "krr", "tag": tag, "features": args.features, "n_features": args.n_features,
           "input_dim": args.input_dim,
           "sigma": m["sigma"], "with_linear": args.with_linear, "ridge": ridge, "recolor": args.recolor,
           "center": not args.no_center, "weighting": args.weighting}
    cpu = lambda t: None if t is None else t.float().cpu()
    torch.save({"state": {k: cpu(v) for k, v in dict(
        mu_txt=m["mu_t"], mu_img=m["mu_i"], omega=m["omega"], bias=m["bias"], W=W,
        phi_mean=m["phi_mean"], y_mean=m["y_mean"], T=T, proj=m["proj"]).items()},
        "cfg": cfg, "txt_model": txt_model_name, "pairs": os.path.abspath(args.pairs), "val": val},
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
               "depth": 0, "width": args.n_features, "epochs": 0, "lr": "", "batch_size": "", "seed": args.seed,
               "best_epoch": "", "val_mean_recall": round(val.get("mean_recall", float("nan")), 4),
               **zs_summary(zs),
               **{f"{n}_{k}": round(v, 4) for n, r in retr.items() for k, v in r.items() if "R@" in k}}
    fields = list(summary)[:11] + SUMMARY_COLS
    summ_path = Path(args.runs_dir) / "summary.csv"
    append_summary(summ_path, summary, fields)
    print(f"\nsaved run to {run_dir}  (summary appended to {summ_path})")


if __name__ == "__main__":
    main()
