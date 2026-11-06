"""
Video-mean (source-group) centering of the Quilt TRAINING pairs for the closed-form fits.
Wired into fit_procrustes.py (orthogonal / linear / cca); fit_krr.py and fit_rcca.py can use the same
two calls (fit_group_centers after pass 1, centers() in pass 2) but are not changed yet.
Group ids come from the img_group list of the pairs file (build_quilt_pairs.source_group); QuiltPairs
is not modified.

Why: Quilt images come in groups that share acquisition nuisance: one YouTube video = one narrator,
slide, stain, scanner and viewer overlay; one tweet = one case. Pathology foundation models encode
slide / source effects strongly (Mishra & Lotter, ML4H 2026: high slide-dependence, low
disease-dependence), and a covariance-based fit spends its capacity on the directions of largest
variance. Centering each training pair on its group's mean keeps only within-group contrasts
("this frame shows mucin, that one glands"), which is closer to what the benchmarks ask.

Default preprocessing (all scripts):   x -> normalise(normalise(x) - mu)       mu  = global Quilt mean
With --group-center video:             x -> normalise(normalise(x) - c_g)      training pairs only
    c_g   = lam_g * mu_g + (1 - lam_g) * mu,      lam_g = n_g / (n_g + kappa)
    mu_g  = weighted mean of the group's L2-normalised features (same pair weights as the fit)
    n_g   = number of TRAINING images of group g (source groups from build_quilt_pairs.source_group)
so  x - c_g = (x - mu_g) + kappa / (n_g + kappa) * (mu_g - mu):  the within-group part is kept, the
between-group part is shrunk, more for large groups (whose offset is well estimated).
    --group-kappa 0       pure within-group centering of multi-image groups
    --group-kappa large   back towards global centering (kappa -> inf is exactly the default)
    groups with fewer than --group-min-images training images (default 2: PubMed / LAION / single-image
    tweets) are centred on mu, exactly as without group centering.

--group-sides image | text centres one side only. Before row renormalisation the cross-covariance
sum_p w_p (x_p - c_t)(y_p - c_i)^T does not depend on whether one or both sides are group-centred
(sum over a group of w_p (y_p - mu_g) is zero), so for orthogonal Procrustes the sides differ only
through the renormalisation; for linear / cca they also change the self-covariances.

--group-center random is the control: same group sizes, images assigned to groups at random (seeded).
A gain of "video" over "random" comes from the real source structure, not from the extra centering.

Caveat: a video is usually about one case, so its mean also carries the case's organ / diagnosis;
within-video centering removes that shared semantics together with the nuisance. kappa trades the two.

Test time is unchanged: Quilt val, ARCH and the benchmarks have no source group and are centred on the
global Quilt mean (or --test-center self), so metrics stay comparable with runs without group centering.
Do NOT centre test sets on a patient / slide mean instead: in BreakHis every patient has a single
tumour type, so a per-patient mean contains the label.

Diagnostics (saved under fit -> group in config.json / results.json):
  between_var_frac_img / _txt   share of the variance (around the global mean) that lies between groups,
                                over images of multi-image groups: how source-dominated each space is.
                                Compare video with random (random gives roughly 1 / group size).
"""
import torch
import torch.nn.functional as F

GROUP_MODES = ("none", "video", "random")


def add_group_args(p):
    g = p.add_argument_group("source-group (video-mean) centering of the training pairs, see group_centering.py")
    g.add_argument("--group-center", choices=GROUP_MODES, default="none",
                   help="video: centre training pairs on their source-group mean; random: same group sizes, "
                        "random membership (control); none: global centering (default)")
    g.add_argument("--group-kappa", type=float, default=0.0,
                   help="shrinkage of the group mean towards the global mean: lam_g = n_g / (n_g + kappa); "
                        "0 = pure within-group centering")
    g.add_argument("--group-min-images", type=int, default=2,
                   help="groups with fewer training images are centred on the global mean")
    g.add_argument("--group-sides", choices=["both", "image", "text"], default="both",
                   help="which side(s) are group-centred")


def group_tag(args):
    """Suffix for run names and the summary.csv 'project' column ('' without group centering)."""
    if getattr(args, "group_center", "none") == "none":
        return ""
    t = f"_gc{args.group_center}_k{args.group_kappa:g}"
    if args.group_sides != "both":
        t += f"_{args.group_sides}"
    if args.group_min_images != 2:
        t += f"_m{args.group_min_images}"
    return t


def load_group_ids(pairs_path, device):
    """Integer source-group id of every image, read from the img_group list of a build_quilt_pairs.py file.

    torch.load(mmap=True) maps the tensors lazily, so only the list of group names is actually read.
    Returns (LongTensor [I] on device, list of group names), or (None, None) without img_group.
    """
    try:
        d = torch.load(pairs_path, map_location="cpu", mmap=True)
    except (RuntimeError, TypeError, ValueError):      # legacy (non-zip) files cannot be memory-mapped
        d = torch.load(pairs_path, map_location="cpu")
    groups = d.get("img_group")
    if groups is None:
        return None, None
    ids = {}
    gid = [ids.setdefault(g, len(ids)) for g in groups]
    return torch.tensor(gid, dtype=torch.long, device=device), list(ids)


def group_cfg(args):
    if getattr(args, "group_center", "none") == "none":
        return {"group_center": "none"}
    return {"group_center": args.group_center, "group_kappa": args.group_kappa,
            "group_min_images": args.group_min_images, "group_sides": args.group_sides}


class GroupCenters:
    """Centres of the training pairs: centers(side, p) -> [len(p), d] for pair indices p ("t" or "i")."""

    def __init__(self, img_gid, C, mu, pair_img, info):
        self.img_gid, self.C, self.mu, self.pair_img, self.info = img_gid, C, mu, pair_img, info

    def centers(self, side, p):
        mu, C = self.mu[side], self.C[side]
        if C is None:                                   # this side is not group-centred
            return mu
        g = self.img_gid[self.pair_img[p]]
        out = mu.expand(len(p), -1).clone()
        m = g >= 0
        out[m] = C[g[m]]
        return out


@torch.no_grad()
def fit_group_centers(data, pairs, w, mu_t, mu_i, args, chunk=65536, seed=0):
    """Shrunk group means of the training pairs (or None without --group-center).

    data:  QuiltPairs; the group ids are read from args.pairs (img_group), so QuiltPairs needs no change
    pairs: indices of the training pairs; w: their weights (same order, summing to 1)
    mu_t, mu_i: global means of the L2-normalised training features; the centres use their dtype.
    """
    mode = getattr(args, "group_center", "none")
    if mode == "none":
        return None
    group_id = getattr(data, "group_id", None)
    if group_id is None:
        group_id, _ = load_group_ids(args.pairs, data.img.device)
    if group_id is None:
        raise SystemExit("--group-center needs img_group in the pairs file; rebuild it with build_quilt_pairs.py")
    if len(group_id) != data.img.shape[0]:
        raise SystemExit(f"img_group has {len(group_id)} entries but the pairs file has {data.img.shape[0]} images")
    if args.group_kappa < 0:
        raise SystemExit("--group-kappa must be >= 0")
    if args.group_min_images < 2 and args.group_kappa == 0:
        raise SystemExit("--group-kappa 0 with --group-min-images < 2 would centre single images to zero")
    dev = data.img.device
    dtype = mu_t.dtype

    # training images and their groups
    train_imgs = torch.unique(data.pair_img[pairs])
    gid = group_id[train_imgs]
    if mode == "random":                                # control: keep the group sizes, shuffle membership
        g = torch.Generator(device="cpu").manual_seed(seed)
        gid = gid[torch.randperm(len(gid), generator=g).to(dev)]
    n_raw = torch.bincount(gid)
    keep = n_raw >= args.group_min_images
    G = int(keep.sum())
    comp = torch.full_like(n_raw, -1)
    comp[keep] = torch.arange(G, device=dev)
    img_gid = torch.full((data.img.shape[0],), -1, dtype=torch.long, device=dev)
    img_gid[train_imgs] = comp[gid]
    n = n_raw[keep].double()
    lam = n / (n + args.group_kappa)                    # kappa = 0 -> 1 (pure within-group)

    sides = {"t": args.group_sides in ("both", "text"), "i": args.group_sides in ("both", "image")}
    mu = {"t": mu_t, "i": mu_i}
    info = {"mode": mode, "kappa": args.group_kappa, "min_images": args.group_min_images,
            "sides": args.group_sides, "n_groups_train": int((n_raw > 0).sum()), "n_groups_centred": G,
            "frac_train_images_centred": float(n.sum() / len(train_imgs)) if G else 0.0}
    if G == 0:
        print(f"group centering ({mode}): no group with >= {args.group_min_images} training images, "
              "falling back to global centering")
        return GroupCenters(img_gid, {"t": None, "i": None}, mu, data.pair_img, info)

    # weighted sums per group, accumulated over the training pairs (float64)
    dt, di = data.txt.shape[1], data.img.shape[1]
    W = torch.zeros(G, dtype=torch.float64, device=dev)
    S = {"t": torch.zeros(G, dt, dtype=torch.float64, device=dev),
         "i": torch.zeros(G, di, dtype=torch.float64, device=dev)}
    tot = {"t": 0.0, "i": 0.0}
    m64 = {s: mu[s].double() for s in mu}
    for s0 in range(0, len(pairs), chunk):
        p, ww = pairs[s0:s0 + chunk], w[s0:s0 + chunk]
        g = img_gid[data.pair_img[p]]
        m = g >= 0
        if not m.any():
            continue
        p, ww, g = p[m], ww[m].double(), g[m]
        feats = {"t": F.normalize(data.txt[data.pair_txt[p]].float(), dim=-1).double(),
                 "i": F.normalize(data.img[data.pair_img[p]].float(), dim=-1).double()}
        W.index_add_(0, g, ww)
        for s in ("t", "i"):
            S[s].index_add_(0, g, ww[:, None] * feats[s])
            tot[s] += float((ww * ((feats[s] - m64[s]) ** 2).sum(1)).sum())

    C = {}
    for s in ("t", "i"):
        M = S[s] / W[:, None]                           # group means
        between = float((W * ((M - m64[s]) ** 2).sum(1)).sum())
        info[f"between_var_frac_{'txt' if s == 't' else 'img'}"] = between / max(tot[s], 1e-30)
        C[s] = (lam[:, None] * M + (1 - lam[:, None]) * m64[s]).to(dtype) if sides[s] else None
        del S[s]
    info.update(frac_pair_weight_centred=float(W.sum()),
                mean_group_size=float(n.mean()), median_group_size=float(n.median()),
                max_group_size=int(n.max()), mean_lambda=float((lam * W).sum() / W.sum()))
    print(f"group centering ({mode}, kappa {args.group_kappa:g}, sides {args.group_sides}): "
          f"{G:,} groups, {info['frac_train_images_centred']:.1%} of training images, "
          f"mean size {info['mean_group_size']:.1f}, mean lambda {info['mean_lambda']:.3f}, "
          f"between-group variance img {info['between_var_frac_img']:.3f} txt {info['between_var_frac_txt']:.3f}")
    return GroupCenters(img_gid, C, mu, data.pair_img, info)