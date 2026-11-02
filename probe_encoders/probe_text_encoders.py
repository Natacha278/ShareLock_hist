"""
Probe text encoders for histopathology:
  A) Class separation  - do descriptions of the same class cluster together,
                         and away from other classes?
  B) Negation          - does "no tumor cells are present" move away from
                         "tumor cells are present"? Does "non-malignant"
                         land near "benign"?

Models are loaded one at a time from WEIGHTS; any model folder that is
missing is skipped. Raw decoders are reported with both mean and
last-token pooling.

Outputs (in ./probe_results/):
  summary.csv                 one row per model/pooling
  negation_<model>.csv        per-triplet negation scores
  heatmap_<model>.png         similarity matrix of class descriptions

Requires: torch, transformers>=4.51, sentence-transformers>=2.7,
          scikit-learn, matplotlib, pandas
"""
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

WEIGHTS = "/project/rrg-josedolz/natgill/weights"
OUT = "probe_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16

# name -> (loader, options)
MODELS = {
    "BioLORD-2023":             ("st", {}),
    "Qwen3-Embedding-8B":       ("st", {"padding_side": "left", "bf16": True}),
    "BMRetriever-7B":           ("decoder", {"poolings": ["last"], "append_eos": True}),
    "BioMistral-7B":            ("decoder", {"poolings": ["mean", "last"]}),
    "Mistral-7B-Instruct-v0.1": ("decoder", {"poolings": ["mean", "last"]}),
    "Qwen3-8B":                 ("decoder", {"poolings": ["mean", "last"]}),
}

# ===========================================================================
# Test data
# ===========================================================================
# A) Each class: several descriptions with deliberately varied wording.
CLASSES = {
    "normal colon mucosa": [
        "Normal colonic mucosa with regularly spaced crypts and abundant goblet cells.",
        "Benign large-bowel mucosa showing test-tube shaped crypts and preserved architecture.",
        "Unremarkable colon lining with orderly glands and intact surface epithelium.",
        "Colonic mucosa without dysplasia; crypts are evenly distributed.",
    ],
    "colorectal adenocarcinoma": [
        "Malignant glands with tall columnar cells, pseudostratified nuclei and dirty necrosis.",
        "Invasive adenocarcinoma of the colon forming irregular, cribriform glands.",
        "Atypical crowded glands infiltrating the submucosa with desmoplastic reaction.",
        "Gland-forming carcinoma with nuclear pleomorphism and frequent mitoses.",
    ],
    "lymphoid tissue": [
        "Dense aggregates of small mature lymphocytes.",
        "Lymphoid follicles with germinal centers.",
        "Sheets of small round cells with dark nuclei and scant cytoplasm, consistent with lymphocytes.",
        "Reactive lymphoid tissue with a mixed population of lymphocytes.",
    ],
    "tumor stroma": [
        "Desmoplastic stroma with spindle-shaped fibroblasts and collagen.",
        "Fibrous connective tissue surrounding tumor nests.",
        "Cancer-associated stroma composed of myofibroblasts in a collagenous matrix.",
        "Loose to dense fibrotic stroma with scattered spindle cells.",
    ],
    "necrosis": [
        "Areas of coagulative necrosis with nuclear debris.",
        "Necrotic tissue with ghost cells and karyorrhexis.",
        "Eosinophilic acellular debris consistent with tumor necrosis.",
        "Extensive necrosis with loss of cellular detail.",
    ],
    "adipose tissue": [
        "Mature adipocytes with large clear vacuoles and thin membranes.",
        "Fat tissue composed of univacuolated cells with peripheral nuclei.",
        "Lobules of mature fat separated by thin fibrous septa.",
        "Adipose tissue without atypia.",
    ],
    "ductal carcinoma in situ": [
        "Ductal carcinoma in situ with cribriform architecture confined to the ducts.",
        "DCIS, high nuclear grade, with comedo-type central necrosis.",
        "Neoplastic epithelial proliferation filling ducts with an intact myoepithelial layer.",
        "Intraductal carcinoma without stromal invasion.",
    ],
    "invasive ductal carcinoma": [
        "Invasive ductal carcinoma infiltrating the breast stroma as cords and nests.",
        "Infiltrating carcinoma of no special type with stromal invasion.",
        "Malignant epithelial cells invading beyond the basement membrane into fibrous stroma.",
        "Invasive breast carcinoma, Nottingham grade 2.",
    ],
}

# Class-name prompts for a zero-shot check (prompt -> description).
PROMPT_TEMPLATE = "An H&E-stained histology image of {}."

# B) Negation triplets: (anchor, same meaning / different words,
#                         opposite meaning / similar words, category)
NEGATION = [
    ("Tumor cells are present.",                 "Malignant cells are identified.",            "No tumor cells are present.",              "explicit no"),
    ("Lymphovascular invasion is identified.",   "Tumor emboli are seen in vessels.",          "Lymphovascular invasion is not identified.", "not identified"),
    ("Perineural invasion is present.",          "Tumor surrounds nerve fibers.",              "Perineural invasion is not present.",       "not present"),
    ("Margins are involved by carcinoma.",       "Carcinoma extends to the resection margin.", "Margins are free of carcinoma.",            "free of"),
    ("Metastatic carcinoma in the lymph node.",  "The lymph node contains tumor deposits.",    "Lymph node negative for metastatic carcinoma.", "negative for"),
    ("The lesion is malignant.",                 "The lesion is cancerous.",                   "The lesion is non-malignant.",              "prefix non-"),
    ("The lesion is benign.",                    "The lesion is non-malignant.",               "The lesion is malignant.",                  "benign = non-malignant"),
    ("No evidence of malignancy.",               "Benign tissue.",                             "Evidence of malignancy.",                   "absence as anchor"),
    ("Dysplasia is present.",                    "Dysplastic epithelium is seen.",             "Dysplasia is absent.",                      "absent"),
    ("Necrosis is seen.",                        "There are areas of necrotic tissue.",        "Necrosis is not seen.",                     "not seen"),
    ("Mitotic figures are frequent.",            "Mitoses are numerous.",                      "Mitotic figures are absent.",               "absent"),
    ("Invasive carcinoma is present.",           "Tumor infiltrates the stroma.",              "Carcinoma in situ without invasion.",       "without"),
    ("Atypical cells are seen.",                 "Cytological atypia is present.",             "No atypical cells are seen.",               "explicit no"),
    ("Positive for malignancy.",                 "Malignant.",                                 "Negative for malignancy.",                  "positive/negative"),
    ("Granulomas are present.",                  "Granulomatous inflammation is seen.",        "Granulomas are not present.",               "not present"),
    ("The tumor is high grade.",                 "Poorly differentiated tumor.",               "The tumor is low grade.",                   "antonym (control)"),
]


# ===========================================================================
# Model loading / embedding
# ===========================================================================
def load_st(path, opts):
    from sentence_transformers import SentenceTransformer
    kw = {"device": DEVICE}
    if opts.get("bf16"):
        kw["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    if "padding_side" in opts:
        kw["tokenizer_kwargs"] = {"padding_side": opts["padding_side"]}
    model = SentenceTransformer(path, **kw)

    def embed(texts):
        e = model.encode(texts, batch_size=BATCH, convert_to_tensor=True,
                         normalize_embeddings=True)
        return {"default": e.float().cpu()}
    return model, embed


def load_decoder(path, opts):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                      device_map="auto")
    model.eval()
    poolings = opts["poolings"]

    @torch.no_grad()
    def embed(texts):
        if opts.get("append_eos"):
            texts = [t + tok.eos_token for t in texts]
        out = {p: [] for p in poolings}
        for i in range(0, len(texts), BATCH):
            batch = tok(texts[i:i + BATCH], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(model.device)
            h = model(**batch).last_hidden_state.float()          # (B, T, D)
            mask = batch["attention_mask"]
            if "mean" in poolings:
                m = mask.unsqueeze(-1).float()
                out["mean"].append(((h * m).sum(1) / m.sum(1)).cpu())
            if "last" in poolings:
                last_idx = mask.sum(1) - 1                          # right padding
                rows = torch.arange(h.size(0), device=h.device)
                out["last"].append(h[rows, last_idx.to(h.device)].cpu())
        return {p: F.normalize(torch.cat(v), dim=-1) for p, v in out.items()}
    return model, embed


# ===========================================================================
# Metrics
# ===========================================================================
def class_separation(E, labels):
    from sklearn.metrics import silhouette_score
    labels = np.array(labels)
    S = (E @ E.T).numpy()
    n = len(labels)
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(n, dtype=bool)
    intra = S[same & off_diag].mean()
    inter = S[~same].mean()

    # Leave-one-out nearest-centroid accuracy
    correct = 0
    classes = np.unique(labels)
    for i in range(n):
        keep = np.arange(n) != i
        cents = torch.stack([E[torch.from_numpy(keep & (labels == c))].mean(0)
                             for c in classes])
        pred = classes[int((F.normalize(cents, dim=-1) @ E[i]).argmax())]
        correct += pred == labels[i]

    sil = silhouette_score(E.numpy(), labels, metric="cosine")
    return {"intra": intra, "inter": inter, "gap": intra - inter,
            "silhouette": sil, "loo_acc": correct / n}, S


def zero_shot(E_desc, E_prompt, labels, class_names):
    pred = (E_desc @ E_prompt.T).argmax(1).numpy()
    return float(np.mean([class_names[p] == l for p, l in zip(pred, labels)]))


def negation_scores(E_a, E_same, E_opp):
    s_same = (E_a * E_same).sum(-1).numpy()
    s_opp = (E_a * E_opp).sum(-1).numpy()
    return s_same, s_opp


def save_heatmap(S, labels, title, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(S, cmap="viridis")
    # one tick per class block
    names, starts = [], []
    for i, l in enumerate(labels):
        if i == 0 or l != labels[i - 1]:
            names.append(l)
            starts.append(i)
    sizes = np.diff(starts + [len(labels)])
    centers = [s + (k - 1) / 2 for s, k in zip(starts, sizes)]
    ax.set_xticks(centers, names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(centers, names, fontsize=8)
    for s in starts[1:]:
        ax.axhline(s - 0.5, color="white", lw=0.8)
        ax.axvline(s - 0.5, color="white", lw=0.8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)

    class_names = list(CLASSES.keys())
    desc_texts, desc_labels = [], []
    for c, descs in CLASSES.items():
        desc_texts += descs
        desc_labels += [c] * len(descs)
    prompt_texts = [PROMPT_TEMPLATE.format(c) for c in class_names]
    anchors = [t[0] for t in NEGATION]
    sames = [t[1] for t in NEGATION]
    opps = [t[2] for t in NEGATION]
    cats = [t[3] for t in NEGATION]

    rows = []
    for name, (loader, opts) in MODELS.items():
        path = os.path.join(WEIGHTS, name)
        if not os.path.isdir(path):
            print(f"[skip] {name}: not found at {path}")
            continue
        print(f"\n######## {name} ########")
        model, embed = (load_st if loader == "st" else load_decoder)(path, opts)

        E_desc = embed(desc_texts)
        E_prompt = embed(prompt_texts)
        E_a, E_s, E_o = embed(anchors), embed(sames), embed(opps)

        for pool in E_desc:
            tag = name if pool == "default" else f"{name} [{pool}]"
            sep, S = class_separation(E_desc[pool], desc_labels)
            zs = zero_shot(E_desc[pool], E_prompt[pool], desc_labels, class_names)
            s_same, s_opp = negation_scores(E_a[pool], E_s[pool], E_o[pool])

            # exclude the antonym control from the negation summary
            neg = np.array([c != "antonym (control)" for c in cats])
            rows.append({
                "model": tag,
                **{k: round(float(v), 4) for k, v in sep.items()},
                "zeroshot_acc": round(zs, 4),
                "neg_acc": round(float((s_same[neg] > s_opp[neg]).mean()), 4),
                "neg_margin": round(float((s_same[neg] - s_opp[neg]).mean()), 4),
                "sim_to_negated": round(float(s_opp[neg].mean()), 4),
            })

            safe = tag.replace(" ", "_").replace("[", "").replace("]", "")
            pd.DataFrame({
                "category": cats, "anchor": anchors,
                "same_meaning": sames, "opposite_meaning": opps,
                "sim_same": s_same.round(4), "sim_opposite": s_opp.round(4),
                "correct": s_same > s_opp,
            }).to_csv(f"{OUT}/negation_{safe}.csv", index=False)
            save_heatmap(S, desc_labels, tag, f"{OUT}/heatmap_{safe}.png")

        del model, embed
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n================ SUMMARY ================")
    print(df.to_string(index=False))
    print(f"\nPer-triplet negation results and heatmaps are in ./{OUT}/")


if __name__ == "__main__":
    main()