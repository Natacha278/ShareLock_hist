"""
Test Qwen3-Embedding-8B on histology vocabulary.
  1) tokenizer fragmentation of pathology terms
  2) cosine similarity on hard pairs (synonyms vs look-alike classes)
  3) mini zero-shot: match a caption to class descriptions
Run on a GPU node, offline.
"""
import torch
from sentence_transformers import SentenceTransformer

PATH = "/project/rrg-josedolz/natgill/weights/Qwen3-Embedding-8B"

model = SentenceTransformer(
    PATH,
    device="cuda",
    model_kwargs={"torch_dtype": torch.bfloat16},
    tokenizer_kwargs={"padding_side": "left"},   # required: last-token pooling
)
print("Embedding dim:", model.get_sentence_embedding_dimension())

# ---------------------------------------------------------------------------
# 1) Tokenizer fragmentation
# ---------------------------------------------------------------------------
terms = [
    "koilocytosis", "psammoma bodies", "signet-ring cell carcinoma",
    "ductal carcinoma in situ", "DCIS", "Gleason pattern 4",
    "lymphovascular invasion", "tumor-infiltrating lymphocytes",
]
print("\n=== Tokenization ===")
for t in terms:
    pieces = model.tokenizer.tokenize(t)
    print(f"{len(pieces):2d} tokens  {t:35s} {pieces}")

# ---------------------------------------------------------------------------
# 2) Hard pairs: synonyms should be HIGH, look-alikes should be LOWER
# ---------------------------------------------------------------------------
pairs = [
    ("synonym",   "DCIS", "ductal carcinoma in situ"),
    ("synonym",   "TILs", "tumor-infiltrating lymphocytes"),
    ("synonym",   "LVI", "lymphovascular invasion"),
    ("look-alike", "Gleason pattern 3", "Gleason pattern 4"),
    ("look-alike", "signet-ring cell carcinoma", "mucinous adenocarcinoma"),
    ("look-alike", "ductal carcinoma in situ", "invasive ductal carcinoma"),
    ("unrelated", "psammoma bodies", "colorectal adenocarcinoma"),
]
a = model.encode([p[1] for p in pairs], normalize_embeddings=True, convert_to_tensor=True)
b = model.encode([p[2] for p in pairs], normalize_embeddings=True, convert_to_tensor=True)
sims = (a * b).sum(-1).tolist()

print("\n=== Hard pairs ===")
for (kind, x, y), s in zip(pairs, sims):
    print(f"{s:.4f}  [{kind:10s}] {x}  <->  {y}")

# ---------------------------------------------------------------------------
# 3) Mini zero-shot: caption (query, with instruction) vs class descriptions
# ---------------------------------------------------------------------------
task = "Given a histopathology image description, retrieve the matching tissue class"
instruct = f"Instruct: {task}\nQuery: "

captions = [
    "Glands lined by tall columnar cells with pseudostratified, cigar-shaped nuclei and dirty necrosis.",
    "Sheets of small round blue cells with scant cytoplasm.",
    "Squamous cells with perinuclear halos and enlarged, wrinkled nuclei.",
]
classes = {
    "colorectal adenocarcinoma": "Colorectal adenocarcinoma: malignant glands with columnar epithelium and luminal necrosis.",
    "small cell carcinoma":      "Small cell carcinoma: small tumor cells with scant cytoplasm, nuclear molding and fine chromatin.",
    "koilocytosis (HPV)":        "Koilocytosis: squamous cells with perinuclear clearing and irregular hyperchromatic nuclei, typical of HPV.",
    "normal lymph node":         "Normal lymph node: lymphoid follicles with germinal centers and sinuses.",
}

q = model.encode(captions, prompt=instruct, normalize_embeddings=True, convert_to_tensor=True)
d = model.encode(list(classes.values()), normalize_embeddings=True, convert_to_tensor=True)
S = q @ d.T
names = list(classes.keys())

print("\n=== Zero-shot (rows = captions, cols = classes) ===")
print(" " * 6 + "  ".join(f"{n[:14]:>14s}" for n in names))
for i, row in enumerate(S.tolist()):
    pred = names[int(torch.tensor(row).argmax())]
    print(f"[{i}]   " + "  ".join(f"{v:14.4f}" for v in row) + f"   -> {pred}")