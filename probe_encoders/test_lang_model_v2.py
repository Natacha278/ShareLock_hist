"""
Cosine similarity between prompts with BioLORD-2023 and BioMistral-7B.
Run on a GPU node (offline): see job script at the bottom of this file.
"""
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

WEIGHTS = "/project/rrg-josedolz/natgill/weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

prompts = [
    "Breast tumor epithelial tissue at 20× magnification, showing densely cellular malignant epithelial proliferation arranged in irregular glands, nests, cords, or solid sheets, with disrupted architecture, variable gland formation, and infiltrative growth within fibrous or desmoplastic stroma."
            ,
    "Normal breast tissue at 20× magnification, showing orderly ducts and lobules composed of regularly arranged epithelial structures with open lumina, embedded within fibrous and adipose stroma. The tissue exhibits organized glandular architecture, relatively uniform epithelial cellularity, and well-defined epithelial–stromal boundaries without dense irregular epithelial proliferation or destructive infiltration."
    "An image showing malignant tissue.",
]

print(prompts)

def print_matrix(name, sim):
    print(f"\n=== {name} ===")
    for i, row in enumerate(sim.tolist()):
        print(f"[{i}] " + "  ".join(f"{v:.4f}" for v in row))


# ---------------------------------------------------------------------------
# 1) BioLORD-2023: a sentence-embedding model, so this is direct
# ---------------------------------------------------------------------------
biolord = SentenceTransformer(f"{WEIGHTS}/BioLORD-2023", device=DEVICE)
emb_biolord = biolord.encode(prompts, convert_to_tensor=True, normalize_embeddings=True)
sim_biolord = emb_biolord @ emb_biolord.T          # normalized -> dot product = cosine
print_matrix("BioLORD-2023", sim_biolord)

del biolord
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 2) BioMistral-7B: a generative (decoder) model, no built-in sentence
#    embedding -> mean-pool the last hidden layer over real (non-pad) tokens
# ---------------------------------------------------------------------------
path = f"{WEIGHTS}/BioMistral-7B"
tok = AutoTokenizer.from_pretrained(path)
tok.pad_token = tok.eos_token                     # Mistral has no pad token
tok.padding_side = "right"

# AutoModel loads only the transformer body (no LM head) -> less memory
biomistral = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
biomistral.eval()

with torch.no_grad():
    batch = tok(prompts, padding=True, return_tensors="pt").to(biomistral.device)
    hidden = biomistral(**batch).last_hidden_state.float()     # (B, T, 4096)
    mask = batch["attention_mask"].unsqueeze(-1).float()       # (B, T, 1)
    emb_mistral = (hidden * mask).sum(1) / mask.sum(1)         # mean pooling
    emb_mistral = F.normalize(emb_mistral, dim=-1)

sim_mistral = emb_mistral @ emb_mistral.T
print_matrix("BioMistral-7B (mean-pooled)", sim_mistral)