# BioLORD
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("/project/rrg-josedolz/natgill/weights/BioLORD-2023", device="cuda")

# # BioMistral
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# path = "/project/rrg-josedolz/natgill/weights/BioMistral-7B"
# tok = AutoTokenizer.from_pretrained(path)
# model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")