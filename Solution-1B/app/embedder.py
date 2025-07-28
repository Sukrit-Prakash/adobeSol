from sentence_transformers import SentenceTransformer
import numpy as np

def get_model(name="paraphrase-MiniLM-L6-v2", device="cpu"):
    model = SentenceTransformer(name, device=device)
    return model

def get_embeddings(model, texts):
    return model.encode(texts, normalize_embeddings=True, batch_size=32)
