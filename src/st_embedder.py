# src/st_embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer

class STEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # good for cosine/IP
        )
        return emb.astype("float32")

def make_model():
    return STEmbedder("all-MiniLM-L6-v2")  # change the model if you like
