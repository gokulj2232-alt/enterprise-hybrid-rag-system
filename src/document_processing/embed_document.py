import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")


def create_embeddings(chunks):
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return np.asarray(embeddings, dtype=np.float32)