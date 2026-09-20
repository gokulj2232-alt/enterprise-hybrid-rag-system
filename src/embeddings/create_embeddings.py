import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"
OUTPUT_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings.npy"
OUTPUT_METADATA = PROJECT_ROOT / "data" / "chunk_metadata.jsonl"

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# --------------------------------------------------
# START
# --------------------------------------------------

print("=" * 60)
print("STARTING EMBEDDING GENERATION")
print("=" * 60)

print(f"Loading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")

# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

print("Loading chunks...")

chunks = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Chunks loaded: {len(chunks):,}")

# --------------------------------------------------
# EXTRACT TEXT
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

print("Creating embeddings...")
print("This may take some time.")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

# --------------------------------------------------
# SAVE EMBEDDINGS
# --------------------------------------------------

np.save(OUTPUT_EMBEDDINGS, embeddings)

# --------------------------------------------------
# SAVE METADATA
# --------------------------------------------------

print("Saving chunk metadata...")

with open(OUTPUT_METADATA, "w", encoding="utf-8") as file:

    for chunk in chunks:

        metadata = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "category": chunk["category"],
            "title": chunk["title"],
            "keyword": chunk["keyword"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"]
        }

        file.write(
            json.dumps(
                metadata,
                ensure_ascii=False
            ) + "\n"
        )

# --------------------------------------------------
# RESULT
# --------------------------------------------------

print()
print("=" * 60)
print("EMBEDDING GENERATION COMPLETE")
print("=" * 60)

print(f"Total chunks       : {len(chunks):,}")
print(f"Embedding shape    : {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Embeddings file    : {OUTPUT_EMBEDDINGS}")
print(f"Metadata file      : {OUTPUT_METADATA}")

print("=" * 60)
print("STAGE 4 EMBEDDINGS COMPLETE")
print("=" * 60)