import numpy as np
import faiss
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings.npy"
INDEX_FILE = PROJECT_ROOT / "data" / "faiss_index.bin"

# --------------------------------------------------
# LOAD EMBEDDINGS
# --------------------------------------------------

print("=" * 60)
print("STARTING FAISS INDEX CREATION")
print("=" * 60)

print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings shape: {embeddings.shape}")

# FAISS requires float32
embeddings = embeddings.astype("float32")

# --------------------------------------------------
# CREATE INDEX
# --------------------------------------------------

dimension = embeddings.shape[1]

print(f"Vector dimension: {dimension}")
print("Creating FAISS index...")

index = faiss.IndexFlatIP(dimension)

# --------------------------------------------------
# ADD VECTORS
# --------------------------------------------------

print("Adding vectors to FAISS...")

index.add(embeddings)

# --------------------------------------------------
# SAVE INDEX
# --------------------------------------------------

faiss.write_index(index, str(INDEX_FILE))

# --------------------------------------------------
# RESULT
# --------------------------------------------------

print()
print("=" * 60)
print("FAISS INDEX CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Vectors indexed : {index.ntotal:,}")
print(f"Dimension       : {dimension}")
print(f"Index file      : {INDEX_FILE}")

print("=" * 60)
print("STAGE 5 VECTOR INDEX COMPLETE")
print("=" * 60)