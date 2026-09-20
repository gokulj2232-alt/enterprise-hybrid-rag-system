import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "faiss_index.bin"
METADATA_FILE = PROJECT_ROOT / "data" / "chunk_metadata.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(str(INDEX_FILE))

# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

print("Loading metadata...")

metadata = []

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            metadata.append(json.loads(line))

print(f"Loaded metadata: {len(metadata):,}")

# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def semantic_search(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        result = metadata[idx].copy()
        result["score"] = float(score)

        results.append(result)

    return results


# --------------------------------------------------
# INTERACTIVE SEARCH
# --------------------------------------------------

print()
print("=" * 60)
print("SEMANTIC SEARCH")
print("=" * 60)

while True:

    query = input("\nEnter your search query (or type 'exit'): ").strip()

    if query.lower() == "exit":
        print("Exiting search...")
        break

    if not query:
        print("Please enter a query.")
        continue

    results = semantic_search(query, top_k=5)

    print()
    print("-" * 60)
    print(f"SEARCH RESULTS FOR: {query}")
    print("-" * 60)

    for rank, result in enumerate(results, start=1):

        print()
        print(f"Result #{rank}")
        print(f"Score     : {result['score']:.4f}")
        print(f"Document  : {result['document_id']}")
        print(f"Category  : {result['category']}")
        print(f"Title     : {result['title']}")
        print(f"Chunk     : {result['chunk_index']}")
        print(f"Text      : {result['text'][:400]}...")

print()
print("SEMANTIC SEARCH CLOSED")