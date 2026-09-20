from pathlib import Path
import json

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index.bin"
METADATA_PATH = PROJECT_ROOT / "data" / "chunk_metadata.jsonl"


def update_faiss_index(embeddings, metadata):
    if len(embeddings) == 0:
        return

    index = faiss.read_index(str(INDEX_PATH))

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "a", encoding="utf-8") as file:
        for item in metadata:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Added {len(embeddings)} vectors to FAISS.")
    print(f"FAISS total vectors: {index.ntotal}")