import pickle
from rank_bm25 import BM25Okapi


CHUNKS_PATH = "models/processed_chunks.pkl"
BM25_PATH = "models/bm25.pkl"


def tokenize(text):
    return text.lower().split()


def main():

    print("=" * 60)
    print("BUILDING BM25 INDEX")
    print("=" * 60)

    print("\n[1/3] Loading processed chunks...")

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    print(f"Loaded chunks: {len(chunks):,}")

    print("\n[2/3] Building BM25 index...")

    documents = [
        tokenize(chunk["chunk_text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(documents)

    print("BM25 index built successfully!")

    print("\n[3/3] Saving BM25 index...")

    with open(BM25_PATH, "wb") as file:
        pickle.dump(bm25, file)

    print(f"BM25 index saved to: {BM25_PATH}")

    print("\n" + "=" * 60)
    print("BM25 INDEX BUILD COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()