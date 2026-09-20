import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    """

    def __init__(self, dimension: int = 384):

        self.dimension = dimension

        # Inner Product works as cosine similarity
        # because our embeddings are normalized.
        self.index = faiss.IndexFlatIP(dimension)

        self.metadata = []

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: list
    ):
        """
        Add embedding vectors and their metadata.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} dimensions, "
                f"got {embeddings.shape[1]}"
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings and metadata "
                "records must be equal."
            )

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):
        """
        Search the vector store using cosine similarity.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} dimensions, "
                f"got {query_embedding.shape[1]}"
            )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            result = self.metadata[index].copy()

            result["score"] = float(score)

            results.append(result)

        return results

    def save(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Save FAISS index and metadata to disk.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(
            metadata_path,
            "wb"
        ) as file:

            pickle.dump(
                self.metadata,
                file
            )

    @classmethod
    def load(
        cls,
        index_path: str,
        metadata_path: str
    ):
        """
        Load FAISS index and metadata from disk.
        """

        store = cls()

        store.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "rb"
        ) as file:

            store.metadata = pickle.load(file)

        return store

    def __len__(self):
        return self.index.ntotal


if __name__ == "__main__":

    print("Testing FAISS Vector Store...")

    # Create test vectors
    test_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0]
        ],
        dtype="float32"
    )

    # Normalize vectors
    faiss.normalize_L2(test_embeddings)

    test_metadata = [
        {
            "document_id": "DOC001",
            "title": "Cloud Computing"
        },
        {
            "document_id": "DOC002",
            "title": "Machine Learning"
        },
        {
            "document_id": "DOC003",
            "title": "Cloud Infrastructure"
        }
    ]

    store = VectorStore(
        dimension=3
    )

    store.add_vectors(
        test_embeddings,
        test_metadata
    )

    # Test query
    query = np.array(
        [[1.0, 0.0, 0.0]],
        dtype="float32"
    )

    faiss.normalize_L2(query)

    results = store.search(
        query,
        top_k=2
    )

    print("\nVector Store Test Successful!")

    print(f"Total vectors: {len(store)}")

    print("\nSearch results:")

    for result in results:
        print(result)