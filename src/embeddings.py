from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around the Sentence Transformers
    embedding model.
    """

    def __init__(self, model_name: str = MODEL_NAME):

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully!")

    def encode(
        self,
        texts,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Convert text into normalized embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert a user query into an embedding vector.
        """

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype("float32")


if __name__ == "__main__":

    model = EmbeddingModel()

    test_texts = [
        "Cloud computing provides scalable computing resources.",
        "Machine learning allows computers to learn from data."
    ]

    embeddings = model.encode(test_texts)

    print("\nEmbedding test completed!")

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Vector dimension: {embeddings.shape[1]}")

    print("\nFirst vector:")
    print(embeddings[0])