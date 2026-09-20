import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/NLP_50K_Upgraded.xlsx")
OUTPUT_FILE = Path("data/NLP_50K_Upgraded_Final.xlsx")

# Rich knowledge for important RAG / AI topics
KNOWLEDGE = {
    "rag": {
        "category": "RAG",
        "content": (
            "Retrieval-Augmented Generation (RAG) is an approach that combines "
            "information retrieval with a large language model. When a user asks "
            "a question, the RAG system searches a knowledge base for relevant "
            "documents. The retrieved documents are provided to the language "
            "model as context, and the model generates an answer using that "
            "context. A typical RAG pipeline includes query processing, document "
            "retrieval, ranking, context construction, answer generation, and "
            "grounding validation. RAG is useful when an application needs to "
            "answer questions using external or private documents rather than "
            "depending only on the model's pretrained knowledge."
        )
    },

    "document chunking": {
        "category": "RAG",
        "content": (
            "Document chunking is the process of dividing a large document into "
            "smaller pieces called chunks before retrieval. Chunking helps a RAG "
            "system retrieve specific sections that are relevant to a user's "
            "question. Common approaches include fixed-size chunking, sentence "
            "based chunking, paragraph based chunking, and overlapping chunks. "
            "Chunk size affects retrieval quality and the amount of context "
            "provided to the language model. Very small chunks may lose context, "
            "while very large chunks may contain unnecessary information."
        )
    },

    "document retrieval": {
        "category": "RAG",
        "content": (
            "Document retrieval is the stage of a RAG system that searches a "
            "knowledge base for information relevant to a user query. Retrieval "
            "can use semantic search, keyword search, BM25, vector similarity, "
            "or a combination of methods. The goal is to return documents or "
            "chunks that contain information useful for answering the question. "
            "Good retrieval is important because the generation model can only "
            "produce a well-grounded answer when relevant evidence is retrieved."
        )
    },

    "embeddings": {
        "category": "Embeddings",
        "content": (
            "Embeddings are numerical vector representations of text, documents, "
            "sentences, or other data. Text with similar meaning can have similar "
            "vector representations. In semantic search and RAG systems, "
            "embeddings allow the system to compare a user's query with document "
            "embeddings and retrieve semantically relevant information. "
            "Embedding quality has a direct effect on semantic retrieval quality."
        )
    },

    "vector database": {
        "category": "Vector Database",
        "content": (
            "A vector database stores and searches numerical vector embeddings. "
            "In RAG applications, documents are converted into embeddings and "
            "stored in a vector index. When a user submits a query, the query is "
            "also converted into an embedding and compared with stored vectors. "
            "The system retrieves vectors that are closest to the query according "
            "to a similarity measure. Vector databases are commonly used for "
            "semantic search, recommendation systems, and retrieval pipelines."
        )
    },

    "semantic search": {
        "category": "Semantic Search",
        "content": (
            "Semantic search retrieves information based on meaning rather than "
            "only exact keyword matches. A query and documents are converted "
            "into embeddings, and the system compares their vector similarity. "
            "This allows semantic search to find relevant documents even when "
            "the wording of the query differs from the wording in the document. "
            "Semantic search is commonly used in RAG systems and intelligent "
            "question-answering applications."
        )
    },

    "bm25": {
        "category": "Information Retrieval",
        "content": (
            "BM25 is a keyword-based information retrieval algorithm used to rank "
            "documents according to their relevance to a query. It considers "
            "factors such as term frequency, inverse document frequency, and "
            "document length. BM25 is useful when exact words or important "
            "keywords need to be matched. In hybrid RAG systems, BM25 can be "
            "combined with semantic vector search to improve retrieval coverage."
        )
    },

    "hybrid search": {
        "category": "Information Retrieval",
        "content": (
            "Hybrid search combines multiple retrieval methods, commonly semantic "
            "vector search and keyword search such as BM25. Semantic search "
            "captures meaning and related concepts, while keyword search can "
            "capture exact terms. Combining their results can improve retrieval "
            "coverage and robustness. Hybrid retrieval is commonly used in "
            "production RAG systems where both semantic similarity and exact "
            "keyword matching are useful."
        )
    },

    "reranking": {
        "category": "RAG",
        "content": (
            "Reranking is a retrieval stage that reorders initially retrieved "
            "documents according to their relevance to the query. A cross-encoder "
            "can examine the query and document together and assign a relevance "
            "score. Reranking helps move the most useful documents toward the "
            "top of the result list before context is sent to the language model. "
            "It can improve the quality of retrieved evidence in RAG pipelines."
        )
    },

    "query processing": {
        "category": "RAG",
        "content": (
            "Query processing prepares a user's question for retrieval. It can "
            "identify the question type, extract important keywords, normalize "
            "the query, and generate related search queries. Query processing "
            "helps the retrieval system understand the user's information need "
            "and can improve the chance of finding relevant documents."
        )
    },

    "context retrieval": {
        "category": "RAG",
        "content": (
            "Context retrieval is the process of selecting relevant information "
            "from a knowledge base and preparing it as context for a language "
            "model. In a RAG system, retrieved chunks are usually combined into "
            "a structured context containing source information and document "
            "content. High-quality context should be relevant to the question "
            "and should avoid unnecessary or unrelated information."
        )
    },

    "llm": {
        "category": "Generative AI",
        "content": (
            "A Large Language Model (LLM) is a machine learning model trained on "
            "large amounts of text to understand and generate natural language. "
            "LLMs can answer questions, summarize text, generate code, and perform "
            "other language tasks. In a RAG system, an LLM receives retrieved "
            "documents as context and generates an answer based on that context."
        )
    },

    "prompt engineering": {
        "category": "Generative AI",
        "content": (
            "Prompt engineering is the practice of designing instructions and "
            "input context for a language model to produce useful and controlled "
            "outputs. A prompt can specify the task, available context, output "
            "format, constraints, and rules. In RAG systems, prompts can instruct "
            "the language model to answer using only the retrieved evidence and "
            "avoid unsupported information."
        )
    },

    "grounding": {
        "category": "RAG",
        "content": (
            "Grounding means ensuring that a generated answer is supported by "
            "retrieved evidence. A grounded RAG system checks whether statements "
            "in the generated answer can be supported by the provided documents. "
            "Grounding validation helps reduce unsupported claims and hallucinated "
            "information."
        )
    },

    "answerability": {
        "category": "RAG",
        "content": (
            "Answerability is the process of determining whether the retrieved "
            "evidence contains enough relevant information to answer a question. "
            "An answerability check can use semantic similarity, keyword overlap, "
            "or other relevance signals. If the evidence is insufficient, a RAG "
            "system can return a controlled response instead of generating an "
            "unsupported answer."
        )
    },
}


def make_content(title, category, keyword):
    """Return rich content for an important topic when available."""
    key = title.strip().lower()

    if key in KNOWLEDGE:
        return KNOWLEDGE[key]["content"]

    # Also match common title variations.
    normalized = key.replace("-", " ").replace("_", " ")

    for topic, info in KNOWLEDGE.items():
        if topic in normalized or normalized in topic:
            return info["content"]

    return None


def main():
    print("Loading dataset...")
    df = pd.read_excel(INPUT_FILE)

    required_columns = [
        "document_id",
        "category",
        "title",
        "content",
        "keyword",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    original_rows = len(df)
    upgraded = 0

    for i in range(len(df)):
        title = str(df.at[i, "title"])
        category = str(df.at[i, "category"])
        keyword = str(df.at[i, "keyword"])

        new_content = make_content(title, category, keyword)

        if new_content:
            df.at[i, "content"] = new_content
            upgraded += 1

    df.to_excel(OUTPUT_FILE, index=False)

    print()
    print("========================================")
    print("DATASET UPGRADE COMPLETED")
    print("========================================")
    print(f"Original rows : {original_rows}")
    print(f"Rows upgraded : {upgraded}")
    print(f"Output file   : {OUTPUT_FILE}")
    print(f"Unique content: {df['content'].nunique()}")
    print("========================================")


if __name__ == "__main__":
    main()