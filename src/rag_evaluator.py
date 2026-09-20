import re
import math
from typing import List, Dict, Any


class RAGEvaluator:
    """
    Evaluation system for the Enterprise Semantic Search + RAG pipeline.

    Metrics:
    1. Retrieval Precision
    2. Context Relevance
    3. Answer Relevance
    4. Faithfulness / Grounding
    5. Answerability
    6. Overall RAG Score
    """

    def __init__(
        self,
        retrieval_weight: float = 0.20,
        context_weight: float = 0.20,
        answer_weight: float = 0.20,
        faithfulness_weight: float = 0.25,
        answerability_weight: float = 0.15,
    ):
        weights = [
            retrieval_weight,
            context_weight,
            answer_weight,
            faithfulness_weight,
            answerability_weight,
        ]

        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Evaluation weights must sum to 1.0")

        self.retrieval_weight = retrieval_weight
        self.context_weight = context_weight
        self.answer_weight = answer_weight
        self.faithfulness_weight = faithfulness_weight
        self.answerability_weight = answerability_weight

    # ============================================================
    # TEXT PROCESSING
    # ============================================================

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

    @staticmethod
    def sentence_split(text: str) -> List[str]:
        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # ============================================================
    # 1. RETRIEVAL PRECISION
    # ============================================================

    def retrieval_precision(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> float:
        """
        Measures how many of the top retrieved documents
        contain meaningful query terms.

        Score range: 0.0 - 1.0
        """

        if not query or not retrieved_results:
            return 0.0

        query_tokens = set(self.tokenize(query))

        if not query_tokens:
            return 0.0

        results = retrieved_results[:top_k]

        relevant_count = 0

        for result in results:
            content = result.get("content", "")

            if not content:
                chunk = result.get("chunk", {})

                if isinstance(chunk, dict):
                    content = (
                        chunk.get("content")
                        or chunk.get("chunk_text")
                        or ""
                    )
                elif isinstance(chunk, str):
                    content = chunk

            document_tokens = set(self.tokenize(content))

            overlap = query_tokens.intersection(document_tokens)

            if len(overlap) >= max(1, math.ceil(len(query_tokens) * 0.3)):
                relevant_count += 1

        return relevant_count / len(results)

    # ============================================================
    # 2. CONTEXT RELEVANCE
    # ============================================================

    def context_relevance(
        self,
        query: str,
        context: str,
    ) -> float:
        """
        Measures how much of the provided context overlaps
        with the query.

        Score range: 0.0 - 1.0
        """

        query_tokens = set(self.tokenize(query))
        context_tokens = set(self.tokenize(context))

        if not query_tokens or not context_tokens:
            return 0.0

        overlap = query_tokens.intersection(context_tokens)

        return min(
            len(overlap) / len(query_tokens),
            1.0
        )

    # ============================================================
    # 3. ANSWER RELEVANCE
    # ============================================================

    def answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        Measures whether the generated answer addresses
        the important terms in the question.

        Score range: 0.0 - 1.0
        """

        if not query or not answer:
            return 0.0

        query_tokens = set(self.tokenize(query))
        answer_tokens = set(self.tokenize(answer))

        if not query_tokens or not answer_tokens:
            return 0.0

        overlap = query_tokens.intersection(answer_tokens)

        score = len(overlap) / len(query_tokens)

        return min(score, 1.0)

    # ============================================================
    # 4. FAITHFULNESS / GROUNDING
    # ============================================================

    def faithfulness(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Measures how strongly each answer sentence is supported
        by the supplied context.

        Score range: 0.0 - 1.0
        """

        if not answer or not context:
            return 0.0

        answer_sentences = self.sentence_split(answer)

        if not answer_sentences:
            return 0.0

        context_tokens = set(self.tokenize(context))

        if not context_tokens:
            return 0.0

        sentence_scores = []

        for sentence in answer_sentences:
            sentence_tokens = set(self.tokenize(sentence))

            if not sentence_tokens:
                continue

            overlap = sentence_tokens.intersection(context_tokens)

            score = len(overlap) / len(sentence_tokens)

            sentence_scores.append(
                min(score, 1.0)
            )

        if not sentence_scores:
            return 0.0

        return sum(sentence_scores) / len(sentence_scores)

    # ============================================================
    # 5. ANSWERABILITY
    # ============================================================

    def answerability_score(
        self,
        answerable: bool,
    ) -> float:
        """
        Converts the existing answerability decision
        into a numerical evaluation metric.
        """

        return 1.0 if answerable else 0.0

    # ============================================================
    # 6. OVERALL RAG SCORE
    # ============================================================

    def overall_score(
        self,
        retrieval_score: float,
        context_score: float,
        answer_score: float,
        faithfulness_score: float,
        answerability_score: float,
    ) -> float:
        """
        Calculates the weighted overall RAG score.

        Score range: 0.0 - 1.0
        """

        score = (
            retrieval_score * self.retrieval_weight
            + context_score * self.context_weight
            + answer_score * self.answer_weight
            + faithfulness_score * self.faithfulness_weight
            + answerability_score * self.answerability_weight
        )

        return round(
            min(max(score, 0.0), 1.0),
            4
        )

    # ============================================================
    # COMPLETE EVALUATION
    # ============================================================

    def evaluate(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        context: str,
        answer: str,
        answerable: bool,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs the complete RAG evaluation.
        """

        retrieval_score = self.retrieval_precision(
            query=query,
            retrieved_results=retrieved_results,
            top_k=top_k,
        )

        context_score = self.context_relevance(
            query=query,
            context=context,
        )

        answer_score = self.answer_relevance(
            query=query,
            answer=answer,
        )

        faithfulness_score = self.faithfulness(
            answer=answer,
            context=context,
        )

        answerability = self.answerability_score(
            answerable
        )

        overall = self.overall_score(
            retrieval_score=retrieval_score,
            context_score=context_score,
            answer_score=answer_score,
            faithfulness_score=faithfulness_score,
            answerability_score=answerability,
        )

        return {
            "retrieval_precision": round(retrieval_score, 4),
            "context_relevance": round(context_score, 4),
            "answer_relevance": round(answer_score, 4),
            "faithfulness": round(faithfulness_score, 4),
            "answerability": round(answerability, 4),
            "overall_rag_score": overall,
        }


# ================================================================
# STANDALONE TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RAG EVALUATION SYSTEM TEST")
    print("=" * 70)

    evaluator = RAGEvaluator()

    query = "What resources does cloud computing provide?"

    retrieved_results = [
        {
            "document_id": "DOC001",
            "title": "Cloud Computing Resources",
            "content": (
                "Cloud computing provides computing resources "
                "over the internet including storage and "
                "processing resources."
            ),
        },
        {
            "document_id": "DOC002",
            "title": "Cloud Computing",
            "content": (
                "Cloud computing provides information about "
                "technology and computing services."
            ),
        },
        {
            "document_id": "DOC003",
            "title": "Cloud Services",
            "content": (
                "Cloud services include computing and storage."
            ),
        },
    ]

    context = (
        "Cloud computing provides computing resources over "
        "the internet including storage and processing resources."
    )

    answer = (
        "Cloud computing provides computing resources over "
        "the internet, including storage and processing resources."
    )

    answerable = True

    results = evaluator.evaluate(
        query=query,
        retrieved_results=retrieved_results,
        context=context,
        answer=answer,
        answerable=answerable,
    )

    print("\nEVALUATION RESULTS")
    print("-" * 70)

    print(
        f"Retrieval Precision : "
        f"{results['retrieval_precision']:.4f}"
    )

    print(
        f"Context Relevance   : "
        f"{results['context_relevance']:.4f}"
    )

    print(
        f"Answer Relevance    : "
        f"{results['answer_relevance']:.4f}"
    )

    print(
        f"Faithfulness        : "
        f"{results['faithfulness']:.4f}"
    )

    print(
        f"Answerability       : "
        f"{results['answerability']:.4f}"
    )

    print("-" * 70)

    print(
        f"Overall RAG Score   : "
        f"{results['overall_rag_score']:.4f}"
    )

    print("=" * 70)
    print("RAG EVALUATION TEST COMPLETE")
    print("=" * 70)