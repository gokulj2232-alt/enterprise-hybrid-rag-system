import re
import numpy as np

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FALLBACK_ANSWER = (
    "I don't know based on the provided documents."
)


class GroundingValidator:
    """
    Grounding validator for the Enterprise Hybrid RAG system.

    Validates generated answers against retrieved evidence.

    Checks:
    1. Sentence-level semantic similarity
    2. Sentence-level lexical overlap
    3. Evidence-wide semantic similarity
    4. Evidence-wide lexical support
    5. Claim-level grounding
    """

    def __init__(
        self,
        semantic_threshold=0.60,
        lexical_threshold=0.10
    ):

        print(
            f"Loading grounding model: "
            f"{EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.semantic_threshold = semantic_threshold
        self.lexical_threshold = lexical_threshold

        print(
            "Grounding Validator initialized successfully!"
        )

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    def normalize_text(self, text):

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =========================================================
    # TOKENIZE
    # =========================================================

    def tokenize(self, text):

        normalized = self.normalize_text(
            text
        )

        return set(
            word
            for word in normalized.split()
            if len(word) > 2
        )

    # =========================================================
    # SPLIT SENTENCES
    # =========================================================

    def split_sentences(self, text):

        text = str(text).strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # =========================================================
    # LEXICAL OVERLAP
    # =========================================================

    def lexical_overlap(
        self,
        answer,
        evidence
    ):

        answer_tokens = self.tokenize(
            answer
        )

        evidence_tokens = self.tokenize(
            evidence
        )

        if not answer_tokens:
            return 0.0

        if not evidence_tokens:
            return 0.0

        common_tokens = (
            answer_tokens
            & evidence_tokens
        )

        return (
            len(common_tokens)
            / len(answer_tokens)
        )

    # =========================================================
    # SEMANTIC SIMILARITY
    # =========================================================

    def semantic_similarity(
        self,
        answer,
        evidence_sentences
    ):

        if not answer:
            return 0.0

        if not evidence_sentences:
            return 0.0

        answer_embedding = self.model.encode(
            [answer],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        evidence_embeddings = self.model.encode(
            evidence_sentences,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        similarities = np.matmul(
            answer_embedding,
            evidence_embeddings.T
        )[0]

        return float(
            np.max(similarities)
        )

    # =========================================================
    # EVIDENCE-WIDE SEMANTIC
    # =========================================================

    def evidence_wide_semantic_similarity(
        self,
        answer,
        evidence
    ):

        if not answer or not evidence:
            return 0.0

        answer_embedding = self.model.encode(
            [answer],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        evidence_embedding = self.model.encode(
            [evidence],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        similarity = np.matmul(
            answer_embedding,
            evidence_embedding.T
        )[0][0]

        return float(similarity)

    # =========================================================
    # EVIDENCE-WIDE LEXICAL
    # =========================================================

    def evidence_wide_lexical_overlap(
        self,
        answer,
        evidence
    ):

        return self.lexical_overlap(
            answer,
            evidence
        )

    # =========================================================
    # VALIDATE ONE SENTENCE
    # =========================================================

    def validate_sentence(
        self,
        sentence,
        evidence_sentences,
        full_evidence
    ):

        # -----------------------------------------------------
        # Individual semantic similarity
        # -----------------------------------------------------

        individual_semantic = (
            self.semantic_similarity(
                sentence,
                evidence_sentences
            )
        )

        # -----------------------------------------------------
        # Individual lexical similarity
        # -----------------------------------------------------

        individual_lexical = 0.0

        for evidence_sentence in evidence_sentences:

            score = self.lexical_overlap(
                sentence,
                evidence_sentence
            )

            individual_lexical = max(
                individual_lexical,
                score
            )

        # -----------------------------------------------------
        # Evidence-wide semantic similarity
        # -----------------------------------------------------

        evidence_wide_semantic = (
            self.evidence_wide_semantic_similarity(
                sentence,
                full_evidence
            )
        )

        # -----------------------------------------------------
        # Evidence-wide lexical similarity
        # -----------------------------------------------------

        evidence_wide_lexical = (
            self.evidence_wide_lexical_overlap(
                sentence,
                full_evidence
            )
        )

        # -----------------------------------------------------
        # Semantic support
        # -----------------------------------------------------

        semantic_supported = (
            individual_semantic
            >= self.semantic_threshold
            or
            evidence_wide_semantic
            >= self.semantic_threshold
        )

        # -----------------------------------------------------
        # Lexical support
        # -----------------------------------------------------

        lexical_supported = (
            individual_lexical
            >= self.lexical_threshold
            or
            evidence_wide_lexical
            >= self.lexical_threshold
        )

        # -----------------------------------------------------
        # Combined grounding
        # -----------------------------------------------------
        #
        # We allow strong semantic evidence OR
        # strong lexical evidence when one signal
        # is slightly weaker.
        #
        # This is useful when the generated answer
        # paraphrases the source document.
        # -----------------------------------------------------

        grounded = (
            (
                semantic_supported
                and lexical_supported
            )
            or
            (
                individual_semantic >= 0.72
                and individual_lexical >= 0.05
            )
            or
            (
                evidence_wide_semantic >= 0.72
                and evidence_wide_lexical >= 0.05
            )
        )

        return {

            "sentence":
                sentence,

            "semantic_score":
                individual_semantic,

            "lexical_score":
                individual_lexical,

            "evidence_wide_semantic_score":
                evidence_wide_semantic,

            "evidence_wide_lexical_score":
                evidence_wide_lexical,

            "is_grounded":
                grounded
        }

    # =========================================================
    # VALIDATE COMPLETE ANSWER
    # =========================================================

    def validate(
        self,
        answer,
        evidence_context
    ):

        # -----------------------------------------------------
        # Empty answer
        # -----------------------------------------------------

        if (
            not answer
            or not answer.strip()
        ):

            return {

                "answer":
                    FALLBACK_ANSWER,

                "semantic_score":
                    0.0,

                "lexical_score":
                    0.0,

                "is_grounded":
                    False,

                "sentence_results":
                    []
            }

        # -----------------------------------------------------
        # Empty evidence
        # -----------------------------------------------------

        if (
            not evidence_context
            or not evidence_context.strip()
        ):

            return {

                "answer":
                    FALLBACK_ANSWER,

                "semantic_score":
                    0.0,

                "lexical_score":
                    0.0,

                "is_grounded":
                    False,

                "sentence_results":
                    []
            }

        # -----------------------------------------------------
        # Split answer
        # -----------------------------------------------------

        answer_sentences = (
            self.split_sentences(
                answer
            )
        )

        # -----------------------------------------------------
        # Split evidence
        # -----------------------------------------------------

        evidence_sentences = (
            self.split_sentences(
                evidence_context
            )
        )

        if not answer_sentences:

            return {

                "answer":
                    FALLBACK_ANSWER,

                "semantic_score":
                    0.0,

                "lexical_score":
                    0.0,

                "is_grounded":
                    False,

                "sentence_results":
                    []
            }

        # =====================================================
        # VALIDATE SENTENCES
        # =====================================================

        sentence_results = []

        for sentence in answer_sentences:

            result = self.validate_sentence(
                sentence,
                evidence_sentences,
                evidence_context
            )

            sentence_results.append(
                result
            )

        # =====================================================
        # GROUNDED SENTENCES
        # =====================================================

        grounded_sentences = [

            result

            for result in sentence_results

            if result["is_grounded"]

        ]

        # =====================================================
        # FINAL GROUNDING DECISION
        # =====================================================

        is_grounded = (
            len(grounded_sentences)
            == len(sentence_results)
        )

        # =====================================================
        # OVERALL SCORES
        # =====================================================

        semantic_scores = [

            result[
                "evidence_wide_semantic_score"
            ]

            for result in sentence_results

        ]

        lexical_scores = [

            result[
                "evidence_wide_lexical_score"
            ]

            for result in sentence_results

        ]

        overall_semantic_score = (

            float(
                np.mean(
                    semantic_scores
                )
            )

            if semantic_scores

            else 0.0
        )

        overall_lexical_score = (

            float(
                np.mean(
                    lexical_scores
                )
            )

            if lexical_scores

            else 0.0
        )

        # =====================================================
        # FINAL ANSWER
        # =====================================================

        if is_grounded:

            final_answer = answer

        else:

            final_answer = FALLBACK_ANSWER

        # =====================================================
        # RETURN RESULT
        # =====================================================

        return {

            "answer":
                final_answer,

            "semantic_score":
                overall_semantic_score,

            "lexical_score":
                overall_lexical_score,

            "is_grounded":
                is_grounded,

            "sentence_results":
                sentence_results
        }


# =============================================================
# STANDALONE TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GROUNDING VALIDATOR TEST")
    print("=" * 60)

    validator = GroundingValidator()

    answer = (
        "Cloud computing provides resources "
        "over the internet."
    )

    evidence = """
    [Evidence 1]
    Document ID: DOC001
    Title: Cloud Computing
    Evidence: Cloud computing provides
    computing resources over the internet.
    """

    result = validator.validate(
        answer,
        evidence
    )

    print("\nGenerated Answer:")
    print(answer)

    print("\n" + "=" * 60)
    print("GROUNDING DETAILS")
    print("=" * 60)

    for item in result[
        "sentence_results"
    ]:

        print(
            f"\nSentence: "
            f"{item['sentence']}"
        )

        print(
            f"Individual Semantic: "
            f"{item['semantic_score']:.4f}"
        )

        print(
            f"Individual Lexical: "
            f"{item['lexical_score']:.4f}"
        )

        print(
            f"Evidence-Wide Semantic: "
            f"{item['evidence_wide_semantic_score']:.4f}"
        )

        print(
            f"Evidence-Wide Lexical: "
            f"{item['evidence_wide_lexical_score']:.4f}"
        )

        print(
            f"Grounded: "
            f"{item['is_grounded']}"
        )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)

    print(
        f"Is Grounded: "
        f"{result['is_grounded']}"
    )

    print(
        f"Semantic Score: "
        f"{result['semantic_score']:.4f}"
    )

    print(
        f"Lexical Score: "
        f"{result['lexical_score']:.4f}"
    )

    print(
        f"Final Answer: "
        f"{result['answer']}"
    )