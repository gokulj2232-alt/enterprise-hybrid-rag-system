import re
import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================
# CONFIGURATION
# =============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FALLBACK_ANSWER = (
    "I don't know based on the provided documents."
)


# =============================================================
# ANSWERABILITY CHECKER
# =============================================================

class AnswerabilityChecker:

    def __init__(
        self,
        semantic_threshold=0.55,
        lexical_threshold=0.15
    ):

        print(
            f"Loading answerability model: "
            f"{EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.semantic_threshold = (
            semantic_threshold
        )

        self.lexical_threshold = (
            lexical_threshold
        )

        print(
            "Answerability Checker initialized successfully!"
        )

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    def normalize_text(
        self,
        text
    ):

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

    def tokenize(
        self,
        text
    ):

        text = self.normalize_text(
            text
        )

        return set(
            word
            for word in text.split()
            if len(word) > 2
        )

    # =========================================================
    # SENTENCE SPLITTING
    # =========================================================

    def split_sentences(
        self,
        text
    ):

        sentences = re.split(
            r"(?<=[.!?])\s+",
            str(text).strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # =========================================================
    # QUESTION TYPE
    # =========================================================

    def detect_question_type(
        self,
        query
    ):

        normalized = self.normalize_text(
            query
        )

        words = normalized.split()

        if not words:

            return "unknown"

        # -----------------------------------------------------
        # WHO
        # -----------------------------------------------------

        if words[0] == "who":

            return "person"

        # -----------------------------------------------------
        # WHICH
        # -----------------------------------------------------

        if words[0] == "which":

            if any(
                word in normalized
                for word in [
                    "company",
                    "companies",
                    "organization",
                    "organizations",
                    "provider",
                    "providers"
                ]
            ):

                return "entity"

            return "selection"

        # -----------------------------------------------------
        # WHEN
        # -----------------------------------------------------

        if words[0] == "when":

            return "date"

        # -----------------------------------------------------
        # WHERE
        # -----------------------------------------------------

        if words[0] == "where":

            return "location"

        # -----------------------------------------------------
        # HOW MANY / HOW MUCH
        # -----------------------------------------------------

        if (
            len(words) >= 2
            and words[0] == "how"
            and words[1] in [
                "many",
                "much"
            ]
        ):

            return "number"

        # -----------------------------------------------------
        # HOW LONG
        # -----------------------------------------------------

        if (
            len(words) >= 2
            and words[0] == "how"
            and words[1] == "long"
        ):

            return "duration"

        # -----------------------------------------------------
        # HOW
        # -----------------------------------------------------

        if words[0] == "how":

            return "process"

        # -----------------------------------------------------
        # WHY
        # -----------------------------------------------------

        if words[0] == "why":

            return "reason"

        # -----------------------------------------------------
        # WHAT
        # -----------------------------------------------------

        if words[0] == "what":

            if any(
                phrase in normalized
                for phrase in [
                    "what is",
                    "what are",
                    "what does",
                    "what do",
                    "what means"
                ]
            ):

                return "definition"

            return "information"

        # -----------------------------------------------------
        # YES / NO FACT QUESTIONS
        # -----------------------------------------------------

        if words[0] in [
            "is",
            "are",
            "does",
            "do",
            "can",
            "could",
            "will"
        ]:

            return "fact"

        return "unknown"

    # =========================================================
    # LEXICAL OVERLAP
    # =========================================================

    def calculate_lexical_overlap(
        self,
        query,
        evidence
    ):

        query_tokens = self.tokenize(
            query
        )

        evidence_tokens = self.tokenize(
            evidence
        )

        if not query_tokens:

            return 0.0

        if not evidence_tokens:

            return 0.0

        common_tokens = (
            query_tokens &
            evidence_tokens
        )

        return (
            len(common_tokens)
            / len(query_tokens)
        )

    # =========================================================
    # SEMANTIC SIMILARITY
    # =========================================================

    def calculate_semantic_similarity(
        self,
        query,
        evidence_sentences
    ):

        if not evidence_sentences:

            return 0.0

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        evidence_embeddings = (
            self.model.encode(
                evidence_sentences,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        )

        similarities = np.matmul(
            query_embedding,
            evidence_embeddings.T
        )[0]

        return float(
            np.max(similarities)
        )

    # =========================================================
    # ENTITY / COMPANY DETECTION
    # =========================================================

    def contains_named_entity(
        self,
        evidence
    ):

        text = str(evidence)

        organization_patterns = [

            r"\b[A-Z][A-Za-z0-9&.-]+"
            r"\s+(Inc|Ltd|LLC|Corporation|Corp)\b",

            r"\b[A-Z][A-Za-z0-9&.-]+"
            r"\s+(Technologies|Technology|Systems|Solutions)\b",

            r"\b[A-Z][A-Za-z0-9&.-]+"
            r"\s+(Company|Companies)\b"
        ]

        for pattern in organization_patterns:

            if re.search(
                pattern,
                text
            ):

                return True

        return False

    # =========================================================
    # DATE DETECTION
    # =========================================================

    def contains_date(
        self,
        evidence
    ):

        patterns = [

            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",

            r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",

            r"\b\d{4}\b",

            r"\b"
            r"(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
            r"\s+\d{1,2}"
            r"\b"
        ]

        for pattern in patterns:

            if re.search(
                pattern,
                evidence,
                flags=re.IGNORECASE
            ):

                return True

        return False

    # =========================================================
    # NUMBER DETECTION
    # =========================================================

    def contains_number(
        self,
        evidence
    ):

        return bool(
            re.search(
                r"\b\d+(\.\d+)?\b",
                str(evidence)
            )
        )

    # =========================================================
    # LOCATION DETECTION
    # =========================================================

    def contains_location(
        self,
        evidence
    ):

        location_words = [

            "located",
            "location",
            "city",
            "country",
            "state",
            "district",
            "street",
            "address",
            "headquarters"
        ]

        normalized = (
            self.normalize_text(
                evidence
            )
        )

        return any(
            word in normalized
            for word in location_words
        )

    # =========================================================
    # REQUIRED EVIDENCE CHECK
    # =========================================================

    def check_required_evidence(
        self,
        question_type,
        evidence
    ):

        if question_type == "entity":

            return self.contains_named_entity(
                evidence
            )

        if question_type == "person":

            return bool(
                re.search(
                    r"\b[A-Z][a-z]+"
                    r"\s+[A-Z][a-z]+\b",
                    str(evidence)
                )
            )

        if question_type == "date":

            return self.contains_date(
                evidence
            )

        if question_type == "number":

            return self.contains_number(
                evidence
            )

        if question_type == "location":

            return self.contains_location(
                evidence
            )

        return True

    # =========================================================
    # MAIN CHECK
    # =========================================================

    def check(
        self,
        query,
        evidence_context,
        uploaded_dataset=False
    ):

        # -----------------------------------------------------
        # EMPTY QUERY
        # -----------------------------------------------------

        if not query or not query.strip():

            return {

                "is_answerable": False,

                "semantic_score": 0.0,

                "lexical_score": 0.0,

                "question_type": "unknown",

                "required_evidence_found": False,

                "reason": "Empty query"
            }

        # -----------------------------------------------------
        # NO EVIDENCE
        # -----------------------------------------------------

        if (
            not evidence_context
            or not evidence_context.strip()
        ):

            return {

                "is_answerable": False,

                "semantic_score": 0.0,

                "lexical_score": 0.0,

                "question_type":
                    self.detect_question_type(
                        query
                    ),

                "required_evidence_found": False,

                "reason":
                    "No evidence available"
            }

        # -----------------------------------------------------
        # SPLIT EVIDENCE
        # -----------------------------------------------------

        evidence_sentences = (
            self.split_sentences(
                evidence_context
            )
        )

        if not evidence_sentences:

            return {

                "is_answerable": False,

                "semantic_score": 0.0,

                "lexical_score": 0.0,

                "question_type":
                    self.detect_question_type(
                        query
                    ),

                "required_evidence_found": False,

                "reason":
                    "No evidence sentences found"
            }

        # -----------------------------------------------------
        # QUESTION TYPE
        # -----------------------------------------------------

        question_type = (
            self.detect_question_type(
                query
            )
        )

        # -----------------------------------------------------
        # SEMANTIC SCORE
        # -----------------------------------------------------

        semantic_score = (
            self.calculate_semantic_similarity(
                query,
                evidence_sentences
            )
        )

        # -----------------------------------------------------
        # LEXICAL SCORE
        # -----------------------------------------------------

        lexical_score = (
            self.calculate_lexical_overlap(
                query,
                evidence_context
            )
        )

        # -----------------------------------------------------
        # REQUIRED EVIDENCE
        # -----------------------------------------------------

        required_evidence_found = (
            self.check_required_evidence(
                question_type,
                evidence_context
            )
        )

        # =====================================================
        # NORMAL DATASET THRESHOLDS
        # =====================================================

        if not uploaded_dataset:

            semantic_threshold = (
                self.semantic_threshold
            )

            lexical_threshold = (
                self.lexical_threshold
            )

            semantic_pass = (
                semantic_score
                >= semantic_threshold
            )

            lexical_pass = (
                lexical_score
                >= lexical_threshold
            )

            is_answerable = (
                semantic_pass
                and lexical_pass
                and required_evidence_found
            )

        # =====================================================
        # UPLOADED DATASET
        # =====================================================

        else:

            semantic_threshold = 0.40

            lexical_threshold = 0.10

            semantic_pass = (
                semantic_score
                >= semantic_threshold
            )

            lexical_pass = (
                lexical_score
                >= lexical_threshold
            )

            # -------------------------------------------------
            # SPECIAL RULE FOR SELECTION QUESTIONS
            # -------------------------------------------------
            #
            # Uploaded datasets are often structured records.
            #
            # Example:
            #
            # product_name: Laptop Pro 15
            # category: Electronics
            # description: High performance laptop
            #              for programming
            #
            # A selection query may have strong lexical evidence
            # while the sentence-level semantic score is slightly
            # below the generic threshold.
            #
            # Therefore:
            #
            # selection +
            # strong lexical overlap +
            # required evidence
            #
            # is sufficient.
            # -------------------------------------------------

            strong_selection_evidence = (

                question_type == "selection"

                and lexical_score >= 0.50

                and required_evidence_found
            )

            if strong_selection_evidence:

                is_answerable = True

                semantic_pass = True

            else:

                is_answerable = (
                    semantic_pass
                    and lexical_pass
                    and required_evidence_found
                )

        # =====================================================
        # REASON
        # =====================================================

        if is_answerable:

            if (
                uploaded_dataset
                and question_type == "selection"
                and lexical_score >= 0.50
            ):

                reason = (
                    "Strong lexical evidence and required "
                    "selection evidence found in the "
                    "uploaded dataset."
                )

            else:

                reason = (
                    "Evidence appears sufficient "
                    "for answering the query."
                )

        elif not semantic_pass:

            reason = (
                "Evidence is not semantically "
                "relevant enough."
            )

        elif not lexical_pass:

            reason = (
                "Evidence has insufficient "
                "lexical overlap."
            )

        elif not required_evidence_found:

            reason = (
                "Evidence does not contain the "
                "required information type."
            )

        else:

            reason = (
                "Evidence is insufficient."
            )

        # =====================================================
        # RETURN RESULT
        # =====================================================

        return {

            "is_answerable":
                is_answerable,

            "semantic_score":
                semantic_score,

            "lexical_score":
                lexical_score,

            "semantic_pass":
                semantic_pass,

            "lexical_pass":
                lexical_pass,

            "question_type":
                question_type,

            "required_evidence_found":
                required_evidence_found,

            "reason":
                reason
        }


# =============================================================
# MODULE TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "ADVANCED ANSWERABILITY CHECKER TEST"
    )

    print("=" * 60)

    checker = AnswerabilityChecker()

    query = (
        "What resources does cloud computing provide?"
    )

    evidence = (
        "Cloud computing provides computing "
        "resources over the internet."
    )

    result = checker.check(
        query,
        evidence
    )

    print(
        f"\nQuery: {query}"
    )

    print(
        f"Question Type: "
        f"{result['question_type']}"
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
        f"Required Evidence: "
        f"{result['required_evidence_found']}"
    )

    print(
        f"Answerable: "
        f"{result['is_answerable']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )

