import re
import numpy as np

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FALLBACK_ANSWER = (
    "I don't know based on the provided documents."
)


class AnswerabilityChecker:
    """
    Conservative answerability checker.

    Checks:
    1. Semantic similarity
    2. Lexical overlap
    3. Question intent
    4. Required evidence type

    The goal is to prevent the LLM from answering questions
    when the retrieved documents do not contain the required
    type of information.
    """

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

        text = self.normalize_text(text)

        return set(
            word
            for word in text.split()
            if len(word) > 2
        )

    # =========================================================
    # SENTENCE SPLITTING
    # =========================================================

    def split_sentences(self, text):

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

    def detect_question_type(self, query):

        normalized = (
            self.normalize_text(query)
        )

        words = normalized.split()

        if not words:
            return "unknown"

        # WHO
        if words[0] == "who":
            return "person"

        # WHICH
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

        # WHEN
        if words[0] in [
            "when"
        ]:
            return "date"

        # WHERE
        if words[0] in [
            "where"
        ]:
            return "location"

        # HOW MANY / HOW MUCH
        if (
            len(words) >= 2
            and words[0] == "how"
            and words[1] in [
                "many",
                "much"
            ]
        ):
            return "number"

        # HOW LONG
        if (
            len(words) >= 2
            and words[0] == "how"
            and words[1] == "long"
        ):
            return "duration"

        # HOW
        if words[0] == "how":
            return "process"

        # WHY
        if words[0] == "why":
            return "reason"

        # WHAT
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

        # YES / NO STYLE
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

        # Common organization/company indicators
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

            # Look for simple person-name pattern.
            # This is intentionally conservative.
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

        # Definition / process / information questions
        # can be supported by semantic + lexical evidence.
        return True

    # =========================================================
    # MAIN CHECK
    # =========================================================

    def check(
        self,
        query,
        evidence_context
    ):

        # -----------------------------------------------------
        # Empty query
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
        # Empty evidence
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
                "reason": "No evidence available"
            }

        # -----------------------------------------------------
        # Evidence sentences
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
        # Question type
        # -----------------------------------------------------

        question_type = (
            self.detect_question_type(
                query
            )
        )

        # -----------------------------------------------------
        # Scores
        # -----------------------------------------------------

        semantic_score = (
            self.calculate_semantic_similarity(
                query,
                evidence_sentences
            )
        )

        lexical_score = (
            self.calculate_lexical_overlap(
                query,
                evidence_context
            )
        )

        # -----------------------------------------------------
        # Basic relevance
        # -----------------------------------------------------

        semantic_pass = (
            semantic_score
            >= self.semantic_threshold
        )

        lexical_pass = (
            lexical_score
            >= self.lexical_threshold
        )

        # -----------------------------------------------------
        # Required evidence
        # -----------------------------------------------------

        required_evidence_found = (
            self.check_required_evidence(
                question_type,
                evidence_context
            )
        )

        # -----------------------------------------------------
        # Final decision
        # -----------------------------------------------------

        is_answerable = (
            semantic_pass
            and lexical_pass
            and required_evidence_found
        )

        # -----------------------------------------------------
        # Reason
        # -----------------------------------------------------

        if is_answerable:

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
# STANDALONE TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ADVANCED ANSWERABILITY CHECKER TEST")
    print("=" * 60)

    checker = AnswerabilityChecker(
        semantic_threshold=0.55,
        lexical_threshold=0.15
    )

    # =========================================================
    # TEST 1
    # =========================================================

    print("\n")
    print("=" * 60)
    print("TEST 1 - SUPPORTED QUERY")
    print("=" * 60)

    query_1 = (
        "What resources does cloud computing provide?"
    )

    evidence_1 = (
        "Cloud computing provides computing "
        "resources over the internet."
    )

    result_1 = checker.check(
        query_1,
        evidence_1
    )

    print(
        f"\nQuery: {query_1}"
    )

    print(
        f"Question Type: "
        f"{result_1['question_type']}"
    )

    print(
        f"Semantic Score: "
        f"{result_1['semantic_score']:.4f}"
    )

    print(
        f"Lexical Score : "
        f"{result_1['lexical_score']:.4f}"
    )

    print(
        f"Required Evidence: "
        f"{result_1['required_evidence_found']}"
    )

    print(
        f"Answerable: "
        f"{result_1['is_answerable']}"
    )

    print(
        f"Reason: "
        f"{result_1['reason']}"
    )

    # =========================================================
    # TEST 2
    # =========================================================

    print("\n")
    print("=" * 60)
    print("TEST 2 - UNSUPPORTED COMPANY QUERY")
    print("=" * 60)

    query_2 = (
        "Which companies provide cloud computing services?"
    )

    evidence_2 = (
        "Cloud computing provides computing "
        "resources over the internet."
    )

    result_2 = checker.check(
        query_2,
        evidence_2
    )

    print(
        f"\nQuery: {query_2}"
    )

    print(
        f"Question Type: "
        f"{result_2['question_type']}"
    )

    print(
        f"Semantic Score: "
        f"{result_2['semantic_score']:.4f}"
    )

    print(
        f"Lexical Score : "
        f"{result_2['lexical_score']:.4f}"
    )

    print(
        f"Required Evidence: "
        f"{result_2['required_evidence_found']}"
    )

    print(
        f"Answerable: "
        f"{result_2['is_answerable']}"
    )

    print(
        f"Reason: "
        f"{result_2['reason']}"
    )

    # =========================================================
    # TEST 3
    # =========================================================

    print("\n")
    print("=" * 60)
    print("TEST 3 - NO EVIDENCE")
    print("=" * 60)

    query_3 = (
        "What is machine learning?"
    )

    evidence_3 = ""

    result_3 = checker.check(
        query_3,
        evidence_3
    )

    print(
        f"\nQuery: {query_3}"
    )

    print(
        f"Question Type: "
        f"{result_3['question_type']}"
    )

    print(
        f"Semantic Score: "
        f"{result_3['semantic_score']:.4f}"
    )

    print(
        f"Lexical Score : "
        f"{result_3['lexical_score']:.4f}"
    )

    print(
        f"Required Evidence: "
        f"{result_3['required_evidence_found']}"
    )

    print(
        f"Answerable: "
        f"{result_3['is_answerable']}"
    )

    print(
        f"Reason: "
        f"{result_3['reason']}"
    )