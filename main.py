import os
import re
import pickle
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/NLP_50K_Document_Dataset.xlsx"
MODEL_DIR = "models"

WORD2VEC_PATH = os.path.join(MODEL_DIR, "word2vec.model")
DOCUMENT_VECTORS_PATH = os.path.join(MODEL_DIR, "document_vectors.npy")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
PROCESSED_DATA_PATH = os.path.join(MODEL_DIR, "processed_data.pkl")


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD / CHECK NLTK RESOURCES
# ============================================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


stop_words = set(stopwords.words("english"))


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_excel(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# STEP 2: STANDARDIZE COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: STANDARDIZING COLUMNS")
print("=" * 70)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

required_columns = [
    "document_id",
    "category",
    "title",
    "content",
    "keyword"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns found.")
print(list(df.columns))


# ============================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: HANDLING MISSING VALUES")
print("=" * 70)

for column in required_columns:
    df[column] = df[column].fillna("").astype(str)

print("Missing values handled.")


# ============================================================
# STEP 4: COMBINE TEXT
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: COMBINING TEXT")
print("=" * 70)

df["combined_text"] = (
    df["title"] + " " +
    df["content"] + " " +
    df["keyword"]
)

print("Combined text created.")


# ============================================================
# STEP 5: TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Keep letters and numbers
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# STEP 6: CLEAN TEXT
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: CLEANING TEXT")
print("=" * 70)

df["clean_text"] = df["combined_text"].apply(clean_text)

print("Text cleaning completed.")


# ============================================================
# STEP 7: TOKENIZATION + STOPWORD REMOVAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: TOKENIZATION AND STOPWORD REMOVAL")
print("=" * 70)


def preprocess_tokens(text):

    tokens = word_tokenize(text)

    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    return tokens


df["tokens"] = df["clean_text"].apply(preprocess_tokens)

df["clean_tokens"] = df["tokens"]

df["token_text"] = df["clean_tokens"].apply(
    lambda tokens: " ".join(tokens)
)

print("Tokenization completed.")
print("Stopword removal completed.")


# ============================================================
# SHOW SAMPLE
# ============================================================

print("\nSample processed document:")
print(df["clean_tokens"].iloc[0][:30])


# ============================================================
# STEP 8: TRAIN WORD2VEC
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: TRAINING WORD2VEC")
print("=" * 70)

sentences = df["clean_tokens"].tolist()

word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=10,
    seed=42
)

print("Word2Vec training completed.")
print(f"Vocabulary size: {len(word2vec_model.wv)}")
print(f"Vector size: {word2vec_model.wv.vector_size}")


# ============================================================
# STEP 9: TRAIN TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: TRAINING TF-IDF")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    df["token_text"]
)

print("TF-IDF training completed.")
print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# ============================================================
# STEP 10: CREATE TF-IDF WEIGHTED DOCUMENT VECTORS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: CREATING DOCUMENT VECTORS")
print("=" * 70)


vocabulary = word2vec_model.wv.key_to_index
idf_values = dict(
    zip(
        tfidf_vectorizer.get_feature_names_out(),
        tfidf_vectorizer.idf_
    )
)


def create_document_vector(tokens):

    vectors = []
    weights = []

    for word in tokens:

        if word in vocabulary:

            vector = word2vec_model.wv[word]

            weight = idf_values.get(word, 1.0)

            vectors.append(vector * weight)
            weights.append(weight)

    if not vectors:
        return np.zeros(
            word2vec_model.wv.vector_size,
            dtype=np.float32
        )

    vectors = np.array(vectors)

    weights = np.array(weights)

    document_vector = np.sum(vectors, axis=0) / np.sum(weights)

    return document_vector.astype(np.float32)


document_vectors = np.vstack(
    [
        create_document_vector(tokens)
        for tokens in df["clean_tokens"]
    ]
)


print("Document vectors created.")
print(f"Document vector shape: {document_vectors.shape}")


# ============================================================
# STEP 11: VALIDATE DOCUMENT VECTORS
# ============================================================

print("\n" + "=" * 70)
print("STEP 10: VALIDATING DOCUMENT VECTORS")
print("=" * 70)

print(f"Rows in dataset: {len(df)}")
print(f"Document vectors: {document_vectors.shape}")

nan_count = np.isnan(document_vectors).sum()
inf_count = np.isinf(document_vectors).sum()

print(f"NaN count: {nan_count}")
print(f"Inf count: {inf_count}")

if len(df) != document_vectors.shape[0]:
    raise ValueError(
        "Dataset row count and document vector count do not match."
    )

if document_vectors.shape[1] != 100:
    raise ValueError(
        "Document vector dimension is not 100."
    )

if nan_count != 0:
    raise ValueError(
        "NaN values found in document vectors."
    )

if inf_count != 0:
    raise ValueError(
        "Infinite values found in document vectors."
    )

print("Document vector validation PASSED.")


# ============================================================
# STEP 12: SAVE TRAINED FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 11: SAVING TRAINED FILES")
print("=" * 70)

word2vec_model.save(WORD2VEC_PATH)

np.save(
    DOCUMENT_VECTORS_PATH,
    document_vectors
)

with open(TFIDF_PATH, "wb") as file:
    pickle.dump(
        tfidf_vectorizer,
        file
    )

df.to_pickle(
    PROCESSED_DATA_PATH
)

print("All trained files saved successfully.")

print(f"\nSaved:")
print(WORD2VEC_PATH)
print(DOCUMENT_VECTORS_PATH)
print(TFIDF_PATH)
print(PROCESSED_DATA_PATH)


# ============================================================
# STEP 13: CHECK SAVED FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 12: CHECKING SAVED FILES")
print("=" * 70)

saved_files = [
    WORD2VEC_PATH,
    DOCUMENT_VECTORS_PATH,
    TFIDF_PATH,
    PROCESSED_DATA_PATH
]

for file_path in saved_files:

    if os.path.exists(file_path):

        size_mb = os.path.getsize(file_path) / (
            1024 * 1024
        )

        print(
            f"FOUND: {file_path} "
            f"({size_mb:.2f} MB)"
        )

    else:

        raise FileNotFoundError(
            f"File was not created: {file_path}"
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAINING VALIDATION")
print("=" * 70)

print(f"Dataset rows: {len(df)}")
print(f"Document vectors: {document_vectors.shape}")
print(f"Word2Vec vocabulary: {len(word2vec_model.wv)}")
print(
    f"TF-IDF vocabulary: "
    f"{len(tfidf_vectorizer.vocabulary_)}"
)
print(f"NaN count: {np.isnan(document_vectors).sum()}")
print(f"Inf count: {np.isinf(document_vectors).sum()}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nYour models folder now contains:")
print("1. word2vec.model")
print("2. document_vectors.npy")
print("3. tfidf_vectorizer.pkl")
print("4. processed_data.pkl")

print("\nYou can now run:")
print("streamlit run app.py")