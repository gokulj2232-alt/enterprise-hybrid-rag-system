import pandas as pd
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "NLP_50K_Clean.xlsx"

OUTPUT_FILE = PROJECT_ROOT / "data" / "NLP_50K_Preprocessed.xlsx"


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    """
    Clean and normalize text for NLP processing.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("STARTING DATA PREPROCESSING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_excel(INPUT_FILE)

print(f"Loaded records: {len(df):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

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


# ============================================================
# CLEAN INDIVIDUAL COLUMNS
# ============================================================

print("\nCleaning text columns...")

df["category"] = df["category"].apply(clean_text)

df["title"] = df["title"].apply(clean_text)

df["content"] = df["content"].apply(clean_text)

df["keyword"] = df["keyword"].apply(clean_text)


# ============================================================
# CREATE SEARCH TEXT
# ============================================================

print("Creating combined search text...")

df["search_text"] = (
    "Category: "
    + df["category"]
    + "\nTitle: "
    + df["title"]
    + "\nContent: "
    + df["content"]
    + "\nKeywords: "
    + df["keyword"]
)


# ============================================================
# REMOVE EMPTY DOCUMENTS
# ============================================================

before_count = len(df)

df = df[
    df["content"].str.strip().ne("")
]

after_count = len(df)

print(
    f"Removed empty documents: "
    f"{before_count - after_count}"
)


# ============================================================
# REMOVE EXACT DUPLICATE DOCUMENTS
# ============================================================

before_count = len(df)

df = df.drop_duplicates(
    subset=["search_text"]
)

after_count = len(df)

print(
    f"Removed duplicate documents: "
    f"{before_count - after_count}"
)


# ============================================================
# RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# CREATE DATA DIRECTORY
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE PREPROCESSED DATASET
# ============================================================

print("\nSaving preprocessed dataset...")

df.to_excel(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Final records : {len(df):,}")

print(
    f"Columns       : "
    f"{list(df.columns)}"
)

print(
    f"Output file   : "
    f"{OUTPUT_FILE}"
)

print()
print("Sample search text:")
print("-" * 60)
print(df["search_text"].iloc[0][:500])

print("=" * 60)
