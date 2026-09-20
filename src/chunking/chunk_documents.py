import json
import re
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "NLP_50K_Preprocessed.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"

# --------------------------------------------------
# CHUNK SETTINGS
# --------------------------------------------------

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --------------------------------------------------
# TEXT CHUNKING FUNCTION
# --------------------------------------------------

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    text = re.sub(r"\s+", " ", str(text)).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------

print("=" * 60)
print("STARTING DOCUMENT CHUNKING")
print("=" * 60)

print("Loading preprocessed dataset...")

df = pd.read_excel(INPUT_FILE)

print(f"Documents loaded: {len(df):,}")

required_columns = [
    "document_id",
    "category",
    "title",
    "content",
    "keyword",
    "search_text"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# --------------------------------------------------
# CREATE CHUNKS
# --------------------------------------------------

print("Creating document chunks...")

total_chunks = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for _, row in df.iterrows():

        text = row["search_text"]

        document_chunks = create_chunks(text)

        for chunk_index, chunk_text in enumerate(document_chunks):

            chunk_record = {
                "chunk_id": f"{row['document_id']}_CHUNK_{chunk_index:03d}",
                "document_id": row["document_id"],
                "category": row["category"],
                "title": row["title"],
                "keyword": row["keyword"],
                "chunk_index": chunk_index,
                "text": chunk_text
            }

            file.write(
                json.dumps(
                    chunk_record,
                    ensure_ascii=False
                ) + "\n"
            )

            total_chunks += 1

        if total_chunks % 10000 < len(document_chunks):
            print(f"Created chunks: {total_chunks:,}")

# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print()
print("=" * 60)
print("DOCUMENT CHUNKING COMPLETE")
print("=" * 60)

print(f"Documents processed : {len(df):,}")
print(f"Total chunks        : {total_chunks:,}")
print(f"Chunk size          : {CHUNK_SIZE} characters")
print(f"Chunk overlap       : {CHUNK_OVERLAP} characters")
print(f"Output file         : {OUTPUT_FILE}")

print("=" * 60)
print("STAGE 3 CHUNKING COMPLETE")
print("=" * 60)