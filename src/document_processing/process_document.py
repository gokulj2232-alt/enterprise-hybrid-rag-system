
from pathlib import Path
import json

from src.document_processing.document_loader import load_document
from src.document_processing.preprocessing.clean_text import clean_text
from src.document_processing.chunker import chunk_text


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# UPLOADED DOCUMENT STORAGE
# ============================================================

UPLOADED_DATA_DIR = PROJECT_ROOT / "data" / "uploads"

UPLOADED_DOCUMENTS_FILE = (
    UPLOADED_DATA_DIR / "uploaded_documents.jsonl"
)

UPLOADED_INDEX_DIR = (
    UPLOADED_DATA_DIR / "index"
)


# ============================================================
# REMOVE OLD UPLOADED INDEX
# ============================================================

def remove_old_uploaded_index():
    """
    Remove only the uploaded-document index.

    IMPORTANT:
    This function NEVER touches the main enterprise index:

        data/faiss_index.bin
    """

    files_to_remove = [
        UPLOADED_INDEX_DIR / "faiss_index.bin",
        UPLOADED_INDEX_DIR / "metadata.json",
        UPLOADED_INDEX_DIR / "bm25.json",
    ]

    for file_path in files_to_remove:

        if not file_path.exists():
            continue

        try:
            file_path.unlink()

            print(
                f"Removed old uploaded index: "
                f"{file_path.name}"
            )

        except OSError as error:

            raise RuntimeError(
                f"Could not remove old uploaded index: "
                f"{file_path}"
            ) from error


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(file_path):
    """
    Process one uploaded document.

    Pipeline:

        Load
          ↓
        Clean
          ↓
        Chunk
          ↓
        Replace uploaded document store
          ↓
        Remove previous uploaded index
          ↓
        Build new uploaded FAISS/BM25 index

    The main enterprise index is never modified.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Uploaded file does not exist: {file_path}"
        )

    print(
        f"\nProcessing uploaded document: "
        f"{file_path.name}"
    )

    # ========================================================
    # 1. LOAD DOCUMENT
    # ========================================================

    document = load_document(file_path)

    if not document:
        raise ValueError(
            "Document loader returned no document."
        )

    if "text" not in document:
        raise ValueError(
            "Loaded document does not contain 'text'."
        )

    # ========================================================
    # 2. CLEAN TEXT
    # ========================================================

    text = clean_text(document["text"])

    if not text:
        raise ValueError(
            "Document contains no usable text."
        )

    # ========================================================
    # 3. CHUNK DOCUMENT
    # ========================================================

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    print(
        f"Created {len(chunks)} chunk(s)."
    )

    # ========================================================
    # 4. CREATE DOCUMENT RECORDS
    # ========================================================

    records = []

    for index, chunk in enumerate(chunks):

        records.append(
            {
                "document_id": (
                    f"{document['document_id']}"
                    f"_chunk_{index}"
                ),
                "content": chunk,
                "source_row": index + 1,
            }
        )

    # ========================================================
    # 5. CREATE DIRECTORIES
    # ========================================================

    UPLOADED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    UPLOADED_INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 6. REPLACE CURRENT UPLOADED DOCUMENT STORE
    # ========================================================

    print(
        "\nReplacing previous uploaded document store..."
    )

    temporary_documents_file = (
        UPLOADED_DOCUMENTS_FILE.with_suffix(
            ".jsonl.tmp"
        )
    )

    try:

        with open(
            temporary_documents_file,
            "w",
            encoding="utf-8"
        ) as file:

            for record in records:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        # Replace the old document store only after
        # successfully writing the new one.

        temporary_documents_file.replace(
            UPLOADED_DOCUMENTS_FILE
        )

    except Exception:

        if temporary_documents_file.exists():
            temporary_documents_file.unlink()

        raise

    print(
        f"Uploaded document store contains "
        f"{len(records)} chunk(s)."
    )

    # ========================================================
    # 7. REMOVE OLD UPLOADED INDEX
    # ========================================================

    print(
        "\nRemoving previous uploaded index..."
    )

    remove_old_uploaded_index()

    # ========================================================
    # 8. BUILD NEW UPLOADED INDEX
    # ========================================================

    from src.uploads.build_uploaded_index import (
        build_uploaded_index
    )

    print(
        "\nBuilding new uploaded-document index..."
    )

    index_result = build_uploaded_index(
        documents_file=str(
            UPLOADED_DOCUMENTS_FILE
        ),
        output_dir=str(
            UPLOADED_INDEX_DIR
        )
    )

    # ========================================================
    # 9. VALIDATE INDEX RESULT
    # ========================================================

    if index_result["documents"] != len(records):

        raise RuntimeError(
            "Uploaded index validation failed: "
            f"expected {len(records)} documents, "
            f"but indexed "
            f"{index_result['documents']}."
        )

    if index_result["dimension"] != 384:

        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{index_result['dimension']}. "
            "Expected 384 for all-MiniLM-L6-v2."
        )

    # ========================================================
    # 10. RESULT
    # ========================================================

    print(
        "\nUploaded document index rebuilt successfully."
    )

    print(
        f"Uploaded documents indexed: "
        f"{index_result['documents']}"
    )

    print(
        f"Embedding dimension: "
        f"{index_result['dimension']}"
    )

    return {
        "document_id": document["document_id"],
        "file_name": document["file_name"],
        "chunks": len(chunks),
        "uploaded_index_documents": (
            index_result["documents"]
        ),
        "uploaded_index_dimension": (
            index_result["dimension"]
        ),
        "faiss_index": index_result["faiss_index"],
        "metadata": index_result["metadata"],
        "bm25": index_result["bm25"],
    }


### 2. `src/uploads/build_uploaded_index.py`


from pathlib import Path
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EXPECTED_DIMENSION = 384


# ============================================================
# ATOMIC FILE REPLACEMENT
# ============================================================

def atomic_replace(source, destination):
    """
    Replace destination with source.

    os.replace() performs the replacement atomically
    on the same filesystem.
    """

    os.replace(
        source,
        destination
    )


# ============================================================
# BUILD UPLOADED INDEX
# ============================================================

def build_uploaded_index(
    documents_file="data/uploads/uploaded_documents.jsonl",
    output_dir="data/uploads/index",
):
    """
    Build a completely fresh FAISS + metadata + BM25
    index for the CURRENT uploaded document only.

    IMPORTANT:

    This function only writes inside:

        data/uploads/index/

    It NEVER modifies:

        data/faiss_index.bin
        data/metadata.json
        data/bm25.json
    """

    documents_file = Path(documents_file)
    output_dir = Path(output_dir)

    # ========================================================
    # 1. VALIDATE DOCUMENT FILE
    # ========================================================

    if not documents_file.exists():

        raise FileNotFoundError(
            f"Uploaded document file not found: "
            f"{documents_file}"
        )

    # ========================================================
    # 2. CREATE OUTPUT DIRECTORY
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 3. OUTPUT FILES
    # ========================================================

    faiss_path = (
        output_dir / "faiss_index.bin"
    )

    metadata_path = (
        output_dir / "metadata.json"
    )

    bm25_path = (
        output_dir / "bm25.json"
    )

    # Temporary files

    temp_faiss = (
        output_dir / "faiss_index.bin.tmp"
    )

    temp_metadata = (
        output_dir / "metadata.json.tmp"
    )

    temp_bm25 = (
        output_dir / "bm25.json.tmp"
    )

    # ========================================================
    # 4. LOAD DOCUMENTS
    # ========================================================

    print(
        f"\nLoading uploaded documents from:"
        f"\n{documents_file}"
    )

    documents = []

    with open(
        documents_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                document = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} in "
                    f"{documents_file}"
                ) from error

            if "document_id" not in document:
                raise ValueError(
                    f"Document on line "
                    f"{line_number} is missing "
                    f"'document_id'."
                )

            if "content" not in document:
                raise ValueError(
                    f"Document on line "
                    f"{line_number} is missing "
                    f"'content'."
                )

            if not document["content"].strip():
                raise ValueError(
                    f"Document on line "
                    f"{line_number} has empty "
                    f"'content'."
                )

            documents.append(document)

    if not documents:

        raise ValueError(
            "No uploaded documents found."
        )

    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    # ========================================================
    # 5. LOAD EMBEDDING MODEL
    # ========================================================

    print(
        f"\nLoading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # ========================================================
    # 6. CREATE EMBEDDINGS
    # ========================================================

    texts = [
        document["content"]
        for document in documents
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} chunk(s)..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # ========================================================
    # 7. VALIDATE EMBEDDINGS
    # ========================================================

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Embedding output must be a 2D array."
        )

    if embeddings.shape[0] != len(documents):

        raise RuntimeError(
            "Embedding/document count mismatch: "
            f"{embeddings.shape[0]} embeddings "
            f"for {len(documents)} documents."
        )

    dimension = embeddings.shape[1]

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    if dimension != EXPECTED_DIMENSION:

        raise RuntimeError(
            f"Unexpected embedding dimension: "
            f"{dimension}. "
            f"Expected {EXPECTED_DIMENSION}."
        )

    # ========================================================
    # 8. BUILD FAISS INDEX
    # ========================================================

    print(
        "\nBuilding uploaded FAISS index..."
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    # ========================================================
    # 9. VALIDATE FAISS INDEX
    # ========================================================

    if index.ntotal != len(documents):

        raise RuntimeError(
            "FAISS/document count mismatch: "
            f"{index.ntotal} vectors for "
            f"{len(documents)} documents."
        )

    # ========================================================
    # 10. CREATE BM25 TOKEN DATA
    # ========================================================

    print(
        "Building uploaded BM25 token data..."
    )

    bm25_documents = []

    for document in documents:

        tokens = (
            document["content"]
            .lower()
            .split()
        )

        bm25_documents.append(
            tokens
        )

    if len(bm25_documents) != len(documents):

        raise RuntimeError(
            "BM25/document count mismatch."
        )

    # ========================================================
    # 11. WRITE TEMPORARY FAISS FILE
    # ========================================================

    print(
        "\nWriting temporary FAISS index..."
    )

    # Remove stale temporary files first.

    for temporary_file in (
        temp_faiss,
        temp_metadata,
        temp_bm25,
    ):

        if temporary_file.exists():

            temporary_file.unlink()

    faiss.write_index(
        index,
        str(temp_faiss)
    )

    # ========================================================
    # 12. WRITE TEMPORARY METADATA
    # ========================================================

    print(
        "Writing temporary metadata..."
    )

    with open(
        temp_metadata,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # 13. WRITE TEMPORARY BM25 DATA
    # ========================================================

    print(
        "Writing temporary BM25 data..."
    )

    with open(
        temp_bm25,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            bm25_documents,
            file,
            ensure_ascii=False
        )

    # ========================================================
    # 14. REPLACE FINAL FILES
    # ========================================================

    print(
        "\nReplacing uploaded index files..."
    )

    try:

        atomic_replace(
            temp_faiss,
            faiss_path
        )

        atomic_replace(
            temp_metadata,
            metadata_path
        )

        atomic_replace(
            temp_bm25,
            bm25_path
        )

    except Exception:

        # Clean up remaining temporary files.

        for temporary_file in (
            temp_faiss,
            temp_metadata,
            temp_bm25,
        ):

            if temporary_file.exists():

                try:
                    temporary_file.unlink()
                except OSError:
                    pass

        raise

    # ========================================================
    # 15. FINAL VALIDATION
    # ========================================================

    if not faiss_path.exists():
        raise RuntimeError(
            "FAISS index was not created."
        )

    if not metadata_path.exists():
        raise RuntimeError(
            "Metadata file was not created."
        )

    if not bm25_path.exists():
        raise RuntimeError(
            "BM25 file was not created."
        )

    # Read the written FAISS index again to make
    # sure the saved file is valid.

    saved_index = faiss.read_index(
        str(faiss_path)
    )

    if saved_index.ntotal != len(documents):

        raise RuntimeError(
            "Saved FAISS index validation failed: "
            f"{saved_index.ntotal} vectors for "
            f"{len(documents)} documents."
        )

    if saved_index.d != EXPECTED_DIMENSION:

        raise RuntimeError(
            "Saved FAISS dimension validation failed: "
            f"{saved_index.d}."
        )

    # ========================================================
    # 16. RESULT
    # ========================================================

    print(
        "\nUploaded index ready."
    )

    print(
        f"Vectors: {saved_index.ntotal}"
    )

    print(
        f"Dimension: {saved_index.d}"
    )

    print(
        f"FAISS: {faiss_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )

    print(
        f"BM25: {bm25_path}"
    )

    return {
        "documents": len(documents),
        "new_documents": len(documents),
        "dimension": saved_index.d,
        "faiss_index": str(faiss_path),
        "metadata": str(metadata_path),
        "bm25": str(bm25_path),
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    result = build_uploaded_index()

    print(
        "\nUploaded index build completed."
    )

    print(
        result
    )
