import pandas as pd
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
TEXT_COLUMN_HINTS = [
    "content",
    "text",
    "description",
    "document",
    "article",
    "body",
    "title",
    "question",
    "answer",
    "summary",
    "keyword",
    "name",
    "category"
]
def load_dataset(file_path):
    file_path = str(file_path).lower()
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please upload CSV or Excel."
        )
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    return df
def detect_text_columns(df):
    """
    Detect columns that are useful for creating searchable documents.
    """
    text_columns = []
    for column in df.columns:
        column_name = str(column).strip().lower()
        # Strong semantic column-name match
        if any(
            hint == column_name or hint in column_name
            for hint in TEXT_COLUMN_HINTS
        ):
            text_columns.append(column)
            continue
        # Otherwise inspect object/string columns
        if pd.api.types.is_string_dtype(df[column]):
            non_empty = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
            )
            if len(non_empty) == 0:
                continue
            # Require some meaningful textual content
            non_empty = non_empty[non_empty != ""]
            if len(non_empty) == 0:
                continue
            unique_ratio = non_empty.nunique() / len(non_empty)
            avg_length = non_empty.str.len().mean()
            if avg_length >= 5 and unique_ratio >= 0.01:
                text_columns.append(column)
    # Remove duplicates while preserving order
    return list(dict.fromkeys(text_columns))
def dataset_to_documents(df):
    text_columns = detect_text_columns(df)
    if not text_columns:
        raise ValueError(
            "No suitable text columns were detected. "
            "The dataset should contain textual information."
        )
    documents = []
    for index, row in df.iterrows():
        parts = []
        for column in text_columns:
            value = row[column]
            if pd.notna(value):
                value = str(value).strip()
                if value:
                    parts.append(
                        f"{column}: {value}"
                    )
        if parts:
            documents.append(
                {
                    "document_id": f"UPLOAD_{index + 1:06d}",
                    "content": " | ".join(parts),
                    "source_row": index + 1
                }
            )
    if not documents:
        raise ValueError(
            "No usable documents could be created."
        )
    return documents, text_columns
