import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    "document_id",
    "category",
    "title",
    "content",
    "keyword"
]


def load_documents(file_path: str) -> pd.DataFrame:
    """
    Load the NLP document dataset from an Excel file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Read Excel dataset
    df = pd.read_excel(file_path)

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Fill missing values
    for column in REQUIRED_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    # Remove empty documents
    df = df[df["content"].str.strip() != ""]

    # Remove duplicate document IDs
    df = df.drop_duplicates(
        subset=["document_id"],
        keep="first"
    )

    # Reset index
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":

    dataset_path = "data/NLP_50K_Document_Dataset.xlsx"

    documents = load_documents(dataset_path)

    print("\nDataset loaded successfully!")
    print(f"Total documents: {len(documents)}")

    print("\nColumns:")
    print(documents.columns.tolist())

    print("\nFirst document:")
    print(documents.iloc[0])