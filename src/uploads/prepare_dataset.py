import json
import os
from src.uploads.dataset_loader import load_dataset, dataset_to_documents
def prepare_uploaded_dataset(file_path, output_dir="data/uploads"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset(file_path)
    documents, text_columns = dataset_to_documents(df)
    output_file = os.path.join(
        output_dir,
        "uploaded_documents.jsonl"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for document in documents:
            f.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                )
                + "\n"
            )
    return {
        "rows": len(df),
        "documents": len(documents),
        "text_columns": text_columns,
        "output_file": output_file
    }
