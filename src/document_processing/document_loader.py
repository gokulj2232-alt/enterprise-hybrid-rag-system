from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xlsx", ".pdf"}


def load_document(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension == ".txt":
        text = file_path.read_text(encoding="utf-8")

    elif extension == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)

    elif extension == ".xlsx":
        import pandas as pd
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)

    elif extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return {
        "document_id": file_path.stem,
        "file_name": file_path.name,
        "file_type": extension,
        "text": text.strip()
    }