import io
from pathlib import Path
from typing import Any

import docx
import pandas as pd
import pymupdf

from llm_client import extract_ontology_json
from models import Ontology

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: PDF, DOCX, XLSX, CSV, TXT.")

    if ext == ".pdf":
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            pages = [page.get_text("text") for page in pdf]
        return "\n".join(pages).strip()

    if ext == ".docx":
        document = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))
        return "\n".join(paragraphs).strip()

    if ext == ".xlsx":
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        parts: list[str] = []
        for sheet_name, df in sheets.items():
            parts.append(f"Sheet: {sheet_name}")
            parts.append(df.fillna("").to_csv(index=False))
        return "\n".join(parts).strip()

    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.fillna("").to_csv(index=False).strip()

    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1").strip()


def extract_ontology(doc_id: str, text: str) -> dict[str, Any]:
    raw_ontology = extract_ontology_json(doc_id=doc_id, text=text)
    validated = Ontology.model_validate(raw_ontology)
    return validated.model_dump()
