import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import docx
import pandas as pd
import pymupdf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}


def extract_text(source: "str | Path | bytes", filename: str = "") -> str:
    """Extract raw text from a file path, Path object, or raw bytes.

    Accepts:
        extract_text(file_path)                   # Mahnoor's CLI usage
        extract_text(file_bytes, filename)         # Seerat's Streamlit UI usage
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return _extract_from_path(path)

    # bytes branch — write to a temp file so pymupdf / docx can open it
    if not filename:
        raise ValueError("filename is required when source is bytes.")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: PDF, DOCX, XLSX, CSV, TXT."
        )
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)
    try:
        return _extract_from_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_from_path(path: Path) -> str:
    """Internal helper: extract text given an existing file Path."""
    ext = path.suffix.lower()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc).strip()

    if ext == ".docx":
        document = docx.Document(path)
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip(" |"):
                    parts.append(row_text)
        return "\n".join(parts).strip()

    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="replace").strip()

    if ext == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        sections = [
            f"Sheet: {name}\n{df.fillna('').to_string(index=False)}"
            for name, df in sheets.items()
        ]
        return "\n\n".join(sections).strip()

    if ext == ".csv":
        df = pd.read_csv(path)
        return df.fillna("").to_string(index=False).strip()

    raise ValueError(
        f"Unsupported file format '{ext}'. Supported: .pdf, .docx, .txt, .xlsx, .csv"
    )


def extract_ontology(doc_id: str, text: str) -> dict[str, Any]:
    """Wrapper used by app.py: extract ontology JSON and add doc_id."""
    result = text_to_ontology(text)
    result["doc_id"] = doc_id
    # Ensure attrs field on every entity (models.py requires it)
    for entity in result.get("entities", []):
        entity.setdefault("attrs", {})
    return result


def text_to_ontology(text: str) -> dict[str, Any]:
    """Use Gemini to extract top 30 entities and relationships as strict JSON."""
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        google_api_key=api_key,
        response_mime_type="application/json",
    )

    prompt = (
        "Extract a formal ontology from the text below.\n"
        "Return ONLY valid JSON with no markdown, no code fences, and no extra commentary.\n"
        "Use this exact schema:\n"
        '{"entities": [{"id": "...", "type": "Person|Org|Concept", "label": "..."}], '
        '"relationships": [{"source": "...", "target": "...", "relation": "..."}]}\n'
        "Rules:\n"
        "- Limit entities to at most the top 30 most important entities.\n"
        '- Every entity "type" must be one of: "Person", "Org", "Concept".\n'
        '- Every entity "id" must be a short unique snake_case identifier.\n'
        '- Every relationship "source" and "target" must reference an entity "id" from "entities".\n\n'
        f"Text:\n{text}"
    )

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        raw = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    else:
        raw = content if isinstance(content, str) else str(content)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Ontology output must be a JSON object.")

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    if not isinstance(entities, list) or not isinstance(relationships, list):
        raise ValueError("Ontology JSON must contain 'entities' and 'relationships' lists.")

    capped_entities = entities[:30]
    valid_ids = {e.get("id") for e in capped_entities if isinstance(e, dict) and e.get("id")}
    filtered_relationships = [
        r
        for r in relationships
        if isinstance(r, dict)
        and r.get("source") in valid_ids
        and r.get("target") in valid_ids
    ]

    return {
        "entities": capped_entities,
        "relationships": filtered_relationships,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <file_path>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    extracted_text = extract_text(input_path)
    ontology = text_to_ontology(extracted_text)

    out_dir = Path("ontologies")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}.json"
    out_path.write_text(json.dumps(ontology, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved ontology to {out_path}")
