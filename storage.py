import json
import re
from pathlib import Path
from typing import Any

from models import Ontology

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGIES_DIR = BASE_DIR / "ontologies"


def ensure_ontologies_dir() -> Path:
    ONTOLOGIES_DIR.mkdir(parents=True, exist_ok=True)
    return ONTOLOGIES_DIR


def sanitize_doc_id(raw_name: str) -> str:
    stem = Path(raw_name).stem.strip()
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_")
    return cleaned or "document"


def ontology_path(doc_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", doc_id):
        raise ValueError(f"Invalid doc_id '{doc_id}'. Use letters, numbers, '-' or '_'.")
    return ensure_ontologies_dir() / f"{doc_id}.json"


def list_documents() -> list[dict[str, Any]]:
    ensure_ontologies_dir()
    docs: list[dict[str, Any]] = []
    for file_path in sorted(ONTOLOGIES_DIR.glob("*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            ontology = Ontology.model_validate(data)
            docs.append(
                {
                    "doc_id": ontology.doc_id,
                    "entity_count": len(ontology.entities),
                    "relationship_count": len(ontology.relationships),
                }
            )
        except Exception as exc:
            docs.append(
                {
                    "doc_id": file_path.stem,
                    "error": f"Invalid ontology JSON: {exc}",
                }
            )
    return docs


def load_ontology(doc_id: str) -> dict[str, Any]:
    path = ontology_path(doc_id)
    if not path.exists():
        raise FileNotFoundError(f"Ontology '{doc_id}' not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    ontology = Ontology.model_validate(data)
    return ontology.model_dump()


def save_ontology(data: dict[str, Any]) -> Path:
    ontology = Ontology.model_validate(data)
    entity_ids = {entity.id for entity in ontology.entities}
    filtered_rels = [
        rel for rel in ontology.relationships if rel.source in entity_ids and rel.target in entity_ids
    ]
    clean = Ontology(
        doc_id=ontology.doc_id,
        entities=ontology.entities,
        relationships=filtered_rels,
    )
    path = ontology_path(clean.doc_id)
    path.write_text(json.dumps(clean.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
