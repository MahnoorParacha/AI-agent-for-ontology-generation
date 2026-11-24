import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

from models import ALLOWED_ENTITY_TYPES, MAX_ENTITIES

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"


def get_model_name() -> str:
    load_dotenv(override=True)
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_client() -> genai.Client:
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError("Missing GEMINI_API_KEY. Set it in your .env file.")
    return genai.Client(api_key=api_key)


def _strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _generate_with_retry(
    client: genai.Client,
    prompt: str,
    config: types.GenerateContentConfig,
    attempts: int = 3,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.models.generate_content(
                model=get_model_name(),
                contents=prompt,
                config=config,
            )
        except APIError as exc:
            last_error = exc
            is_transient = exc.code in (429, 500, 503, 504)
            if not is_transient or attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise last_error  # pragma: no cover


def extract_ontology_json(doc_id: str, text: str) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Document contains no readable text to extract.")

    client = get_client()
    allowed_types = "|".join(ALLOWED_ENTITY_TYPES)
    prompt = f"""Extract a formal business ontology from the document text below.

Rules:
- Return ONLY valid JSON. Do not include markdown fences or commentary.
- Extract at most {MAX_ENTITIES} of the most important entities.
- Every entity "type" MUST be one of: {allowed_types}
- Use short unique entity IDs like "e1", "e2", ...
- Only create relationships whose "source" and "target" match entity IDs in "entities".
- Use the exact JSON shape:
{{
  "doc_id": "{doc_id}",
  "entities": [
    {{"id": "e1", "type": "Org", "label": "Example", "attrs": {{}}}}
  ],
  "relationships": [
    {{"source": "e1", "target": "e2", "relation": "owns"}}
  ]
}}

Document text:
{text[:30000]}
"""

    response = _generate_with_retry(
        client=client,
        prompt=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    raw = response.text or ""
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Extracted ontology must be a JSON object.")

    parsed["doc_id"] = doc_id
    entities = parsed.get("entities", [])
    if isinstance(entities, list) and len(entities) > MAX_ENTITIES:
        entities = entities[:MAX_ENTITIES]
        parsed["entities"] = entities

    valid_ids = {e.get("id") for e in parsed.get("entities", []) if isinstance(e, dict)}
    parsed["relationships"] = [
        r
        for r in parsed.get("relationships", [])
        if isinstance(r, dict) and r.get("source") in valid_ids and r.get("target") in valid_ids
    ]
    return parsed


def answer_from_ontology(ontology: dict[str, Any], question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    client = get_client()
    ontology_json = json.dumps(ontology, indent=2, ensure_ascii=False)
    prompt = f"""You are an ontology query assistant.
Answer the question using ONLY the facts explicitly present in the ontology JSON below.
Rules:
- Never guess, infer external facts, or hallucinate.
- If the ontology JSON does not contain enough information to answer, reply: "Not found in this ontology."
- Cite the relevant entity labels or relationships from the JSON in your answer.

Ontology JSON:
{ontology_json}

Question: {question}
"""

    response = _generate_with_retry(
        client=client,
        prompt=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return (response.text or "").strip() or "Not found in this ontology."
