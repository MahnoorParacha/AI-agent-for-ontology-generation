# Ontology Layer AI Agent

Business files in → agent-ready ontology out, served through MCP.

## Overview

This project turns business documents (`PDF`, `DOCX`, `XLSX`, `CSV`, `TXT`) into a structured JSON ontology using Gemini, lets you review/edit/delete entities and relationships in a Streamlit UI with an interactive `pyvis` knowledge graph, and exposes the saved ontologies to AI agents through an MCP server.

## Capabilities

- **Multi-format ingestion:** `PDF`, `DOCX`, `XLSX`, `CSV`, `TXT`
- **Structured JSON ontology:** Stored locally in `ontologies/<doc_id>.json` using the schema:
  ```json
  {
    "doc_id": "example_doc",
    "entities": [
      {"id": "e1", "type": "Person|Org|Concept|Document|Other", "label": "Acme Corp", "attrs": {}}
    ],
    "relationships": [
      {"source": "e1", "target": "e2", "relation": "employs"}
    ]
  }
  ```
- **Human review & graph UI:** Inline `Accept` / edit / delete tables for entities and relationships, automatic cleanup of dangling relationships, and an interactive directed `pyvis` knowledge graph.
- **MCP server (`mcp_server.py`):** Exposes three tools via the official `mcp` Python SDK (`MCPServer`):
  - `list_documents()` — lists saved `doc_id`s with entity and relationship counts
  - `get_ontology(doc_id)` — returns the exact ontology JSON for a document
  - `ask_ontology(doc_id, question)` — answers questions strictly from the saved ontology JSON

## Project Structure

- `app.py` — Streamlit UI (upload, review/edit tables, save, knowledge graph)
- `extractor.py` — File text extraction (`pymupdf`, `python-docx`, `openpyxl`, `pandas`) and ontology extraction
- `llm_client.py` — Gemini API client (`google-genai`) with retry handling for extraction and `ask_ontology`
- `storage.py` — Local JSON storage and validation under `ontologies/*.json`
- `graph_view.py` — `pyvis` HTML graph builder
- `models.py` — Pydantic models and schema constraints (max 30 entities per extraction)
- `mcp_server.py` — MCP server exposing `list_documents`, `get_ontology`, and `ask_ontology`
- `requirements.txt` — Python dependencies
- `.env.example` — Environment template (`GEMINI_API_KEY`, `GEMINI_MODEL`)

## Setup & Run

1. **Create a virtual environment and install dependencies:**
   ```zsh
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Configure your Gemini API key:**
   ```zsh
   cp .env.example .env
   ```
   Edit `.env` and set `GEMINI_API_KEY` (and `GEMINI_MODEL` if needed).

3. **Run the Streamlit UI:**
   ```zsh
   .venv/bin/streamlit run app.py
   ```

4. **Run the MCP Server (stdio transport):**
   ```zsh
   .venv/bin/python mcp_server.py
   ```

5. **Example MCP Client Configuration:**
   ```json
   {
     "mcpServers": {
       "ontology-layer": {
         "command": "/path/to/Internship_Project/.venv/bin/python",
         "args": ["/path/to/Internship_Project/mcp_server.py"]
       }
     }
   }
   ```
