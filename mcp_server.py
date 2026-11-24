from typing import Any

from mcp.server.mcpserver import MCPServer

from llm_client import answer_from_ontology
from storage import list_documents as store_list_documents
from storage import load_ontology

mcp = MCPServer("ontology-layer")


@mcp.tool()
def list_documents() -> list[dict[str, Any]]:
    """List all saved ontology documents with entity and relationship counts."""
    return store_list_documents()


@mcp.tool()
def get_ontology(doc_id: str) -> dict[str, Any]:
    """Return the complete ontology JSON for the given doc_id."""
    return load_ontology(doc_id)


@mcp.tool()
def ask_ontology(doc_id: str, question: str) -> str:
    """Answer a question strictly from the saved ontology JSON for doc_id."""
    ontology = load_ontology(doc_id)
    return answer_from_ontology(ontology=ontology, question=question)


if __name__ == "__main__":
    mcp.run()
