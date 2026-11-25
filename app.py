import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from extractor import extract_text, text_to_ontology
from graph_view import TYPE_COLORS, build_graph_html
from storage import list_documents, load_ontology, sanitize_doc_id, save_ontology

SUPPORTED_EXTENSIONS = ["pdf", "docx", "xlsx", "csv", "txt"]

st.set_page_config(page_title="Ontology Layer AI Agent", layout="wide")
st.title("Ontology Layer AI Agent")
st.caption("Business files in → agent-ready ontology out, served through MCP.")

ONTOLOGIES_DIR = Path(__file__).parent / "ontologies"
ONTOLOGIES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = ("Person", "Org", "Concept", "Document", "Other")


def _entities_to_df(entities: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "accept": True,
            "id": e.get("id", ""),
            "type": e.get("type", "Other"),
            "label": e.get("label", ""),
            "attrs": json.dumps(e.get("attrs", {}), ensure_ascii=False),
        }
        for e in entities
    ]
    if not rows:
        return pd.DataFrame(columns=["accept", "id", "type", "label", "attrs"])
    return pd.DataFrame(rows)


def _rels_to_df(relationships: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "accept": True,
            "source": r.get("source", ""),
            "target": r.get("target", ""),
            "relation": r.get("relation", ""),
        }
        for r in relationships
    ]
    if not rows:
        return pd.DataFrame(columns=["accept", "source", "target", "relation"])
    return pd.DataFrame(rows)


def _parse_edited(
    doc_id: str,
    entities_df: pd.DataFrame,
    rels_df: pd.DataFrame,
) -> tuple[dict[str, Any], int]:
    entities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, row in entities_df.iterrows():
        if not bool(row.get("accept", True)):
            continue
        eid = str(row.get("id", "")).strip()
        label = str(row.get("label", "")).strip()
        etype = str(row.get("type", "Other")).strip()
        raw_attrs = row.get("attrs", "{}")

        if not eid or not label:
            raise ValueError(f"Row {idx + 1}: 'id' and 'label' cannot be empty.")
        if eid in seen_ids:
            raise ValueError(f"Duplicate entity id '{eid}'.")
        if etype not in ALLOWED_TYPES:
            raise ValueError(f"Entity '{eid}' has invalid type '{etype}'.")
        try:
            attrs_text = "{}" if pd.isna(raw_attrs) or str(raw_attrs).strip() == "" else str(raw_attrs)
            attrs = json.loads(attrs_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Entity '{eid}' has invalid JSON in attrs: {exc}") from exc
        if not isinstance(attrs, dict):
            raise ValueError(f"Entity '{eid}' attrs must be a JSON object.")

        seen_ids.add(eid)
        entities.append({"id": eid, "type": etype, "label": label, "attrs": attrs})

    relationships: list[dict[str, Any]] = []
    pruned = 0
    for idx, row in rels_df.iterrows():
        if not bool(row.get("accept", True)):
            continue
        src = str(row.get("source", "")).strip()
        tgt = str(row.get("target", "")).strip()
        rel = str(row.get("relation", "")).strip()
        if not src or not tgt or not rel:
            raise ValueError(f"Relationship row {idx + 1}: all fields required.")
        if src not in seen_ids or tgt not in seen_ids:
            pruned += 1
            continue
        relationships.append({"source": src, "target": tgt, "relation": rel})

    ontology = {"doc_id": doc_id, "entities": entities, "relationships": relationships}
    return ontology, pruned


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Supported: PDF, DOCX, XLSX, CSV, TXT",
        type=SUPPORTED_EXTENSIONS,
    )
    custom_doc_id = st.text_input(
        "Document ID (optional)",
        value=sanitize_doc_id(uploaded_file.name) if uploaded_file else "",
    )

    if st.button("Extract Ontology", type="primary", use_container_width=True, disabled=uploaded_file is None):
        if uploaded_file is not None:
            doc_id = sanitize_doc_id(custom_doc_id.strip() or uploaded_file.name)
            try:
                with st.spinner("Extracting text and building ontology..."):
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)
                    try:
                        raw_text = extract_text(tmp_path)
                        raw_ontology = text_to_ontology(raw_text)
                    finally:
                        tmp_path.unlink(missing_ok=True)

                raw_ontology["doc_id"] = doc_id
                for entity in raw_ontology.get("entities", []):
                    entity.setdefault("attrs", {})

                save_ontology(raw_ontology)
                st.session_state["selected_doc_id"] = doc_id
                st.success(f"Saved '{doc_id}' ({len(raw_ontology['entities'])} entities).")
                st.rerun()
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")

    st.divider()
    st.header("2. Saved Ontologies")
    docs = list_documents()
    valid_doc_ids = [d["doc_id"] for d in docs if "error" not in d]
    for d in docs:
        if "error" in d:
            st.warning(f"{d['doc_id']}: {d['error']}")

    if valid_doc_ids:
        current = st.session_state.get("selected_doc_id")
        default_idx = valid_doc_ids.index(current) if current in valid_doc_ids else 0
        selected_doc = st.selectbox("Select document", options=valid_doc_ids, index=default_idx)
        st.session_state["selected_doc_id"] = selected_doc
    else:
        selected_doc = None
        st.info("No ontologies saved yet. Upload a file to begin.")


# ── Main area ────────────────────────────────────────────────────────────────
if selected_doc:
    try:
        ontology = load_ontology(selected_doc)
    except Exception as exc:
        st.error(f"Failed to load ontology '{selected_doc}': {exc}")
        st.stop()

    col_left, col_right = st.columns([1.15, 1.0], gap="large")

    with col_left:
        st.subheader(f"Review & Edit: `{selected_doc}`")
        st.caption("Uncheck accept or delete rows to remove items, edit cells inline, then Save Changes.")

        st.markdown("**Entities**")
        edited_entities_df = st.data_editor(
            _entities_to_df(ontology["entities"]),
            key=f"entities_{selected_doc}",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "accept": st.column_config.CheckboxColumn("Accept", default=True),
                "id": st.column_config.TextColumn("ID", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=list(ALLOWED_TYPES), required=True),
                "label": st.column_config.TextColumn("Label", required=True),
                "attrs": st.column_config.TextColumn("Attrs (JSON)"),
            },
        )

        st.markdown("**Relationships**")
        edited_rels_df = st.data_editor(
            _rels_to_df(ontology["relationships"]),
            key=f"rels_{selected_doc}",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "accept": st.column_config.CheckboxColumn("Accept", default=True),
                "source": st.column_config.TextColumn("Source ID", required=True),
                "target": st.column_config.TextColumn("Target ID", required=True),
                "relation": st.column_config.TextColumn("Relation", required=True),
            },
        )

        if st.button("Save Changes", type="primary"):
            try:
                updated, pruned = _parse_edited(selected_doc, edited_entities_df, edited_rels_df)
                save_ontology(updated)
                msg = f"Saved '{selected_doc}' ({len(updated['entities'])} entities, {len(updated['relationships'])} relationships)."
                if pruned:
                    msg += f" Removed {pruned} dangling relationship(s)."
                st.success(msg)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save: {exc}")

    with col_right:
        st.subheader("Knowledge Graph")
        legend = " | ".join(
            f"<span style='color:{c};font-weight:600'>{t}</span>"
            for t, c in TYPE_COLORS.items()
        )
        st.markdown(legend, unsafe_allow_html=True)

        if ontology["entities"]:
            components.html(build_graph_html(ontology), height=540, scrolling=True)
        else:
            st.info("No entities to display.")

        with st.expander("Raw Ontology JSON"):
            st.json(ontology)
else:
    st.info("Upload a PDF, Word, Excel, CSV, or TXT file from the sidebar to generate your first ontology.")
