import html
import json
from typing import Any

from pyvis.network import Network

TYPE_COLORS: dict[str, str] = {
    "Person": "#4F46E5",
    "Org": "#059669",
    "Concept": "#D97706",
    "Document": "#2563EB",
    "Other": "#6B7280",
}


def build_graph_html(ontology: dict[str, Any], height: str = "520px") -> str:
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#111827",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=140)

    entities = ontology.get("entities", [])
    relationships = ontology.get("relationships", [])

    entity_ids: set[str] = set()
    for entity in entities:
        eid = str(entity.get("id", ""))
        if not eid:
            continue
        entity_ids.add(eid)
        etype = str(entity.get("type", "Other"))
        label = str(entity.get("label", eid))
        attrs = entity.get("attrs", {})
        attrs_pretty = json.dumps(attrs, ensure_ascii=False) if attrs else "{}"
        tooltip = f"ID: {html.escape(eid)}<br>Type: {html.escape(etype)}<br>Attrs: {html.escape(attrs_pretty)}"
        net.add_node(
            eid,
            label=label,
            title=tooltip,
            color=TYPE_COLORS.get(etype, TYPE_COLORS["Other"]),
            shape="dot",
            size=20,
        )

    for rel in relationships:
        src = str(rel.get("source", ""))
        tgt = str(rel.get("target", ""))
        relation = str(rel.get("relation", ""))
        if src in entity_ids and tgt in entity_ids:
            net.add_edge(src, tgt, label=relation, title=relation, arrows="to")

    return net.generate_html()
