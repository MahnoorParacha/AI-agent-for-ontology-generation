from typing import Any, Literal
from pydantic import BaseModel, Field

EntityType = Literal["Person", "Org", "Concept", "Document", "Other"]
ALLOWED_ENTITY_TYPES: tuple[str, ...] = ("Person", "Org", "Concept", "Document", "Other")
MAX_ENTITIES = 30


class Entity(BaseModel):
    id: str = Field(..., min_length=1)
    type: EntityType
    label: str = Field(..., min_length=1)
    attrs: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)


class Ontology(BaseModel):
    doc_id: str = Field(..., min_length=1)
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
