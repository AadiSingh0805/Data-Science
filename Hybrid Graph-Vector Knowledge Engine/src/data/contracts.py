"""Data contracts and schemas for GraphRAG Knowledge Engine.
Enforces strict typing, ontology enumeration, and citation auditability.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class EntityType(str, Enum):
    """Permitted entity node types in the knowledge graph ontology."""
    COMPANY = "COMPANY"
    PRODUCT = "PRODUCT"
    FACILITY = "FACILITY"
    RISK_EVENT = "RISK_EVENT"
    TECHNOLOGY = "TECHNOLOGY"
    GEO_LOCATION = "GEO_LOCATION"


class RelationType(str, Enum):
    """Permitted relationship edge verbs in the knowledge graph ontology.
    Strictly enumerated to avoid predicate explosion and graph fragmentation.
    """
    SUPPLIES_TO = "SUPPLIES_TO"
    MANUFACTURES_FOR = "MANUFACTURES_FOR"
    DEPENDS_ON = "DEPENDS_ON"
    EXPOSED_TO_RISK = "EXPOSED_TO_RISK"
    LOCATED_IN = "LOCATED_IN"
    INCORPORATES = "INCORPORATES"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    COMPETES_WITH = "COMPETES_WITH"


class Entity(BaseModel):
    """Node contract in the knowledge graph."""
    id: str = Field(..., description="Normalized unique entity ID, e.g., 'company:nvidia'")
    name: str = Field(..., description="Canonical entity display name, e.g., 'NVIDIA Corporation'")
    entity_type: EntityType = Field(..., description="Ontological category of the entity")
    description: Optional[str] = Field(default="", description="Contextual description of the entity")

    @classmethod
    def create_id(cls, entity_type: EntityType, name: str) -> str:
        """Helper to create standardized entity IDs."""
        clean_name = name.lower().strip()
        for char in [".", ",", "(", ")", "[", "]", "'", '"', "/"]:
            clean_name = clean_name.replace(char, "")
        clean_name = clean_name.replace(" ", "_")
        return f"{entity_type.value.lower()}:{clean_name}"


class Relationship(BaseModel):
    """Edge contract in the knowledge graph.
    Every relationship MUST be grounded by a verbatim quote to prevent hallucinations.
    """
    source_id: str = Field(..., description="Source entity ID (subject)")
    target_id: str = Field(..., description="Target entity ID (object)")
    relation_type: RelationType = Field(..., description="Strictly typed relationship predicate")
    verbatim_quote: str = Field(
        ...,
        min_length=10,
        description="Verbatim sentence from the source document grounding this factual edge"
    )
    source_doc_id: str = Field(..., description="Source document identifier (e.g. 'NVDA_2024_10K.txt')")
    chunk_id: int = Field(default=0, ge=0, description="Zero-indexed chunk position within source document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")

    @model_validator(mode="after")
    def validate_no_self_loops(self):
        if self.source_id == self.target_id:
            raise ValueError(f"Self-referential relationships are disallowed: {self.source_id}")
        return self


class KnowledgeGraphExtraction(BaseModel):
    """Contract for structured LLM triplet extraction outputs."""
    entities: List[Entity] = Field(default_factory=list, description="Extracted unique entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Extracted grounded relationships")

    @model_validator(mode="after")
    def validate_referential_integrity(self):
        """Ensures that all edge endpoints exist within the entities list."""
        entity_ids = {e.id for e in self.entities}
        for rel in self.relationships:
            if rel.source_id not in entity_ids:
                raise ValueError(f"Relationship source_id '{rel.source_id}' missing from entities.")
            if rel.target_id not in entity_ids:
                raise ValueError(f"Relationship target_id '{rel.target_id}' missing from entities.")
        return self


class DocumentChunk(BaseModel):
    """Contract for text chunks prepared for hybrid indexing (Vector + Graph)."""
    chunk_id: int = Field(..., ge=0)
    doc_id: str = Field(...)
    content: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)

