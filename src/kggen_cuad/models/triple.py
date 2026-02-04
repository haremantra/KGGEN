"""
Triple and entity models for knowledge graph representation.

Based on the KGGen methodology and CUAD legal domain ontology.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """
    Valid entity types for the legal knowledge graph.

    Based on the PRD schema with 8 node types.
    """

    PARTY = "Party"
    IP_ASSET = "IPAsset"
    OBLIGATION = "Obligation"
    RESTRICTION = "Restriction"
    LIABILITY_PROVISION = "LiabilityProvision"
    TEMPORAL = "Temporal"
    JURISDICTION = "Jurisdiction"
    CONTRACT_CLAUSE = "ContractClause"


class PredicateType(str, Enum):
    """
    Valid predicate/relationship types for the knowledge graph.

    Based on the PRD schema with 10 edge types.
    """

    LICENSES_TO = "LICENSES_TO"
    OWNS = "OWNS"
    ASSIGNS = "ASSIGNS"
    HAS_OBLIGATION = "HAS_OBLIGATION"
    SUBJECT_TO_RESTRICTION = "SUBJECT_TO_RESTRICTION"
    HAS_LIABILITY = "HAS_LIABILITY"
    GOVERNED_BY = "GOVERNED_BY"
    CONTAINS_CLAUSE = "CONTAINS_CLAUSE"
    EFFECTIVE_ON = "EFFECTIVE_ON"
    TERMINATES_ON = "TERMINATES_ON"


class Entity(BaseModel):
    """
    Represents an entity (node) in the knowledge graph.

    Entities are extracted from contracts and can be of various types
    defined by the legal domain ontology.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Entity name/label")
    entity_type: EntityType = Field(..., description="Type of entity")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional entity properties"
    )

    # Source information
    contract_id: UUID | None = Field(default=None, description="Source contract ID")
    source_text: str | None = Field(
        default=None, description="Original text span from contract"
    )
    source_page: int | None = Field(default=None, description="Page number in source")

    # Extraction metadata
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )

    # Resolution (from Stage 3)
    is_canonical: bool = Field(
        default=False, description="Whether this is a canonical entity"
    )
    canonical_id: UUID | None = Field(
        default=None, description="ID of canonical entity if this is an alias"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Known aliases for this entity"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        use_enum_values = True

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Normalize entity name by stripping whitespace."""
        return v.strip() if isinstance(v, str) else v

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j node properties."""
        props = {
            "id": str(self.id),
            "name": self.name,
            "entity_type": self.entity_type,
            "confidence_score": self.confidence_score,
            "is_canonical": self.is_canonical,
        }
        if self.contract_id:
            props["contract_id"] = str(self.contract_id)
        if self.source_text:
            props["source_text"] = self.source_text
        if self.source_page:
            props["source_page"] = self.source_page
        props.update(self.properties)
        return props


class Predicate(BaseModel):
    """
    Represents a predicate/relationship type with properties.
    """

    type: PredicateType = Field(..., description="Relationship type")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Relationship properties"
    )

    # Resolution
    is_canonical: bool = Field(default=False)
    canonical_form: str | None = Field(default=None)

    class Config:
        use_enum_values = True


class Triple(BaseModel):
    """
    Represents a knowledge graph triple (subject-predicate-object).

    This is the fundamental unit of the knowledge graph, representing
    a relationship between two entities.
    """

    id: UUID = Field(default_factory=uuid4)

    # Subject
    subject: Entity = Field(..., description="Subject entity")
    subject_text: str = Field(..., description="Subject text representation")

    # Predicate
    predicate: PredicateType = Field(..., description="Relationship type")
    predicate_properties: dict[str, Any] = Field(
        default_factory=dict, description="Relationship properties"
    )

    # Object
    object: Entity = Field(..., description="Object entity")
    object_text: str = Field(..., description="Object text representation")

    # Source information
    contract_id: UUID | None = Field(default=None, description="Source contract ID")
    cuad_label: str | None = Field(
        default=None, description="Associated CUAD label category"
    )
    source_text: str | None = Field(
        default=None, description="Original text from contract"
    )
    source_page: int | None = Field(default=None, description="Page number in source")

    # Extraction metadata
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    llm_model: str | None = Field(default=None, description="LLM used for extraction")

    # Resolution (from Stage 3)
    is_canonical: bool = Field(
        default=False, description="Whether this uses canonical forms"
    )
    canonical_id: UUID | None = Field(
        default=None, description="ID of canonical triple if this is a variant"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        use_enum_values = True

    @classmethod
    def from_texts(
        cls,
        subject_text: str,
        subject_type: EntityType,
        predicate: PredicateType,
        object_text: str,
        object_type: EntityType,
        contract_id: UUID | None = None,
        **kwargs: Any,
    ) -> "Triple":
        """
        Create a triple from text representations.

        Convenience method for creating triples when full Entity objects
        are not yet available.
        """
        subject = Entity(name=subject_text, entity_type=subject_type)
        obj = Entity(name=object_text, entity_type=object_type)
        return cls(
            subject=subject,
            subject_text=subject_text,
            predicate=predicate,
            object=obj,
            object_text=object_text,
            contract_id=contract_id,
            **kwargs,
        )

    def to_tuple(self) -> tuple[str, str, str]:
        """Return as (subject, predicate, object) tuple."""
        return (self.subject_text, self.predicate, self.object_text)

    def to_neo4j_relationship(self) -> dict[str, Any]:
        """Convert to Neo4j relationship format."""
        return {
            "id": str(self.id),
            "type": self.predicate,
            "properties": {
                "confidence_score": self.confidence_score,
                "contract_id": str(self.contract_id) if self.contract_id else None,
                "cuad_label": self.cuad_label,
                "is_canonical": self.is_canonical,
                **self.predicate_properties,
            },
        }

    def to_embedding_text(self) -> str:
        """
        Generate text representation for embedding.

        Combines subject, predicate, and object into a natural language
        representation suitable for semantic embedding.
        """
        predicate_text = self.predicate.replace("_", " ").lower()
        return f"{self.subject_text} {predicate_text} {self.object_text}"
