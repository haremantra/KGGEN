"""
Entity and Triple model classes for knowledge graph representation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EntityType(str, Enum):
    """Types of entities extracted from contracts."""
    PARTY = "party"
    IP_ASSET = "ip_asset"
    OBLIGATION = "obligation"
    RESTRICTION = "restriction"
    LIABILITY_PROVISION = "liability_provision"
    TEMPORAL = "temporal"
    JURISDICTION = "jurisdiction"
    CONTRACT_CLAUSE = "contract_clause"


class PredicateType(str, Enum):
    """Types of predicates (relations) between entities."""
    LICENSES_TO = "licenses_to"
    OWNS = "owns"
    ASSIGNS = "assigns"
    HAS_OBLIGATION = "has_obligation"
    SUBJECT_TO_RESTRICTION = "subject_to_restriction"
    HAS_LIABILITY = "has_liability"
    GOVERNED_BY = "governed_by"
    CONTAINS_CLAUSE = "contains_clause"
    EFFECTIVE_ON = "effective_on"
    TERMINATES_ON = "terminates_on"


@dataclass
class Entity:
    """
    An entity extracted from a contract.

    Entities represent key concepts like parties, IP assets,
    obligations, restrictions, etc.
    """
    id: UUID
    name: str
    entity_type: EntityType
    normalized_name: str
    properties: dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    source_contract_id: UUID | None = None
    source_edit_id: UUID | None = None
    confidence: float = 0.8
    aliases: list[str] = field(default_factory=list)
    canonical_id: UUID | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # CUAD label mapping
    cuad_labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.aliases is None:
            self.aliases = []
        if self.cuad_labels is None:
            self.cuad_labels = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "entity_type": self.entity_type.value,
            "normalized_name": self.normalized_name,
            "properties": self.properties,
            "source_text": self.source_text[:500] if self.source_text else "",
            "source_contract_id": str(self.source_contract_id) if self.source_contract_id else None,
            "source_edit_id": str(self.source_edit_id) if self.source_edit_id else None,
            "confidence": self.confidence,
            "aliases": self.aliases,
            "canonical_id": str(self.canonical_id) if self.canonical_id else None,
            "created_at": self.created_at.isoformat(),
            "cuad_labels": self.cuad_labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create from dictionary representation."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            normalized_name=data["normalized_name"],
            properties=data.get("properties", {}),
            source_text=data.get("source_text", ""),
            source_contract_id=UUID(data["source_contract_id"]) if data.get("source_contract_id") else None,
            source_edit_id=UUID(data["source_edit_id"]) if data.get("source_edit_id") else None,
            confidence=data.get("confidence", 0.8),
            aliases=data.get("aliases", []),
            canonical_id=UUID(data["canonical_id"]) if data.get("canonical_id") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            cuad_labels=data.get("cuad_labels", []),
        )


@dataclass
class Triple:
    """
    A triple representing a relation between two entities.

    Triples form the edges of the knowledge graph.
    """
    id: UUID
    subject_id: UUID
    predicate: PredicateType
    object_id: UUID
    confidence: float = 0.8
    properties: dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    source_contract_id: UUID | None = None
    source_edit_id: UUID | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # CUAD label mapping
    cuad_labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.cuad_labels is None:
            self.cuad_labels = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "subject_id": str(self.subject_id),
            "predicate": self.predicate.value,
            "object_id": str(self.object_id),
            "confidence": self.confidence,
            "properties": self.properties,
            "source_text": self.source_text[:500] if self.source_text else "",
            "source_contract_id": str(self.source_contract_id) if self.source_contract_id else None,
            "source_edit_id": str(self.source_edit_id) if self.source_edit_id else None,
            "created_at": self.created_at.isoformat(),
            "cuad_labels": self.cuad_labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Triple":
        """Create from dictionary representation."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            subject_id=UUID(data["subject_id"]) if isinstance(data["subject_id"], str) else data["subject_id"],
            predicate=PredicateType(data["predicate"]),
            object_id=UUID(data["object_id"]) if isinstance(data["object_id"], str) else data["object_id"],
            confidence=data.get("confidence", 0.8),
            properties=data.get("properties", {}),
            source_text=data.get("source_text", ""),
            source_contract_id=UUID(data["source_contract_id"]) if data.get("source_contract_id") else None,
            source_edit_id=UUID(data["source_edit_id"]) if data.get("source_edit_id") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            cuad_labels=data.get("cuad_labels", []),
        )
