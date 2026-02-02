"""
Knowledge graph model classes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from kggen_cuad.models.triple import Entity, Triple


@dataclass
class GraphStatistics:
    """Statistics about the knowledge graph."""
    entity_count: int = 0
    triple_count: int = 0
    entity_types: dict[str, int] = field(default_factory=dict)
    predicate_types: dict[str, int] = field(default_factory=dict)
    contracts_processed: int = 0
    edits_processed: int = 0
    avg_confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = {}
        if self.predicate_types is None:
            self.predicate_types = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entity_count": self.entity_count,
            "triple_count": self.triple_count,
            "entity_types": self.entity_types,
            "predicate_types": self.predicate_types,
            "contracts_processed": self.contracts_processed,
            "edits_processed": self.edits_processed,
            "avg_confidence": self.avg_confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SubGraph:
    """
    A subset of the knowledge graph, typically returned from queries.
    """
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    source_query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.triples is None:
            self.triples = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "triples": [t.to_dict() for t in self.triples],
            "source_query": self.source_query,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeGraph:
    """
    The complete knowledge graph containing all entities and triples.
    """
    entities: dict[UUID, Entity] = field(default_factory=dict)
    triples: list[Triple] = field(default_factory=list)
    statistics: GraphStatistics = field(default_factory=GraphStatistics)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.triples is None:
            self.triples = []
        if self.metadata is None:
            self.metadata = {}

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        self.updated_at = datetime.utcnow()
        self._update_statistics()

    def add_triple(self, triple: Triple) -> None:
        """Add a triple to the graph."""
        self.triples.append(triple)
        self.updated_at = datetime.utcnow()
        self._update_statistics()

    def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a given type."""
        return [e for e in self.entities.values() if e.entity_type.value == entity_type]

    def get_triples_for_entity(self, entity_id: UUID) -> list[Triple]:
        """Get all triples involving an entity."""
        return [t for t in self.triples if t.subject_id == entity_id or t.object_id == entity_id]

    def get_subgraph(self, entity_ids: list[UUID], hops: int = 1) -> SubGraph:
        """
        Get a subgraph starting from the given entities.

        Args:
            entity_ids: Starting entity IDs
            hops: Number of hops to expand

        Returns:
            SubGraph containing the relevant entities and triples
        """
        current_ids = set(entity_ids)
        all_ids = set(entity_ids)

        for _ in range(hops):
            new_ids = set()
            for triple in self.triples:
                if triple.subject_id in current_ids:
                    new_ids.add(triple.object_id)
                if triple.object_id in current_ids:
                    new_ids.add(triple.subject_id)
            current_ids = new_ids - all_ids
            all_ids.update(new_ids)

        entities = [self.entities[eid] for eid in all_ids if eid in self.entities]
        triples = [t for t in self.triples
                   if t.subject_id in all_ids and t.object_id in all_ids]

        return SubGraph(entities=entities, triples=triples)

    def _update_statistics(self) -> None:
        """Update graph statistics."""
        entity_types: dict[str, int] = {}
        predicate_types: dict[str, int] = {}
        total_confidence = 0.0

        for entity in self.entities.values():
            et = entity.entity_type.value
            entity_types[et] = entity_types.get(et, 0) + 1
            total_confidence += entity.confidence

        for triple in self.triples:
            pt = triple.predicate.value
            predicate_types[pt] = predicate_types.get(pt, 0) + 1
            total_confidence += triple.confidence

        total_items = len(self.entities) + len(self.triples)
        avg_confidence = total_confidence / total_items if total_items > 0 else 0.0

        self.statistics = GraphStatistics(
            entity_count=len(self.entities),
            triple_count=len(self.triples),
            entity_types=entity_types,
            predicate_types=predicate_types,
            avg_confidence=avg_confidence,
            updated_at=datetime.utcnow(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entities": {str(k): v.to_dict() for k, v in self.entities.items()},
            "triples": [t.to_dict() for t in self.triples],
            "statistics": self.statistics.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """Create from dictionary representation."""
        entities = {
            UUID(k): Entity.from_dict(v)
            for k, v in data.get("entities", {}).items()
        }
        triples = [Triple.from_dict(t) for t in data.get("triples", [])]

        kg = cls(
            entities=entities,
            triples=triples,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )
        kg._update_statistics()
        return kg
