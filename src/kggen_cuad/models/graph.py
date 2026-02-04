"""
Knowledge graph container models.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from kggen_cuad.models.triple import Entity, Triple


class GraphStatistics(BaseModel):
    """Statistics about a knowledge graph."""

    node_count: int = 0
    edge_count: int = 0
    entity_type_counts: dict[str, int] = Field(default_factory=dict)
    predicate_type_counts: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    canonical_ratio: float = 0.0


class SubGraph(BaseModel):
    """
    A subset of a knowledge graph, typically returned from queries.

    Used for representing query results and context for LLM.
    """

    nodes: list[Entity] = Field(default_factory=list)
    edges: list[Triple] = Field(default_factory=list)
    query: str | None = Field(default=None, description="Query that produced this subgraph")
    relevance_scores: dict[str, float] = Field(
        default_factory=dict, description="Relevance scores for nodes/edges"
    )

    @computed_field
    @property
    def node_count(self) -> int:
        """Number of nodes in the subgraph."""
        return len(self.nodes)

    @computed_field
    @property
    def edge_count(self) -> int:
        """Number of edges in the subgraph."""
        return len(self.edges)

    def to_context_string(self) -> str:
        """
        Convert subgraph to a string context for LLM prompts.

        Returns a formatted string representation suitable for
        including in LLM context.
        """
        lines = ["# Knowledge Graph Context", "", "## Relevant Triples"]

        for edge in self.edges:
            score = self.relevance_scores.get(str(edge.id), 0.0)
            line = f"- ({edge.subject_text}) --[{edge.predicate}]--> ({edge.object_text})"
            if score > 0:
                line += f" [relevance: {score:.2f}]"
            lines.append(line)

        if not self.edges:
            lines.append("- No relevant triples found")

        lines.append("")
        lines.append("## Entities")

        for node in self.nodes:
            props = ", ".join(f"{k}={v}" for k, v in node.properties.items())
            line = f"- {node.name} ({node.entity_type})"
            if props:
                line += f" [{props}]"
            lines.append(line)

        return "\n".join(lines)


class KnowledgeGraph(BaseModel):
    """
    Complete knowledge graph container.

    Holds all entities and triples for a set of contracts,
    along with metadata and statistics.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(default="KGGEN-CUAD Knowledge Graph")

    # Graph content
    entities: dict[UUID, Entity] = Field(
        default_factory=dict, description="Entity ID to Entity mapping"
    )
    triples: list[Triple] = Field(default_factory=list, description="All triples")

    # Source tracking
    contract_ids: list[UUID] = Field(
        default_factory=list, description="Source contract IDs"
    )

    # Processing stage
    stage: str = Field(
        default="extraction",
        description="Current processing stage: extraction, aggregation, resolution",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

    def add_entity(self, entity: Entity) -> Entity:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        self.updated_at = datetime.utcnow()
        return entity

    def add_triple(self, triple: Triple) -> Triple:
        """Add a triple to the graph, also adding subject/object entities."""
        # Add subject and object entities if not present
        if triple.subject.id not in self.entities:
            self.entities[triple.subject.id] = triple.subject
        if triple.object.id not in self.entities:
            self.entities[triple.object.id] = triple.object

        self.triples.append(triple)
        self.updated_at = datetime.utcnow()
        return triple

    def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Get an entity by name (case-insensitive)."""
        name_lower = name.lower()
        for entity in self.entities.values():
            if entity.name.lower() == name_lower:
                return entity
        return None

    def get_triples_for_entity(self, entity_id: UUID) -> list[Triple]:
        """Get all triples involving an entity (as subject or object)."""
        return [
            t
            for t in self.triples
            if t.subject.id == entity_id or t.object.id == entity_id
        ]

    def get_neighbors(self, entity_id: UUID, hops: int = 1) -> SubGraph:
        """
        Get neighboring entities within N hops.

        Returns a SubGraph containing all entities and edges
        reachable within the specified number of hops.
        """
        visited_entities: set[UUID] = {entity_id}
        collected_triples: list[Triple] = []

        current_frontier = {entity_id}

        for _ in range(hops):
            next_frontier: set[UUID] = set()

            for eid in current_frontier:
                for triple in self.triples:
                    if triple.subject.id == eid and triple.object.id not in visited_entities:
                        visited_entities.add(triple.object.id)
                        next_frontier.add(triple.object.id)
                        collected_triples.append(triple)
                    elif triple.object.id == eid and triple.subject.id not in visited_entities:
                        visited_entities.add(triple.subject.id)
                        next_frontier.add(triple.subject.id)
                        collected_triples.append(triple)

            current_frontier = next_frontier

        nodes = [self.entities[eid] for eid in visited_entities if eid in self.entities]
        return SubGraph(nodes=nodes, edges=collected_triples)

    def compute_statistics(self) -> GraphStatistics:
        """Compute statistics about the graph."""
        entity_type_counts: dict[str, int] = {}
        predicate_type_counts: dict[str, int] = {}
        total_confidence = 0.0
        canonical_count = 0

        for entity in self.entities.values():
            entity_type = str(entity.entity_type)
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            if entity.is_canonical:
                canonical_count += 1

        for triple in self.triples:
            predicate = str(triple.predicate)
            predicate_type_counts[predicate] = predicate_type_counts.get(predicate, 0) + 1
            total_confidence += triple.confidence_score

        node_count = len(self.entities)
        edge_count = len(self.triples)

        return GraphStatistics(
            node_count=node_count,
            edge_count=edge_count,
            entity_type_counts=entity_type_counts,
            predicate_type_counts=predicate_type_counts,
            avg_confidence=total_confidence / edge_count if edge_count > 0 else 0.0,
            canonical_ratio=canonical_count / node_count if node_count > 0 else 0.0,
        )

    def merge(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """
        Merge another knowledge graph into this one.

        Used during Stage 2 aggregation.
        """
        # Add all entities from other graph
        for entity_id, entity in other.entities.items():
            if entity_id not in self.entities:
                self.entities[entity_id] = entity

        # Add all triples from other graph
        existing_triple_ids = {t.id for t in self.triples}
        for triple in other.triples:
            if triple.id not in existing_triple_ids:
                self.triples.append(triple)

        # Merge contract IDs
        self.contract_ids.extend(
            cid for cid in other.contract_ids if cid not in self.contract_ids
        )

        self.updated_at = datetime.utcnow()
        return self

    def to_subgraph(self, max_nodes: int | None = None) -> SubGraph:
        """Convert entire graph to a SubGraph (optionally limited)."""
        nodes = list(self.entities.values())
        edges = self.triples

        if max_nodes and len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            # Filter edges to only include those with both endpoints in nodes
            node_ids = {n.id for n in nodes}
            edges = [
                e for e in edges
                if e.subject.id in node_ids and e.object.id in node_ids
            ]

        return SubGraph(nodes=nodes, edges=edges)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "node_count": len(self.entities),
            "edge_count": len(self.triples),
            "contract_ids": [str(cid) for cid in self.contract_ids],
            "stage": self.stage,
            "statistics": self.compute_statistics().model_dump(),
        }
