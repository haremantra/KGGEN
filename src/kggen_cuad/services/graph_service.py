"""
Graph service for Neo4j operations and knowledge graph management.
"""

from functools import lru_cache
from typing import Any
from uuid import UUID

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.triple import Entity, Triple, EntityType, PredicateType
from kggen_cuad.models.graph import KnowledgeGraph, SubGraph, GraphStatistics
from kggen_cuad.storage.neo4j_adapter import get_neo4j_adapter, Neo4jAdapter

logger = structlog.get_logger(__name__)


class GraphService:
    """
    Service for knowledge graph operations.

    Provides high-level operations for building, querying, and managing
    the contract knowledge graph.
    """

    def __init__(self):
        self.settings = get_settings()
        self._neo4j: Neo4jAdapter | None = None

    @property
    def neo4j(self) -> Neo4jAdapter:
        """Get Neo4j adapter."""
        if self._neo4j is None:
            self._neo4j = get_neo4j_adapter()
        return self._neo4j

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def add_entity(
        self,
        entity: Entity,
        contract_id: UUID | None = None,
    ) -> str:
        """
        Add an entity to the knowledge graph.

        Returns the entity ID.
        """
        properties = {
            "id": str(entity.id),
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "normalized_name": entity.normalized_name,
            "source_contract_id": str(contract_id) if contract_id else None,
            **entity.properties,
        }

        await self.neo4j.create_entity(
            entity_type=entity.entity_type.value,
            properties=properties,
        )

        logger.debug(
            "entity_added",
            entity_id=str(entity.id),
            name=entity.name,
            type=entity.entity_type.value,
        )

        return str(entity.id)

    async def add_entities_batch(
        self,
        entities: list[Entity],
        contract_id: UUID | None = None,
    ) -> list[str]:
        """Add multiple entities in batch."""
        entity_ids = []

        for entity in entities:
            entity_id = await self.add_entity(entity, contract_id)
            entity_ids.append(entity_id)

        logger.info("entities_batch_added", count=len(entity_ids))
        return entity_ids

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        result = await self.neo4j.get_entity(entity_id)
        if not result:
            return None

        return Entity(
            id=UUID(result["id"]),
            name=result["name"],
            entity_type=EntityType(result["entity_type"]),
            normalized_name=result.get("normalized_name"),
            properties={
                k: v for k, v in result.items()
                if k not in ["id", "name", "entity_type", "normalized_name"]
            },
        )

    async def find_entities_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Find entities by name (fuzzy match)."""
        results = await self.neo4j.find_entities_by_name(
            name=name,
            entity_type=entity_type.value if entity_type else None,
            limit=limit,
        )

        entities = []
        for result in results:
            entity = Entity(
                id=UUID(result["id"]),
                name=result["name"],
                entity_type=EntityType(result["entity_type"]),
                normalized_name=result.get("normalized_name"),
            )
            entities.append(entity)

        return entities

    async def get_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
    ) -> list[Entity]:
        """Get all entities of a specific type."""
        results = await self.neo4j.get_entities_by_type(
            entity_type=entity_type.value,
            limit=limit,
        )

        return [
            Entity(
                id=UUID(r["id"]),
                name=r["name"],
                entity_type=entity_type,
                normalized_name=r.get("normalized_name"),
            )
            for r in results
        ]

    # =========================================================================
    # Triple Operations
    # =========================================================================

    async def add_triple(
        self,
        triple: Triple,
        contract_id: UUID | None = None,
    ) -> str:
        """
        Add a triple (relationship) to the knowledge graph.

        Returns the triple ID.
        """
        properties = {
            "id": str(triple.id),
            "confidence": triple.confidence,
            "source_contract_id": str(contract_id) if contract_id else None,
            **triple.properties,
        }

        await self.neo4j.create_triple(
            subject_id=str(triple.subject_id),
            predicate=triple.predicate.value,
            object_id=str(triple.object_id),
            properties=properties,
        )

        logger.debug(
            "triple_added",
            triple_id=str(triple.id),
            predicate=triple.predicate.value,
        )

        return str(triple.id)

    async def add_triples_batch(
        self,
        triples: list[Triple],
        contract_id: UUID | None = None,
    ) -> list[str]:
        """Add multiple triples in batch."""
        triple_ids = []

        for triple in triples:
            triple_id = await self.add_triple(triple, contract_id)
            triple_ids.append(triple_id)

        logger.info("triples_batch_added", count=len(triple_ids))
        return triple_ids

    async def get_triple(self, triple_id: str) -> Triple | None:
        """Get a triple by ID."""
        result = await self.neo4j.get_triple(triple_id)
        if not result:
            return None

        return Triple(
            id=UUID(result["id"]),
            subject_id=UUID(result["subject_id"]),
            predicate=PredicateType(result["predicate"]),
            object_id=UUID(result["object_id"]),
            confidence=result.get("confidence", 1.0),
        )

    async def get_triples_for_entity(
        self,
        entity_id: str,
        direction: str = "both",
        predicate: PredicateType | None = None,
    ) -> list[Triple]:
        """
        Get all triples involving an entity.

        Args:
            entity_id: The entity ID
            direction: "outgoing", "incoming", or "both"
            predicate: Optional predicate filter
        """
        results = await self.neo4j.get_triples_for_entity(
            entity_id=entity_id,
            direction=direction,
            predicate=predicate.value if predicate else None,
        )

        return [
            Triple(
                id=UUID(r["id"]),
                subject_id=UUID(r["subject_id"]),
                predicate=PredicateType(r["predicate"]),
                object_id=UUID(r["object_id"]),
                confidence=r.get("confidence", 1.0),
            )
            for r in results
        ]

    # =========================================================================
    # Subgraph Operations
    # =========================================================================

    async def get_entity_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
        max_nodes: int = 50,
    ) -> SubGraph:
        """
        Get the neighborhood subgraph around an entity.

        Returns entities and triples within the specified depth.
        """
        result = await self.neo4j.get_neighborhood(
            entity_id=entity_id,
            depth=depth,
            max_nodes=max_nodes,
        )

        entities = []
        for node in result.get("nodes", []):
            entity = Entity(
                id=UUID(node["id"]),
                name=node["name"],
                entity_type=EntityType(node["entity_type"]),
                normalized_name=node.get("normalized_name"),
            )
            entities.append(entity)

        triples = []
        for edge in result.get("edges", []):
            triple = Triple(
                id=UUID(edge["id"]),
                subject_id=UUID(edge["subject_id"]),
                predicate=PredicateType(edge["predicate"]),
                object_id=UUID(edge["object_id"]),
                confidence=edge.get("confidence", 1.0),
            )
            triples.append(triple)

        return SubGraph(
            entities=entities,
            triples=triples,
            center_entity_id=UUID(entity_id),
            depth=depth,
        )

    async def get_contract_subgraph(
        self,
        contract_id: UUID,
    ) -> SubGraph:
        """Get all entities and triples from a specific contract."""
        result = await self.neo4j.get_contract_subgraph(str(contract_id))

        entities = [
            Entity(
                id=UUID(node["id"]),
                name=node["name"],
                entity_type=EntityType(node["entity_type"]),
            )
            for node in result.get("nodes", [])
        ]

        triples = [
            Triple(
                id=UUID(edge["id"]),
                subject_id=UUID(edge["subject_id"]),
                predicate=PredicateType(edge["predicate"]),
                object_id=UUID(edge["object_id"]),
            )
            for edge in result.get("edges", [])
        ]

        return SubGraph(
            entities=entities,
            triples=triples,
            source_contract_id=contract_id,
        )

    async def get_path_between_entities(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 5,
    ) -> list[list[dict[str, Any]]]:
        """
        Find paths between two entities.

        Returns list of paths, where each path is a list of
        alternating entities and relationships.
        """
        return await self.neo4j.find_paths(
            start_id=start_entity_id,
            end_id=end_entity_id,
            max_depth=max_depth,
        )

    # =========================================================================
    # Graph Statistics
    # =========================================================================

    async def get_statistics(self) -> GraphStatistics:
        """Get knowledge graph statistics."""
        stats = await self.neo4j.get_statistics()

        return GraphStatistics(
            total_entities=stats.get("total_entities", 0),
            total_triples=stats.get("total_triples", 0),
            entities_by_type=stats.get("entities_by_type", {}),
            triples_by_predicate=stats.get("triples_by_predicate", {}),
            contracts_processed=stats.get("contracts_processed", 0),
            average_triples_per_contract=stats.get("avg_triples_per_contract", 0.0),
        )

    # =========================================================================
    # Graph Traversal Queries
    # =========================================================================

    async def get_parties_with_obligations(
        self,
        contract_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Get all parties and their obligations."""
        query = """
        MATCH (p:Party)-[:HAS_OBLIGATION]->(o:Obligation)
        WHERE $contract_id IS NULL OR p.source_contract_id = $contract_id
        RETURN p.name as party, collect(o.name) as obligations
        """
        return await self.neo4j.run_query(
            query,
            {"contract_id": str(contract_id) if contract_id else None},
        )

    async def get_ip_licensing_structure(
        self,
        contract_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Get IP assets and their licensing relationships."""
        query = """
        MATCH (licensor:Party)-[l:LICENSES_TO]->(licensee:Party)
        OPTIONAL MATCH (licensor)-[:OWNS]->(ip:IPAsset)
        WHERE $contract_id IS NULL OR licensor.source_contract_id = $contract_id
        RETURN
            licensor.name as licensor,
            licensee.name as licensee,
            collect(DISTINCT ip.name) as ip_assets,
            l.license_type as license_type
        """
        return await self.neo4j.run_query(
            query,
            {"contract_id": str(contract_id) if contract_id else None},
        )

    async def get_restrictions_by_party(
        self,
        party_name: str,
    ) -> list[dict[str, Any]]:
        """Get all restrictions applicable to a party."""
        query = """
        MATCH (p:Party {name: $party_name})-[:SUBJECT_TO_RESTRICTION]->(r:Restriction)
        RETURN r.name as restriction, r.type as restriction_type, r.properties as details
        """
        return await self.neo4j.run_query(query, {"party_name": party_name})

    async def get_liability_provisions(
        self,
        contract_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Get liability provisions and their associated parties."""
        query = """
        MATCH (p:Party)-[:HAS_LIABILITY]->(l:LiabilityProvision)
        WHERE $contract_id IS NULL OR p.source_contract_id = $contract_id
        RETURN
            p.name as party,
            l.name as provision,
            l.cap_amount as liability_cap,
            l.provision_type as type
        """
        return await self.neo4j.run_query(
            query,
            {"contract_id": str(contract_id) if contract_id else None},
        )

    async def get_temporal_structure(
        self,
        contract_id: UUID,
    ) -> list[dict[str, Any]]:
        """Get temporal relationships (effective dates, terminations, etc.)."""
        query = """
        MATCH (c:ContractClause)-[r]->(t:Temporal)
        WHERE c.source_contract_id = $contract_id
        RETURN
            c.name as clause,
            type(r) as relationship,
            t.name as temporal_reference,
            t.date as date
        ORDER BY t.date
        """
        return await self.neo4j.run_query(query, {"contract_id": str(contract_id)})

    # =========================================================================
    # Entity Resolution Support
    # =========================================================================

    async def merge_entities(
        self,
        canonical_id: str,
        alias_ids: list[str],
    ) -> bool:
        """
        Merge duplicate entities into a canonical entity.

        Redirects all relationships from aliases to the canonical entity.
        """
        try:
            await self.neo4j.merge_entities(canonical_id, alias_ids)
            logger.info(
                "entities_merged",
                canonical=canonical_id,
                aliases=len(alias_ids),
            )
            return True
        except Exception as e:
            logger.error("entity_merge_failed", error=str(e))
            return False

    async def add_entity_alias(
        self,
        entity_id: str,
        alias: str,
    ) -> bool:
        """Add an alias to an entity."""
        return await self.neo4j.add_entity_alias(entity_id, alias)

    # =========================================================================
    # Cleanup Operations
    # =========================================================================

    async def delete_contract_data(self, contract_id: UUID) -> int:
        """Delete all entities and triples from a contract."""
        count = await self.neo4j.delete_contract_data(str(contract_id))
        logger.info("contract_data_deleted", contract_id=str(contract_id), count=count)
        return count

    async def clear_all(self) -> bool:
        """Clear all data from the knowledge graph. Use with caution!"""
        await self.neo4j.clear_all()
        logger.warning("knowledge_graph_cleared")
        return True


@lru_cache()
def get_graph_service() -> GraphService:
    """Get cached graph service instance."""
    return GraphService()
