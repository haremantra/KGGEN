"""
Neo4j graph database adapter.
"""

from functools import lru_cache
from typing import Any
from uuid import UUID

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable

from kggen_cuad.config import get_settings
from kggen_cuad.models.triple import Entity, EntityType, PredicateType, Triple
from kggen_cuad.models.graph import KnowledgeGraph, SubGraph

logger = structlog.get_logger(__name__)


class Neo4jAdapter:
    """
    Neo4j graph database adapter.

    Handles all graph operations for storing and querying the knowledge graph.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the Neo4j driver."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.info("neo4j_connected", uri=self.uri)

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    @property
    def driver(self) -> AsyncDriver:
        """Get the Neo4j driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not connected. Call connect() first.")
        return self._driver

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            await self.connect()
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            return True
        except ServiceUnavailable as e:
            logger.error("neo4j_health_check_failed", error=str(e))
            return False

    # =========================================================================
    # Schema Setup
    # =========================================================================

    async def setup_schema(self) -> None:
        """Create indexes and constraints for the knowledge graph."""
        await self.connect()
        async with self.driver.session() as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (c:Contract) REQUIRE c.id IS UNIQUE",
            ]

            # Create indexes for common queries
            indexes = [
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                "CREATE INDEX entity_contract IF NOT EXISTS FOR (e:Entity) ON (e.contract_id)",
                "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
            ]

            for query in constraints + indexes:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning("schema_setup_warning", query=query, error=str(e))

        logger.info("neo4j_schema_setup_complete")

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity node in Neo4j."""
        await self.connect()
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.entity_type = $entity_type,
            e.contract_id = $contract_id,
            e.confidence_score = $confidence_score,
            e.is_canonical = $is_canonical,
            e.properties = $properties
        RETURN e
        """

        async with self.driver.session() as session:
            await session.run(
                query,
                {
                    "id": str(entity.id),
                    "name": entity.name,
                    "entity_type": entity.entity_type.value if isinstance(entity.entity_type, EntityType) else entity.entity_type,
                    "contract_id": str(entity.contract_id) if entity.contract_id else None,
                    "confidence_score": entity.confidence_score,
                    "is_canonical": entity.is_canonical,
                    "properties": str(entity.properties),
                },
            )

        logger.debug("entity_created", entity_id=str(entity.id), name=entity.name)
        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        await self.connect()
        query = "MATCH (e:Entity {id: $id}) RETURN e"

        async with self.driver.session() as session:
            result = await session.run(query, {"id": str(entity_id)})
            record = await result.single()
            if record:
                return self._record_to_entity(record["e"])
            return None

    async def search_entities(
        self,
        query_text: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by name."""
        await self.connect()
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query)
        YIELD node, score
        WHERE ($entity_type IS NULL OR node.entity_type = $entity_type)
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                {
                    "query": query_text,
                    "entity_type": entity_type.value if entity_type else None,
                    "limit": limit,
                },
            )
            records = await result.data()
            return [self._record_to_entity(r["node"]) for r in records]

    # =========================================================================
    # Triple/Relationship Operations
    # =========================================================================

    async def create_triple(self, triple: Triple) -> Triple:
        """Create a relationship (triple) in Neo4j."""
        await self.connect()

        # First ensure both entities exist
        await self.create_entity(triple.subject)
        await self.create_entity(triple.object)

        # Create the relationship
        predicate = triple.predicate.value if isinstance(triple.predicate, PredicateType) else triple.predicate
        query = f"""
        MATCH (s:Entity {{id: $subject_id}})
        MATCH (o:Entity {{id: $object_id}})
        MERGE (s)-[r:{predicate}]->(o)
        SET r.id = $triple_id,
            r.confidence_score = $confidence_score,
            r.contract_id = $contract_id,
            r.cuad_label = $cuad_label
        RETURN r
        """

        async with self.driver.session() as session:
            await session.run(
                query,
                {
                    "subject_id": str(triple.subject.id),
                    "object_id": str(triple.object.id),
                    "triple_id": str(triple.id),
                    "confidence_score": triple.confidence_score,
                    "contract_id": str(triple.contract_id) if triple.contract_id else None,
                    "cuad_label": triple.cuad_label,
                },
            )

        logger.debug(
            "triple_created",
            triple_id=str(triple.id),
            predicate=predicate,
        )
        return triple

    async def get_triples_for_entity(
        self,
        entity_id: UUID,
        direction: str = "both",  # "outgoing", "incoming", or "both"
    ) -> list[Triple]:
        """Get all triples involving an entity."""
        await self.connect()

        if direction == "outgoing":
            query = """
            MATCH (s:Entity {id: $entity_id})-[r]->(o:Entity)
            RETURN s, type(r) as predicate, r, o
            """
        elif direction == "incoming":
            query = """
            MATCH (s:Entity)-[r]->(o:Entity {id: $entity_id})
            RETURN s, type(r) as predicate, r, o
            """
        else:
            query = """
            MATCH (e:Entity {id: $entity_id})
            OPTIONAL MATCH (e)-[r1]->(o1:Entity)
            OPTIONAL MATCH (s2:Entity)-[r2]->(e)
            WITH e,
                 collect(DISTINCT {s: e, r: r1, o: o1, p: type(r1)}) as outgoing,
                 collect(DISTINCT {s: s2, r: r2, o: e, p: type(r2)}) as incoming
            RETURN outgoing + incoming as relationships
            """

        async with self.driver.session() as session:
            result = await session.run(query, {"entity_id": str(entity_id)})
            records = await result.data()

            triples = []
            if direction in ("outgoing", "incoming"):
                for r in records:
                    triple = self._record_to_triple(r)
                    if triple:
                        triples.append(triple)
            else:
                # Handle combined query
                if records and records[0].get("relationships"):
                    for rel in records[0]["relationships"]:
                        if rel.get("s") and rel.get("o") and rel.get("p"):
                            triple = self._record_to_triple(rel)
                            if triple:
                                triples.append(triple)

            return triples

    # =========================================================================
    # Graph Operations
    # =========================================================================

    async def import_graph(self, graph: KnowledgeGraph) -> int:
        """Import a complete knowledge graph into Neo4j."""
        await self.connect()

        # Create all entities
        for entity in graph.entities.values():
            await self.create_entity(entity)

        # Create all triples
        for triple in graph.triples:
            await self.create_triple(triple)

        logger.info(
            "graph_imported",
            entities=len(graph.entities),
            triples=len(graph.triples),
        )
        return len(graph.triples)

    async def get_subgraph(
        self,
        entity_ids: list[UUID],
        hops: int = 2,
        max_nodes: int = 100,
    ) -> SubGraph:
        """Get a subgraph around specified entities with N-hop expansion."""
        await self.connect()

        query = """
        MATCH (start:Entity)
        WHERE start.id IN $entity_ids
        CALL apoc.path.subgraphAll(start, {
            maxLevel: $hops,
            limit: $max_nodes
        })
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    {
                        "entity_ids": [str(eid) for eid in entity_ids],
                        "hops": hops,
                        "max_nodes": max_nodes,
                    },
                )
                record = await result.single()

                if record:
                    nodes = [self._record_to_entity(n) for n in record["nodes"]]
                    edges = []
                    for rel in record["relationships"]:
                        triple = self._relationship_to_triple(rel)
                        if triple:
                            edges.append(triple)
                    return SubGraph(nodes=nodes, edges=edges)
        except Exception as e:
            logger.warning("subgraph_query_failed", error=str(e))

        # Fallback: simple expansion
        return await self._simple_expansion(entity_ids, hops)

    async def _simple_expansion(
        self,
        entity_ids: list[UUID],
        hops: int,
    ) -> SubGraph:
        """Simple N-hop expansion without APOC."""
        all_entities: dict[UUID, Entity] = {}
        all_triples: list[Triple] = []
        current_ids = set(entity_ids)

        for _ in range(hops):
            new_ids: set[UUID] = set()
            for eid in current_ids:
                entity = await self.get_entity(eid)
                if entity:
                    all_entities[eid] = entity

                triples = await self.get_triples_for_entity(eid)
                for triple in triples:
                    if triple.id not in {t.id for t in all_triples}:
                        all_triples.append(triple)
                    new_ids.add(triple.subject.id)
                    new_ids.add(triple.object.id)

            current_ids = new_ids - set(all_entities.keys())

        return SubGraph(
            nodes=list(all_entities.values()),
            edges=all_triples,
        )

    async def get_graph_statistics(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        await self.connect()

        query = """
        MATCH (e:Entity)
        WITH count(e) as node_count
        MATCH ()-[r]->()
        WITH node_count, count(r) as edge_count
        RETURN node_count, edge_count
        """

        async with self.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                return {
                    "node_count": record["node_count"],
                    "edge_count": record["edge_count"],
                }
            return {"node_count": 0, "edge_count": 0}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _record_to_entity(self, node: Any) -> Entity:
        """Convert Neo4j node to Entity model."""
        props = dict(node) if hasattr(node, "__iter__") else {}
        return Entity(
            id=UUID(props.get("id", str(UUID()))),
            name=props.get("name", ""),
            entity_type=EntityType(props.get("entity_type", "Party")),
            contract_id=UUID(props["contract_id"]) if props.get("contract_id") else None,
            confidence_score=props.get("confidence_score", 1.0),
            is_canonical=props.get("is_canonical", False),
        )

    def _record_to_triple(self, record: dict[str, Any]) -> Triple | None:
        """Convert query record to Triple model."""
        try:
            subject = self._record_to_entity(record["s"])
            obj = self._record_to_entity(record["o"])
            predicate = record.get("predicate") or record.get("p")

            return Triple(
                subject=subject,
                subject_text=subject.name,
                predicate=PredicateType(predicate) if predicate in [p.value for p in PredicateType] else PredicateType.HAS_OBLIGATION,
                object=obj,
                object_text=obj.name,
            )
        except Exception:
            return None

    def _relationship_to_triple(self, rel: Any) -> Triple | None:
        """Convert Neo4j relationship to Triple model."""
        try:
            props = dict(rel) if hasattr(rel, "__iter__") else {}
            return Triple(
                id=UUID(props.get("id", str(UUID()))),
                subject=Entity(name="", entity_type=EntityType.PARTY),
                subject_text="",
                predicate=PredicateType.HAS_OBLIGATION,
                object=Entity(name="", entity_type=EntityType.PARTY),
                object_text="",
            )
        except Exception:
            return None


@lru_cache()
def get_neo4j_adapter() -> Neo4jAdapter:
    """Get cached Neo4j adapter instance."""
    return Neo4jAdapter()
