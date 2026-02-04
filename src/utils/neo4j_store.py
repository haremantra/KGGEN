"""Neo4j graph database storage for knowledge graphs."""

from neo4j import GraphDatabase
from ..config import settings
from ..models.schema import KGNode, KGEdge, ExtractionResult, NodeType, EdgeType


class Neo4jStore:
    """Store and query knowledge graphs in Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """Initialize Neo4j connection.

        Args:
            uri: Neo4j bolt URI. Defaults to settings.neo4j_uri.
            user: Neo4j username. Defaults to settings.neo4j_user.
            password: Neo4j password. Defaults to settings.neo4j_password.
        """
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._driver = None

    @property
    def driver(self):
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        return self._driver

    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints for node uniqueness
            for node_type in NodeType:
                try:
                    session.run(f"""
                        CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type.value})
                        REQUIRE n.id IS UNIQUE
                    """)
                except Exception:
                    pass

            # Create indexes for common queries
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (n:Party) ON (n.name)
            """)
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (n:Contract) ON (n.source_contract_id)
            """)

    def store_extraction_result(self, result: ExtractionResult):
        """Store an extraction result in Neo4j.

        Args:
            result: The extraction result to store.
        """
        with self.driver.session() as session:
            # Store entities as nodes
            for entity in result.entities:
                self._create_node(session, entity)

            # Store triples as relationships
            for triple in result.triples:
                self._create_relationship_from_triple(
                    session, triple, result.contract_id
                )

    def _create_node(self, session, node: KGNode):
        """Create a node in Neo4j."""
        query = f"""
            MERGE (n:{node.type.value} {{id: $id}})
            SET n.name = $name,
                n.source_contract_id = $source_contract_id,
                n.cuad_label = $cuad_label,
                n.confidence_score = $confidence_score,
                n.properties = $properties
            RETURN n
        """
        session.run(
            query,
            id=node.id,
            name=node.name,
            source_contract_id=node.source_contract_id,
            cuad_label=node.cuad_label,
            confidence_score=node.confidence_score,
            properties=str(node.properties),
        )

    def _create_relationship_from_triple(self, session, triple, contract_id: str):
        """Create a relationship from a triple."""
        # Map predicate to edge type
        edge_type = self._map_predicate_to_edge_type(triple.predicate)

        query = f"""
            MERGE (s:Entity {{name: $subject}})
            MERGE (o:Entity {{name: $object}})
            MERGE (s)-[r:{edge_type}]->(o)
            SET r.source_contract_id = $contract_id,
                r.confidence = $confidence,
                r.properties = $properties
            RETURN r
        """
        session.run(
            query,
            subject=triple.subject,
            object=triple.object,
            contract_id=contract_id,
            confidence=triple.confidence,
            properties=str(triple.properties),
        )

    def _map_predicate_to_edge_type(self, predicate: str) -> str:
        """Map a predicate string to an edge type."""
        predicate_lower = predicate.lower().replace(" ", "_").replace("-", "_")

        mapping = {
            "licenses_to": "LICENSES_TO",
            "license": "LICENSES_TO",
            "licenses": "LICENSES_TO",
            "owns": "OWNS",
            "own": "OWNS",
            "assigns": "ASSIGNS",
            "assign": "ASSIGNS",
            "has_obligation": "HAS_OBLIGATION",
            "obligation": "HAS_OBLIGATION",
            "must": "HAS_OBLIGATION",
            "shall": "HAS_OBLIGATION",
            "subject_to_restriction": "SUBJECT_TO_RESTRICTION",
            "restricted": "SUBJECT_TO_RESTRICTION",
            "cannot": "SUBJECT_TO_RESTRICTION",
            "has_liability": "HAS_LIABILITY",
            "liability": "HAS_LIABILITY",
            "governed_by": "GOVERNED_BY",
            "governing_law": "GOVERNED_BY",
            "contains_clause": "CONTAINS_CLAUSE",
            "contains": "CONTAINS_CLAUSE",
            "effective_on": "EFFECTIVE_ON",
            "effective": "EFFECTIVE_ON",
            "terminates_on": "TERMINATES_ON",
            "expires": "TERMINATES_ON",
            "expiration": "TERMINATES_ON",
        }

        return mapping.get(predicate_lower, predicate_lower.upper())

    def get_contract_graph(self, contract_id: str) -> dict:
        """Retrieve the knowledge graph for a specific contract.

        Args:
            contract_id: The contract identifier.

        Returns:
            Dict with 'nodes' and 'edges' keys.
        """
        with self.driver.session() as session:
            # Get all nodes for this contract
            nodes_result = session.run("""
                MATCH (n)
                WHERE n.source_contract_id = $contract_id
                RETURN n
            """, contract_id=contract_id)

            nodes = []
            for record in nodes_result:
                node = record["n"]
                nodes.append({
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "labels": list(node.labels),
                    "properties": dict(node),
                })

            # Get all relationships for this contract
            edges_result = session.run("""
                MATCH (s)-[r]->(o)
                WHERE r.source_contract_id = $contract_id
                RETURN s, r, o, type(r) as rel_type
            """, contract_id=contract_id)

            edges = []
            for record in edges_result:
                edges.append({
                    "source": record["s"].get("name"),
                    "target": record["o"].get("name"),
                    "type": record["rel_type"],
                    "properties": dict(record["r"]),
                })

            return {"nodes": nodes, "edges": edges}

    def query_cypher(self, query: str, parameters: dict | None = None) -> list:
        """Execute a raw Cypher query.

        Args:
            query: The Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of result records.
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def get_statistics(self) -> dict:
        """Get graph statistics.

        Returns:
            Dict with node and edge counts.
        """
        with self.driver.session() as session:
            node_count = session.run(
                "MATCH (n) RETURN count(n) as count"
            ).single()["count"]

            edge_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]

            label_counts = {}
            for node_type in NodeType:
                result = session.run(
                    f"MATCH (n:{node_type.value}) RETURN count(n) as count"
                )
                label_counts[node_type.value] = result.single()["count"]

            return {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "nodes_by_type": label_counts,
            }
