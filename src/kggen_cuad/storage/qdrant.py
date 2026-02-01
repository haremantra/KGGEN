"""
Qdrant vector database adapter for semantic search.
"""

from functools import lru_cache
from typing import Any
from uuid import UUID

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from kggen_cuad.config import get_settings

logger = structlog.get_logger(__name__)


class QdrantAdapter:
    """
    Qdrant vector database adapter.

    Handles vector storage and similarity search for embeddings.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ):
        settings = get_settings()
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.dimension = settings.embedding_dimension

        self._client: QdrantClient | None = None

    def connect(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
            logger.info("qdrant_connected", host=self.host, port=self.port)
        return self._client

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client."""
        return self.connect()

    def close(self) -> None:
        """Close the Qdrant client."""
        if self._client:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error("qdrant_health_check_failed", error=str(e))
            return False

    # =========================================================================
    # Collection Management
    # =========================================================================

    def create_collection(self, recreate: bool = False) -> bool:
        """Create the embeddings collection."""
        try:
            if recreate:
                try:
                    self.client.delete_collection(self.collection_name)
                except UnexpectedResponse:
                    pass  # Collection doesn't exist

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.dimension,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="entity_type",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="contract_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )

            logger.info("qdrant_collection_created", collection=self.collection_name)
            return True

        except UnexpectedResponse as e:
            if "already exists" in str(e):
                logger.info("qdrant_collection_exists", collection=self.collection_name)
                return True
            raise

    def ensure_collection(self) -> None:
        """Ensure collection exists, creating if necessary."""
        try:
            self.client.get_collection(self.collection_name)
        except UnexpectedResponse:
            self.create_collection()

    # =========================================================================
    # Vector Operations
    # =========================================================================

    def upsert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> int:
        """Upsert vectors into the collection."""
        self.ensure_collection()

        points = []
        for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
            payload = payloads[i] if payloads else {}
            points.append(
                qdrant_models.PointStruct(
                    id=vec_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.debug("vectors_upserted", count=len(points))
        return len(points)

    def upsert_entity_embedding(
        self,
        entity_id: UUID,
        embedding: list[float],
        entity_name: str,
        entity_type: str,
        contract_id: UUID | None = None,
    ) -> None:
        """Upsert an entity embedding."""
        payload = {
            "entity_id": str(entity_id),
            "entity_name": entity_name,
            "entity_type": entity_type,
            "embedding_type": "entity",
        }
        if contract_id:
            payload["contract_id"] = str(contract_id)

        self.upsert_vectors(
            vectors=[embedding],
            ids=[str(entity_id)],
            payloads=[payload],
        )

    def upsert_triple_embedding(
        self,
        triple_id: UUID,
        embedding: list[float],
        subject: str,
        predicate: str,
        obj: str,
        contract_id: UUID | None = None,
        cuad_label: str | None = None,
    ) -> None:
        """Upsert a triple embedding."""
        payload = {
            "triple_id": str(triple_id),
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "embedding_type": "triple",
        }
        if contract_id:
            payload["contract_id"] = str(contract_id)
        if cuad_label:
            payload["cuad_label"] = cuad_label

        self.upsert_vectors(
            vectors=[embedding],
            ids=[str(triple_id)],
            payloads=[payload],
        )

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.

        Returns list of dicts with 'id', 'score', and 'payload' keys.
        """
        self.ensure_collection()

        # Build filter conditions
        filter_conditions = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value),
                        )
                    )
            if conditions:
                filter_conditions = qdrant_models.Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in results
        ]

    def search_entities(
        self,
        query_vector: list[float],
        top_k: int = 10,
        entity_type: str | None = None,
        contract_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar entities."""
        filters = {"embedding_type": "entity"}
        if entity_type:
            filters["entity_type"] = entity_type
        if contract_id:
            filters["contract_id"] = str(contract_id)

        return self.search(query_vector, top_k=top_k, filters=filters)

    def search_triples(
        self,
        query_vector: list[float],
        top_k: int = 10,
        contract_id: UUID | None = None,
        cuad_label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar triples."""
        filters = {"embedding_type": "triple"}
        if contract_id:
            filters["contract_id"] = str(contract_id)
        if cuad_label:
            filters["cuad_label"] = cuad_label

        return self.search(query_vector, top_k=top_k, filters=filters)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def delete_by_contract(self, contract_id: UUID) -> int:
        """Delete all vectors associated with a contract."""
        self.ensure_collection()

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="contract_id",
                            match=qdrant_models.MatchValue(value=str(contract_id)),
                        )
                    ]
                )
            ),
        )

        logger.info("vectors_deleted_by_contract", contract_id=str(contract_id))
        return result.status if hasattr(result, "status") else 0

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except UnexpectedResponse:
            return {"name": self.collection_name, "status": "not_found"}


@lru_cache()
def get_qdrant_adapter() -> QdrantAdapter:
    """Get cached Qdrant adapter instance."""
    return QdrantAdapter()
