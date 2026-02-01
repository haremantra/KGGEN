"""
Search service for hybrid BM25 + semantic search.

Implements the retrieval strategy from KGGen methodology:
- BM25 for lexical matching
- Semantic embeddings for meaning-based search
- Fusion with configurable weights
"""

from functools import lru_cache
from typing import Any
from uuid import UUID

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.triple import Entity, Triple
from kggen_cuad.models.graph import SubGraph
from kggen_cuad.services.embedding_service import get_embedding_service, EmbeddingService
from kggen_cuad.storage.qdrant import get_qdrant_adapter, QdrantAdapter

logger = structlog.get_logger(__name__)


class SearchService:
    """
    Hybrid search service combining BM25 and semantic search.

    Uses the fusion approach from KGGen:
    - k=16 for retrieval
    - 0.5 weight for each method
    """

    def __init__(self):
        self.settings = get_settings()
        self._embedding_service: EmbeddingService | None = None
        self._qdrant: QdrantAdapter | None = None

        # Search parameters from KGGen
        self.retrieval_k = 16
        self.bm25_weight = 0.5
        self.semantic_weight = 0.5

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get embedding service."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    @property
    def qdrant(self) -> QdrantAdapter:
        """Get Qdrant adapter."""
        if self._qdrant is None:
            self._qdrant = get_qdrant_adapter()
        return self._qdrant

    # =========================================================================
    # Entity Search
    # =========================================================================

    async def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        contract_ids: list[UUID] | None = None,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[tuple[Entity, float]]:
        """
        Search for entities using hybrid retrieval.

        Returns list of (entity, score) tuples.
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)

        # Semantic search in Qdrant
        semantic_results = await self.qdrant.search_entities(
            query_embedding=query_embedding,
            entity_types=entity_types,
            contract_ids=[str(c) for c in contract_ids] if contract_ids else None,
            limit=self.retrieval_k,
        )

        # BM25 search (simulated with Qdrant text search or fallback)
        bm25_results = await self._bm25_entity_search(
            query=query,
            entity_types=entity_types,
            contract_ids=contract_ids,
            limit=self.retrieval_k,
        )

        # Fuse results
        fused = self._fuse_results(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
        )

        # Filter and limit
        results = [
            (entity, score)
            for entity, score in fused
            if score >= threshold
        ][:limit]

        logger.debug(
            "entity_search_complete",
            query=query[:50],
            results=len(results),
        )

        return results

    async def _bm25_entity_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        contract_ids: list[UUID] | None = None,
        limit: int = 16,
    ) -> list[tuple[Entity, float]]:
        """
        BM25-style search for entities.

        Uses keyword matching with TF-IDF-like scoring.
        """
        # Use Qdrant's sparse vector search or full-text search
        results = await self.qdrant.text_search_entities(
            query=query,
            entity_types=entity_types,
            contract_ids=[str(c) for c in contract_ids] if contract_ids else None,
            limit=limit,
        )

        return results

    def _fuse_results(
        self,
        semantic_results: list[tuple[Entity, float]],
        bm25_results: list[tuple[Entity, float]],
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[tuple[Entity, float]]:
        """
        Fuse semantic and BM25 results using weighted combination.

        Uses Reciprocal Rank Fusion (RRF) for score combination.
        """
        # Build score maps
        entity_scores: dict[str, dict[str, Any]] = {}

        # Add semantic results
        for rank, (entity, score) in enumerate(semantic_results):
            entity_id = str(entity.id)
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {
                    "entity": entity,
                    "semantic_rank": rank + 1,
                    "semantic_score": score,
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                }
            else:
                entity_scores[entity_id]["semantic_rank"] = rank + 1
                entity_scores[entity_id]["semantic_score"] = score

        # Add BM25 results
        for rank, (entity, score) in enumerate(bm25_results):
            entity_id = str(entity.id)
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {
                    "entity": entity,
                    "semantic_rank": None,
                    "semantic_score": 0.0,
                    "bm25_rank": rank + 1,
                    "bm25_score": score,
                }
            else:
                entity_scores[entity_id]["bm25_rank"] = rank + 1
                entity_scores[entity_id]["bm25_score"] = score

        # Calculate fused scores using RRF
        k = 60  # RRF constant
        fused_results = []

        for entity_id, data in entity_scores.items():
            # RRF score
            rrf_score = 0.0

            if data["semantic_rank"] is not None:
                rrf_score += semantic_weight / (k + data["semantic_rank"])

            if data["bm25_rank"] is not None:
                rrf_score += bm25_weight / (k + data["bm25_rank"])

            fused_results.append((data["entity"], rrf_score))

        # Sort by fused score
        fused_results.sort(key=lambda x: x[1], reverse=True)

        return fused_results

    # =========================================================================
    # Triple Search
    # =========================================================================

    async def search_triples(
        self,
        query: str,
        predicate_types: list[str] | None = None,
        contract_ids: list[UUID] | None = None,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[tuple[Triple, float]]:
        """
        Search for triples using hybrid retrieval.

        Returns list of (triple, score) tuples.
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)

        # Semantic search
        semantic_results = await self.qdrant.search_triples(
            query_embedding=query_embedding,
            predicate_types=predicate_types,
            contract_ids=[str(c) for c in contract_ids] if contract_ids else None,
            limit=self.retrieval_k,
        )

        # BM25 search
        bm25_results = await self._bm25_triple_search(
            query=query,
            predicate_types=predicate_types,
            contract_ids=contract_ids,
            limit=self.retrieval_k,
        )

        # Fuse results
        fused = self._fuse_triple_results(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
        )

        # Filter and limit
        results = [
            (triple, score)
            for triple, score in fused
            if score >= threshold
        ][:limit]

        logger.debug(
            "triple_search_complete",
            query=query[:50],
            results=len(results),
        )

        return results

    async def _bm25_triple_search(
        self,
        query: str,
        predicate_types: list[str] | None = None,
        contract_ids: list[UUID] | None = None,
        limit: int = 16,
    ) -> list[tuple[Triple, float]]:
        """BM25-style search for triples."""
        results = await self.qdrant.text_search_triples(
            query=query,
            predicate_types=predicate_types,
            contract_ids=[str(c) for c in contract_ids] if contract_ids else None,
            limit=limit,
        )
        return results

    def _fuse_triple_results(
        self,
        semantic_results: list[tuple[Triple, float]],
        bm25_results: list[tuple[Triple, float]],
    ) -> list[tuple[Triple, float]]:
        """Fuse triple search results."""
        triple_scores: dict[str, dict[str, Any]] = {}

        for rank, (triple, score) in enumerate(semantic_results):
            triple_id = str(triple.id)
            if triple_id not in triple_scores:
                triple_scores[triple_id] = {
                    "triple": triple,
                    "semantic_rank": rank + 1,
                    "bm25_rank": None,
                }
            else:
                triple_scores[triple_id]["semantic_rank"] = rank + 1

        for rank, (triple, score) in enumerate(bm25_results):
            triple_id = str(triple.id)
            if triple_id not in triple_scores:
                triple_scores[triple_id] = {
                    "triple": triple,
                    "semantic_rank": None,
                    "bm25_rank": rank + 1,
                }
            else:
                triple_scores[triple_id]["bm25_rank"] = rank + 1

        k = 60
        fused_results = []

        for triple_id, data in triple_scores.items():
            rrf_score = 0.0
            if data["semantic_rank"] is not None:
                rrf_score += self.semantic_weight / (k + data["semantic_rank"])
            if data["bm25_rank"] is not None:
                rrf_score += self.bm25_weight / (k + data["bm25_rank"])
            fused_results.append((data["triple"], rrf_score))

        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results

    # =========================================================================
    # Context Retrieval
    # =========================================================================

    async def retrieve_context(
        self,
        query: str,
        contract_ids: list[UUID] | None = None,
        max_entities: int = 10,
        max_triples: int = 20,
    ) -> dict[str, Any]:
        """
        Retrieve context for a query (entities + triples).

        Used for query answering - retrieves relevant graph context.
        """
        # Search entities
        entity_results = await self.search_entities(
            query=query,
            contract_ids=contract_ids,
            limit=max_entities,
        )

        # Search triples
        triple_results = await self.search_triples(
            query=query,
            contract_ids=contract_ids,
            limit=max_triples,
        )

        # Format context
        entities = [
            {
                "name": entity.name,
                "type": entity.entity_type.value,
                "score": score,
            }
            for entity, score in entity_results
        ]

        triples = [
            {
                "subject": str(triple.subject_id),
                "predicate": triple.predicate.value,
                "object": str(triple.object_id),
                "score": score,
            }
            for triple, score in triple_results
        ]

        return {
            "entities": entities,
            "triples": triples,
            "query": query,
        }

    async def retrieve_context_formatted(
        self,
        query: str,
        contract_ids: list[UUID] | None = None,
        max_entities: int = 10,
        max_triples: int = 20,
    ) -> str:
        """
        Retrieve and format context as a string for LLM prompts.
        """
        context = await self.retrieve_context(
            query=query,
            contract_ids=contract_ids,
            max_entities=max_entities,
            max_triples=max_triples,
        )

        lines = ["Relevant Entities:"]
        for entity in context["entities"]:
            lines.append(f"  - {entity['name']} ({entity['type']})")

        lines.append("\nRelevant Relationships:")
        for triple in context["triples"]:
            pred_readable = triple["predicate"].replace("_", " ").lower()
            lines.append(f"  - {triple['subject']} {pred_readable} {triple['object']}")

        return "\n".join(lines)

    # =========================================================================
    # Similarity Search
    # =========================================================================

    async def find_similar_entities(
        self,
        entity: Entity,
        limit: int = 10,
        threshold: float = 0.7,
        exclude_self: bool = True,
    ) -> list[tuple[Entity, float]]:
        """
        Find entities similar to a given entity.

        Used for entity resolution and deduplication.
        """
        # Get entity embedding
        embedding = self.embedding_service.embed_entity(
            entity_name=entity.name,
            entity_type=entity.entity_type.value,
        )

        # Search for similar
        results = await self.qdrant.search_entities(
            query_embedding=embedding,
            entity_types=[entity.entity_type.value],
            limit=limit + (1 if exclude_self else 0),
        )

        # Filter
        filtered = []
        for found_entity, score in results:
            if exclude_self and str(found_entity.id) == str(entity.id):
                continue
            if score >= threshold:
                filtered.append((found_entity, score))

        return filtered[:limit]

    async def find_duplicate_candidates(
        self,
        entities: list[Entity],
        similarity_threshold: float = 0.8,
    ) -> list[list[Entity]]:
        """
        Find groups of potentially duplicate entities.

        Returns clusters of entities that may refer to the same thing.
        """
        if not entities:
            return []

        # Generate embeddings for all entities
        embeddings = [
            self.embedding_service.embed_entity(e.name, e.entity_type.value)
            for e in entities
        ]

        # Cluster embeddings
        labels = self.embedding_service.cluster_embeddings(
            embeddings=embeddings,
            n_clusters=min(len(entities) // 2, 128),
        )

        # Group by cluster
        clusters: dict[int, list[Entity]] = {}
        for entity, label in zip(entities, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(entity)

        # Filter to clusters with 2+ entities
        duplicate_groups = [
            group for group in clusters.values()
            if len(group) >= 2
        ]

        logger.info(
            "duplicate_candidates_found",
            total_entities=len(entities),
            groups=len(duplicate_groups),
        )

        return duplicate_groups

    # =========================================================================
    # Indexing
    # =========================================================================

    async def index_entity(
        self,
        entity: Entity,
        contract_id: UUID | None = None,
    ) -> bool:
        """Index an entity for search."""
        # Generate embedding
        embedding = self.embedding_service.embed_entity(
            entity_name=entity.name,
            entity_type=entity.entity_type.value,
        )

        # Store in Qdrant
        await self.qdrant.upsert_entity(
            entity_id=str(entity.id),
            embedding=embedding,
            payload={
                "name": entity.name,
                "entity_type": entity.entity_type.value,
                "normalized_name": entity.normalized_name,
                "contract_id": str(contract_id) if contract_id else None,
            },
        )

        return True

    async def index_triple(
        self,
        triple: Triple,
        subject_name: str,
        object_name: str,
        contract_id: UUID | None = None,
    ) -> bool:
        """Index a triple for search."""
        # Generate embedding
        embedding = self.embedding_service.embed_triple(
            subject=subject_name,
            predicate=triple.predicate.value,
            obj=object_name,
        )

        # Store in Qdrant
        await self.qdrant.upsert_triple(
            triple_id=str(triple.id),
            embedding=embedding,
            payload={
                "subject_id": str(triple.subject_id),
                "predicate": triple.predicate.value,
                "object_id": str(triple.object_id),
                "subject_name": subject_name,
                "object_name": object_name,
                "contract_id": str(contract_id) if contract_id else None,
            },
        )

        return True

    async def index_entities_batch(
        self,
        entities: list[Entity],
        contract_id: UUID | None = None,
    ) -> int:
        """Index multiple entities in batch."""
        count = 0
        for entity in entities:
            if await self.index_entity(entity, contract_id):
                count += 1
        logger.info("entities_indexed", count=count)
        return count


@lru_cache()
def get_search_service() -> SearchService:
    """Get cached search service instance."""
    return SearchService()
