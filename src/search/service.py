"""Hybrid search service — BM25 + semantic with Reciprocal Rank Fusion."""

from functools import lru_cache
from typing import Any

from ..config import get_settings
from ..utils.embedding import get_embedding_service
from .backends import InMemoryVectorBackend, QdrantVectorBackend


class SearchService:
    """Hybrid search combining BM25 keyword matching and semantic vector search."""

    def __init__(self):
        self._settings = get_settings()
        self._embedding = None
        self._vector_backend = None
        self._bm25_corpus: list[tuple[str, dict]] = []  # (text, payload) pairs
        self._bm25_index = None

        self.bm25_weight = self._settings.search_bm25_weight
        self.semantic_weight = self._settings.search_semantic_weight
        self.retrieval_k = 16

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = get_embedding_service()
        return self._embedding

    @property
    def vector_backend(self):
        if self._vector_backend is None:
            if self._settings.qdrant_enabled:
                qdrant = QdrantVectorBackend()
                if qdrant.available:
                    self._vector_backend = qdrant
                else:
                    self._vector_backend = InMemoryVectorBackend()
            else:
                self._vector_backend = InMemoryVectorBackend()
        return self._vector_backend

    def _rebuild_bm25(self):
        """Rebuild BM25 index from corpus."""
        if not self._bm25_corpus:
            self._bm25_index = None
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [text.lower().split() for text, _ in self._bm25_corpus]
            self._bm25_index = BM25Okapi(tokenized)
        except ImportError:
            self._bm25_index = None

    def index_entity(self, entity_id: str, name: str, entity_type: str, payload: dict | None = None) -> None:
        """Index an entity for search."""
        text = f"{name} ({entity_type})"
        emb = self.embedding.embed_entity(name, entity_type)
        full_payload = {"name": name, "entity_type": entity_type, "kind": "entity", **(payload or {})}

        self.vector_backend.index(entity_id, emb, full_payload)
        self._bm25_corpus.append((text, {"id": entity_id, **full_payload}))
        self._bm25_index = None  # invalidate

    def index_triple(
        self, triple_id: str, subject: str, predicate: str, obj: str, payload: dict | None = None
    ) -> None:
        """Index a triple for search."""
        pred_text = predicate.replace("_", " ").lower()
        text = f"{subject} {pred_text} {obj}"
        emb = self.embedding.embed_triple(subject, predicate, obj)
        full_payload = {
            "subject": subject, "predicate": predicate, "object": obj,
            "kind": "triple", **(payload or {}),
        }

        self.vector_backend.index(triple_id, emb, full_payload)
        self._bm25_corpus.append((text, {"id": triple_id, **full_payload}))
        self._bm25_index = None

    def search_entities(
        self, query: str, limit: int = 10, threshold: float = 0.0
    ) -> list[tuple[dict, float]]:
        """Search entities using hybrid retrieval. Returns (payload, score)."""
        return self._hybrid_search(query, kind="entity", limit=limit, threshold=threshold)

    def search_triples(
        self, query: str, limit: int = 10, threshold: float = 0.0
    ) -> list[tuple[dict, float]]:
        """Search triples using hybrid retrieval."""
        return self._hybrid_search(query, kind="triple", limit=limit, threshold=threshold)

    def _hybrid_search(
        self, query: str, kind: str | None = None, limit: int = 10, threshold: float = 0.0
    ) -> list[tuple[dict, float]]:
        """Run hybrid BM25 + semantic search with RRF fusion."""
        # Semantic search
        query_emb = self.embedding.embed(query)
        semantic_raw = self.vector_backend.search(query_emb, limit=self.retrieval_k * 2)

        # Filter by kind if specified
        semantic_results = []
        for id, score, payload in semantic_raw:
            if kind and payload.get("kind") != kind:
                continue
            semantic_results.append((id, score, payload))
        semantic_results = semantic_results[:self.retrieval_k]

        # BM25 search
        bm25_results = self._bm25_search(query, kind=kind, limit=self.retrieval_k)

        # Fuse via RRF
        return self._rrf_fuse(semantic_results, bm25_results, limit, threshold)

    def _bm25_search(
        self, query: str, kind: str | None = None, limit: int = 16
    ) -> list[tuple[str, float, dict]]:
        """BM25 keyword search."""
        if self._bm25_index is None:
            self._rebuild_bm25()
        if self._bm25_index is None or not self._bm25_corpus:
            return []

        tokens = query.lower().split()
        scores = self._bm25_index.get_scores(tokens)

        scored = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            _, payload = self._bm25_corpus[i]
            if kind and payload.get("kind") != kind:
                continue
            scored.append((payload.get("id", ""), float(score), payload))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _rrf_fuse(
        self,
        semantic: list[tuple[str, float, dict]],
        bm25: list[tuple[str, float, dict]],
        limit: int,
        threshold: float,
    ) -> list[tuple[dict, float]]:
        """Reciprocal Rank Fusion."""
        k = 60  # RRF constant
        scores: dict[str, dict[str, Any]] = {}

        for rank, (id, score, payload) in enumerate(semantic):
            scores[id] = {
                "payload": payload,
                "semantic_rank": rank + 1,
                "bm25_rank": None,
            }

        for rank, (id, score, payload) in enumerate(bm25):
            if id in scores:
                scores[id]["bm25_rank"] = rank + 1
            else:
                scores[id] = {
                    "payload": payload,
                    "semantic_rank": None,
                    "bm25_rank": rank + 1,
                }

        fused = []
        for id, data in scores.items():
            rrf_score = 0.0
            if data["semantic_rank"] is not None:
                rrf_score += self.semantic_weight / (k + data["semantic_rank"])
            if data["bm25_rank"] is not None:
                rrf_score += self.bm25_weight / (k + data["bm25_rank"])

            if rrf_score >= threshold:
                fused.append((data["payload"], rrf_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:limit]

    def retrieve_context(
        self, query: str, max_entities: int = 10, max_triples: int = 20
    ) -> dict[str, Any]:
        """Retrieve KG context for a query (entities + triples)."""
        entities = self.search_entities(query, limit=max_entities)
        triples = self.search_triples(query, limit=max_triples)

        return {
            "entities": [
                {"name": p.get("name", ""), "type": p.get("entity_type", ""), "score": s}
                for p, s in entities
            ],
            "triples": [
                {"subject": p.get("subject", ""), "predicate": p.get("predicate", ""),
                 "object": p.get("object", ""), "score": s}
                for p, s in triples
            ],
            "query": query,
        }

    def retrieve_context_formatted(
        self, query: str, max_entities: int = 10, max_triples: int = 20
    ) -> str:
        """Retrieve and format context as string for LLM prompts."""
        ctx = self.retrieve_context(query, max_entities, max_triples)

        lines = ["Relevant Entities:"]
        for e in ctx["entities"]:
            lines.append(f"  - {e['name']} ({e['type']})")

        lines.append("\nRelevant Relationships:")
        for t in ctx["triples"]:
            pred = t["predicate"].replace("_", " ").lower()
            lines.append(f"  - {t['subject']} {pred} {t['object']}")

        return "\n".join(lines)


@lru_cache()
def get_search_service() -> SearchService:
    """Get cached search service singleton."""
    return SearchService()
