"""Embedding service for semantic search and entity resolution.

Provides S-BERT embeddings with caching support (in-memory or Redis).
"""

import hashlib
from collections import OrderedDict
from functools import lru_cache
from typing import Protocol

import numpy as np

from ..config import get_settings


class CacheBackend(Protocol):
    """Protocol for embedding cache backends."""

    def get(self, key: str) -> list[float] | None: ...
    def set(self, key: str, value: list[float]) -> None: ...


class InMemoryCacheBackend:
    """LRU in-memory cache for embeddings (max 10k entries)."""

    def __init__(self, max_size: int = 10_000):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> list[float] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: list[float]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value


class RedisCacheBackend:
    """Redis cache backend with graceful fallback to no-op."""

    def __init__(self, redis_url: str, prefix: str = "emb:"):
        self._prefix = prefix
        self._client = None
        try:
            import redis
            self._client = redis.from_url(redis_url, decode_responses=False)
            self._client.ping()
        except Exception:
            self._client = None

    def _key(self, key: str) -> str:
        return self._prefix + key

    def get(self, key: str) -> list[float] | None:
        if self._client is None:
            return None
        try:
            data = self._client.get(self._key(key))
            if data is not None:
                return np.frombuffer(data, dtype=np.float32).tolist()
        except Exception:
            pass
        return None

    def set(self, key: str, value: list[float]) -> None:
        if self._client is None:
            return
        try:
            arr = np.array(value, dtype=np.float32)
            self._client.set(self._key(key), arr.tobytes(), ex=86400)
        except Exception:
            pass


class EmbeddingService:
    """S-BERT embedding service with lazy model loading and caching."""

    def __init__(self):
        self._settings = get_settings()
        self._model = None
        self._cache: CacheBackend = self._init_cache()

    def _init_cache(self) -> CacheBackend:
        if self._settings.redis_enabled:
            backend = RedisCacheBackend(self._settings.redis_url)
            if backend._client is not None:
                return backend
        return InMemoryCacheBackend()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self._settings.embedding_model}")
            self._model = SentenceTransformer(self._settings.embedding_model)
            print("Embedding model loaded.")
        return self._model

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(
            f"{self._settings.embedding_model}:{text}".encode()
        ).hexdigest()

    def embed(self, text: str, use_cache: bool = True) -> list[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            return [0.0] * self._settings.embedding_dimension

        key = self._cache_key(text)
        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        embedding = self.model.encode(text, convert_to_numpy=True)
        result = embedding.tolist()

        if use_cache:
            self._cache.set(key, result)

        return result

    def embed_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        results: list[tuple[int, list[float]]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        if use_cache:
            for i, text in enumerate(texts):
                key = self._cache_key(text)
                cached = self._cache.get(key)
                if cached is not None:
                    results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = list(texts)
            uncached_indices = list(range(len(texts)))

        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(uncached_texts) > 100,
            )
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                emb_list = emb.tolist()
                results.append((idx, emb_list))
                if use_cache:
                    self._cache.set(self._cache_key(text), emb_list)

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def embed_entity(self, entity_name: str, entity_type: str) -> list[float]:
        """Generate embedding for an entity (name + type)."""
        return self.embed(f"{entity_name} ({entity_type})")

    def embed_triple(self, subject: str, predicate: str, obj: str) -> list[float]:
        """Generate embedding for a knowledge graph triple."""
        pred_text = predicate.replace("_", " ").lower()
        return self.embed(f"{subject} {pred_text} {obj}")

    def compute_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def find_similar(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Find most similar embeddings. Returns (index, score) tuples."""
        if not candidate_embeddings:
            return []

        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []
        query_normalized = query / q_norm

        c_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        c_norms = np.where(c_norms == 0, 1, c_norms)
        candidates_normalized = candidates / c_norms

        similarities = np.dot(candidates_normalized, query_normalized)
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices[:top_k]:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((int(idx), score))
        return results

    def cluster_embeddings(
        self,
        embeddings: list[list[float]],
        n_clusters: int | None = None,
    ) -> list[int]:
        """Cluster embeddings using k-means."""
        from sklearn.cluster import KMeans

        if not embeddings:
            return []

        n = len(embeddings)
        if n_clusters is None:
            n_clusters = min(128, max(2, int(np.sqrt(n))))

        if n_clusters >= n:
            return list(range(n))

        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        return labels.tolist()


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service singleton."""
    return EmbeddingService()
