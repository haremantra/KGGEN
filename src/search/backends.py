"""Vector search backends — InMemory (with size guards) and Qdrant (optional)."""

import warnings
from typing import Protocol

import numpy as np

from ..config import get_settings


class VectorBackend(Protocol):
    """Protocol for vector storage backends."""

    def index(self, id: str, embedding: list[float], payload: dict) -> None: ...
    def search(self, query_embedding: list[float], limit: int = 10) -> list[tuple[str, float, dict]]: ...
    def __len__(self) -> int: ...


class InMemoryVectorBackend:
    """In-memory vector search with cosine similarity and size guards.

    Warns at warn_threshold vectors, raises RuntimeError at max_vectors.
    """

    def __init__(self):
        self._settings = get_settings()
        self._vectors: dict[str, tuple[np.ndarray, dict]] = {}
        self._warn_threshold = self._settings.search_inmemory_warn_threshold
        self._max_vectors = self._settings.search_inmemory_max_vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def index(self, id: str, embedding: list[float], payload: dict) -> None:
        """Index a vector with size guard checks."""
        count = len(self._vectors)

        if count >= self._max_vectors and id not in self._vectors:
            raise RuntimeError(
                f"InMemoryVectorBackend exceeded {self._max_vectors} vectors. "
                f"Configure Qdrant (QDRANT_HOST/QDRANT_PORT in .env) for large-scale indexing."
            )

        if count >= self._warn_threshold and id not in self._vectors:
            warnings.warn(
                f"InMemoryVectorBackend has {count} vectors. "
                f"Consider switching to Qdrant for production use.",
                ResourceWarning,
                stacklevel=2,
            )

        self._vectors[id] = (np.array(embedding, dtype=np.float32), payload)

    def search(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[tuple[str, float, dict]]:
        """Search by cosine similarity."""
        if not self._vectors:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []
        query_normalized = query / q_norm

        results = []
        for id, (vec, payload) in self._vectors.items():
            v_norm = np.linalg.norm(vec)
            if v_norm == 0:
                continue
            score = float(np.dot(vec / v_norm, query_normalized))
            results.append((id, score, payload))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def delete(self, id: str) -> None:
        self._vectors.pop(id, None)


class QdrantVectorBackend:
    """Qdrant vector search backend (optional, graceful fallback)."""

    def __init__(self, collection_name: str = "kggen"):
        self._settings = get_settings()
        self._collection = collection_name
        self._client = None
        self._dimension = self._settings.embedding_dimension

        if self._settings.qdrant_enabled:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                self._client = QdrantClient(
                    host=self._settings.qdrant_host,
                    port=self._settings.qdrant_port,
                )
                # Ensure collection exists
                collections = [c.name for c in self._client.get_collections().collections]
                if self._collection not in collections:
                    self._client.create_collection(
                        collection_name=self._collection,
                        vectors_config=VectorParams(
                            size=self._dimension, distance=Distance.COSINE
                        ),
                    )
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def __len__(self) -> int:
        if not self._client:
            return 0
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0

    def index(self, id: str, embedding: list[float], payload: dict) -> None:
        if not self._client:
            return
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=id, vector=embedding, payload=payload)],
        )

    def search(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[tuple[str, float, dict]]:
        if not self._client:
            return []
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=limit,
        )
        return [(str(r.id), r.score, r.payload or {}) for r in results]

    def delete(self, id: str) -> None:
        if not self._client:
            return
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[id]),
        )
