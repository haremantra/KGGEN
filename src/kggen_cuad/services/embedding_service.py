"""
Embedding service for semantic search and entity resolution.
"""

from functools import lru_cache
from typing import Any

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from kggen_cuad.config import get_settings
from kggen_cuad.storage.redis_cache import get_redis_cache

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.

    Uses sentence-transformers with caching support.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.embedding_model
        self.dimension = self.settings.embedding_dimension
        self._model: SentenceTransformer | None = None
        self._cache = get_redis_cache()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name)
        return self._model

    def embed(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use Redis caching

        Returns:
            List of floats representing the embedding vector
        """
        if not text.strip():
            return [0.0] * self.dimension

        # Check cache
        if use_cache:
            cached = self._cache.get_embedding(text, self.model_name)
            if cached is not None:
                return cached

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding_list = embedding.tolist()

        # Cache result
        if use_cache:
            self._cache.set_embedding(text, self.model_name, embedding_list)

        return embedding_list

    def embed_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use Redis caching
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        if use_cache:
            for i, text in enumerate(texts):
                cached = self._cache.get_embedding(text, self.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(uncached_texts) > 100,
            )

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                emb_list = emb.tolist()
                embeddings.append((idx, emb_list))

                # Cache result
                if use_cache:
                    self._cache.set_embedding(text, self.model_name, emb_list)

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return [e[1] for e in embeddings]

    def embed_entity(self, entity_name: str, entity_type: str) -> list[float]:
        """
        Generate embedding for an entity.

        Combines entity name and type for better disambiguation.
        """
        text = f"{entity_name} ({entity_type})"
        return self.embed(text)

    def embed_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> list[float]:
        """
        Generate embedding for a knowledge graph triple.

        Creates a natural language representation of the triple.
        """
        predicate_text = predicate.replace("_", " ").lower()
        text = f"{subject} {predicate_text} {obj}"
        return self.embed(text)

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Normalize
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def find_similar(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings from candidates.

        Returns list of (index, similarity_score) tuples.
        """
        if not candidate_embeddings:
            return []

        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        # Normalize
        query_norm = query / np.linalg.norm(query)
        candidate_norms = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        # Compute similarities
        similarities = np.dot(candidate_norms, query_norm)

        # Get top-k above threshold
        results = []
        sorted_indices = np.argsort(similarities)[::-1]

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
        """
        Cluster embeddings using k-means.

        Args:
            embeddings: List of embedding vectors
            n_clusters: Number of clusters (default: sqrt(n) or 128)

        Returns:
            List of cluster assignments
        """
        from sklearn.cluster import KMeans

        if not embeddings:
            return []

        n = len(embeddings)
        if n_clusters is None:
            # Default: min(128, sqrt(n))
            n_clusters = min(128, max(2, int(np.sqrt(n))))

        if n_clusters >= n:
            return list(range(n))

        X = np.array(embeddings)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)

        logger.info(
            "embeddings_clustered",
            n_points=n,
            n_clusters=n_clusters,
        )

        return labels.tolist()

    def get_cluster_representatives(
        self,
        embeddings: list[list[float]],
        labels: list[int],
    ) -> dict[int, int]:
        """
        Find the representative embedding for each cluster.

        Returns dict mapping cluster_id -> representative_index.
        """
        from collections import defaultdict

        if not embeddings or not labels:
            return {}

        # Group embeddings by cluster
        clusters: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        representatives = {}
        embeddings_array = np.array(embeddings)

        for cluster_id, indices in clusters.items():
            if len(indices) == 1:
                representatives[cluster_id] = indices[0]
            else:
                # Find centroid
                cluster_embeddings = embeddings_array[indices]
                centroid = cluster_embeddings.mean(axis=0)

                # Find closest to centroid
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                closest_idx = indices[np.argmin(distances)]
                representatives[cluster_id] = closest_idx

        return representatives


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    return EmbeddingService()
