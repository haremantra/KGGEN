"""Tests for src/utils/embedding.py — cache backends, similarity, clustering."""

import pytest
import numpy as np
from src.utils.embedding import InMemoryCacheBackend, EmbeddingService


class TestInMemoryCacheBackend:

    def test_set_and_get(self):
        cache = InMemoryCacheBackend(max_size=10)
        cache.set("key1", [1.0, 2.0, 3.0])
        assert cache.get("key1") == [1.0, 2.0, 3.0]

    def test_miss_returns_none(self):
        cache = InMemoryCacheBackend(max_size=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = InMemoryCacheBackend(max_size=3)
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])
        cache.set("d", [4.0])  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == [2.0]
        assert cache.get("d") == [4.0]

    def test_lru_access_refreshes(self):
        cache = InMemoryCacheBackend(max_size=3)
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])
        cache.get("a")  # refresh "a"
        cache.set("d", [4.0])  # should evict "b" (oldest unreferenced)
        assert cache.get("a") == [1.0]
        assert cache.get("b") is None


class TestEmbeddingServiceMath:
    """Test math functions without loading S-BERT model."""

    def test_compute_similarity_identical(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        v = [1.0, 0.0, 0.0]
        assert abs(svc.compute_similarity(v, v) - 1.0) < 1e-6

    def test_compute_similarity_orthogonal(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(svc.compute_similarity(a, b)) < 1e-6

    def test_compute_similarity_opposite(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(svc.compute_similarity(a, b) - (-1.0)) < 1e-6

    def test_compute_similarity_zero_vector(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert svc.compute_similarity(a, b) == 0.0

    def test_find_similar_ranking(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        results = svc.find_similar(query, candidates, top_k=3)
        assert len(results) == 3
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_similar_top_k(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        results = svc.find_similar(query, candidates, top_k=1)
        assert len(results) == 1

    def test_find_similar_threshold(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0]]  # second is orthogonal
        results = svc.find_similar(query, candidates, threshold=0.5)
        assert len(results) == 1

    def test_find_similar_empty(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        results = svc.find_similar([1.0, 0.0], [], top_k=10)
        assert results == []
