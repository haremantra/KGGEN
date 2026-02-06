"""Tests for src/search/service.py — RRF fusion, BM25, hybrid search."""

import pytest
from unittest.mock import MagicMock, patch

from src.search.service import SearchService


@pytest.fixture
def search_service(mock_embedding_service, monkeypatch):
    """SearchService with mocked embedding and in-memory backend."""
    mock_settings = MagicMock()
    mock_settings.search_bm25_weight = 0.3
    mock_settings.search_semantic_weight = 0.7
    mock_settings.search_inmemory_warn_threshold = 50000
    mock_settings.search_inmemory_max_vectors = 100000
    mock_settings.qdrant_enabled = False
    mock_settings.embedding_dimension = 384
    monkeypatch.setattr("src.search.service.get_settings", lambda: mock_settings)
    monkeypatch.setattr("src.search.backends.get_settings", lambda: mock_settings)

    svc = SearchService()
    return svc


class TestIndexing:

    def test_index_entity(self, search_service):
        search_service.index_entity("e1", "ACME Corp", "Party", {"role": "Licensor"})
        # Uses shared _bm25_corpus for both entities and triples
        assert len(search_service._bm25_corpus) == 1

    def test_index_triple(self, search_service):
        search_service.index_triple("t1", "ACME", "LICENSES_TO", "BETA")
        assert len(search_service._bm25_corpus) == 1

    def test_bm25_invalidated_on_index(self, search_service):
        search_service.index_entity("e1", "ACME Corp", "Party")
        # _bm25_index is set to None when corpus changes (lazy rebuild)
        assert search_service._bm25_index is None


class TestRRFFusion:

    def test_rrf_fuse_combines(self, search_service):
        semantic = [
            ("id1", 0.95, {"name": "A"}),
            ("id2", 0.80, {"name": "B"}),
        ]
        bm25 = [
            ("id2", 1.5, {"name": "B"}),
            ("id3", 1.0, {"name": "C"}),
        ]
        results = search_service._rrf_fuse(semantic, bm25, limit=10, threshold=0.0)
        ids = [r[0]["name"] for r in results]
        # id2 appears in both, should rank high
        assert "B" in ids

    def test_rrf_fuse_empty(self, search_service):
        results = search_service._rrf_fuse([], [], limit=10, threshold=0.0)
        assert results == []

    def test_rrf_fuse_respects_limit(self, search_service):
        semantic = [(f"id{i}", 0.9 - i * 0.1, {"name": f"s{i}"}) for i in range(10)]
        bm25 = []
        results = search_service._rrf_fuse(semantic, bm25, limit=3, threshold=0.0)
        assert len(results) <= 3


class TestHybridSearch:

    def test_search_entities(self, search_service):
        search_service.index_entity("e1", "ACME Corp", "Party")
        search_service.index_entity("e2", "liability cap provision", "LiabilityProvision")
        results = search_service.search_entities("ACME", limit=5)
        assert isinstance(results, list)

    def test_search_triples(self, search_service):
        search_service.index_triple("t1", "ACME", "LICENSES_TO", "BETA")
        results = search_service.search_triples("license", limit=5)
        assert isinstance(results, list)


class TestContextRetrieval:

    def test_retrieve_context(self, search_service):
        search_service.index_entity("e1", "ACME Corp", "Party")
        search_service.index_triple("t1", "ACME", "LICENSES_TO", "BETA")
        ctx = search_service.retrieve_context("ACME", max_entities=5, max_triples=5)
        assert "entities" in ctx
        assert "triples" in ctx

    def test_retrieve_context_formatted(self, search_service):
        search_service.index_entity("e1", "ACME Corp", "Party")
        text = search_service.retrieve_context_formatted("ACME")
        assert isinstance(text, str)
