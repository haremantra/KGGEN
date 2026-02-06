"""Tests for src/query/service.py — RAG pipeline, specialized queries."""

import pytest
from unittest.mock import MagicMock

from src.query.service import QueryService, QueryResponse


@pytest.fixture
def query_service(mock_embedding_service, mock_llm_service, monkeypatch):
    """QueryService with mocked LLM and search."""
    mock_settings = MagicMock()
    mock_settings.search_bm25_weight = 0.3
    mock_settings.search_semantic_weight = 0.7
    mock_settings.search_inmemory_warn_threshold = 50000
    mock_settings.search_inmemory_max_vectors = 100000
    mock_settings.qdrant_enabled = False
    mock_settings.embedding_dimension = 384
    monkeypatch.setattr("src.query.service.get_settings", lambda: mock_settings)
    monkeypatch.setattr("src.search.service.get_settings", lambda: mock_settings)
    monkeypatch.setattr("src.search.backends.get_settings", lambda: mock_settings)

    svc = QueryService()
    # Inject mock LLM and search
    svc._llm = mock_llm_service
    mock_search = MagicMock()
    mock_search.retrieve_context_formatted.return_value = "Entity: ACME Corp (Party)\nTriple: ACME LICENSES_TO BETA"
    mock_search.retrieve_context.return_value = {
        "entities": [{"name": "ACME Corp", "type": "Party"}],
        "triples": [{"subject": "ACME", "predicate": "LICENSES_TO", "object": "BETA"}],
    }
    svc._search = mock_search
    return svc


class TestQueryResponse:

    def test_to_dict(self):
        r = QueryResponse(query="test?", answer="answer", confidence=0.9,
                          sources=[{"s": "A"}], model_used="test-model")
        d = r.to_dict()
        assert d["query"] == "test?"
        assert d["confidence"] == 0.9
        assert d["model_used"] == "test-model"


class TestQuery:

    def test_returns_answer(self, query_service):
        result = query_service.query("What is the license type?")
        assert isinstance(result, QueryResponse)
        assert result.answer != ""

    def test_includes_sources(self, query_service):
        result = query_service.query("What is the license type?")
        assert isinstance(result.sources, list)

    def test_empty_context(self, query_service):
        query_service._search.retrieve_context_formatted.return_value = ""
        query_service._search.retrieve_context.return_value = {"entities": [], "triples": []}
        result = query_service.query("Unknown question?")
        assert isinstance(result, QueryResponse)

    def test_caching(self, query_service):
        r1 = query_service.query("Same question?")
        r2 = query_service.query("Same question?")
        # Second call should use cache (same object)
        assert r1 is r2

    def test_no_cache(self, query_service):
        r1 = query_service.query("Question?", use_cache=False)
        r2 = query_service.query("Question?", use_cache=False)
        # Without cache, should make two calls
        assert query_service._llm.answer_query.call_count >= 2


class TestSpecializedQueries:

    def test_query_licensing(self, query_service):
        result = query_service.query_licensing()
        assert isinstance(result, QueryResponse)

    def test_query_obligations(self, query_service):
        result = query_service.query_obligations()
        assert isinstance(result, QueryResponse)

    def test_query_obligations_with_party(self, query_service):
        result = query_service.query_obligations(party_name="ACME Corp")
        assert isinstance(result, QueryResponse)

    def test_query_restrictions(self, query_service):
        result = query_service.query_restrictions()
        assert isinstance(result, QueryResponse)

    def test_query_liability(self, query_service):
        result = query_service.query_liability()
        assert isinstance(result, QueryResponse)

    def test_query_termination(self, query_service):
        result = query_service.query_termination()
        assert isinstance(result, QueryResponse)

    def test_query_governing_law(self, query_service):
        result = query_service.query_governing_law()
        assert isinstance(result, QueryResponse)


class TestMultiContract:

    def test_compare(self, query_service):
        result = query_service.compare("licensing")
        assert isinstance(result, QueryResponse)

    def test_suggest_queries(self, query_service):
        suggestions = query_service.suggest_queries()
        assert isinstance(suggestions, list)
        assert len(suggestions) == 10
