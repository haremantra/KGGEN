"""Integration tests for FastAPI API endpoints via TestClient."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "kggen-cuad-api"

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestContractEndpoints:

    def test_list_contracts_empty(self, client):
        resp = client.get("/api/contracts")
        assert resp.status_code == 200
        # Should return empty list or similar
        data = resp.json()
        assert isinstance(data, (list, dict))

    def test_contract_not_found(self, client):
        resp = client.get("/api/contracts/nonexistent-999/risks")
        assert resp.status_code in (404, 400, 422)


class TestSearchEndpoints:

    def test_search_entities(self, client, mock_embedding_service, monkeypatch):
        mock_search = MagicMock()
        mock_search.search_entities.return_value = [
            ({"name": "ACME", "type": "Party"}, 0.95),
        ]
        # The import is inside the endpoint, so monkeypatch the source module
        monkeypatch.setattr("src.search.service.get_search_service", lambda: mock_search)

        resp = client.post("/api/search/entities", json={"query": "ACME"})
        assert resp.status_code == 200

    def test_search_triples(self, client, mock_embedding_service, monkeypatch):
        mock_search = MagicMock()
        mock_search.search_triples.return_value = [
            ({"subject": "A", "predicate": "LICENSES_TO", "object": "B"}, 0.9),
        ]
        monkeypatch.setattr("src.search.service.get_search_service", lambda: mock_search)

        resp = client.post("/api/search/triples", json={"query": "license"})
        assert resp.status_code == 200


class TestQueryEndpoints:

    def test_query(self, client, monkeypatch):
        mock_query_svc = MagicMock()
        mock_query_svc.query.return_value = MagicMock(
            to_dict=lambda: {
                "query": "What license?",
                "answer": "Non-exclusive",
                "confidence": 0.9,
                "sources": [],
                "model_used": "test",
            },
        )
        # The import is inside the endpoint, so monkeypatch the source module
        monkeypatch.setattr("src.query.service.get_query_service", lambda: mock_query_svc)

        resp = client.post("/api/query", json={"question": "What license?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
