"""Tests for src/search/backends.py — InMemoryVectorBackend with size guards."""

import warnings
import pytest
from unittest.mock import MagicMock

from src.search.backends import InMemoryVectorBackend


@pytest.fixture
def backend(monkeypatch):
    """InMemoryVectorBackend with default thresholds."""
    mock_settings = MagicMock()
    mock_settings.search_inmemory_warn_threshold = 50000
    mock_settings.search_inmemory_max_vectors = 100000
    monkeypatch.setattr("src.search.backends.get_settings", lambda: mock_settings)
    return InMemoryVectorBackend()


@pytest.fixture
def small_backend(monkeypatch):
    """InMemoryVectorBackend with low thresholds for testing guards."""
    mock_settings = MagicMock()
    mock_settings.search_inmemory_warn_threshold = 5
    mock_settings.search_inmemory_max_vectors = 10
    monkeypatch.setattr("src.search.backends.get_settings", lambda: mock_settings)
    return InMemoryVectorBackend()


class TestIndexAndSearch:

    def test_index_and_len(self, backend):
        for i in range(5):
            backend.index(f"id{i}", [float(i), 0.0, 0.0], {"name": f"item{i}"})
        assert len(backend) == 5

    def test_search_cosine_ranking(self, backend):
        backend.index("a", [1.0, 0.0, 0.0], {"name": "exact"})
        backend.index("b", [0.5, 0.5, 0.0], {"name": "partial"})
        backend.index("c", [0.0, 1.0, 0.0], {"name": "orthogonal"})

        results = backend.search([1.0, 0.0, 0.0], limit=3)
        assert results[0][0] == "a"  # exact match first
        assert results[0][1] > results[1][1]  # higher score

    def test_search_limit(self, backend):
        for i in range(10):
            backend.index(f"id{i}", [float(i), 1.0, 0.0], {"i": i})
        results = backend.search([1.0, 0.0, 0.0], limit=3)
        assert len(results) == 3

    def test_search_empty_index(self, backend):
        assert backend.search([1.0, 0.0], limit=5) == []

    def test_search_zero_query(self, backend):
        backend.index("a", [1.0, 0.0], {"name": "test"})
        assert backend.search([0.0, 0.0], limit=5) == []

    def test_payload_preserved(self, backend):
        backend.index("a", [1.0, 0.0], {"name": "test", "type": "entity"})
        results = backend.search([1.0, 0.0], limit=1)
        assert results[0][2]["name"] == "test"
        assert results[0][2]["type"] == "entity"


class TestDelete:

    def test_delete_existing(self, backend):
        backend.index("a", [1.0, 0.0], {"name": "test"})
        assert len(backend) == 1
        backend.delete("a")
        assert len(backend) == 0

    def test_delete_nonexistent(self, backend):
        backend.delete("nonexistent")  # should not raise

    def test_update_existing_id(self, backend):
        backend.index("a", [1.0, 0.0], {"v": 1})
        backend.index("a", [0.0, 1.0], {"v": 2})
        assert len(backend) == 1
        results = backend.search([0.0, 1.0], limit=1)
        assert results[0][2]["v"] == 2


class TestSizeGuards:

    def test_warn_threshold(self, small_backend):
        for i in range(5):
            small_backend.index(f"id{i}", [float(i)], {})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            small_backend.index("id5", [5.0], {})
            assert len(w) == 1
            assert issubclass(w[0].category, ResourceWarning)
            assert "Consider switching to Qdrant" in str(w[0].message)

    def test_max_vectors_error(self, small_backend):
        for i in range(10):
            small_backend.index(f"id{i}", [float(i)], {})

        with pytest.raises(RuntimeError, match="exceeded"):
            small_backend.index("id10", [10.0], {})

    def test_update_existing_no_warn(self, small_backend):
        for i in range(8):
            small_backend.index(f"id{i}", [float(i)], {})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            small_backend.index("id0", [99.0], {})  # update existing
            # Should not warn since it's an update, not a new vector
            warns = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(warns) == 0
