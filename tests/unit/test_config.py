"""Tests for src/config.py — Settings, defaults, thresholds."""

import pytest
from src.config import Settings, settings, get_settings, DEFAULT_AGGREGATION_THRESHOLDS


class TestDefaultAggregationThresholds:

    def test_keys(self):
        expected = {"Party", "Obligation", "Restriction", "IPAsset",
                    "Temporal", "Jurisdiction", "LiabilityProvision", "ContractClause"}
        assert set(DEFAULT_AGGREGATION_THRESHOLDS.keys()) == expected

    def test_values_in_range(self):
        for entity_type, threshold in DEFAULT_AGGREGATION_THRESHOLDS.items():
            assert 0.7 <= threshold <= 1.0, f"{entity_type} threshold {threshold} out of range"

    def test_party_threshold(self):
        assert DEFAULT_AGGREGATION_THRESHOLDS["Party"] == 0.85

    def test_temporal_threshold(self):
        assert DEFAULT_AGGREGATION_THRESHOLDS["Temporal"] == 0.90

    def test_obligation_threshold(self):
        assert DEFAULT_AGGREGATION_THRESHOLDS["Obligation"] == 0.78

    def test_contract_clause_threshold(self):
        assert DEFAULT_AGGREGATION_THRESHOLDS["ContractClause"] == 0.75


class TestSettings:

    def test_get_settings_returns_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_returns_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_embedding_model_default(self):
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_embedding_dimension_default(self):
        assert settings.embedding_dimension == 384

    def test_search_inmemory_warn_threshold_default(self):
        assert settings.search_inmemory_warn_threshold == 50000

    def test_search_inmemory_max_vectors_default(self):
        assert settings.search_inmemory_max_vectors == 100000

    def test_postgres_url_property(self):
        url = settings.postgres_url
        assert "postgresql://" in url

    def test_primary_llm_provider_default(self):
        assert settings.primary_llm_provider == "anthropic"

    def test_fallback_llm_provider_default(self):
        assert settings.fallback_llm_provider == "openai"
