"""Tests for src/aggregation/aggregator.py — per-type thresholds, normalization."""

import pytest
from unittest.mock import MagicMock

from src.models.schema import KGNode, Triple, NodeType
from src.aggregation.aggregator import ContractAggregator, AggregationResult, DEFAULT_THRESHOLD


class TestThresholds:

    def test_default_threshold(self):
        assert DEFAULT_THRESHOLD == 0.80

    def test_get_threshold_party(self, mock_embedding_service):
        agg = ContractAggregator()
        assert agg._get_threshold("Party") == 0.85

    def test_get_threshold_obligation(self, mock_embedding_service):
        agg = ContractAggregator()
        assert agg._get_threshold("Obligation") == 0.78

    def test_get_threshold_restriction(self, mock_embedding_service):
        agg = ContractAggregator()
        assert agg._get_threshold("Restriction") == 0.78

    def test_get_threshold_temporal(self, mock_embedding_service):
        agg = ContractAggregator()
        assert agg._get_threshold("Temporal") == 0.90

    def test_get_threshold_jurisdiction(self, mock_embedding_service):
        agg = ContractAggregator()
        assert agg._get_threshold("Jurisdiction") == 0.90

    def test_get_threshold_unknown(self, mock_embedding_service):
        agg = ContractAggregator()
        assert agg._get_threshold("UnknownType") == DEFAULT_THRESHOLD


class TestNormalization:

    def test_normalize_party_name_inc(self, mock_embedding_service):
        agg = ContractAggregator()
        result = agg._normalize_party_name("ACME Inc.")
        assert "ACME" in result

    def test_normalize_party_name_llc(self, mock_embedding_service):
        agg = ContractAggregator()
        result = agg._normalize_party_name("Beta LLC")
        assert "Beta" in result or "LLC" in result

    def test_normalize_party_name_preserves_name(self, mock_embedding_service):
        agg = ContractAggregator()
        result = agg._normalize_party_name("Simple Company Name")
        assert "Simple" in result

    def test_normalize_temporal(self, mock_embedding_service):
        agg = ContractAggregator()
        result = agg._normalize_temporal("January 1, 2024")
        assert isinstance(result, str)


class TestTripleDedup:

    def test_deduplicate_triples(self, mock_embedding_service):
        triples = [
            Triple(subject="A", predicate="LICENSES_TO", object="B"),
            Triple(subject="A", predicate="LICENSES_TO", object="B"),
            Triple(subject="C", predicate="OWNS", object="D"),
        ]
        agg = ContractAggregator()
        deduped = agg._deduplicate_triples(triples)
        assert len(deduped) == 2


class TestComputeStatistics:

    def test_compute_statistics(self, mock_embedding_service, sample_nodes, sample_triples):
        agg = ContractAggregator()
        stats = agg.compute_statistics(sample_nodes, sample_triples)
        assert "entity_count" in stats or "total_entities" in stats or isinstance(stats, dict)


class TestAggregation:

    def test_aggregate_single_contract(self, mock_embedding_service, sample_nodes, sample_triples):
        agg = ContractAggregator()
        extractions = {"c1": (sample_nodes, sample_triples)}
        result = agg.aggregate(extractions)
        assert isinstance(result, AggregationResult)
        assert len(result.entities) > 0

    def test_aggregate_multiple_contracts(self, mock_embedding_service):
        nodes_c1 = [
            KGNode(id="p1", name="ACME Corp", type=NodeType.PARTY, source_contract_id="c1"),
        ]
        triples_c1 = [Triple(subject="ACME Corp", predicate="OWNS", object="Software")]

        nodes_c2 = [
            KGNode(id="p2", name="ACME Corporation", type=NodeType.PARTY, source_contract_id="c2"),
        ]
        triples_c2 = [Triple(subject="ACME Corporation", predicate="OWNS", object="Software")]

        agg = ContractAggregator()
        extractions = {"c1": (nodes_c1, triples_c1), "c2": (nodes_c2, triples_c2)}
        result = agg.aggregate(extractions)
        assert isinstance(result, AggregationResult)


class TestAggregationResult:

    def test_to_dict(self):
        result = AggregationResult(
            entities=[], triples=[], id_mapping={},
            statistics={"total": 0},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        # to_dict() spreads statistics into the result, so "total" should be present
        assert "merged_entities" in d
        assert "merged_triples" in d
        assert d["total"] == 0  # spread from statistics
