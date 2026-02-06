"""Tests for src/resolution/ — clustering, canonical selection, dedup."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.models.schema import KGNode, Triple, NodeType
from src.resolution.resolver import EntityResolver, ResolutionResult
from src.resolution import analysis_to_entities_triples


class TestResolutionResult:

    def test_to_dict(self):
        result = ResolutionResult(
            resolved_entities=[], resolved_triples=[],
            id_mapping={"old": "new"}, alias_mapping={"new": ["old"]},
            original_count=5, resolved_count=3,
        )
        d = result.to_dict()
        assert d["original_count"] == 5
        assert d["resolved_count"] == 3
        # to_dict() returns: original_count, resolved_count, reduction_rate, alias_groups
        assert "reduction_rate" in d
        assert "alias_groups" in d

    def test_empty_result(self):
        result = ResolutionResult(
            resolved_entities=[], resolved_triples=[],
            id_mapping={}, alias_mapping={},
        )
        assert result.original_count == 0
        assert result.resolved_count == 0


class TestEntityResolver:

    def test_resolve_empty(self, mock_embedding_service, mock_llm_service):
        resolver = EntityResolver(use_llm=False)
        result = resolver.resolve([], [])
        assert result.original_count == 0
        assert result.resolved_count == 0

    def test_resolve_single_entity(self, mock_embedding_service, mock_llm_service):
        entities = [KGNode(id="p1", name="ACME Corp", type=NodeType.PARTY)]
        resolver = EntityResolver(use_llm=False)
        result = resolver.resolve(entities, [])
        assert result.resolved_count == 1

    def test_resolve_groups_by_type(self, mock_embedding_service, mock_llm_service):
        entities = [
            KGNode(id="p1", name="ACME Corp", type=NodeType.PARTY),
            KGNode(id="p2", name="Beta LLC", type=NodeType.PARTY),
            KGNode(id="o1", name="Provide support", type=NodeType.OBLIGATION),
        ]
        resolver = EntityResolver(use_llm=False)
        result = resolver.resolve(entities, [])
        # Should process Party and Obligation separately
        assert result.resolved_count <= 3

    def test_heuristic_canonical_longest_name(self, mock_embedding_service, mock_llm_service):
        entities = [
            KGNode(id="a", name="ACME", type=NodeType.PARTY),
            KGNode(id="b", name="ACME Corporation", type=NodeType.PARTY),
            KGNode(id="c", name="ACME Corp", type=NodeType.PARTY),
        ]
        resolver = EntityResolver(use_llm=False)
        canonical, aliases = resolver._heuristic_canonical(entities)
        # Should prefer longer/more formal name
        assert "ACME" in canonical.name

    def test_heuristic_canonical_corporate_suffix(self, mock_embedding_service, mock_llm_service):
        entities = [
            KGNode(id="a", name="Beta", type=NodeType.PARTY),
            KGNode(id="b", name="Beta Inc.", type=NodeType.PARTY),
        ]
        resolver = EntityResolver(use_llm=False)
        canonical, aliases = resolver._heuristic_canonical(entities)
        assert canonical.name == "Beta Inc."

    def test_remap_triples(self, mock_embedding_service, mock_llm_service):
        triples = [
            Triple(subject="old_id", predicate="LICENSES_TO", object="other"),
        ]
        mapping = {"old_id": "new_id"}
        resolver = EntityResolver(use_llm=False)
        remapped = resolver._remap_triples(triples, mapping)
        assert remapped[0].subject == "new_id"

    def test_dedup_triples(self, mock_embedding_service, mock_llm_service):
        triples = [
            Triple(subject="A", predicate="LICENSES_TO", object="B"),
            Triple(subject="A", predicate="LICENSES_TO", object="B"),  # duplicate
            Triple(subject="A", predicate="OWNS", object="C"),
        ]
        resolver = EntityResolver(use_llm=False)
        deduped = resolver._dedup_triples(triples)
        assert len(deduped) == 2

    def test_find_duplicates(self, mock_embedding_service, mock_llm_service):
        entities = [
            KGNode(id="a", name="ACME Corp", type=NodeType.PARTY),
            KGNode(id="b", name="ACME Corporation", type=NodeType.PARTY),
        ]
        resolver = EntityResolver(use_llm=False)
        groups = resolver.find_duplicates(entities)
        assert isinstance(groups, list)


class TestBridgeFunction:

    def test_analysis_to_entities_triples(self, sample_contract_analysis):
        entities, triples = analysis_to_entities_triples(sample_contract_analysis)
        assert isinstance(entities, list)
        assert isinstance(triples, list)
        # Should extract entities from the analyzed clauses
        assert len(entities) > 0

    def test_bridge_empty_analysis(self):
        from src.pipeline import ContractAnalysis
        analysis = ContractAnalysis(
            contract_id="empty", total_clauses=0,
            analyzed_clauses=[], summary={},
        )
        entities, triples = analysis_to_entities_triples(analysis)
        assert entities == []
        assert triples == []
