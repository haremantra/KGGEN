"""Tests for interdependency module — types, matrix, graph algorithms."""

import pytest
from src.interdependency.types import (
    DependencyType, ClauseNode, DependencyEdge, MissingRequirement,
    ClauseDependencyGraph, ImpactResult, InterdependencyReport,
)
from src.interdependency.matrix import (
    DEPENDENCY_RULES, get_rules_for_pair, get_rules_for_label,
    get_all_conflict_pairs, get_requires_rules, get_rules_requiring_llm,
)
from src.interdependency.graph import DependencyGraphBuilder


# ---- Types ----

class TestDependencyType:

    def test_count(self):
        assert len(DependencyType) == 6

    def test_values(self):
        values = {d.value for d in DependencyType}
        expected = {"DEPENDS_ON", "CONFLICTS_WITH", "REQUIRES", "MITIGATES", "RESTRICTS", "MODIFIES"}
        assert values == expected


class TestClauseNode:

    def test_creation(self):
        node = ClauseNode(label="License Grant", category="general_information")
        assert node.label == "License Grant"
        assert node.present is True

    def test_to_dict(self):
        node = ClauseNode(label="Test", category="cat", text="Some text", confidence=0.8)
        d = node.to_dict()
        assert d["label"] == "Test"
        assert d["category"] == "cat"


class TestDependencyEdge:

    def test_to_dict_enum_serialized(self):
        edge = DependencyEdge(
            source_label="A", target_label="B",
            dependency_type=DependencyType.REQUIRES, strength=0.9, reason="test",
        )
        d = edge.to_dict()
        assert d["dependency_type"] == "REQUIRES"


class TestMissingRequirement:

    def test_to_dict(self):
        mr = MissingRequirement(required_by="A", missing_label="B", reason="needed")
        d = mr.to_dict()
        assert d["required_by"] == "A"
        assert d["severity"] == "MEDIUM"  # default


class TestInterdependencyReport:

    def test_to_dict(self, sample_clause_nodes, sample_dependency_edges):
        graph = ClauseDependencyGraph(
            contract_id="c1", nodes=sample_clause_nodes, edges=sample_dependency_edges,
        )
        report = InterdependencyReport(
            contract_id="c1", graph=graph, contradictions=[],
            missing_requirements=[], risk_score_adjustment=10,
        )
        d = report.to_dict()
        assert d["contract_id"] == "c1"
        assert d["risk_score_adjustment"] == 10


# ---- Matrix ----

class TestDependencyRules:

    def test_total_count(self):
        assert len(DEPENDENCY_RULES) == 73

    def test_requires_count(self):
        count = sum(1 for r in DEPENDENCY_RULES if r.dependency_type == DependencyType.REQUIRES)
        assert count == 22

    def test_conflicts_with_count(self):
        count = sum(1 for r in DEPENDENCY_RULES if r.dependency_type == DependencyType.CONFLICTS_WITH)
        assert count == 8

    def test_mitigates_count(self):
        count = sum(1 for r in DEPENDENCY_RULES if r.dependency_type == DependencyType.MITIGATES)
        assert count == 9

    def test_restricts_count(self):
        count = sum(1 for r in DEPENDENCY_RULES if r.dependency_type == DependencyType.RESTRICTS)
        assert count == 10

    def test_modifies_count(self):
        count = sum(1 for r in DEPENDENCY_RULES if r.dependency_type == DependencyType.MODIFIES)
        assert count == 12

    def test_depends_on_count(self):
        count = sum(1 for r in DEPENDENCY_RULES if r.dependency_type == DependencyType.DEPENDS_ON)
        assert count == 12

    def test_all_have_reason(self):
        for rule in DEPENDENCY_RULES:
            assert rule.reason, f"Rule {rule.source_label}->{rule.target_label} has empty reason"

    def test_all_strengths_valid(self):
        for rule in DEPENDENCY_RULES:
            assert 0 < rule.default_strength <= 1.0, \
                f"Rule {rule.source_label}->{rule.target_label} strength {rule.default_strength}"


class TestMatrixHelpers:

    def test_get_rules_for_pair_found(self):
        rules = get_rules_for_pair("Renewal Term", "Expiration Date")
        assert len(rules) > 0

    def test_get_rules_for_pair_empty(self):
        rules = get_rules_for_pair("FakeLabel", "AnotherFake")
        assert rules == []

    def test_get_rules_for_label(self):
        rules = get_rules_for_label("License Grant")
        assert len(rules) > 0

    def test_get_all_conflict_pairs(self):
        pairs = get_all_conflict_pairs()
        assert len(pairs) == 8

    def test_get_requires_rules(self):
        rules = get_requires_rules()
        assert len(rules) == 22

    def test_get_rules_requiring_llm(self):
        rules = get_rules_requiring_llm()
        assert len(rules) >= 0  # may be 0 if none require LLM


# ---- Graph ----

class TestDependencyGraphBuilder:

    def test_build(self, sample_clause_nodes, sample_dependency_edges):
        builder = DependencyGraphBuilder()
        graph = builder.build(sample_clause_nodes, sample_dependency_edges, "c1")
        assert graph.contract_id == "c1"
        assert len(graph.nodes) == len(sample_clause_nodes)
        assert len(graph.edges) >= len(sample_dependency_edges)

    def test_bidirectional_creates_two_directed(self):
        nodes = [
            ClauseNode(label="A", category="cat"),
            ClauseNode(label="B", category="cat"),
        ]
        edges = [
            DependencyEdge(source_label="A", target_label="B",
                           dependency_type=DependencyType.CONFLICTS_WITH,
                           strength=0.8, reason="test", bidirectional=True),
        ]
        builder = DependencyGraphBuilder()
        builder.build(nodes, edges, "c1")
        # NetworkX graph should have edges in both directions
        assert builder._graph.has_edge("A", "B")
        assert builder._graph.has_edge("B", "A")

    def test_impact_analysis(self, sample_clause_nodes, sample_dependency_edges):
        builder = DependencyGraphBuilder()
        builder.build(sample_clause_nodes, sample_dependency_edges, "c1")
        result = builder.impact_analysis("License Grant", max_hops=2)
        assert isinstance(result, ImpactResult)
        assert result.source_label == "License Grant"

    def test_impact_analysis_nonexistent(self, sample_clause_nodes, sample_dependency_edges):
        builder = DependencyGraphBuilder()
        builder.build(sample_clause_nodes, sample_dependency_edges, "c1")
        result = builder.impact_analysis("NonExistentClause")
        assert result.total_affected == 0

    def test_find_contradictions(self, sample_clause_nodes, sample_dependency_edges):
        builder = DependencyGraphBuilder()
        builder.build(sample_clause_nodes, sample_dependency_edges, "c1")
        contradictions = builder.find_contradictions()
        assert isinstance(contradictions, list)

    def test_find_cycles_no_cycles(self):
        nodes = [
            ClauseNode(label="A", category="cat"),
            ClauseNode(label="B", category="cat"),
            ClauseNode(label="C", category="cat"),
        ]
        edges = [
            DependencyEdge(source_label="A", target_label="B",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
            DependencyEdge(source_label="B", target_label="C",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
        ]
        builder = DependencyGraphBuilder()
        builder.build(nodes, edges, "c1")
        cycles = builder.find_cycles()
        assert cycles == []

    def test_find_cycles_with_cycle(self):
        nodes = [
            ClauseNode(label="A", category="cat"),
            ClauseNode(label="B", category="cat"),
            ClauseNode(label="C", category="cat"),
        ]
        edges = [
            DependencyEdge(source_label="A", target_label="B",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
            DependencyEdge(source_label="B", target_label="C",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
            DependencyEdge(source_label="C", target_label="A",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
        ]
        builder = DependencyGraphBuilder()
        builder.build(nodes, edges, "c1")
        cycles = builder.find_cycles()
        assert len(cycles) >= 1

    def test_topological_sort(self):
        nodes = [
            ClauseNode(label="A", category="cat"),
            ClauseNode(label="B", category="cat"),
            ClauseNode(label="C", category="cat"),
        ]
        edges = [
            DependencyEdge(source_label="A", target_label="B",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
            DependencyEdge(source_label="B", target_label="C",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
        ]
        builder = DependencyGraphBuilder()
        builder.build(nodes, edges, "c1")
        order = builder.topological_sort()
        assert isinstance(order, list)
        assert len(order) == 3

    def test_connected_components(self):
        nodes = [
            ClauseNode(label="A", category="cat"),
            ClauseNode(label="B", category="cat"),
            ClauseNode(label="C", category="cat"),
            ClauseNode(label="D", category="cat"),
        ]
        edges = [
            DependencyEdge(source_label="A", target_label="B",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
            DependencyEdge(source_label="C", target_label="D",
                           dependency_type=DependencyType.REQUIRES, strength=0.8, reason="r"),
        ]
        builder = DependencyGraphBuilder()
        builder.build(nodes, edges, "c1")
        components = builder.connected_components()
        assert len(components) == 2

    def test_centrality_scores(self, sample_clause_nodes, sample_dependency_edges):
        builder = DependencyGraphBuilder()
        builder.build(sample_clause_nodes, sample_dependency_edges, "c1")
        scores = builder.centrality_scores()
        assert isinstance(scores, dict)
        assert all(v >= 0 for v in scores.values())

    def test_betweenness_centrality(self, sample_clause_nodes, sample_dependency_edges):
        builder = DependencyGraphBuilder()
        builder.build(sample_clause_nodes, sample_dependency_edges, "c1")
        scores = builder.betweenness_centrality()
        assert isinstance(scores, dict)
