"""Tests for src/portfolio/analyzer.py — portfolio risk, gap analysis, comparison."""

import pytest
from unittest.mock import MagicMock

from src.pipeline import ContractAnalysis, AnalyzedClause, ExtractedValue
from src.risk.assessor import RiskAssessment, RiskFinding
from src.risk.rules import RiskSeverity
from src.portfolio.analyzer import (
    PortfolioAnalyzer, ContractSummary, PortfolioRiskSummary,
    ClauseComparison, ContractComparison, GapAnalysis, PortfolioAnalysis,
    STANDARD_PROTECTIONS,
)


def _make_analysis(contract_id, labels):
    """Helper to create ContractAnalysis with specified labels."""
    clauses = [
        AnalyzedClause(
            text=f"Clause for {label}", cuad_label=label,
            label_confidence=0.85, category="general_information",
        )
        for label in labels
    ]
    return ContractAnalysis(
        contract_id=contract_id, total_clauses=len(clauses),
        analyzed_clauses=clauses, summary={"labels_found": len(clauses)},
    )


def _make_risk(contract_id, score, findings=None, missing=None):
    """Helper to create RiskAssessment."""
    level = "LOW"
    if score >= 75:
        level = "CRITICAL"
    elif score >= 50:
        level = "HIGH"
    elif score >= 25:
        level = "MEDIUM"
    return RiskAssessment(
        contract_id=contract_id, overall_risk_score=score, risk_level=level,
        findings=findings or [], missing_clause_risks=missing or [],
    )


@pytest.fixture
def portfolio_analyzer(monkeypatch):
    """PortfolioAnalyzer with mocked pipeline and risk assessor."""
    # Avoid initializing real pipeline/assessor
    monkeypatch.setattr(
        "src.portfolio.analyzer.ContractAnalysisPipeline.__init__",
        lambda self, *a, **kw: None,
    )
    monkeypatch.setattr(
        "src.portfolio.analyzer.RiskAssessor.__init__",
        lambda self, *a, **kw: None,
    )
    analyzer = PortfolioAnalyzer(use_llm=False)
    analyzer.pipeline = MagicMock()
    analyzer.risk_assessor = MagicMock()
    analyzer._analyses = {}
    analyzer._risk_assessments = {}
    return analyzer


class TestContractSummary:

    def test_creation(self):
        cs = ContractSummary(
            contract_id="c1", risk_score=45, risk_level="MEDIUM",
            labels_found=["License Grant", "Cap On Liability"],
            critical_findings=0, high_findings=2,
            missing_protections=["Source Code Escrow"],
        )
        assert cs.contract_id == "c1"
        assert cs.risk_score == 45

    def test_to_dict(self):
        cs = ContractSummary(
            contract_id="c1", risk_score=45, risk_level="MEDIUM",
            labels_found=["A"], critical_findings=0, high_findings=1,
            missing_protections=[],
        )
        d = cs.to_dict()
        assert d["risk_score"] == 45
        assert isinstance(d["labels_found"], list)


class TestPortfolioRiskSummary:

    def test_average_risk_score_single(self, portfolio_analyzer):
        analysis = _make_analysis("c1", ["License Grant", "Cap On Liability"])
        risk = _make_risk("c1", 40)
        portfolio_analyzer.add_analysis(analysis, risk)
        result = portfolio_analyzer.get_portfolio_analysis()
        assert result.risk_summary.average_risk_score == 40.0

    def test_average_risk_score_multiple(self, portfolio_analyzer):
        for i, (score, labels) in enumerate([
            (30, ["License Grant"]),
            (50, ["Cap On Liability"]),
            (70, ["Non-Compete"]),
        ]):
            analysis = _make_analysis(f"c{i}", labels)
            risk = _make_risk(f"c{i}", score)
            portfolio_analyzer.add_analysis(analysis, risk)

        result = portfolio_analyzer.get_portfolio_analysis()
        assert result.risk_summary.average_risk_score == 50.0

    def test_risk_level_distribution(self, portfolio_analyzer):
        configs = [
            ("c1", 20, ["A"]),   # LOW
            ("c2", 30, ["B"]),   # MEDIUM
            ("c3", 60, ["C"]),   # HIGH
            ("c4", 80, ["D"]),   # CRITICAL
        ]
        for cid, score, labels in configs:
            portfolio_analyzer.add_analysis(_make_analysis(cid, labels), _make_risk(cid, score))

        result = portfolio_analyzer.get_portfolio_analysis()
        levels = result.risk_summary.contracts_by_risk_level
        assert levels.get("LOW", 0) == 1
        assert levels.get("MEDIUM", 0) == 1
        assert levels.get("HIGH", 0) == 1
        assert levels.get("CRITICAL", 0) == 1

    def test_most_common_risk_labels(self, portfolio_analyzer):
        for i in range(3):
            analysis = _make_analysis(f"c{i}", ["License Grant", "Cap On Liability"])
            risk = _make_risk(f"c{i}", 40)
            portfolio_analyzer.add_analysis(analysis, risk)

        result = portfolio_analyzer.get_portfolio_analysis()
        labels = [item[0] for item in result.risk_summary.most_common_risks]
        assert "License Grant" in labels
        assert "Cap On Liability" in labels

    def test_top_highest_risk_contracts(self, portfolio_analyzer):
        for i, score in enumerate([20, 80, 50, 90, 30]):
            portfolio_analyzer.add_analysis(
                _make_analysis(f"c{i}", ["A"]), _make_risk(f"c{i}", score),
            )

        result = portfolio_analyzer.get_portfolio_analysis()
        highest = result.risk_summary.highest_risk_contracts
        assert len(highest) <= 5
        # First should be the highest-scoring contract
        assert highest[0] == "c3"  # score 90


class TestGapAnalysis:

    def test_standard_protections(self):
        assert len(STANDARD_PROTECTIONS) == 10
        assert "Cap On Liability" in STANDARD_PROTECTIONS

    def test_clause_coverage_percentage(self, portfolio_analyzer):
        # 3 contracts: License Grant present in c1+c2, Cap On Liability in c1+c3
        for cid, labels in [
            ("c1", ["License Grant", "Cap On Liability"]),
            ("c2", ["License Grant"]),
            ("c3", ["Cap On Liability"]),
        ]:
            portfolio_analyzer.add_analysis(
                _make_analysis(cid, labels), _make_risk(cid, 40),
            )

        result = portfolio_analyzer.get_portfolio_analysis()
        lg_coverage = result.clause_coverage.get("License Grant")
        if lg_coverage:
            # 2 out of 3 = 66.7%
            assert 60 <= lg_coverage.coverage_percentage <= 70

    def test_gap_analysis_finds_missing(self, portfolio_analyzer):
        # Contract with very few protections
        analysis = _make_analysis("c1", ["License Grant"])
        risk = _make_risk("c1", 60, missing=[
            RiskFinding(label="Cap On Liability", severity=RiskSeverity.CRITICAL,
                        reason="MISSING", recommendation="Add cap"),
        ])
        portfolio_analyzer.add_analysis(analysis, risk)

        result = portfolio_analyzer.get_portfolio_analysis()
        gaps = result.gap_analysis.gap_by_contract["c1"]
        # Should be missing most STANDARD_PROTECTIONS
        assert len(gaps) > 5


class TestContractComparison:

    def test_shared_and_unique_clauses(self, portfolio_analyzer):
        a = _make_analysis("cA", ["License Grant", "Cap On Liability", "Governing Law"])
        b = _make_analysis("cB", ["License Grant", "Non-Compete", "Governing Law"])
        portfolio_analyzer.add_analysis(a, _make_risk("cA", 40))
        portfolio_analyzer.add_analysis(b, _make_risk("cB", 50))

        comparison = portfolio_analyzer.compare_contracts("cA", "cB")
        assert isinstance(comparison, ContractComparison)
        assert "License Grant" in comparison.shared_clauses
        assert "Cap On Liability" in comparison.only_in_a
        assert "Non-Compete" in comparison.only_in_b

    def test_risk_comparison(self, portfolio_analyzer):
        a = _make_analysis("cA", ["A"])
        b = _make_analysis("cB", ["B"])
        portfolio_analyzer.add_analysis(a, _make_risk("cA", 30))
        portfolio_analyzer.add_analysis(b, _make_risk("cB", 70))

        comparison = portfolio_analyzer.compare_contracts("cA", "cB")
        assert comparison.risk_comparison["difference"] == -40  # 30 - 70


class TestSerialization:

    def test_portfolio_analysis_to_dict(self, portfolio_analyzer):
        portfolio_analyzer.add_analysis(
            _make_analysis("c1", ["License Grant"]), _make_risk("c1", 40),
        )
        result = portfolio_analyzer.get_portfolio_analysis()
        d = result.to_dict()
        assert "contracts" in d
        assert "risk_summary" in d
        assert "gap_analysis" in d
