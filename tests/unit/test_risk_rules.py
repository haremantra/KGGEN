"""Tests for src/risk/rules.py and src/risk/assessor.py — Risk scoring logic."""

import pytest
from src.risk.rules import (
    RiskSeverity, RiskRule, RISK_RULES, MISSING_CLAUSE_RISKS,
    get_rule_for_label, get_missing_clause_rule,
    get_high_risk_labels, get_labels_requiring_llm,
)
from src.risk.assessor import (
    RiskAssessor, RiskAssessment, RiskFinding, SEVERITY_WEIGHTS,
)


class TestRiskSeverityEnum:

    def test_count(self):
        assert len(RiskSeverity) == 5

    def test_values(self):
        values = {s.value for s in RiskSeverity}
        assert values == {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}


class TestRiskRules:

    def test_count(self):
        assert len(RISK_RULES) == 40

    def test_all_have_reason(self):
        for label, rule in RISK_RULES.items():
            assert rule.reason, f"{label} has empty reason"

    def test_all_have_recommendation(self):
        for label, rule in RISK_RULES.items():
            assert rule.recommendation, f"{label} has empty recommendation"

    def test_all_have_valid_severity(self):
        for label, rule in RISK_RULES.items():
            assert isinstance(rule.severity, RiskSeverity), f"{label} has invalid severity"


class TestMissingClauseRisks:

    def test_count(self):
        assert len(MISSING_CLAUSE_RISKS) == 8

    def test_cap_on_liability_is_critical(self):
        rule = MISSING_CLAUSE_RISKS["Cap On Liability"]
        assert rule.severity == RiskSeverity.CRITICAL


class TestHelperFunctions:

    def test_get_rule_for_label_exists(self):
        rule = get_rule_for_label("Exclusivity")
        assert rule is not None
        assert rule.severity == RiskSeverity.CRITICAL

    def test_get_rule_for_label_missing(self):
        assert get_rule_for_label("FakeLabel") is None

    def test_get_missing_clause_rule_exists(self):
        rule = get_missing_clause_rule("Cap On Liability")
        assert rule is not None
        assert rule.severity == RiskSeverity.CRITICAL

    def test_get_missing_clause_rule_missing(self):
        assert get_missing_clause_rule("FakeLabel") is None

    def test_get_high_risk_labels(self):
        labels = get_high_risk_labels()
        assert len(labels) > 0
        for label in labels:
            rule = RISK_RULES[label]
            assert rule.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)

    def test_get_labels_requiring_llm(self):
        labels = get_labels_requiring_llm()
        assert len(labels) == 10
        for label in labels:
            assert RISK_RULES[label].requires_llm is True


class TestSeverityWeights:

    def test_critical(self):
        assert SEVERITY_WEIGHTS[RiskSeverity.CRITICAL] == 25

    def test_high(self):
        assert SEVERITY_WEIGHTS[RiskSeverity.HIGH] == 15

    def test_medium(self):
        assert SEVERITY_WEIGHTS[RiskSeverity.MEDIUM] == 8

    def test_low(self):
        assert SEVERITY_WEIGHTS[RiskSeverity.LOW] == 3

    def test_info(self):
        assert SEVERITY_WEIGHTS[RiskSeverity.INFO] == 0


class TestRiskAssessor:

    def test_score_to_level_low(self):
        assessor = RiskAssessor(use_llm=False)
        assert assessor._score_to_level(0) == "LOW"
        assert assessor._score_to_level(24) == "LOW"

    def test_score_to_level_medium(self):
        assessor = RiskAssessor(use_llm=False)
        assert assessor._score_to_level(25) == "MEDIUM"
        assert assessor._score_to_level(49) == "MEDIUM"

    def test_score_to_level_high(self):
        assessor = RiskAssessor(use_llm=False)
        assert assessor._score_to_level(50) == "HIGH"
        assert assessor._score_to_level(74) == "HIGH"

    def test_score_to_level_critical(self):
        assessor = RiskAssessor(use_llm=False)
        assert assessor._score_to_level(75) == "CRITICAL"
        assert assessor._score_to_level(100) == "CRITICAL"

    def test_calculate_risk_score_empty(self):
        assessor = RiskAssessor(use_llm=False)
        assert assessor._calculate_risk_score([], []) == 0

    def test_calculate_risk_score_capped_at_100(self):
        assessor = RiskAssessor(use_llm=False)
        many_findings = [
            RiskFinding(label=f"l{i}", severity=RiskSeverity.CRITICAL,
                        reason="r", recommendation="rec", confidence=1.0)
            for i in range(50)
        ]
        score = assessor._calculate_risk_score(many_findings, [])
        assert score <= 100

    def test_assess_no_llm(self, sample_contract_analysis):
        assessor = RiskAssessor(use_llm=False)
        result = assessor.assess(sample_contract_analysis)
        assert isinstance(result, RiskAssessment)
        assert 0 <= result.overall_risk_score <= 100
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert result.contract_id == "test-contract-001"

    def test_assess_finds_present_clause_risks(self, sample_contract_analysis):
        assessor = RiskAssessor(use_llm=False)
        result = assessor.assess(sample_contract_analysis)
        labels = {f.label for f in result.findings}
        assert "License Grant" in labels or "Cap On Liability" in labels

    def test_assess_finds_missing_clauses(self, sample_contract_analysis):
        assessor = RiskAssessor(use_llm=False)
        result = assessor.assess(sample_contract_analysis)
        missing_labels = {f.label for f in result.missing_clause_risks}
        # Contract has Cap On Liability so it shouldn't be missing
        assert "Cap On Liability" not in missing_labels


class TestSerialization:

    def test_risk_finding_to_dict(self):
        f = RiskFinding(label="Test", severity=RiskSeverity.HIGH,
                        reason="Reason", recommendation="Rec",
                        clause_text="clause", confidence=0.9, source="rule")
        d = f.to_dict()
        assert d["label"] == "Test"
        assert d["severity"] == "HIGH"
        assert d["source"] == "rule"

    def test_risk_assessment_to_dict(self, sample_risk_assessment):
        d = sample_risk_assessment.to_dict()
        assert d["contract_id"] == "test-contract-001"
        assert d["overall_risk_score"] == 45
        assert isinstance(d["findings"], list)
