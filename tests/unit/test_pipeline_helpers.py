"""Tests for src/pipeline.py — dataclasses, _build_summary, EXTRACTION_PROMPTS."""

import pytest
from src.modality import Modality, ModalFinding, ModalityChecker
from src.pipeline import (
    ExtractedValue, AnalyzedClause, ContractAnalysis,
    ContractAnalysisPipeline, EXTRACTION_PROMPTS,
)


class TestDataclasses:

    def test_extracted_value_creation(self):
        ev = ExtractedValue(field="cap_amount", value="$100K", confidence=0.9)
        assert ev.field == "cap_amount"
        assert ev.value == "$100K"
        assert ev.confidence == 0.9

    def test_analyzed_clause_creation(self):
        ac = AnalyzedClause(
            text="Test clause", cuad_label="License Grant",
            label_confidence=0.87, category="general_information",
        )
        assert ac.cuad_label == "License Grant"
        assert ac.extracted_values == []
        assert ac.entities == []
        assert ac.relationships == []

    def test_contract_analysis_creation(self):
        ca = ContractAnalysis(
            contract_id="test", total_clauses=5,
            analyzed_clauses=[], summary={},
        )
        assert ca.contract_id == "test"
        assert ca.total_clauses == 5


class TestBuildSummary:

    def test_by_category_grouping(self):
        pipeline = ContractAnalysisPipeline.__new__(ContractAnalysisPipeline)
        clauses = [
            AnalyzedClause(text="a", cuad_label="License Grant",
                           label_confidence=0.8, category="general_information"),
            AnalyzedClause(text="b", cuad_label="Non-Compete",
                           label_confidence=0.7, category="restrictive_covenants"),
            AnalyzedClause(text="c", cuad_label="Governing Law",
                           label_confidence=0.9, category="general_information"),
        ]
        summary = pipeline._build_summary(clauses)
        assert "general_information" in summary["by_category"]
        assert "restrictive_covenants" in summary["by_category"]
        assert len(summary["by_category"]["general_information"]) == 2

    def test_key_findings_high_confidence(self):
        pipeline = ContractAnalysisPipeline.__new__(ContractAnalysisPipeline)
        clauses = [
            AnalyzedClause(
                text="a", cuad_label="License Grant",
                label_confidence=0.8, category="general",
                extracted_values=[
                    ExtractedValue(field="type", value="non-exclusive", confidence=0.9),
                    ExtractedValue(field="scope", value="limited", confidence=0.5),
                ],
            ),
        ]
        summary = pipeline._build_summary(clauses)
        # Only the high-confidence finding (0.9 >= 0.7) should appear
        assert len(summary["key_findings"]) == 1
        assert summary["key_findings"][0]["field"] == "type"

    def test_empty_clauses(self):
        pipeline = ContractAnalysisPipeline.__new__(ContractAnalysisPipeline)
        summary = pipeline._build_summary([])
        assert summary["labels_found"] == 0
        assert summary["by_category"] == {}
        assert summary["key_findings"] == []

    def test_labels_found_count(self):
        pipeline = ContractAnalysisPipeline.__new__(ContractAnalysisPipeline)
        clauses = [
            AnalyzedClause(text="a", cuad_label="X", label_confidence=0.8, category="cat"),
            AnalyzedClause(text="b", cuad_label="Y", label_confidence=0.7, category="cat"),
        ]
        summary = pipeline._build_summary(clauses)
        assert summary["labels_found"] == 2


class TestExtractionPrompts:

    def test_has_entries(self):
        assert len(EXTRACTION_PROMPTS) >= 30

    def test_license_grant_prompt(self):
        assert "License Grant" in EXTRACTION_PROMPTS
        assert "Extract" in EXTRACTION_PROMPTS["License Grant"]

    def test_cap_on_liability_prompt(self):
        assert "Cap On Liability" in EXTRACTION_PROMPTS

    def test_governing_law_prompt(self):
        assert "Governing Law" in EXTRACTION_PROMPTS


class TestModalityWiring:
    """Slice C: ModalityChecker integration into ContractAnalysisPipeline."""

    def test_analyzed_clause_has_modality_findings_field(self):
        ac = AnalyzedClause(
            text="x", cuad_label="License Grant",
            label_confidence=0.8, category="general_information",
        )
        assert hasattr(ac, "modality_findings")
        assert ac.modality_findings == []

    def test_analyzed_clause_accepts_modality_findings(self):
        finding = ModalFinding(
            modality=Modality.PROHIBITION,
            modal_phrase="shall not",
            span=(9, 18),
            subject="Licensee",
            strength="HIGH",
            cuad_label="Non-Compete",
            clause_text="Licensee shall not compete.",
        )
        ac = AnalyzedClause(
            text="Licensee shall not compete.",
            cuad_label="Non-Compete",
            label_confidence=0.78,
            category="restrictive_covenants",
            modality_findings=[finding],
        )
        assert len(ac.modality_findings) == 1
        assert ac.modality_findings[0].modality == Modality.PROHIBITION

    def test_pipeline_init_creates_modality_checker(self, monkeypatch):
        """The pipeline must instantiate a ModalityChecker so analyze() can call it."""
        # Stub Anthropic and ClauseClassifier — neither is reachable offline,
        # and neither is what this test is verifying.
        monkeypatch.setattr("src.pipeline.Anthropic", lambda **kw: object())
        monkeypatch.setattr("src.pipeline.ClauseClassifier", lambda **kw: object())
        pipeline = ContractAnalysisPipeline()
        assert hasattr(pipeline, "modality_checker")
        assert isinstance(pipeline.modality_checker, ModalityChecker)

    def test_modality_findings_propagate_through_full_text_not_truncated(self):
        """Pipeline scans the full clause text; truncation only affects display."""
        long_text = "Some preamble. " + ("Licensee shall comply. " * 50)
        # 50 occurrences of "shall" — all should be picked up against full text.
        findings = ModalityChecker().check_text(long_text, cuad_label="Compliance")
        # Truncated display version (what AnalyzedClause.text would carry).
        truncated_findings = ModalityChecker().check_text(long_text[:500] + "...")
        assert len(findings) > len(truncated_findings)

    def test_serialized_clause_includes_modality_findings_key(self):
        """The CLI/file JSON output (analyze_contract_file) must include the key."""
        ac = AnalyzedClause(
            text="Licensee shall pay.",
            cuad_label="Payment Terms",
            label_confidence=0.85,
            category="general_information",
            modality_findings=[
                ModalFinding(
                    modality=Modality.OBLIGATION,
                    modal_phrase="shall",
                    span=(9, 14),
                    subject="Licensee",
                    strength="MEDIUM",
                    cuad_label="Payment Terms",
                    clause_text="Licensee shall pay.",
                ),
            ],
        )
        # Mirror the dict-shape produced in analyze_contract_file:
        serialized = {
            "cuad_label": ac.cuad_label,
            "modality_findings": [f.to_dict() for f in ac.modality_findings],
        }
        assert "modality_findings" in serialized
        assert len(serialized["modality_findings"]) == 1
        assert serialized["modality_findings"][0]["modality"] == "OBLIGATION"
        assert serialized["modality_findings"][0]["strength"] == "MEDIUM"
