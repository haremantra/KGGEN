"""Executable spec for the PolarityChecker (FM-B06 / FM-D02 detector).

Polarity = where the modal force points (which party is bound, permitted, or
prohibited). These tests pin the decision-support behavior of the checker:
per-clause profiles, MUTUAL_DRIFT detection, DUPLICATE_INCONSISTENCY across
clauses, and subject normalization.
"""

import json

import pytest

from src.modality import Modality, ModalFinding
from src.pipeline import AnalyzedClause, ContractAnalysis
from src.polarity import (
    ClausePolarityProfile,
    PolarityChecker,
    PolarityFinding,
    PolarityKind,
    PolarityReport,
    PolaritySeverity,
    PolarityVerdict,
    SubjectKind,
)
from src.polarity.checker import _classify_subject_kind, _normalize_subject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mf(modality, subject, phrase="shall", strength="HIGH", clause_text=""):
    """Compact ModalFinding constructor for tests."""
    return ModalFinding(
        modality=modality,
        modal_phrase=phrase,
        span=(0, len(phrase)),
        subject=subject,
        strength=strength,
        clause_text=clause_text or phrase,
    )


def _clause(label, category="general_information", text="", findings=None):
    return AnalyzedClause(
        text=text or label,
        cuad_label=label,
        label_confidence=0.85,
        category=category,
        modality_findings=findings or [],
    )


# ---------------------------------------------------------------------------
# Subject normalization & classification
# ---------------------------------------------------------------------------

class TestSubjectNormalization:

    @pytest.mark.parametrize("raw,expected", [
        ("Licensee", "licensee"),
        ("The Licensee", "licensee"),
        ("the licensee", "licensee"),
        ("THE  LICENSEE", "licensee"),
        ("a Vendor", "vendor"),
        ("an Agent", "agent"),
        ("Either Party", "either party"),
        ("  spaced  ", "spaced"),
    ])
    def test_normalize(self, raw, expected):
        assert _normalize_subject(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
    def test_empty_returns_none(self, raw):
        assert _normalize_subject(raw) is None


class TestSubjectKindClassification:

    @pytest.mark.parametrize("subject,expected", [
        ("licensee", SubjectKind.NAMED),
        ("vendor", SubjectKind.NAMED),
        ("acme corp", SubjectKind.NAMED),
        ("either party", SubjectKind.MUTUAL),
        ("both parties", SubjectKind.MUTUAL),
        ("the parties", SubjectKind.MUTUAL),
        ("each party", SubjectKind.MUTUAL),
        ("mutually", SubjectKind.MUTUAL),
        ("neither party", SubjectKind.NEGATIVE),
        ("no party", SubjectKind.NEGATIVE),
    ])
    def test_classification(self, subject, expected):
        assert _classify_subject_kind(subject) == expected

    def test_none_classifies_as_none(self):
        assert _classify_subject_kind(None) == SubjectKind.NONE
        assert _classify_subject_kind("") == SubjectKind.NONE


# ---------------------------------------------------------------------------
# Per-clause profiles
# ---------------------------------------------------------------------------

class TestProfileClause:

    def test_single_party_obligated(self):
        clause = _clause("Payment Terms", findings=[
            _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
        ])
        prof = PolarityChecker().profile_clause(clause)
        assert prof.verdict == PolarityVerdict.SINGLE_PARTY_OBLIGATED
        assert prof.subjects == ["licensee"]
        assert prof.modality_counts == {"OBLIGATION": 1}

    def test_single_party_prohibited(self):
        clause = _clause("Non-Compete", findings=[
            _mf(Modality.PROHIBITION, "Licensee", phrase="shall not"),
        ])
        prof = PolarityChecker().profile_clause(clause)
        assert prof.verdict == PolarityVerdict.SINGLE_PARTY_PROHIBITED

    def test_single_party_permitted(self):
        clause = _clause("Termination", findings=[
            _mf(Modality.PERMISSION, "Licensee", phrase="may"),
        ])
        prof = PolarityChecker().profile_clause(clause)
        assert prof.verdict == PolarityVerdict.SINGLE_PARTY_PERMITTED

    def test_mutual(self):
        clause = _clause("Termination", findings=[
            _mf(Modality.PERMISSION, "Either Party", phrase="may"),
            _mf(Modality.PERMISSION, "either party", phrase="may"),
        ])
        prof = PolarityChecker().profile_clause(clause)
        assert prof.verdict == PolarityVerdict.MUTUAL
        assert "either party" in prof.subjects

    def test_mixed_when_multiple_modalities(self):
        clause = _clause("License Grant", findings=[
            _mf(Modality.OBLIGATION, "Licensor", phrase="shall"),
            _mf(Modality.PROHIBITION, "Licensee", phrase="shall not"),
        ])
        prof = PolarityChecker().profile_clause(clause)
        assert prof.verdict == PolarityVerdict.MIXED

    def test_undetermined_when_empty(self):
        clause = _clause("Recitals", findings=[])
        prof = PolarityChecker().profile_clause(clause)
        assert prof.verdict == PolarityVerdict.UNDETERMINED

    def test_subjects_deduplicated(self):
        clause = _clause("Payment Terms", findings=[
            _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
            _mf(Modality.OBLIGATION, "the Licensee", phrase="shall"),
            _mf(Modality.OBLIGATION, "Licensee", phrase="must"),
        ])
        prof = PolarityChecker().profile_clause(clause)
        assert prof.subjects == ["licensee"]
        assert prof.modality_counts == {"OBLIGATION": 3}


# ---------------------------------------------------------------------------
# Within-clause: MUTUAL_DRIFT
# ---------------------------------------------------------------------------

class TestMutualDrift:

    def test_mutual_plus_named_yields_drift(self):
        clause = _clause("Termination", findings=[
            _mf(Modality.PERMISSION, "Either Party", phrase="may"),
            _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
        ])
        findings = PolarityChecker().check_clause(clause)
        assert len(findings) == 1
        f = findings[0]
        assert f.kind == PolarityKind.MUTUAL_DRIFT
        assert f.fm_code == "FM-B06"
        assert f.severity == PolaritySeverity.MODERATE
        assert "either party" in f.subjects
        assert "licensee" in f.subjects

    def test_negative_plus_named_yields_drift(self):
        clause = _clause("Liability", findings=[
            _mf(Modality.PROHIBITION, "Neither Party", phrase="shall"),
            _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
        ])
        findings = PolarityChecker().check_clause(clause)
        assert len(findings) == 1
        assert findings[0].kind == PolarityKind.MUTUAL_DRIFT

    def test_only_mutual_no_drift(self):
        clause = _clause("Termination", findings=[
            _mf(Modality.PERMISSION, "Either Party", phrase="may"),
            _mf(Modality.OBLIGATION, "Both Parties", phrase="shall"),
        ])
        assert PolarityChecker().check_clause(clause) == []

    def test_only_named_no_drift(self):
        clause = _clause("Payment Terms", findings=[
            _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
            _mf(Modality.OBLIGATION, "Customer", phrase="must"),
        ])
        assert PolarityChecker().check_clause(clause) == []

    def test_empty_findings_no_drift(self):
        assert PolarityChecker().check_clause(_clause("Recitals")) == []


# ---------------------------------------------------------------------------
# Cross-clause: DUPLICATE_INCONSISTENCY
# ---------------------------------------------------------------------------

class TestDuplicateInconsistency:

    def test_same_label_conflicting_subjects(self):
        analysis = ContractAnalysis(
            contract_id="c1", total_clauses=2, summary={},
            analyzed_clauses=[
                _clause("Cap On Liability", findings=[
                    _mf(Modality.PROHIBITION, "Licensor", phrase="shall not"),
                ]),
                _clause("Cap On Liability", findings=[
                    _mf(Modality.PROHIBITION, "Licensee", phrase="shall not"),
                ]),
            ],
        )
        report = PolarityChecker().check_analysis(analysis)
        dups = [f for f in report.findings if f.kind == PolarityKind.DUPLICATE_INCONSISTENCY]
        assert len(dups) == 1
        assert dups[0].fm_code == "FM-D02"
        assert dups[0].cuad_label == "Cap On Liability"
        assert {"licensor", "licensee"} <= set(dups[0].subjects)

    def test_same_label_conflicting_modalities(self):
        analysis = ContractAnalysis(
            contract_id="c1", total_clauses=2, summary={},
            analyzed_clauses=[
                _clause("Termination For Convenience", findings=[
                    _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
                ]),
                _clause("Termination For Convenience", findings=[
                    _mf(Modality.PROHIBITION, "Licensee", phrase="shall not"),
                ]),
            ],
        )
        report = PolarityChecker().check_analysis(analysis)
        dups = [f for f in report.findings if f.kind == PolarityKind.DUPLICATE_INCONSISTENCY]
        assert len(dups) == 1

    def test_unique_labels_no_duplicate_finding(self):
        analysis = ContractAnalysis(
            contract_id="c1", total_clauses=2, summary={},
            analyzed_clauses=[
                _clause("License Grant", findings=[
                    _mf(Modality.OBLIGATION, "Licensor", phrase="shall"),
                ]),
                _clause("Cap On Liability", findings=[
                    _mf(Modality.PROHIBITION, "Licensee", phrase="shall not"),
                ]),
            ],
        )
        report = PolarityChecker().check_analysis(analysis)
        dups = [f for f in report.findings if f.kind == PolarityKind.DUPLICATE_INCONSISTENCY]
        assert dups == []

    def test_same_label_consistent_no_finding(self):
        """Same label, same subject, same modality across two clauses → no flag."""
        analysis = ContractAnalysis(
            contract_id="c1", total_clauses=2, summary={},
            analyzed_clauses=[
                _clause("Insurance", findings=[
                    _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
                ]),
                _clause("Insurance", findings=[
                    _mf(Modality.OBLIGATION, "Licensee", phrase="shall"),
                ]),
            ],
        )
        report = PolarityChecker().check_analysis(analysis)
        dups = [f for f in report.findings if f.kind == PolarityKind.DUPLICATE_INCONSISTENCY]
        assert dups == []


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------

class TestReport:

    def test_report_has_one_profile_per_clause(self):
        analysis = ContractAnalysis(
            contract_id="c1", total_clauses=3, summary={},
            analyzed_clauses=[
                _clause("A"), _clause("B"), _clause("C"),
            ],
        )
        report = PolarityChecker().check_analysis(analysis)
        assert len(report.profiles) == 3

    def test_counts_by_severity_includes_all_levels(self):
        analysis = ContractAnalysis(
            contract_id="c1", total_clauses=1, summary={},
            analyzed_clauses=[_clause("X", findings=[
                _mf(Modality.PERMISSION, "Either Party"),
                _mf(Modality.OBLIGATION, "Licensee"),
            ])],
        )
        report = PolarityChecker().check_analysis(analysis)
        for sev in PolaritySeverity:
            assert sev.value in report.counts_by_severity
        # We emitted exactly one MODERATE drift.
        assert report.counts_by_severity["MODERATE"] >= 1
        assert report.counts_by_severity["CRITICAL"] == 0

    def test_report_roundtrips_through_json(self):
        analysis = ContractAnalysis(
            contract_id="c1", total_clauses=1, summary={},
            analyzed_clauses=[_clause("X", findings=[
                _mf(Modality.OBLIGATION, "Licensee"),
            ])],
        )
        report = PolarityChecker().check_analysis(analysis)
        blob = json.dumps(report.to_dict())
        restored = json.loads(blob)
        assert restored["contract_id"] == "c1"
        assert "profiles" in restored
        assert "findings" in restored

    def test_empty_analysis_yields_empty_report(self):
        analysis = ContractAnalysis(
            contract_id="empty", total_clauses=0,
            analyzed_clauses=[], summary={},
        )
        report = PolarityChecker().check_analysis(analysis)
        assert report.profiles == []
        assert report.findings == []
        assert report.counts_by_severity == {sev.value: 0 for sev in PolaritySeverity}


# ---------------------------------------------------------------------------
# Integration with sample fixture (modality_findings populated manually)
# ---------------------------------------------------------------------------

class TestAgainstSampleFixture:
    """Augment the conftest sample_contract_analysis with modality findings,
    then sanity-check polarity output."""

    def test_non_compete_clause_yields_single_party_prohibited(
        self, sample_contract_analysis,
    ):
        # Augment fixture in place: Non-Compete clause → PROHIBITION on Licensee.
        non_compete = next(
            c for c in sample_contract_analysis.analyzed_clauses
            if c.cuad_label == "Non-Compete"
        )
        non_compete.modality_findings = [
            _mf(Modality.PROHIBITION, "Licensee", phrase="shall not",
                clause_text=non_compete.text),
        ]
        report = PolarityChecker().check_analysis(sample_contract_analysis)
        non_compete_profile = next(
            p for p in report.profiles if p.cuad_label == "Non-Compete"
        )
        assert non_compete_profile.verdict == PolarityVerdict.SINGLE_PARTY_PROHIBITED
        assert "licensee" in non_compete_profile.subjects
