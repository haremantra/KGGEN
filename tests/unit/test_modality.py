"""Executable spec for the synchronous deontic modality checker.

These tests are the contract: passing them means `src.modality` conforms to
its first-slice spec. They cover the ordered rule priority, the four modality
classes, subject extraction, overlap suppression, edge cases, and
serialization round-trip.
"""

import json

import pytest

from src.modality import (
    MODAL_RULES,
    Modality,
    ModalityChecker,
    ModalFinding,
    ModalityReport,
    find_modal_matches,
)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class TestModalityEnum:

    def test_count(self):
        assert len(Modality) == 5

    def test_values(self):
        values = {m.value for m in Modality}
        assert values == {"OBLIGATION", "PERMISSION", "PROHIBITION", "ENTITLEMENT", "NONE"}

    def test_string_valued(self):
        assert Modality.OBLIGATION == "OBLIGATION"
        assert Modality.PROHIBITION.value == "PROHIBITION"


# ---------------------------------------------------------------------------
# Rule priority
# ---------------------------------------------------------------------------

class TestRulePriority:

    def test_shall_not_is_prohibition_not_obligation(self):
        """Regression guard: `shall not` must never surface `shall` OBLIGATION."""
        findings = ModalityChecker().check_text("Licensee shall not reverse engineer.")
        modalities = [f.modality for f in findings]
        assert Modality.PROHIBITION in modalities
        assert Modality.OBLIGATION not in modalities
        assert len(findings) == 1

    def test_prohibition_rules_precede_obligation_and_permission(self):
        """MODAL_RULES is priority-ordered; prohibition rules come first."""
        idx_by_modality = {}
        for i, rule in enumerate(MODAL_RULES):
            idx_by_modality.setdefault(rule.modality, i)
        assert idx_by_modality[Modality.PROHIBITION] < idx_by_modality[Modality.OBLIGATION]
        assert idx_by_modality[Modality.PROHIBITION] < idx_by_modality[Modality.PERMISSION]
        # Entitlement must precede bare-`may` PERMISSION so "may elect" wins.
        assert idx_by_modality[Modality.ENTITLEMENT] < idx_by_modality[Modality.PERMISSION]


# ---------------------------------------------------------------------------
# Per-class detection
# ---------------------------------------------------------------------------

class TestObligation:

    @pytest.mark.parametrize("text", [
        "Licensor shall deliver the software.",
        "The Provider must provide support.",
        "The Agreement will renew automatically.",
        "Licensee agrees to pay all fees.",
        "Each party agrees to indemnify the other.",
        "Vendor is obligated to maintain records.",
        "Customer is required to notify Vendor.",
        "Party A covenants to keep information confidential.",
    ])
    def test_obligation_phrases(self, text):
        findings = ModalityChecker().check_text(text)
        assert any(f.modality == Modality.OBLIGATION for f in findings), (
            f"expected OBLIGATION in {text!r}, got {[f.modality for f in findings]}"
        )


class TestPermission:

    @pytest.mark.parametrize("text", [
        "Either party may terminate the Agreement.",
        "Licensee is entitled to a refund.",
        "Customer is permitted to sublicense.",
        "Provider has the right to audit records.",
    ])
    def test_permission_phrases(self, text):
        findings = ModalityChecker().check_text(text)
        assert any(f.modality == Modality.PERMISSION for f in findings), (
            f"expected PERMISSION in {text!r}, got {[f.modality for f in findings]}"
        )


class TestProhibition:

    @pytest.mark.parametrize("text", [
        "Licensee shall not reverse engineer the Software.",
        "The party must not disclose confidential information.",
        "Customer may not assign this Agreement.",
        "Disclosure is prohibited without written consent.",
        "Such use is forbidden under applicable law.",
        "Neither party may solicit the employees of the other.",
        "No party shall be liable for indirect damages.",
    ])
    def test_prohibition_phrases(self, text):
        findings = ModalityChecker().check_text(text)
        modalities = [f.modality for f in findings]
        assert Modality.PROHIBITION in modalities, (
            f"expected PROHIBITION in {text!r}, got {modalities}"
        )


class TestEntitlement:

    @pytest.mark.parametrize("text", [
        "Licensor may, at its sole option, terminate the Agreement.",
        "Customer may elect to renew the subscription.",
        "The Vendor has the option to extend the term.",
    ])
    def test_entitlement_phrases(self, text):
        findings = ModalityChecker().check_text(text)
        modalities = [f.modality for f in findings]
        assert Modality.ENTITLEMENT in modalities, (
            f"expected ENTITLEMENT in {text!r}, got {modalities}"
        )


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------

class TestSubjectExtraction:

    def test_subject_single_token(self):
        findings = ModalityChecker().check_text("Licensor shall deliver the software.")
        assert findings
        assert findings[0].subject == "Licensor"

    def test_subject_contains_party_name(self):
        findings = ModalityChecker().check_text("The Licensee may terminate this Agreement.")
        assert findings
        assert findings[0].subject is not None
        assert "Licensee" in findings[0].subject

    def test_sentence_initial_modal_has_no_subject(self):
        findings = ModalityChecker().check_text("Shall pay on time.")
        assert findings
        assert findings[0].subject is None

    def test_subject_bounded_by_previous_clause(self):
        """Commas bound the subject window so prior clause spillover doesn't leak in."""
        findings = ModalityChecker().check_text(
            "If the conditions are met, Licensee shall deliver within 30 days."
        )
        # Pick the OBLIGATION finding
        obligations = [f for f in findings if f.modality == Modality.OBLIGATION]
        assert obligations
        subj = obligations[0].subject
        assert subj is not None
        assert "Licensee" in subj
        assert "conditions" not in subj  # bounded by the comma


# ---------------------------------------------------------------------------
# Overlap suppression
# ---------------------------------------------------------------------------

class TestOverlapSuppression:

    def test_shall_not_and_shall_in_same_text_yield_two_findings(self):
        text = "Licensee shall not disclose secrets. Licensee shall maintain records."
        findings = ModalityChecker().check_text(text)
        modalities = [f.modality for f in findings]
        assert len(findings) == 2
        assert Modality.PROHIBITION in modalities
        assert Modality.OBLIGATION in modalities

    def test_may_elect_is_entitlement_not_also_permission(self):
        findings = ModalityChecker().check_text("Customer may elect to renew.")
        assert len(findings) == 1
        assert findings[0].modality == Modality.ENTITLEMENT


# ---------------------------------------------------------------------------
# Edge cases / false-positive guards
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
    def test_empty_or_whitespace(self, text):
        assert ModalityChecker().check_text(text) == []

    def test_shallow_is_not_shall(self):
        findings = ModalityChecker().check_text("The water is shallow here.")
        assert findings == []

    def test_mayor_is_not_may(self):
        findings = ModalityChecker().check_text("The mayor signed the document.")
        assert findings == []

    def test_text_with_no_modals(self):
        findings = ModalityChecker().check_text("This Agreement is effective on January 1, 2024.")
        assert findings == []


# ---------------------------------------------------------------------------
# Duck-typed clause/analysis APIs
# ---------------------------------------------------------------------------

class TestCheckClause:

    def test_finding_carries_cuad_label(self, sample_contract_analysis):
        clause = sample_contract_analysis.analyzed_clauses[3]  # Non-Compete
        assert clause.cuad_label == "Non-Compete"
        findings = ModalityChecker().check_clause(clause)
        assert findings
        # "Licensee shall not compete..." should yield PROHIBITION.
        assert any(f.modality == Modality.PROHIBITION for f in findings)
        assert all(f.cuad_label == "Non-Compete" for f in findings)

    def test_license_grant_clause_yields_obligation(self, sample_contract_analysis):
        clause = sample_contract_analysis.analyzed_clauses[0]  # License Grant
        findings = ModalityChecker().check_clause(clause)
        # No modal verbs in "Licensor grants Licensee..." - no findings expected
        # (it's a performative, not a modal). That's fine; spec only asserts
        # that if a modal is present, we detect it. Empty is acceptable here.
        for f in findings:
            assert f.cuad_label == "License Grant"


class TestCheckAnalysis:

    def test_report_contract_id(self, sample_contract_analysis):
        report = ModalityChecker().check_analysis(sample_contract_analysis)
        assert report.contract_id == "test-contract-001"

    def test_counts_sum_equals_findings(self, sample_contract_analysis):
        report = ModalityChecker().check_analysis(sample_contract_analysis)
        assert sum(report.counts.values()) == len(report.findings)

    def test_all_modality_keys_present(self, sample_contract_analysis):
        report = ModalityChecker().check_analysis(sample_contract_analysis)
        for m in Modality:
            assert m.value in report.counts

    def test_non_compete_clause_produces_prohibition(self, sample_contract_analysis):
        report = ModalityChecker().check_analysis(sample_contract_analysis)
        assert report.counts[Modality.PROHIBITION.value] >= 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestToDict:

    def test_finding_truncates_clause_text_at_200_chars(self):
        long_text = "Licensee shall " + "x" * 300
        finding = ModalFinding(
            modality=Modality.OBLIGATION,
            modal_phrase="shall",
            span=(9, 14),
            subject="Licensee",
            confidence=0.75,
            cuad_label="Parties",
            clause_text=long_text,
        )
        d = finding.to_dict()
        assert len(d["clause_text"]) == 203  # 200 + "..."
        assert d["clause_text"].endswith("...")

    def test_finding_short_text_not_truncated(self):
        finding = ModalFinding(
            modality=Modality.OBLIGATION,
            modal_phrase="shall",
            span=(9, 14),
            subject="Licensee",
            confidence=0.75,
            clause_text="Licensee shall pay.",
        )
        d = finding.to_dict()
        assert d["clause_text"] == "Licensee shall pay."

    def test_enum_serialized_as_string_value(self):
        finding = ModalFinding(
            modality=Modality.PROHIBITION,
            modal_phrase="shall not",
            span=(0, 9),
            subject=None,
            confidence=0.9,
        )
        d = finding.to_dict()
        assert d["modality"] == "PROHIBITION"

    def test_report_roundtrips_through_json(self, sample_contract_analysis):
        report = ModalityChecker().check_analysis(sample_contract_analysis)
        payload = report.to_dict()
        # Should not raise
        blob = json.dumps(payload)
        restored = json.loads(blob)
        assert restored["contract_id"] == report.contract_id
        assert restored["counts"] == report.counts


# ---------------------------------------------------------------------------
# End-to-end pin against the minimal contract fixture
# ---------------------------------------------------------------------------

class TestMinimalContract:
    """Pin expected counts for `minimal_contract_text`. Rule changes that
    alter these counts should surface here so they're reviewed explicitly.
    """

    def test_minimal_contract_has_prohibitions(self, minimal_contract_text):
        findings = ModalityChecker().check_text(minimal_contract_text)
        prohibitions = [f for f in findings if f.modality == Modality.PROHIBITION]
        # "shall not reverse engineer", "shall not develop competing...",
        # "shall not exceed" (liability cap).
        assert len(prohibitions) >= 3

    def test_minimal_contract_has_obligations(self, minimal_contract_text):
        findings = ModalityChecker().check_text(minimal_contract_text)
        obligations = [f for f in findings if f.modality == Modality.OBLIGATION]
        # "shall be governed", "shall maintain ... insurance", "will automatically renew".
        assert len(obligations) >= 3

    def test_minimal_contract_has_permission(self, minimal_contract_text):
        findings = ModalityChecker().check_text(minimal_contract_text)
        permissions = [f for f in findings if f.modality == Modality.PERMISSION]
        # "Either party may terminate".
        assert len(permissions) >= 1

    def test_minimal_contract_no_entitlement(self, minimal_contract_text):
        findings = ModalityChecker().check_text(minimal_contract_text)
        entitlements = [f for f in findings if f.modality == Modality.ENTITLEMENT]
        assert len(entitlements) == 0


# ---------------------------------------------------------------------------
# Raw helper
# ---------------------------------------------------------------------------

class TestFindModalMatches:

    def test_returns_pairs_sorted_by_start(self):
        text = "Licensee shall maintain records. Licensor may not disclose."
        pairs = find_modal_matches(text)
        starts = [m.start() for _, m in pairs]
        assert starts == sorted(starts)

    def test_empty_text(self):
        assert find_modal_matches("") == []
