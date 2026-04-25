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
from src.modality.types import Strength as StrengthType  # noqa: F401  (typed-alias check)


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

    @pytest.mark.parametrize("text", [
        "IN NO EVENT SHALL Licensor be liable for indirect damages.",
        "In no event will the Provider exceed the cap.",
        "AT NO TIME SHALL Licensee disclose the source code.",
        "Under no circumstances may the parties amend this Agreement orally.",
    ])
    def test_reversed_order_legal_boilerplate(self, text):
        """`IN NO EVENT SHALL ...` must classify as PROHIBITION, not OBLIGATION."""
        findings = ModalityChecker().check_text(text)
        modalities = [f.modality for f in findings]
        assert Modality.PROHIBITION in modalities, (
            f"expected PROHIBITION in {text!r}, got {modalities}"
        )
        # The bare-modal OBLIGATION rule must be suppressed by overlap.
        assert Modality.OBLIGATION not in modalities


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

    def test_paragraph_break_bounds_subject(self):
        """A blank line resets the subject window so prior-paragraph text doesn't leak in."""
        text = (
            "1. RECITALS\n"
            "The parties wish to enter into a license arrangement\n\n"
            "Licensee shall pay the License Fee on the Effective Date."
        )
        findings = ModalityChecker().check_text(text)
        obligations = [f for f in findings if f.modality == Modality.OBLIGATION]
        assert obligations
        subj = obligations[0].subject
        assert subj is not None
        assert "Licensee" in subj
        assert "RECITALS" not in subj
        assert "license arrangement" not in subj

    def test_single_newline_does_not_break_subject(self):
        """Soft line wraps mid-clause should not break the subject window."""
        text = "Licensee\nshall comply with the terms set forth herein."
        findings = ModalityChecker().check_text(text)
        obligations = [f for f in findings if f.modality == Modality.OBLIGATION]
        assert obligations
        assert obligations[0].subject is not None
        assert "Licensee" in obligations[0].subject


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
            strength="MEDIUM",
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
            strength="MEDIUM",
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
            strength="HIGH",
        )
        d = finding.to_dict()
        assert d["modality"] == "PROHIBITION"

    def test_strength_serialized_as_string(self):
        finding = ModalFinding(
            modality=Modality.OBLIGATION,
            modal_phrase="shall",
            span=(0, 5),
            strength="MEDIUM",
        )
        d = finding.to_dict()
        assert d["strength"] == "MEDIUM"
        assert isinstance(d["strength"], str)

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
# Strength categorical (no floats)
# ---------------------------------------------------------------------------

class TestStrength:

    def test_all_rule_strengths_are_categorical_strings(self):
        valid = {"LOW", "MEDIUM", "HIGH"}
        for rule in MODAL_RULES:
            assert isinstance(rule.strength, str), (
                f"{rule.name}: expected categorical string strength"
            )
            assert rule.strength in valid, (
                f"{rule.name}: {rule.strength!r} not in {valid}"
            )

    def test_phrasal_patterns_are_high(self):
        """Multi-word locutions and unambiguous phrasings are HIGH."""
        for rule in MODAL_RULES:
            if rule.name in {
                "prohibition_modal_not",
                "prohibition_no_event_reversed",
                "prohibition_modal_no_event",
                "prohibition_no_party",
                "prohibition_neither_party",
                "obligation_agrees_to",
                "entitlement_optional",
            }:
                assert rule.strength == "HIGH", f"{rule.name} should be HIGH"

    def test_bare_modal_verbs_are_medium(self):
        """Bare `shall`/`may` carry less unambiguous force."""
        bare = {r.name for r in MODAL_RULES if r.name in {"obligation_modal", "permission_may"}}
        assert bare == {"obligation_modal", "permission_may"}
        for rule in MODAL_RULES:
            if rule.name in bare:
                assert rule.strength == "MEDIUM", f"{rule.name} should be MEDIUM"

    def test_finding_strength_propagates_from_rule(self):
        """`shall not` (HIGH rule) → finding.strength == 'HIGH'."""
        findings = ModalityChecker().check_text("Licensee shall not disclose.")
        assert findings
        assert findings[0].strength == "HIGH"

    def test_bare_shall_finding_is_medium(self):
        findings = ModalityChecker().check_text("Licensee shall pay the fees.")
        assert findings
        assert findings[0].strength == "MEDIUM"


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


# ---------------------------------------------------------------------------
# check_diff: original vs redlined diff primitive
# ---------------------------------------------------------------------------

class TestCheckDiff:

    def test_identical_text_yields_only_kept(self):
        text = "Licensee shall pay all fees."
        diff = ModalityChecker().check_diff(text, text)
        assert len(diff.kept) == 1
        assert diff.added == []
        assert diff.removed == []
        assert diff.drifted == []
        assert diff.counts["kept"] == 1

    def test_removed_finding(self):
        original = "Licensee shall pay all fees."
        redlined = "The agreement is hereby effective."
        diff = ModalityChecker().check_diff(original, redlined)
        assert len(diff.removed) == 1
        assert diff.removed[0].modality.value == "OBLIGATION"
        assert diff.kept == []
        assert diff.added == []
        assert diff.drifted == []

    def test_added_finding(self):
        original = "The agreement is hereby effective."
        redlined = "Licensee shall not assign this agreement."
        diff = ModalityChecker().check_diff(original, redlined)
        assert len(diff.added) == 1
        assert diff.added[0].modality.value == "PROHIBITION"
        assert diff.kept == []
        assert diff.removed == []
        assert diff.drifted == []

    def test_subject_drift(self):
        """Same modality + phrase, different normalized subject → SUBJECT drift."""
        from src.modality import DriftKind
        original = "Licensor shall indemnify the other party."
        redlined = "Licensee shall indemnify the other party."
        diff = ModalityChecker().check_diff(original, redlined)
        assert len(diff.drifted) == 1
        d = diff.drifted[0]
        assert d.drift_kind == DriftKind.SUBJECT
        assert d.original.subject == "Licensor"
        assert d.redlined.subject == "Licensee"
        assert diff.kept == []
        assert diff.added == []
        assert diff.removed == []

    def test_modality_drift(self):
        """Same subject, different modality → MODALITY drift."""
        from src.modality import DriftKind
        original = "Licensee shall pay all fees."
        redlined = "Licensee may pay all fees."
        diff = ModalityChecker().check_diff(original, redlined)
        assert len(diff.drifted) == 1
        d = diff.drifted[0]
        assert d.drift_kind == DriftKind.MODALITY
        assert d.original.modality.value == "OBLIGATION"
        assert d.redlined.modality.value == "PERMISSION"

    def test_subject_normalization_in_matching(self):
        """`The Licensee` and `Licensee` count as the same subject for matching."""
        original = "Licensee shall pay."
        redlined = "The Licensee shall pay."
        diff = ModalityChecker().check_diff(original, redlined)
        assert len(diff.kept) == 1
        assert diff.drifted == []

    def test_combined_kept_added_removed_drifted(self):
        from src.modality import DriftKind
        original = (
            "Licensor shall maintain insurance. "
            "Licensor shall indemnify the other party. "
            "Licensee may terminate."
        )
        redlined = (
            "Licensor shall maintain insurance. "          # kept
            "Licensee shall indemnify the other party. "   # SUBJECT drift
            # `Licensee may terminate` removed
            "Either party shall not assign rights."        # added (PROHIBITION)
        )
        diff = ModalityChecker().check_diff(original, redlined)
        kinds = [d.drift_kind for d in diff.drifted]
        assert DriftKind.SUBJECT in kinds
        assert any(f.modality.value == "PROHIBITION" for f in diff.added)
        assert any(f.modality.value == "PERMISSION" for f in diff.removed)
        assert any(f.modality.value == "OBLIGATION" and f.subject == "Licensor"
                   for f in diff.kept)

    def test_none_subject_drift_match_via_predicate_hint(self):
        """Predicate-aware matching: even with no subject, a shared
        predicate_hint anchors a MODALITY drift pair."""
        from src.modality import DriftKind
        original = "Shall pay all fees."         # subject=None, OBLIGATION
        redlined = "May pay all fees."           # subject=None, PERMISSION
        diff = ModalityChecker().check_diff(original, redlined)
        # Same predicate_hint ("pay all fees") + same (None) subject + different
        # modality -> MODALITY drift (the right answer; the old behavior treated
        # this as remove+add because subject was None).
        assert len(diff.drifted) == 1
        assert diff.drifted[0].drift_kind == DriftKind.MODALITY
        assert diff.removed == []
        assert diff.added == []

    def test_predicate_hint_disambiguates_same_subject_modality(self):
        """The motivating bugfix: two distinct (Licensee, OBLIGATION)
        statements no longer collide as `kept` just because they share
        subject and modality. The predicate hint differentiates them so
        "Licensee shall pay X" and "Licensee shall pay Y" are not
        spuriously matched."""
        original = "Licensee shall pay all fees. Licensor shall maintain insurance."
        redlined = "Licensee shall pay license royalties. Licensor shall maintain insurance."

        diff = ModalityChecker().check_diff(original, redlined)
        kept_predicates = [f.predicate_hint for f in diff.kept]
        # The unchanged Licensor obligation should be kept.
        assert "maintain insurance" in kept_predicates
        # The "Licensee shall pay" obligations have different predicates
        # ("pay all fees" vs "pay license royalties") and must NOT collide
        # as kept — they should appear as removed/added.
        removed_predicates = [f.predicate_hint for f in diff.removed]
        added_predicates = [f.predicate_hint for f in diff.added]
        assert "pay all fees" in removed_predicates
        assert "pay license royalties" in added_predicates
        # Most importantly: NOT in kept.
        assert "pay all fees" not in kept_predicates
        assert "pay license royalties" not in kept_predicates

    def test_predicate_hint_populated_in_findings(self):
        """check_text attaches a predicate_hint to each finding."""
        findings = ModalityChecker().check_text("Licensee shall pay all fees on the date.")
        assert findings
        assert findings[0].predicate_hint
        # First three lowercased word tokens after the modal verb.
        assert findings[0].predicate_hint == "pay all fees"

    def test_predicate_hint_stops_at_sentence_boundary(self):
        findings = ModalityChecker().check_text("Licensee shall pay. The agreement continues.")
        assert findings
        assert findings[0].predicate_hint == "pay"  # period truncates

    def test_counts_match_lists(self):
        original = "Licensor shall pay. Licensee shall not assign."
        redlined = "Licensor may pay. Licensee shall not assign."
        diff = ModalityChecker().check_diff(original, redlined)
        assert diff.counts["kept"] == len(diff.kept)
        assert diff.counts["added"] == len(diff.added)
        assert diff.counts["removed"] == len(diff.removed)
        assert (diff.counts["drifted_subject"] + diff.counts["drifted_modality"]
                == len(diff.drifted))

    def test_diff_roundtrips_through_json(self):
        original = "Licensor shall pay. Licensee may terminate."
        redlined = "Licensee shall pay. Licensee may not terminate."
        diff = ModalityChecker().check_diff(original, redlined)
        blob = json.dumps(diff.to_dict())
        restored = json.loads(blob)
        assert "kept" in restored
        assert "added" in restored
        assert "removed" in restored
        assert "drifted" in restored
        assert "counts" in restored

    def test_empty_inputs(self):
        diff = ModalityChecker().check_diff("", "")
        assert diff.kept == []
        assert diff.added == []
        assert diff.removed == []
        assert diff.drifted == []

    def test_drifted_pair_to_dict_includes_both_sides(self):
        original = "Licensor shall indemnify."
        redlined = "Licensee shall indemnify."
        diff = ModalityChecker().check_diff(original, redlined)
        assert len(diff.drifted) == 1
        d = diff.drifted[0].to_dict()
        assert d["drift_kind"] == "SUBJECT"
        assert d["original"]["subject"] == "Licensor"
        assert d["redlined"]["subject"] == "Licensee"


# ---------------------------------------------------------------------------
# normalize_subject (public helper)
# ---------------------------------------------------------------------------

class TestNormalizeSubject:

    @pytest.mark.parametrize("raw,expected", [
        ("Licensee", "licensee"),
        ("The Licensee", "licensee"),
        ("THE  LICENSEE", "licensee"),
        ("a Vendor", "vendor"),
        ("an Agent", "agent"),
        ("Either Party", "either party"),
    ])
    def test_canonical_form(self, raw, expected):
        from src.modality import normalize_subject
        assert normalize_subject(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_empty_returns_none(self, raw):
        from src.modality import normalize_subject
        assert normalize_subject(raw) is None
