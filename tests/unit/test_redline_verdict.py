"""Executable spec for the redline-compliance verdict layer.

Each test pins one verdict assignment scenario from the observed-redline
taxonomy. The spec is the test list; passing it means the assigner
matches the skill's vocabulary.
"""

import json

import pytest

from src.modality import Modality, ModalityChecker
from src.redline import (
    Recommendation,
    RedlineComplianceReport,
    RedlineVerdictAssigner,
    Verdict,
    VerdictAssignment,
)


# ---------------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------------

class TestInsert:

    def test_faithful_when_added_finding_matches(self):
        diff = ModalityChecker().check_diff(
            "The agreement is hereby effective.",
            "The agreement is hereby effective. Licensee shall not assign rights.",
        )
        rec = Recommendation(
            rec_id="REC-001", rec_type="INSERT",
            summary="Add anti-assignment prohibition on Licensee.",
            expected_modality=Modality.PROHIBITION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec], contract_id="c1")
        assert len(report.verdicts) == 1
        v = report.verdicts[0]
        assert v.rec_id == "REC-001"
        assert v.verdict == Verdict.FAITHFUL
        assert len(v.matched_findings) == 1

    def test_missed_when_added_does_not_match_and_selected(self):
        diff = ModalityChecker().check_diff(
            "Original text.",
            "Licensor may terminate.",  # PERMISSION on Licensor; not the rec
        )
        rec = Recommendation(
            rec_id="REC-002", rec_type="INSERT",
            expected_modality=Modality.PROHIBITION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        v = report.verdicts[0]
        assert v.verdict == Verdict.MISSED

    def test_expected_skip_when_unselected(self):
        diff = ModalityChecker().check_diff("A.", "A.")
        rec = Recommendation(
            rec_id="REC-003", rec_type="INSERT",
            expected_modality=Modality.PROHIBITION,
            expected_subject="Licensee",
            selected=False,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        v = report.verdicts[0]
        assert v.verdict == Verdict.EXPECTED_SKIP


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

class TestDelete:

    def test_faithful_when_finding_removed(self):
        diff = ModalityChecker().check_diff(
            "Licensee shall pay all fees on the Effective Date.",
            "The agreement is hereby effective.",  # all original modality stripped
        )
        rec = Recommendation(
            rec_id="REC-010", rec_type="DELETE",
            summary="Remove the payment obligation on Licensee.",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        v = report.verdicts[0]
        assert v.verdict == Verdict.FAITHFUL
        assert len(v.matched_findings) == 1

    def test_missed_when_obligation_remains(self):
        diff = ModalityChecker().check_diff(
            "Licensee shall pay all fees.",
            "Licensee shall pay all fees.",  # nothing removed
        )
        rec = Recommendation(
            rec_id="REC-011", rec_type="DELETE",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        assert report.verdicts[0].verdict == Verdict.MISSED


# ---------------------------------------------------------------------------
# MODIFY
# ---------------------------------------------------------------------------

class TestModify:

    def test_faithful_when_drift_pair_matches_both_endpoints(self):
        """Memo: change Licensee's obligation to a permission. Redline does so."""
        diff = ModalityChecker().check_diff(
            "Licensee shall pay all fees.",     # OBLIGATION on Licensee
            "Licensee may pay all fees.",        # PERMISSION on Licensee
        )
        rec = Recommendation(
            rec_id="REC-020", rec_type="MODIFY",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.PERMISSION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        v = report.verdicts[0]
        assert v.verdict == Verdict.FAITHFUL
        assert len(v.drift_pairs) == 1

    def test_drifted_when_redlined_side_does_not_match(self):
        """Memo: change Licensee's obligation to a permission. Redline
        instead inverts the party (Licensor obligated). Original side
        matches; redlined side does not -> DRIFTED."""
        diff = ModalityChecker().check_diff(
            "Licensee shall pay all fees.",
            "Licensor shall pay all fees.",
        )
        rec = Recommendation(
            rec_id="REC-021", rec_type="MODIFY",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.PERMISSION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        assert report.verdicts[0].verdict == Verdict.DRIFTED

    def test_missed_when_no_drift_pair_matches_before_state(self):
        diff = ModalityChecker().check_diff("A.", "A.")
        rec = Recommendation(
            rec_id="REC-022", rec_type="MODIFY",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.PERMISSION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        assert report.verdicts[0].verdict == Verdict.MISSED

    def test_faithful_via_removed_added_pair_fallback(self):
        """When predicate-aware diffing classifies a MODIFY as remove+add
        (rather than as a drift pair), the verdict layer should still
        recognize a FAITHFUL match if both endpoints align."""
        # Memo: change the predicate of Licensee's obligation. Modality
        # and subject stay the same, but the predicate changed -> diff
        # produces remove+add, not a drift pair.
        diff = ModalityChecker().check_diff(
            "Licensee shall pay all fees.",
            "Licensee shall pay license royalties.",
        )
        # Confirm the diff classified this as remove+add (not drifted).
        assert diff.drifted == []
        assert len(diff.removed) >= 1
        assert len(diff.added) >= 1

        rec = Recommendation(
            rec_id="REC-023", rec_type="MODIFY",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.OBLIGATION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        assert report.verdicts[0].verdict == Verdict.FAITHFUL

    def test_drifted_via_removed_only_fallback(self):
        """Predicate-aware diff: if the original-state finding is removed
        but no expected-state finding was added, that's DRIFTED (the
        redline acted on the right clause but produced something else)."""
        diff = ModalityChecker().check_diff(
            "Licensee shall pay all fees.",
            "Licensor may pay all fees.",  # totally different (subject + modality)
        )
        rec = Recommendation(
            rec_id="REC-024", rec_type="MODIFY",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.OBLIGATION,  # memo wanted obligation kept
            expected_subject="Licensee",            # memo wanted Licensee kept
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        assert report.verdicts[0].verdict == Verdict.DRIFTED


# ---------------------------------------------------------------------------
# PHANTOM
# ---------------------------------------------------------------------------

class TestPhantom:

    def test_unmatched_added_finding_becomes_phantom(self):
        """Redline adds something the memo never asked for."""
        diff = ModalityChecker().check_diff(
            "Original.",
            "Under no circumstances shall Licensor be liable.",  # added PROHIBITION
        )
        # Empty rec list -> the addition is unaccounted for.
        report = RedlineVerdictAssigner().assign(diff, [], contract_id="c1")
        assert len(report.phantom_findings) == 1
        assert report.phantom_findings[0].modality == Modality.PROHIBITION
        assert report.counts_by_verdict["PHANTOM"] == 1

    def test_subject_drift_without_rec_becomes_phantom(self):
        """A SUBJECT drift that no MODIFY rec accounts for is a PHANTOM."""
        diff = ModalityChecker().check_diff(
            "Licensor shall indemnify the other party.",
            "Licensee shall indemnify the other party.",
        )
        report = RedlineVerdictAssigner().assign(diff, [])
        # The drift pair's redlined side becomes a phantom.
        assert len(report.phantom_findings) >= 1


# ---------------------------------------------------------------------------
# Aggregate scoring (skill-aligned formulas)
# ---------------------------------------------------------------------------

class TestScoring:

    def test_fidelity_and_confabulation_when_perfect(self):
        diff = ModalityChecker().check_diff(
            "A.",
            "Licensee shall not assign.",
        )
        rec = Recommendation(
            rec_id="REC-100", rec_type="INSERT",
            expected_modality=Modality.PROHIBITION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        assert report.fidelity_score == 1.0
        assert report.confabulation_score == 0.0
        assert report.total_selected == 1

    def test_fidelity_partial_credit_for_drifted(self):
        diff = ModalityChecker().check_diff(
            "Licensee shall pay.",
            "Licensor shall pay.",
        )
        rec = Recommendation(
            rec_id="REC-101", rec_type="MODIFY",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.PERMISSION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        # 1 DRIFTED of 1 selected -> fidelity 0.5
        assert report.fidelity_score == 0.5
        # confabulation = (missed + drifted + phantom) / total = (0 + 1 + 0) / 1
        assert report.confabulation_score == 1.0

    def test_zero_selected_recs_yields_zero_scores(self):
        diff = ModalityChecker().check_diff("A.", "A.")
        rec = Recommendation(
            rec_id="REC-x", rec_type="INSERT",
            expected_modality=Modality.PROHIBITION,
            expected_subject="Licensee",
            selected=False,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        assert report.fidelity_score == 0.0
        assert report.confabulation_score == 0.0
        assert report.total_selected == 0


# ---------------------------------------------------------------------------
# RESTRUCTURE silently excluded
# ---------------------------------------------------------------------------

class TestRestructure:

    def test_restructure_recs_omitted_from_report(self):
        diff = ModalityChecker().check_diff("A.", "A.")
        recs = [
            Recommendation(rec_id="REC-200", rec_type="RESTRUCTURE",
                           summary="reorder sections", selected=True),
            Recommendation(rec_id="REC-201", rec_type="INSERT",
                           expected_modality=Modality.PROHIBITION,
                           expected_subject="Licensee", selected=True),
        ]
        report = RedlineVerdictAssigner().assign(diff, recs)
        rec_ids = {v.rec_id for v in report.verdicts}
        assert "REC-200" not in rec_ids
        assert "REC-201" in rec_ids
        # RESTRUCTURE excluded from total_selected too
        assert report.total_selected == 1


# ---------------------------------------------------------------------------
# Integration: HARM-POL canonical case (memo says A, redline does B-on-wrong-party)
# ---------------------------------------------------------------------------

class TestHarmPolCase:
    """The skill's headline scenario: a MODIFY rec asks for a permission
    on the Licensee, but the redline grants the permission to Licensor
    (party misattribution / risk polarity inversion)."""

    def test_polarity_inversion_yields_drifted(self):
        diff = ModalityChecker().check_diff(
            "Licensee shall indemnify the other party.",
            "Licensor shall indemnify the other party.",
        )
        rec = Recommendation(
            rec_id="REC-POL", rec_type="MODIFY",
            summary="Licensee's indemnification should remain on Licensee but become permissive.",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.PERMISSION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec])
        v = report.verdicts[0]
        assert v.verdict == Verdict.DRIFTED
        assert len(v.drift_pairs) == 1
        # The redlined side bound the wrong party.
        assert v.drift_pairs[0].redlined.subject == "Licensor"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_report_roundtrips_through_json(self):
        diff = ModalityChecker().check_diff("A.", "Licensee shall not disclose.")
        rec = Recommendation(
            rec_id="REC-300", rec_type="INSERT",
            expected_modality=Modality.PROHIBITION,
            expected_subject="Licensee",
            selected=True,
        )
        report = RedlineVerdictAssigner().assign(diff, [rec], contract_id="c1")
        blob = json.dumps(report.to_dict())
        restored = json.loads(blob)
        assert restored["contract_id"] == "c1"
        assert restored["verdicts"][0]["verdict"] == "FAITHFUL"
        assert "fidelity_score" in restored
        assert "confabulation_score" in restored

    def test_recommendation_to_dict(self):
        rec = Recommendation(
            rec_id="REC-9", rec_type="MODIFY",
            summary="x",
            before_modality=Modality.OBLIGATION,
            before_subject="Licensee",
            expected_modality=Modality.PERMISSION,
            expected_subject="Licensee",
            selected=True,
        )
        d = rec.to_dict()
        assert d["rec_id"] == "REC-9"
        assert d["rec_type"] == "MODIFY"
        assert d["before_modality"] == "OBLIGATION"
        assert d["expected_modality"] == "PERMISSION"


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_diff_and_no_recs(self):
        diff = ModalityChecker().check_diff("", "")
        report = RedlineVerdictAssigner().assign(diff, [])
        assert report.verdicts == []
        assert report.phantom_findings == []
        assert report.fidelity_score == 0.0

    def test_counts_by_verdict_includes_all_keys(self):
        diff = ModalityChecker().check_diff("A.", "A.")
        report = RedlineVerdictAssigner().assign(diff, [])
        for v in Verdict:
            assert v.value in report.counts_by_verdict
