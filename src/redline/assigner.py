"""RedlineVerdictAssigner.

Maps a ``ModalityDiff`` plus a list of ``Recommendation`` instances to a
``RedlineComplianceReport`` with FAITHFUL / MISSED / DRIFTED / PHANTOM /
EXPECTED_SKIP verdicts. Stateless; no LLM, no I/O.

Matching strategy per rec_type:

INSERT
    Look for a ``ModalityDiff.added`` finding matching
    (expected_modality, expected_subject) — both required.
    Found      → FAITHFUL (mark finding as consumed).
    Not found  → MISSED if selected, EXPECTED_SKIP if not.

DELETE
    Look for a ``ModalityDiff.removed`` finding matching
    (before_modality, before_subject) — falling back to
    (expected_modality, expected_subject) for callers that didn't
    bother distinguishing.
    Found     → FAITHFUL.
    Not found → MISSED / EXPECTED_SKIP as above.

MODIFY
    Look for a ``DriftPair`` whose ``original`` matches the rec's
    before-state and whose ``redlined`` matches its expected state.
    Both endpoints match exactly       → FAITHFUL.
    Original matches, redlined doesn't → DRIFTED.
    No matching pair found             → MISSED / EXPECTED_SKIP.

RESTRUCTURE recs are excluded from the report — structural changes
aren't observable from modality findings alone.

After all recs are processed, any remaining ``added`` findings or
``drifted.redlined`` sides not consumed by a FAITHFUL/DRIFTED match are
collected as ``phantom_findings`` (FM-C03 confabulation candidates).
"""

from __future__ import annotations

from collections import Counter

from ..modality import normalize_subject
from ..modality.types import DriftPair, ModalFinding, ModalityDiff
from .types import (
    Recommendation,
    RedlineComplianceReport,
    Verdict,
    VerdictAssignment,
)


def _matches_finding(
    finding: ModalFinding,
    expected_modality,
    expected_subject_norm: str | None,
) -> bool:
    """Whether a finding matches an expected (modality, normalized_subject)."""
    if expected_modality is not None and finding.modality != expected_modality:
        return False
    if expected_subject_norm is not None:
        if normalize_subject(finding.subject) != expected_subject_norm:
            return False
    return True


class RedlineVerdictAssigner:
    """Assigns FAITHFUL / MISSED / DRIFTED / PHANTOM / EXPECTED_SKIP verdicts."""

    def assign(
        self,
        diff: ModalityDiff,
        recommendations: list[Recommendation],
        contract_id: str = "",
    ) -> RedlineComplianceReport:
        # Working pools — we mark consumed entries as we match recs.
        added_used = [False] * len(diff.added)
        removed_used = [False] * len(diff.removed)
        drift_used = [False] * len(diff.drifted)

        verdicts: list[VerdictAssignment] = []

        for rec in recommendations:
            if rec.rec_type == "RESTRUCTURE":
                continue  # not observable from modality alone

            if rec.rec_type == "INSERT":
                verdicts.append(
                    self._assign_insert(rec, diff, added_used)
                )
            elif rec.rec_type == "DELETE":
                verdicts.append(
                    self._assign_delete(rec, diff, removed_used)
                )
            elif rec.rec_type == "MODIFY":
                verdicts.append(
                    self._assign_modify(rec, diff, drift_used)
                )

        # Anything left in `added` or `drifted.redlined` that wasn't claimed
        # by a FAITHFUL/DRIFTED verdict is a PHANTOM candidate — it appeared
        # in the redline without a matching memo recommendation.
        phantoms: list[ModalFinding] = []
        for i, f in enumerate(diff.added):
            if not added_used[i]:
                phantoms.append(f)
        for i, dp in enumerate(diff.drifted):
            if not drift_used[i]:
                phantoms.append(dp.redlined)

        # Tallies
        counts: Counter = Counter(v.verdict.value for v in verdicts)
        counts_by_verdict = {v.value: counts.get(v.value, 0) for v in Verdict}
        # PHANTOM is tracked separately in phantom_findings; reflect the count.
        counts_by_verdict[Verdict.PHANTOM.value] = len(phantoms)

        total_selected = sum(1 for r in recommendations
                             if r.selected and r.rec_type != "RESTRUCTURE")
        faithful = counts_by_verdict[Verdict.FAITHFUL.value]
        drifted = counts_by_verdict[Verdict.DRIFTED.value]
        missed = counts_by_verdict[Verdict.MISSED.value]
        phantom = counts_by_verdict[Verdict.PHANTOM.value]

        # Skill-aligned formulas:
        #   fidelity = (FAITHFUL + 0.5 * DRIFTED) / total_selected
        #   confabulation = (MISSED + DRIFTED + PHANTOM) / total_selected
        if total_selected > 0:
            fidelity_score = (faithful + 0.5 * drifted) / total_selected
            confabulation_score = (missed + drifted + phantom) / total_selected
        else:
            fidelity_score = 0.0
            confabulation_score = 0.0

        return RedlineComplianceReport(
            contract_id=contract_id,
            verdicts=verdicts,
            phantom_findings=phantoms,
            counts_by_verdict=counts_by_verdict,
            confabulation_score=confabulation_score,
            fidelity_score=fidelity_score,
            total_selected=total_selected,
        )

    # ------------------------------------------------------------------ #
    # Per-rec_type assignment helpers
    # ------------------------------------------------------------------ #

    def _assign_insert(
        self,
        rec: Recommendation,
        diff: ModalityDiff,
        added_used: list[bool],
    ) -> VerdictAssignment:
        expected_subj_norm = normalize_subject(rec.expected_subject)
        for i, f in enumerate(diff.added):
            if added_used[i]:
                continue
            if _matches_finding(f, rec.expected_modality, expected_subj_norm):
                added_used[i] = True
                return VerdictAssignment(
                    rec_id=rec.rec_id, rec_type="INSERT",
                    verdict=Verdict.FAITHFUL,
                    matched_findings=[f],
                    notes="INSERT rec satisfied by an `added` finding.",
                )
        return VerdictAssignment(
            rec_id=rec.rec_id, rec_type="INSERT",
            verdict=Verdict.MISSED if rec.selected else Verdict.EXPECTED_SKIP,
            notes=("No `added` finding matches the expected (modality, subject)."
                   if rec.selected else "Unselected; no redline expected."),
        )

    def _assign_delete(
        self,
        rec: Recommendation,
        diff: ModalityDiff,
        removed_used: list[bool],
    ) -> VerdictAssignment:
        # Prefer before_* if provided; fall back to expected_*.
        target_modality = rec.before_modality or rec.expected_modality
        target_subject = rec.before_subject or rec.expected_subject
        target_subj_norm = normalize_subject(target_subject)

        for i, f in enumerate(diff.removed):
            if removed_used[i]:
                continue
            if _matches_finding(f, target_modality, target_subj_norm):
                removed_used[i] = True
                return VerdictAssignment(
                    rec_id=rec.rec_id, rec_type="DELETE",
                    verdict=Verdict.FAITHFUL,
                    matched_findings=[f],
                    notes="DELETE rec satisfied by a `removed` finding.",
                )
        return VerdictAssignment(
            rec_id=rec.rec_id, rec_type="DELETE",
            verdict=Verdict.MISSED if rec.selected else Verdict.EXPECTED_SKIP,
            notes=("No `removed` finding matches the original (modality, subject)."
                   if rec.selected else "Unselected; no removal expected."),
        )

    def _assign_modify(
        self,
        rec: Recommendation,
        diff: ModalityDiff,
        drift_used: list[bool],
    ) -> VerdictAssignment:
        before_subj_norm = normalize_subject(rec.before_subject)
        expected_subj_norm = normalize_subject(rec.expected_subject)

        # First pass: a drift pair whose ORIGINAL side matches the rec's
        # before-state.
        for i, dp in enumerate(diff.drifted):
            if drift_used[i]:
                continue
            if not _matches_finding(dp.original, rec.before_modality, before_subj_norm):
                continue
            # Original side matches. Check the redlined side.
            if _matches_finding(dp.redlined, rec.expected_modality, expected_subj_norm):
                drift_used[i] = True
                return VerdictAssignment(
                    rec_id=rec.rec_id, rec_type="MODIFY",
                    verdict=Verdict.FAITHFUL,
                    drift_pairs=[dp],
                    notes="MODIFY rec satisfied: drift pair matches both endpoints.",
                )
            else:
                drift_used[i] = True
                return VerdictAssignment(
                    rec_id=rec.rec_id, rec_type="MODIFY",
                    verdict=Verdict.DRIFTED,
                    drift_pairs=[dp],
                    notes=("MODIFY rec found a drift pair on the original "
                           "side, but the redlined side doesn't match the "
                           "expected (modality, subject)."),
                )

        return VerdictAssignment(
            rec_id=rec.rec_id, rec_type="MODIFY",
            verdict=Verdict.MISSED if rec.selected else Verdict.EXPECTED_SKIP,
            notes=("No drift pair matches the rec's before-state."
                   if rec.selected else "Unselected; no modification expected."),
        )
