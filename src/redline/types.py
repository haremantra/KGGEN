"""Data models for the redline-compliance verdict layer.

Bridges ``src.modality.ModalityDiff`` (an original-vs-redlined diff
primitive) with the observed-redline-analysis domain vocabulary:
FAITHFUL / MISSED / DRIFTED / PHANTOM / EXPECTED_SKIP.

This module deliberately stays caller-driven: it does NOT parse memos.
The caller constructs ``Recommendation`` instances describing what each
memo rec asks for, then ``RedlineVerdictAssigner.assign`` matches them
against a ``ModalityDiff`` and emits one ``VerdictAssignment`` per
supported recommendation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from ..modality.types import DriftPair, ModalFinding, Modality


class Verdict(str, Enum):
    """Per-recommendation verdict, matching observed-redline-analysis."""
    FAITHFUL = "FAITHFUL"
    MISSED = "MISSED"
    DRIFTED = "DRIFTED"
    PHANTOM = "PHANTOM"
    EXPECTED_SKIP = "EXPECTED_SKIP"


RecType = Literal["MODIFY", "INSERT", "DELETE", "RESTRUCTURE"]


@dataclass
class Recommendation:
    """A single memo recommendation expressed as expected modality state.

    Fields:
        rec_id: stable identifier (e.g., "REC-001").
        rec_type: MODIFY / INSERT / DELETE / RESTRUCTURE.
        summary: one-line description for human review.
        selected: whether the memo flagged this rec for redlining.
        expected_modality: the modality the redline should produce
            (for MODIFY/INSERT) or the original-state modality
            (for DELETE).
        expected_subject: the normalized party who should be bound.
            (Compared against ``modality.normalize_subject``.)
        before_modality / before_subject: for MODIFY only — the original
            state's modality/subject prior to the change. Lets the
            assigner verify that a drift pair matches both endpoints.
    """
    rec_id: str
    rec_type: RecType
    summary: str = ""
    selected: bool = True
    expected_modality: Modality | None = None
    expected_subject: str | None = None
    before_modality: Modality | None = None
    before_subject: str | None = None

    def to_dict(self) -> dict:
        return {
            "rec_id": self.rec_id,
            "rec_type": self.rec_type,
            "summary": self.summary,
            "selected": self.selected,
            "expected_modality": (
                self.expected_modality.value if self.expected_modality else None
            ),
            "expected_subject": self.expected_subject,
            "before_modality": (
                self.before_modality.value if self.before_modality else None
            ),
            "before_subject": self.before_subject,
        }


@dataclass
class VerdictAssignment:
    """Verdict for a single recommendation, with supporting evidence."""
    rec_id: str
    rec_type: RecType
    verdict: Verdict
    matched_findings: list[ModalFinding] = field(default_factory=list)
    drift_pairs: list[DriftPair] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "rec_id": self.rec_id,
            "rec_type": self.rec_type,
            "verdict": self.verdict.value,
            "matched_findings": [f.to_dict() for f in self.matched_findings],
            "drift_pairs": [d.to_dict() for d in self.drift_pairs],
            "notes": self.notes,
        }


@dataclass
class RedlineComplianceReport:
    """Aggregated verdict report for one contract's redline against a memo."""
    contract_id: str
    verdicts: list[VerdictAssignment] = field(default_factory=list)
    phantom_findings: list[ModalFinding] = field(default_factory=list)
    counts_by_verdict: dict[str, int] = field(default_factory=dict)
    confabulation_score: float = 0.0
    fidelity_score: float = 0.0
    total_selected: int = 0

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "phantom_findings": [f.to_dict() for f in self.phantom_findings],
            "counts_by_verdict": dict(self.counts_by_verdict),
            "confabulation_score": self.confabulation_score,
            "fidelity_score": self.fidelity_score,
            "total_selected": self.total_selected,
        }
