"""Redline-compliance verdict layer.

Bridges ``src.modality.ModalityDiff`` (an original-vs-redlined diff
primitive) with the observed-redline-analysis domain vocabulary:
FAITHFUL / MISSED / DRIFTED / PHANTOM / EXPECTED_SKIP.

Caller-driven: this module does NOT parse memos. Construct a list of
``Recommendation`` instances representing what the memo asked for, then
hand it to ``RedlineVerdictAssigner.assign`` along with a ModalityDiff.

Diff matching strategy
======================

``ModalityChecker.check_diff`` is predicate-aware: it matches findings
by ``(normalized_subject, modality, predicate_hint)`` where
``predicate_hint`` is the next ~3 lowercased tokens after the modal
verb. This prevents two distinct statements with the same
``(subject, modality)`` — e.g. "Licensee shall pay" and "Licensee shall
indemnify" — from spuriously collapsing into a single ``kept`` match.

For MODIFY recs, the verdict assigner first looks for a ``DriftPair``
whose endpoints match the rec's before/after states. If none, it falls
back to pairing a ``removed`` finding (matching before-state) with an
``added`` finding (matching expected-state) — the case where the change
was a predicate edit rather than a modality/subject change.

Forward trajectory (deferred)
=============================

- Memo parsing — extract Recommendation list from a structured memo doc.
- Event-sourced emission — write each verdict assignment to a JSONL log
  for replay/audit (mirrors the observed-redline-analysis coordinator).
- Self-review checks: silent-failure audit, deletion verification,
  scalar verification, denominator validation.
- Cross-contract aggregation of confabulation patterns.
"""

from .types import (
    Recommendation,
    RecType,
    RedlineComplianceReport,
    Verdict,
    VerdictAssignment,
)
from .assigner import RedlineVerdictAssigner

__all__ = [
    "Recommendation",
    "RecType",
    "RedlineComplianceReport",
    "RedlineVerdictAssigner",
    "Verdict",
    "VerdictAssignment",
]
