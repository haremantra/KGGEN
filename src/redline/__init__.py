"""Redline-compliance verdict layer.

Bridges ``src.modality.ModalityDiff`` (an original-vs-redlined diff
primitive) with the observed-redline-analysis domain vocabulary:
FAITHFUL / MISSED / DRIFTED / PHANTOM / EXPECTED_SKIP.

Caller-driven: this module does NOT parse memos. Construct a list of
``Recommendation`` instances representing what the memo asked for, then
hand it to ``RedlineVerdictAssigner.assign`` along with a ModalityDiff.

Known limitation inherited from the diff primitive
==================================================

``ModalityChecker.check_diff`` matches findings by
``(normalized_subject, modality)`` and falls back to
``(modality, modal_phrase)`` and ``(normalized_subject)``. It does NOT
inspect the verb that follows the modal, so two genuinely different
statements that share the same ``(subject, modality)`` — e.g.
"Licensee shall pay" and "Licensee shall indemnify" — collide as a
single ``kept`` match.

In contracts where each ``(subject, modality)`` pair is unique within
the original, the verdict layer is precise. When the original repeats
``(subject, modality)`` across distinct statements, expect false
positives in ``kept`` and false negatives in ``drifted``. A
predicate-aware matching upgrade (capture the next ~3 tokens after the
modal verb and include them in the match key) is a planned follow-up.

Forward trajectory (deferred)
=============================

- Predicate-aware diff matching (see above).
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
