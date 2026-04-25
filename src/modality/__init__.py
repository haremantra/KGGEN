"""Deontic modality checker for contract clauses.

Synchronous, rule-based detection of obligation / permission / prohibition /
entitlement modality. First slice: standalone module whose pytest suite serves
as the executable spec. Not yet wired into the main analysis pipeline.

Forward trajectory
==================

Modality + subject extraction is the primitive for detecting **party
misattribution** ("FM-B06" in redline-compliance taxonomies) and **risk
polarity inversion** — e.g., an indemnification clause that obligates the
licensee instead of the licensor. A clause with the right modal verb on the
wrong party reads as legally competent and passes surface-level review, so
this primitive carries weight beyond its size.

Planned downstream uses (not in this slice):

* ``ContractAnalysisPipeline`` integration: attach ``modality_findings`` to
  each ``AnalyzedClause`` so risk and interdependency analyses can read them.
* ``RiskAssessor`` / ``DependencyDetector``: compare ``(modality, subject)``
  pairs across clauses to flag contradictions and polarity inversions.
* Diff-against-baseline (``check_diff(original, redlined)``) for redline
  fidelity / confabulation detection.

Strength is intentionally categorical (``LOW`` / ``MEDIUM`` / ``HIGH``) — see
``src/extraction/extractor.py`` ``CONFIDENCE_MAP`` for the same convention.
Numeric confidence in a regex-only layer would imply false precision.
"""

from .types import Modality, ModalFinding, ModalityReport, Strength
from .rules import ModalRule, MODAL_RULES, find_modal_matches
from .checker import ModalityChecker

__all__ = [
    "Modality",
    "ModalFinding",
    "ModalityReport",
    "Strength",
    "ModalRule",
    "MODAL_RULES",
    "find_modal_matches",
    "ModalityChecker",
]
