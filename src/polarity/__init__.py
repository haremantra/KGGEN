"""Party-misattribution / polarity-inversion detection.

Decision-support layer over ``src/modality``. Consumes the ``modality_findings``
attached to each ``AnalyzedClause`` by the pipeline and surfaces signals worth
human review — primarily MUTUAL_DRIFT (within-clause) and DUPLICATE_INCONSISTENCY
(cross-clause) using the redline-compliance FM taxonomy (FM-B06, FM-D02).

This module never claims a finding is autonomously "wrong" — that requires a
memo/baseline comparison (see future ``check_diff`` slice). It marks signals
worth a lawyer's attention.
"""

from .types import (
    ClausePolarityProfile,
    PolarityFinding,
    PolarityKind,
    PolarityReport,
    PolaritySeverity,
    PolarityVerdict,
    SubjectKind,
)
from .checker import PolarityChecker

__all__ = [
    "ClausePolarityProfile",
    "PolarityChecker",
    "PolarityFinding",
    "PolarityKind",
    "PolarityReport",
    "PolaritySeverity",
    "PolarityVerdict",
    "SubjectKind",
]
