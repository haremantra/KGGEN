"""Data models for polarity analysis.

Polarity = where the modal force points (which party is bound, permitted, or
prohibited). This module is decision-support: it surfaces signals worth a
human review, not autonomous alarms. Severity tops out at HIGH for now;
CRITICAL is reserved for memo-vs-redline diffs (see slice 5).

FM codes follow the redline-compliance taxonomy:
- FM-B06: Party Misattribution (obligation/prohibition assigned to wrong party).
- FM-D02: Duplicate Implementation (same recommendation in multiple clauses
  with inconsistent language).
"""

from dataclasses import dataclass, field
from enum import Enum


class PolaritySeverity(str, Enum):
    """Severity of a polarity finding."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolarityKind(str, Enum):
    """What kind of polarity signal a finding represents."""
    MUTUAL_DRIFT = "MUTUAL_DRIFT"
    DUPLICATE_INCONSISTENCY = "DUPLICATE_INCONSISTENCY"


class PolarityVerdict(str, Enum):
    """Per-clause modality+subject summary."""
    SINGLE_PARTY_OBLIGATED = "SINGLE_PARTY_OBLIGATED"
    SINGLE_PARTY_PROHIBITED = "SINGLE_PARTY_PROHIBITED"
    SINGLE_PARTY_PERMITTED = "SINGLE_PARTY_PERMITTED"
    MUTUAL = "MUTUAL"
    MIXED = "MIXED"
    UNDETERMINED = "UNDETERMINED"


class SubjectKind(str, Enum):
    """Coarse classification of a normalized subject string."""
    NAMED = "NAMED"          # specific party, e.g. "licensee", "licensor"
    MUTUAL = "MUTUAL"        # "either party", "both parties", "the parties"
    NEGATIVE = "NEGATIVE"    # "neither party", "no party"
    NONE = "NONE"            # subject was None / empty


@dataclass
class ClausePolarityProfile:
    """Per-clause modality+subject summary."""
    cuad_label: str
    category: str
    subjects: list[str] = field(default_factory=list)            # normalized
    subject_kinds: list[str] = field(default_factory=list)       # SubjectKind values
    modality_counts: dict[str, int] = field(default_factory=dict)
    verdict: PolarityVerdict = PolarityVerdict.UNDETERMINED

    def to_dict(self) -> dict:
        return {
            "cuad_label": self.cuad_label,
            "category": self.category,
            "subjects": list(self.subjects),
            "subject_kinds": list(self.subject_kinds),
            "modality_counts": dict(self.modality_counts),
            "verdict": self.verdict.value,
        }


@dataclass
class PolarityFinding:
    """A polarity signal worth human review."""
    kind: PolarityKind
    severity: PolaritySeverity
    fm_code: str
    cuad_label: str
    subjects: list[str] = field(default_factory=list)
    description: str = ""
    evidence_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "fm_code": self.fm_code,
            "cuad_label": self.cuad_label,
            "subjects": list(self.subjects),
            "description": self.description,
            "evidence_phrases": list(self.evidence_phrases),
        }


@dataclass
class PolarityReport:
    """Full polarity analysis for a contract."""
    contract_id: str
    profiles: list[ClausePolarityProfile] = field(default_factory=list)
    findings: list[PolarityFinding] = field(default_factory=list)
    counts_by_severity: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "profiles": [p.to_dict() for p in self.profiles],
            "findings": [f.to_dict() for f in self.findings],
            "counts_by_severity": dict(self.counts_by_severity),
        }
