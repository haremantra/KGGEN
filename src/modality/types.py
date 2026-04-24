"""Data models for deontic modality analysis.

Defines the modality taxonomy (obligation / permission / prohibition /
entitlement) and the finding/report dataclasses returned by the checker.
"""

from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    """Deontic modality classes for legal clauses."""
    OBLIGATION = "OBLIGATION"
    PERMISSION = "PERMISSION"
    PROHIBITION = "PROHIBITION"
    ENTITLEMENT = "ENTITLEMENT"
    NONE = "NONE"


@dataclass
class ModalFinding:
    """A single modal phrase detected in a piece of clause text."""
    modality: Modality
    modal_phrase: str
    span: tuple[int, int]
    subject: str | None = None
    confidence: float = 1.0
    cuad_label: str | None = None
    clause_text: str = ""

    def to_dict(self) -> dict:
        text = self.clause_text
        if len(text) > 200:
            text = text[:200] + "..."
        return {
            "modality": self.modality.value,
            "modal_phrase": self.modal_phrase,
            "span": list(self.span),
            "subject": self.subject,
            "confidence": self.confidence,
            "cuad_label": self.cuad_label,
            "clause_text": text,
        }


@dataclass
class ModalityReport:
    """Aggregated modality findings across a contract analysis."""
    contract_id: str
    findings: list[ModalFinding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "findings": [f.to_dict() for f in self.findings],
            "counts": dict(self.counts),
        }
