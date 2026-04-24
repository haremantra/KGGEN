"""Synchronous, rule-based deontic modality checker.

No LLM, no network, no file I/O. Duck-types on ``AnalyzedClause`` and
``ContractAnalysis`` shapes from ``src.pipeline`` to stay import-light.
"""

import re

from .rules import find_modal_matches
from .types import ModalFinding, Modality, ModalityReport


_SUBJECT_BOUNDARY = re.compile(r"[.;:,!?]")
_WORD = re.compile(r"\S+")


def _extract_subject(text: str, modal_start: int) -> str | None:
    """Return up to 6 words preceding ``modal_start`` within the same clause.

    Returns None if the modal is sentence-initial (nothing meaningful before it).
    Clause boundaries are terminated by ``. ; : , ! ?``.
    """
    pre = text[:modal_start]
    # Walk backwards to nearest clause boundary
    boundary = 0
    for m in _SUBJECT_BOUNDARY.finditer(pre):
        boundary = m.end()
    segment = pre[boundary:].strip()
    if not segment:
        return None

    words = segment.split()
    if not words:
        return None
    tail = words[-6:]
    subject = " ".join(tail).strip()
    return subject or None


class ModalityChecker:
    """Detects deontic modality in clause text using ordered regex rules."""

    def __init__(self) -> None:
        # Pure rule-based; no heavy init.
        pass

    def check_text(
        self, text: str, cuad_label: str | None = None
    ) -> list[ModalFinding]:
        """Scan a single string and return its modal findings."""
        if not text or not text.strip():
            return []

        findings: list[ModalFinding] = []
        for rule, match in find_modal_matches(text):
            start, end = match.span()
            findings.append(ModalFinding(
                modality=rule.modality,
                modal_phrase=match.group(0),
                span=(start, end),
                subject=_extract_subject(text, start),
                confidence=rule.strength,
                cuad_label=cuad_label,
                clause_text=text,
            ))
        return findings

    def check_clause(self, clause) -> list[ModalFinding]:
        """Scan an ``AnalyzedClause``-shaped object (duck-typed on .text/.cuad_label)."""
        text = getattr(clause, "text", "") or ""
        label = getattr(clause, "cuad_label", None)
        return self.check_text(text, cuad_label=label)

    def check_analysis(self, analysis) -> ModalityReport:
        """Scan a ``ContractAnalysis``-shaped object and aggregate counts.

        Duck-typed on ``.contract_id`` and ``.analyzed_clauses``.
        """
        contract_id = getattr(analysis, "contract_id", "") or ""
        clauses = getattr(analysis, "analyzed_clauses", []) or []

        all_findings: list[ModalFinding] = []
        for clause in clauses:
            all_findings.extend(self.check_clause(clause))

        counts: dict[str, int] = {m.value: 0 for m in Modality}
        for f in all_findings:
            counts[f.modality.value] = counts.get(f.modality.value, 0) + 1

        return ModalityReport(
            contract_id=contract_id,
            findings=all_findings,
            counts=counts,
        )
