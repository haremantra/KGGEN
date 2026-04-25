"""Synchronous, rule-based deontic modality checker.

No LLM, no network, no file I/O. Duck-types on ``AnalyzedClause`` and
``ContractAnalysis`` shapes from ``src.pipeline`` to stay import-light.
"""

import re
from collections import Counter

from .rules import find_modal_matches
from .types import (
    DriftKind,
    DriftPair,
    ModalFinding,
    Modality,
    ModalityDiff,
    ModalityReport,
)


# Subject extraction stops at sentence/clause punctuation OR at a paragraph
# break (blank line). Single newlines are intentionally NOT boundaries —
# legal prose often line-wraps mid-clause without breaking the subject.
_SUBJECT_BOUNDARY = re.compile(r"[.;:,!?]|\n\s*\n")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_subject(subject: str | None) -> str | None:
    """Lowercase, strip leading article, collapse whitespace.

    Returns None for None or all-whitespace input. Public so other modules
    (e.g. ``src.polarity``) can match findings on the same canonical form
    without duplicating the rule.
    """
    if subject is None:
        return None
    s = subject.strip().lower()
    if not s:
        return None
    s = _WHITESPACE.sub(" ", s)
    s = _LEADING_ARTICLE.sub("", s, count=1)
    return s or None


def _extract_subject(text: str, modal_start: int) -> str | None:
    """Return up to 6 words preceding ``modal_start`` within the same clause.

    Returns None if the modal is sentence-initial (nothing meaningful before it).
    Clause boundaries are terminated by ``. ; : , ! ?`` or a blank line.
    """
    pre = text[:modal_start]
    # Walk forward through all boundaries; the last one before modal_start
    # marks the start of our subject window.
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
                strength=rule.strength,
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

    def check_diff(
        self,
        original_text: str,
        redlined_text: str,
        cuad_label: str | None = None,
    ) -> ModalityDiff:
        """Compare modality findings between two text states.

        Returns a structured diff classifying each finding as kept, added,
        removed, or drifted (subject changed and/or modality changed). Use
        as the primitive for FAITHFUL/MISSED/DRIFTED/PHANTOM verdicts in a
        redline-compliance layer.

        Greedy 1-to-1 matching across four passes:
            1. Exact (normalized_subject, modality) -> kept.
            2. Same (modality, modal_phrase) but different subject -> SUBJECT drift.
            3. Same (normalized_subject, modal_phrase) but different modality -> BOTH-style.
               Treated as MODALITY drift (the more salient axis).
            4. Same normalized_subject, different modality -> MODALITY drift.
            5. Anything left in original -> removed; anything left in redlined -> added.
        """
        orig = self.check_text(original_text, cuad_label=cuad_label)
        red = self.check_text(redlined_text, cuad_label=cuad_label)

        used_orig = [False] * len(orig)
        used_red = [False] * len(red)

        kept: list[ModalFinding] = []
        drifted: list[DriftPair] = []

        def _norm_phrase(p: str) -> str:
            return (p or "").lower()

        # Pass 1: exact (norm_subject, modality)
        for i, of in enumerate(orig):
            o_key = (normalize_subject(of.subject), of.modality)
            for j, rf in enumerate(red):
                if used_red[j]:
                    continue
                r_key = (normalize_subject(rf.subject), rf.modality)
                if o_key == r_key:
                    kept.append(of)
                    used_orig[i] = True
                    used_red[j] = True
                    break

        # Pass 2: same (modality, phrase), different subject -> SUBJECT drift
        for i, of in enumerate(orig):
            if used_orig[i]:
                continue
            for j, rf in enumerate(red):
                if used_red[j]:
                    continue
                if (of.modality == rf.modality
                        and _norm_phrase(of.modal_phrase) == _norm_phrase(rf.modal_phrase)
                        and normalize_subject(of.subject) != normalize_subject(rf.subject)):
                    drifted.append(DriftPair(
                        original=of, redlined=rf, drift_kind=DriftKind.SUBJECT,
                    ))
                    used_orig[i] = True
                    used_red[j] = True
                    break

        # Pass 3: same normalized subject (non-None), different modality -> MODALITY drift
        for i, of in enumerate(orig):
            if used_orig[i]:
                continue
            o_subj = normalize_subject(of.subject)
            if o_subj is None:
                continue
            for j, rf in enumerate(red):
                if used_red[j]:
                    continue
                if (normalize_subject(rf.subject) == o_subj
                        and of.modality != rf.modality):
                    drifted.append(DriftPair(
                        original=of, redlined=rf, drift_kind=DriftKind.MODALITY,
                    ))
                    used_orig[i] = True
                    used_red[j] = True
                    break

        removed = [of for i, of in enumerate(orig) if not used_orig[i]]
        added = [rf for j, rf in enumerate(red) if not used_red[j]]

        # Summary counts.
        counts = {
            "kept": len(kept),
            "added": len(added),
            "removed": len(removed),
            "drifted_subject": sum(1 for d in drifted if d.drift_kind == DriftKind.SUBJECT),
            "drifted_modality": sum(1 for d in drifted if d.drift_kind == DriftKind.MODALITY),
        }

        return ModalityDiff(
            kept=kept,
            added=added,
            removed=removed,
            drifted=drifted,
            counts=counts,
        )
