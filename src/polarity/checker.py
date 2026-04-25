"""PolarityChecker — decision-support detector for party misattribution signals.

Consumes ``ContractAnalysis.analyzed_clauses`` (each carrying ``modality_findings``
populated by the pipeline; see ``src/modality``) and produces a ``PolarityReport``
with per-clause profiles and review-worthy findings.

Two finding kinds in this slice:

- ``MUTUAL_DRIFT`` — within a single clause, modality findings span both mutual
  subjects ("either party", "neither party") and named-party subjects.
  Possible signal that a memo's mutual recommendation drifted into unilateral
  language during implementation.

- ``DUPLICATE_INCONSISTENCY`` — the same CUAD label appears in multiple
  AnalyzedClauses with conflicting subject/modality patterns.

Out of scope (deferred):

- A static expected-direction map per CUAD label (e.g., "Indemnification
  obligates Vendor"). Requires per-label legal judgment and role assignment.
- Memo-vs-redline diff. Belongs in the diff-against-baseline slice.
"""

import re
from collections import Counter, defaultdict

from ..modality import normalize_subject as _normalize_subject

from .types import (
    ClausePolarityProfile,
    PolarityFinding,
    PolarityKind,
    PolarityReport,
    PolaritySeverity,
    PolarityVerdict,
    SubjectKind,
)


_MUTUAL_PATTERNS = re.compile(
    r"^(?:either\s+party|both\s+parties|the\s+parties|each\s+party|the\s+two\s+parties|mutual(?:ly)?)$"
)
_NEGATIVE_PATTERNS = re.compile(
    r"^(?:neither\s+party|no\s+party)$"
)


def _classify_subject_kind(normalized: str | None) -> SubjectKind:
    """Return SubjectKind for a normalized subject string."""
    if not normalized:
        return SubjectKind.NONE
    if _MUTUAL_PATTERNS.search(normalized):
        return SubjectKind.MUTUAL
    if _NEGATIVE_PATTERNS.search(normalized):
        return SubjectKind.NEGATIVE
    return SubjectKind.NAMED


def _verdict_from(modality_counts: dict[str, int], subject_kinds: set[str]) -> PolarityVerdict:
    """Derive a PolarityVerdict from modality + subject-kind distribution."""
    if not modality_counts:
        return PolarityVerdict.UNDETERMINED

    total = sum(modality_counts.values())
    if total == 0:
        return PolarityVerdict.UNDETERMINED

    # Mutual subjects dominate → MUTUAL.
    if SubjectKind.MUTUAL.value in subject_kinds and SubjectKind.NAMED.value not in subject_kinds:
        return PolarityVerdict.MUTUAL

    nonzero = {m: c for m, c in modality_counts.items() if c > 0}
    if len(nonzero) == 1 and SubjectKind.NAMED.value in subject_kinds:
        only = next(iter(nonzero))
        if only == "OBLIGATION":
            return PolarityVerdict.SINGLE_PARTY_OBLIGATED
        if only == "PROHIBITION":
            return PolarityVerdict.SINGLE_PARTY_PROHIBITED
        if only == "PERMISSION":
            return PolarityVerdict.SINGLE_PARTY_PERMITTED

    if len(nonzero) >= 2:
        return PolarityVerdict.MIXED

    return PolarityVerdict.UNDETERMINED


class PolarityChecker:
    """Detects party-misattribution signals from modality + subject data."""

    def __init__(self) -> None:
        # Stateless — no LLM, no model load.
        pass

    def profile_clause(self, clause) -> ClausePolarityProfile:
        """Build a polarity profile for a single AnalyzedClause."""
        cuad_label = getattr(clause, "cuad_label", "") or ""
        category = getattr(clause, "category", "") or ""
        findings = getattr(clause, "modality_findings", []) or []

        norm_subjects: list[str] = []
        kinds: list[str] = []
        modality_counts: dict[str, int] = defaultdict(int)

        for f in findings:
            n = _normalize_subject(getattr(f, "subject", None))
            if n is not None:
                norm_subjects.append(n)
            kinds.append(_classify_subject_kind(n).value)
            mod = getattr(f, "modality", None)
            mod_value = mod.value if hasattr(mod, "value") else str(mod)
            modality_counts[mod_value] += 1

        # Dedupe subjects while preserving first-seen order.
        seen: set[str] = set()
        deduped: list[str] = []
        for s in norm_subjects:
            if s not in seen:
                seen.add(s)
                deduped.append(s)

        verdict = _verdict_from(dict(modality_counts), set(kinds))

        return ClausePolarityProfile(
            cuad_label=cuad_label,
            category=category,
            subjects=deduped,
            subject_kinds=sorted(set(kinds)),
            modality_counts=dict(modality_counts),
            verdict=verdict,
        )

    def check_clause(self, clause) -> list[PolarityFinding]:
        """Detect within-clause polarity signals (MUTUAL_DRIFT)."""
        findings = getattr(clause, "modality_findings", []) or []
        if not findings:
            return []

        kinds = {_classify_subject_kind(_normalize_subject(getattr(f, "subject", None))).value
                 for f in findings}

        results: list[PolarityFinding] = []

        # MUTUAL_DRIFT: clause has both mutual/negative AND named subjects.
        has_mutual_or_negative = bool(
            kinds & {SubjectKind.MUTUAL.value, SubjectKind.NEGATIVE.value}
        )
        has_named = SubjectKind.NAMED.value in kinds
        if has_mutual_or_negative and has_named:
            cuad_label = getattr(clause, "cuad_label", "") or ""
            phrases = [getattr(f, "modal_phrase", "") for f in findings]
            named_subjects = sorted({
                _normalize_subject(getattr(f, "subject", None))
                for f in findings
                if _classify_subject_kind(
                    _normalize_subject(getattr(f, "subject", None))
                ) == SubjectKind.NAMED
            } - {None})
            mutual_subjects = sorted({
                _normalize_subject(getattr(f, "subject", None))
                for f in findings
                if _classify_subject_kind(
                    _normalize_subject(getattr(f, "subject", None))
                ) in {SubjectKind.MUTUAL, SubjectKind.NEGATIVE}
            } - {None})
            results.append(PolarityFinding(
                kind=PolarityKind.MUTUAL_DRIFT,
                severity=PolaritySeverity.MODERATE,
                fm_code="FM-B06",
                cuad_label=cuad_label,
                subjects=list(mutual_subjects) + list(named_subjects),
                description=(
                    f"Clause '{cuad_label}' mixes mutual-language subjects "
                    f"({', '.join(mutual_subjects)}) with named-party subjects "
                    f"({', '.join(named_subjects)}). Possible drift from a mutual "
                    f"recommendation into unilateral language."
                ),
                evidence_phrases=[p for p in phrases if p],
            ))

        return results

    def check_analysis(self, analysis) -> PolarityReport:
        """Build profiles + findings across the full ContractAnalysis."""
        contract_id = getattr(analysis, "contract_id", "") or ""
        clauses = getattr(analysis, "analyzed_clauses", []) or []

        profiles: list[ClausePolarityProfile] = []
        findings: list[PolarityFinding] = []

        # Per-clause: profile + within-clause checks.
        for clause in clauses:
            profiles.append(self.profile_clause(clause))
            findings.extend(self.check_clause(clause))

        # Cross-clause: same CUAD label appearing in ≥2 clauses with conflicting
        # (named subject, modality) patterns.
        by_label: dict[str, list[ClausePolarityProfile]] = defaultdict(list)
        for prof in profiles:
            if prof.cuad_label:
                by_label[prof.cuad_label].append(prof)

        for label, profs in by_label.items():
            if len(profs) < 2:
                continue
            # Flatten (named_subject, modality) pairs across the duplicates.
            pairs: set[tuple[str, str]] = set()
            for p in profs:
                named = [s for s in p.subjects
                         if _classify_subject_kind(s) == SubjectKind.NAMED]
                for s in named:
                    for mod, count in p.modality_counts.items():
                        if count > 0:
                            pairs.add((s, mod))
            if not pairs:
                continue
            # Inconsistency: same modality on different named subjects, OR
            # different modalities on the same named subject.
            subjects = {s for s, _ in pairs}
            modalities = {m for _, m in pairs}
            inconsistent = (len(subjects) >= 2 or len(modalities) >= 2)
            if inconsistent:
                findings.append(PolarityFinding(
                    kind=PolarityKind.DUPLICATE_INCONSISTENCY,
                    severity=PolaritySeverity.MODERATE,
                    fm_code="FM-D02",
                    cuad_label=label,
                    subjects=sorted(subjects),
                    description=(
                        f"CUAD label '{label}' appears in {len(profs)} clauses "
                        f"with conflicting (subject, modality) patterns: "
                        f"{sorted(pairs)}."
                    ),
                ))

        # Severity counts.
        sev_counter: Counter = Counter(f.severity.value for f in findings)
        # Ensure all severities are present (zero where absent) for consumers
        # that index by key.
        counts_by_severity = {sev.value: sev_counter.get(sev.value, 0)
                              for sev in PolaritySeverity}

        return PolarityReport(
            contract_id=contract_id,
            profiles=profiles,
            findings=findings,
            counts_by_severity=counts_by_severity,
        )
