"""Regex rules for deontic modality detection.

Rules are ordered by priority: prohibitions first (so "shall not" wins over
"shall"), then obligations, entitlements, and permissions. Strength is
categorical (HIGH/MEDIUM/LOW) — explicit phrasal patterns are HIGH, bare
modal verbs are MEDIUM. LOW is reserved for future LLM-derived weak signals.
"""

import re
from dataclasses import dataclass
from typing import Literal

from .types import Modality


Strength = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class ModalRule:
    """A single modality regex rule."""
    pattern: re.Pattern
    modality: Modality
    strength: Strength
    name: str


def _compile(pat: str) -> re.Pattern:
    return re.compile(pat, re.IGNORECASE)


# Priority-ordered rules. First match wins per span; overlapping lower-priority
# matches are suppressed by ``find_modal_matches``.
MODAL_RULES: list[ModalRule] = [
    # ---------- PROHIBITION (highest priority) ----------
    ModalRule(
        pattern=_compile(r"\b(?:shall|must|will|may|can)\s+not\b"),
        modality=Modality.PROHIBITION,
        strength="HIGH",
        name="prohibition_modal_not",
    ),
    # Reversed-order legal boilerplate: "IN NO EVENT SHALL ...",
    # "AT NO TIME SHALL ...", "UNDER NO CIRCUMSTANCES SHALL ...".
    ModalRule(
        pattern=_compile(
            r"\b(?:in\s+no\s+event|at\s+no\s+time|under\s+no\s+circumstances)"
            r"\s+(?:shall|will|may|can)\b"
        ),
        modality=Modality.PROHIBITION,
        strength="HIGH",
        name="prohibition_no_event_reversed",
    ),
    # Forward-order variant kept for completeness.
    ModalRule(
        pattern=_compile(
            r"\b(?:shall|will|may|can)\s+(?:in\s+no\s+event|at\s+no\s+time)\b"
        ),
        modality=Modality.PROHIBITION,
        strength="HIGH",
        name="prohibition_modal_no_event",
    ),
    ModalRule(
        pattern=_compile(r"\bis\s+(?:not\s+permitted|prohibited|forbidden)\b"),
        modality=Modality.PROHIBITION,
        strength="HIGH",
        name="prohibition_is_prohibited",
    ),
    ModalRule(
        pattern=_compile(r"\bno\s+party\s+(?:shall|may|will)\b"),
        modality=Modality.PROHIBITION,
        strength="HIGH",
        name="prohibition_no_party",
    ),
    ModalRule(
        pattern=_compile(r"\bneither\s+party\s+(?:shall|may|will)\b"),
        modality=Modality.PROHIBITION,
        strength="HIGH",
        name="prohibition_neither_party",
    ),

    # ---------- ENTITLEMENT (optional rights — before PERMISSION so that
    # "may elect" doesn't get swallowed by the bare "may" permission rule) ----
    ModalRule(
        pattern=_compile(
            r"\b(?:at\s+its\s+(?:sole\s+)?option|may\s+(?:elect|choose)|has\s+the\s+option)\b"
        ),
        modality=Modality.ENTITLEMENT,
        strength="HIGH",
        name="entitlement_optional",
    ),

    # ---------- OBLIGATION ----------
    ModalRule(
        pattern=_compile(
            r"\b(?:agrees?|undertakes?|covenants?)\s+to\b"
            r"|\bis\s+(?:obligated|required)\s+to\b"
        ),
        modality=Modality.OBLIGATION,
        strength="HIGH",
        name="obligation_agrees_to",
    ),
    ModalRule(
        # Bare shall/must/will — negated forms already consumed by
        # PROHIBITION rules via overlap suppression.
        pattern=_compile(r"\b(?:shall|must|will)\b"),
        modality=Modality.OBLIGATION,
        strength="MEDIUM",
        name="obligation_modal",
    ),

    # ---------- PERMISSION ----------
    ModalRule(
        pattern=_compile(
            r"\bis\s+(?:entitled|permitted)\s+to\b|\bhas\s+the\s+right\s+to\b"
        ),
        modality=Modality.PERMISSION,
        strength="HIGH",
        name="permission_entitled_to",
    ),
    ModalRule(
        pattern=_compile(r"\bmay\b"),
        modality=Modality.PERMISSION,
        strength="MEDIUM",
        name="permission_may",
    ),
]


def find_modal_matches(text: str) -> list[tuple[ModalRule, re.Match]]:
    """Scan ``text`` for modal phrases in priority order.

    Returns a list of ``(rule, match)`` pairs sorted by start position.
    Overlapping matches from lower-priority rules are suppressed so that, e.g.,
    ``shall not`` surfaces once as PROHIBITION rather than also matching the
    bare ``shall`` OBLIGATION rule.
    """
    if not text:
        return []

    accepted: list[tuple[ModalRule, re.Match]] = []
    claimed_spans: list[tuple[int, int]] = []

    for rule in MODAL_RULES:
        for m in rule.pattern.finditer(text):
            start, end = m.span()
            if any(not (end <= cs or start >= ce) for cs, ce in claimed_spans):
                continue  # overlaps a higher-priority match
            accepted.append((rule, m))
            claimed_spans.append((start, end))

    accepted.sort(key=lambda pair: pair[1].start())
    return accepted
