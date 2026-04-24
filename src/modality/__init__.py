"""Deontic modality checker for contract clauses.

Synchronous, rule-based detection of obligation / permission / prohibition /
entitlement modality. First slice: standalone module whose pytest suite serves
as the executable spec. Not yet wired into the main analysis pipeline.
"""

from .types import Modality, ModalFinding, ModalityReport
from .rules import ModalRule, MODAL_RULES, find_modal_matches
from .checker import ModalityChecker

__all__ = [
    "Modality",
    "ModalFinding",
    "ModalityReport",
    "ModalRule",
    "MODAL_RULES",
    "find_modal_matches",
    "ModalityChecker",
]
