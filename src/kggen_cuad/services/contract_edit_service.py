"""
Contract edit tracking and retrieval service.

Provides functionality to:
- Track changes/edits made to contracts
- Store version history
- Retrieve last edits as input for KGGEN labeling
- Compute diffs between contract versions
"""

import difflib
import re
from datetime import datetime
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.contract import Contract, ContractEdit, ContractSection, EditType

logger = structlog.get_logger(__name__)


# CUAD label categories for automatic classification of edits
CUAD_LABEL_KEYWORDS = {
    # General Information
    "Effective Date": ["effective date", "commencement date", "start date"],
    "Expiration Date": ["expiration", "termination date", "end date", "expires"],
    "Renewal Term": ["renewal", "renew", "automatic renewal", "auto-renew"],
    "Notice Period": ["notice period", "days notice", "prior notice", "written notice"],
    "Parties": ["party", "parties", "between", "licensor", "licensee", "hereinafter"],
    "Governing Law": ["governing law", "jurisdiction", "laws of", "governed by"],

    # Restrictive Covenants
    "Non-Compete": ["non-compete", "noncompete", "compete", "competition"],
    "Exclusivity": ["exclusive", "exclusivity", "sole right"],
    "Anti-Assignment": ["assignment", "assign", "transfer", "assignable"],
    "No-Solicit": ["solicit", "solicitation", "recruit", "hire"],

    # Revenue/Risk
    "Cap on Liability": ["cap on liability", "limitation of liability", "maximum liability", "aggregate liability"],
    "Uncapped Liability": ["uncapped", "unlimited liability", "no cap"],
    "Liquidated Damages": ["liquidated damages", "predetermined damages"],
    "Revenue Commitment": ["revenue", "minimum commitment", "payment commitment"],
    "Audit Rights": ["audit", "inspection", "examine records"],

    # Intellectual Property
    "IP Ownership": ["intellectual property", "ip ownership", "proprietary rights"],
    "IP Assignment": ["assign ip", "assignment of ip", "ip assignment", "transfer of intellectual property"],
    "Joint IP Ownership": ["joint ownership", "jointly owned", "co-ownership"],
    "License Grant": ["license grant", "grant of license", "licensed to", "grants to"],
    "Source Code Escrow": ["source code escrow", "escrow", "code deposit"],

    # Indemnification
    "Indemnification": ["indemnify", "indemnification", "hold harmless", "defend"],

    # Insurance
    "Insurance": ["insurance", "coverage", "policy", "insured"],

    # Termination
    "Termination for Convenience": ["termination for convenience", "terminate without cause"],
    "Termination for Cause": ["termination for cause", "material breach", "terminate for breach"],

    # Confidentiality
    "Confidentiality": ["confidential", "confidentiality", "non-disclosure", "nda", "proprietary information"],

    # Warranty
    "Warranty": ["warranty", "warrants", "representation", "guarantees"],
}


def classify_edit_labels(text: str) -> list[str]:
    """
    Classify which CUAD label categories are affected by an edit.

    Args:
        text: The text of the edit (old + new combined)

    Returns:
        List of affected CUAD label categories
    """
    text_lower = text.lower()
    affected_labels = []

    for label, keywords in CUAD_LABEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                affected_labels.append(label)
                break

    return affected_labels


def determine_edit_type(old_text: str, new_text: str, affected_labels: list[str]) -> EditType:
    """
    Determine the type of edit based on content analysis.

    Args:
        old_text: Text before the edit
        new_text: Text after the edit
        affected_labels: CUAD labels affected by this edit

    Returns:
        The determined EditType
    """
    # Check for additions/removals
    if not old_text.strip() and new_text.strip():
        # New content added
        if any(label in affected_labels for label in ["Non-Compete", "Exclusivity", "Anti-Assignment"]):
            return EditType.RESTRICTION_ADDED
        if any(label in affected_labels for label in ["Revenue Commitment", "Audit Rights"]):
            return EditType.OBLIGATION_ADDED
        return EditType.CLAUSE_ADDED

    if old_text.strip() and not new_text.strip():
        # Content removed
        if any(label in affected_labels for label in ["Non-Compete", "Exclusivity", "Anti-Assignment"]):
            return EditType.RESTRICTION_REMOVED
        if any(label in affected_labels for label in ["Revenue Commitment", "Audit Rights"]):
            return EditType.OBLIGATION_REMOVED
        return EditType.CLAUSE_REMOVED

    # Content modified
    if "Parties" in affected_labels:
        return EditType.PARTY_CHANGED
    if any(label in affected_labels for label in ["Cap on Liability", "Uncapped Liability", "Liquidated Damages"]):
        return EditType.LIABILITY_MODIFIED
    if "Effective Date" in affected_labels:
        return EditType.EFFECTIVE_DATE_CHANGED
    if any(label in affected_labels for label in ["Expiration Date", "Termination for Convenience", "Termination for Cause"]):
        return EditType.TERMINATION_MODIFIED
    if any(label in affected_labels for label in ["IP Ownership", "IP Assignment", "Joint IP Ownership", "License Grant"]):
        return EditType.IP_CLAUSE_MODIFIED
    if any(label in affected_labels for label in ["Renewal Term", "Notice Period"]):
        return EditType.TERM_MODIFIED

    return EditType.CLAUSE_MODIFIED


class ContractEditService:
    """
    Service for tracking and retrieving contract edits.

    Provides methods to:
    - Record contract edits
    - Compute diffs between versions
    - Retrieve last edits for KGGEN processing
    - Classify edits by CUAD label categories
    """

    def __init__(self):
        self.settings = get_settings()
        # In-memory storage for edits (would be replaced with database in production)
        self._edits: dict[UUID, ContractEdit] = {}
        # Contract version tracking
        self._contract_versions: dict[UUID, list[Contract]] = {}
        # Index: contract_id -> list of edit_ids
        self._contract_edits: dict[UUID, list[UUID]] = {}

    def record_edit(
        self,
        contract: Contract,
        old_text: str,
        new_text: str,
        section_title: str = "",
        section_start: int = 0,
        section_end: int = 0,
        edited_by: str = "",
        edit_reason: str = "",
        context_chars: int = 500,
    ) -> ContractEdit:
        """
        Record a contract edit.

        Args:
            contract: The contract being edited
            old_text: Text before the edit
            new_text: Text after the edit
            section_title: Title of the affected section
            section_start: Start character position
            section_end: End character position
            edited_by: User who made the edit
            edit_reason: Reason for the edit
            context_chars: Number of context characters to capture

        Returns:
            The recorded ContractEdit
        """
        # Get context around the edit
        context_before = ""
        context_after = ""

        if contract.raw_text and section_start > 0:
            start = max(0, section_start - context_chars)
            context_before = contract.raw_text[start:section_start]

        if contract.raw_text and section_end < len(contract.raw_text):
            end = min(len(contract.raw_text), section_end + context_chars)
            context_after = contract.raw_text[section_end:end]

        # Classify affected labels
        combined_text = f"{old_text} {new_text}"
        affected_labels = classify_edit_labels(combined_text)

        # Determine edit type
        edit_type = determine_edit_type(old_text, new_text, affected_labels)

        # Create the edit record
        edit = ContractEdit(
            id=uuid4(),
            contract_id=contract.id,
            edit_type=edit_type,
            section_title=section_title,
            section_start_char=section_start,
            section_end_char=section_end,
            old_text=old_text,
            new_text=new_text,
            context_before=context_before,
            context_after=context_after,
            affected_labels=affected_labels,
            edited_by=edited_by,
            edit_reason=edit_reason,
            timestamp=datetime.utcnow(),
            from_version=contract.version,
            to_version=contract.version + 1,
        )

        # Store the edit
        self._edits[edit.id] = edit

        # Update contract-edits index
        if contract.id not in self._contract_edits:
            self._contract_edits[contract.id] = []
        self._contract_edits[contract.id].append(edit.id)

        logger.info(
            "edit_recorded",
            edit_id=str(edit.id),
            contract_id=str(contract.id),
            edit_type=edit_type.value,
            affected_labels=affected_labels,
        )

        return edit

    def compute_diff(
        self,
        old_contract: Contract,
        new_contract: Contract,
        context_chars: int = 500,
    ) -> list[ContractEdit]:
        """
        Compute the diff between two contract versions and create edit records.

        Args:
            old_contract: Previous version of the contract
            new_contract: New version of the contract
            context_chars: Number of context characters to capture

        Returns:
            List of ContractEdit records for each change
        """
        old_text = old_contract.raw_text or ""
        new_text = new_contract.raw_text or ""

        # Use difflib to find changes
        differ = difflib.SequenceMatcher(None, old_text, new_text)
        edits = []

        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == "equal":
                continue

            old_segment = old_text[i1:i2]
            new_segment = new_text[j1:j2]

            # Skip trivial whitespace-only changes
            if not old_segment.strip() and not new_segment.strip():
                continue

            # Find section title (look for nearby headings)
            section_title = self._find_section_title(new_text, j1)

            edit = self.record_edit(
                contract=new_contract,
                old_text=old_segment,
                new_text=new_segment,
                section_title=section_title,
                section_start=j1,
                section_end=j2,
                context_chars=context_chars,
            )
            edits.append(edit)

        logger.info(
            "diff_computed",
            contract_id=str(new_contract.id),
            edits_found=len(edits),
        )

        return edits

    def _find_section_title(self, text: str, position: int, search_range: int = 1000) -> str:
        """
        Find the nearest section title before a position in the text.
        """
        search_start = max(0, position - search_range)
        search_text = text[search_start:position]

        # Common section heading patterns
        patterns = [
            r'\n(\d+\.?\s+[A-Z][A-Za-z\s]+)\n',  # "1. License Grant"
            r'\n([A-Z][A-Z\s]+:)',  # "LICENSE GRANT:"
            r'\n(Section\s+\d+[:\.]?\s+[A-Za-z\s]+)',  # "Section 1: Definitions"
            r'\n(Article\s+[IVX\d]+[:\.]?\s+[A-Za-z\s]+)',  # "Article I: Definitions"
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                return matches[-1].group(1).strip()

        return "Unknown Section"

    def get_edit(self, edit_id: UUID) -> ContractEdit | None:
        """Get an edit by ID."""
        return self._edits.get(edit_id)

    def get_edits_for_contract(
        self,
        contract_id: UUID,
        limit: int | None = None,
        unprocessed_only: bool = False,
    ) -> list[ContractEdit]:
        """
        Get all edits for a contract.

        Args:
            contract_id: The contract ID
            limit: Maximum number of edits to return
            unprocessed_only: Only return unprocessed edits

        Returns:
            List of ContractEdit records, most recent first
        """
        edit_ids = self._contract_edits.get(contract_id, [])
        edits = [self._edits[eid] for eid in edit_ids if eid in self._edits]

        if unprocessed_only:
            edits = [e for e in edits if not e.processed]

        # Sort by timestamp, most recent first
        edits.sort(key=lambda e: e.timestamp, reverse=True)

        if limit:
            edits = edits[:limit]

        return edits

    def get_last_edits(
        self,
        contract_id: UUID,
        count: int = 10,
        unprocessed_only: bool = True,
    ) -> list[ContractEdit]:
        """
        Get the last N edits for a contract, suitable for KGGEN input.

        This is the primary method for retrieving contract edits as input
        to the KGGEN labeling pipeline.

        Args:
            contract_id: The contract ID
            count: Number of edits to retrieve
            unprocessed_only: Only return unprocessed edits

        Returns:
            List of ContractEdit records ready for KGGEN processing
        """
        return self.get_edits_for_contract(
            contract_id=contract_id,
            limit=count,
            unprocessed_only=unprocessed_only,
        )

    def get_edits_for_labeling(
        self,
        contract_id: UUID,
        label_categories: list[str] | None = None,
    ) -> list[ContractEdit]:
        """
        Get edits filtered by CUAD label categories.

        Useful for targeted extraction of specific contract elements.

        Args:
            contract_id: The contract ID
            label_categories: Optional list of CUAD labels to filter by

        Returns:
            List of ContractEdit records matching the specified labels
        """
        edits = self.get_edits_for_contract(contract_id)

        if label_categories:
            label_set = set(label_categories)
            edits = [
                e for e in edits
                if any(label in label_set for label in e.affected_labels)
            ]

        return edits

    def prepare_edits_for_kggen(
        self,
        edits: list[ContractEdit],
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Prepare edit records for KGGEN extraction pipeline input.

        Formats edits into a structure suitable for LLM-based extraction.

        Args:
            edits: List of ContractEdit records
            include_context: Whether to include surrounding context

        Returns:
            List of dictionaries ready for KGGEN processing
        """
        prepared = []

        for edit in edits:
            text_for_extraction = edit.new_text
            if include_context:
                text_for_extraction = edit.get_full_context()

            prepared.append({
                "edit_id": str(edit.id),
                "contract_id": str(edit.contract_id),
                "edit_type": edit.edit_type.value,
                "section_title": edit.section_title,
                "text": text_for_extraction,
                "old_text": edit.old_text,
                "new_text": edit.new_text,
                "affected_labels": edit.affected_labels,
                "cuad_label_hints": self._get_label_hints(edit.affected_labels),
                "from_version": edit.from_version,
                "to_version": edit.to_version,
            })

        return prepared

    def _get_label_hints(self, labels: list[str]) -> str:
        """
        Generate extraction hints based on CUAD labels.

        These hints guide the LLM during entity/relation extraction.
        """
        hints = []

        if "Parties" in labels:
            hints.append("Extract party names and their roles (licensor, licensee, etc.)")
        if any(l in labels for l in ["IP Ownership", "IP Assignment", "License Grant"]):
            hints.append("Extract IP assets, ownership relationships, and licensing terms")
        if any(l in labels for l in ["Cap on Liability", "Liquidated Damages"]):
            hints.append("Extract liability provisions, caps, and damage amounts")
        if any(l in labels for l in ["Effective Date", "Expiration Date"]):
            hints.append("Extract temporal entities (dates, periods, durations)")
        if any(l in labels for l in ["Non-Compete", "Exclusivity", "Anti-Assignment"]):
            hints.append("Extract restrictive covenants and their scope")
        if "Governing Law" in labels:
            hints.append("Extract jurisdiction and governing law provisions")
        if any(l in labels for l in ["Revenue Commitment", "Audit Rights"]):
            hints.append("Extract obligations and compliance requirements")

        return " | ".join(hints) if hints else "Extract all relevant entities and relations"

    def mark_edit_processed(
        self,
        edit_id: UUID,
        entities_extracted: list[UUID],
        triples_extracted: list[UUID],
    ) -> None:
        """
        Mark an edit as processed by KGGEN.

        Args:
            edit_id: The edit ID
            entities_extracted: List of extracted entity IDs
            triples_extracted: List of extracted triple IDs
        """
        edit = self._edits.get(edit_id)
        if edit:
            edit.processed = True
            edit.entities_extracted = entities_extracted
            edit.triples_extracted = triples_extracted

            logger.info(
                "edit_marked_processed",
                edit_id=str(edit_id),
                entities=len(entities_extracted),
                triples=len(triples_extracted),
            )

    def get_unprocessed_count(self, contract_id: UUID) -> int:
        """Get count of unprocessed edits for a contract."""
        edits = self.get_edits_for_contract(contract_id, unprocessed_only=True)
        return len(edits)

    def clear_edits(self, contract_id: UUID) -> int:
        """
        Clear all edits for a contract.

        Returns the number of edits removed.
        """
        edit_ids = self._contract_edits.get(contract_id, [])
        count = len(edit_ids)

        for edit_id in edit_ids:
            self._edits.pop(edit_id, None)

        self._contract_edits[contract_id] = []

        logger.info("edits_cleared", contract_id=str(contract_id), count=count)
        return count


@lru_cache()
def get_contract_edit_service() -> ContractEditService:
    """Get cached contract edit service instance."""
    return ContractEditService()
