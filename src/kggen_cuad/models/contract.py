"""
Contract model classes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class ContractStatus(str, Enum):
    """Contract processing status."""
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    AGGREGATING = "aggregating"
    RESOLVING = "resolving"
    COMPLETED = "completed"
    FAILED = "failed"


class EditType(str, Enum):
    """Types of contract edits that can be tracked."""
    CLAUSE_ADDED = "clause_added"
    CLAUSE_REMOVED = "clause_removed"
    CLAUSE_MODIFIED = "clause_modified"
    PARTY_CHANGED = "party_changed"
    TERM_MODIFIED = "term_modified"
    OBLIGATION_ADDED = "obligation_added"
    OBLIGATION_REMOVED = "obligation_removed"
    RESTRICTION_ADDED = "restriction_added"
    RESTRICTION_REMOVED = "restriction_removed"
    LIABILITY_MODIFIED = "liability_modified"
    EFFECTIVE_DATE_CHANGED = "effective_date_changed"
    TERMINATION_MODIFIED = "termination_modified"
    IP_CLAUSE_MODIFIED = "ip_clause_modified"
    GENERAL_MODIFICATION = "general_modification"


@dataclass
class ContractSection:
    """
    A section or chunk of a contract.

    Tracks position within the original document for traceability.
    """
    contract_id: UUID
    title: str
    text: str
    start_page: int
    end_page: int
    start_char: int
    end_char: int
    section_type: str = "page"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Contract:
    """
    A contract document with extracted text and metadata.
    """
    id: UUID
    cuad_id: str
    filename: str
    raw_text: str
    page_count: int
    word_count: int
    sections: list[ContractSection] = field(default_factory=list)
    contract_type: str | None = None
    parties: list[str] = field(default_factory=list)
    status: ContractStatus = ContractStatus.UPLOADED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if self.sections is None:
            self.sections = []
        if self.parties is None:
            self.parties = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "cuad_id": self.cuad_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "contract_type": self.contract_type,
            "parties": self.parties,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Contract":
        """Create from dictionary representation."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            cuad_id=data["cuad_id"],
            filename=data["filename"],
            raw_text=data.get("raw_text", ""),
            page_count=data["page_count"],
            word_count=data["word_count"],
            contract_type=data.get("contract_type"),
            parties=data.get("parties", []),
            status=ContractStatus(data.get("status", "uploaded")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            version=data.get("version", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ContractEdit:
    """
    Tracks a single edit/change made to a contract.

    Used to provide edit history as input to KGGEN for incremental
    knowledge graph updates and contract labeling.
    """
    id: UUID = field(default_factory=uuid4)
    contract_id: UUID = field(default_factory=uuid4)
    edit_type: EditType = EditType.GENERAL_MODIFICATION

    # The affected section/clause
    section_title: str = ""
    section_start_char: int = 0
    section_end_char: int = 0

    # Before and after text for the edit
    old_text: str = ""
    new_text: str = ""

    # Context around the edit (for LLM processing)
    context_before: str = ""
    context_after: str = ""

    # CUAD label categories affected by this edit
    affected_labels: list[str] = field(default_factory=list)

    # Edit metadata
    edited_by: str = ""
    edit_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Version tracking
    from_version: int = 1
    to_version: int = 2

    # Processing status
    processed: bool = False
    entities_extracted: list[UUID] = field(default_factory=list)
    triples_extracted: list[UUID] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.affected_labels is None:
            self.affected_labels = []
        if self.entities_extracted is None:
            self.entities_extracted = []
        if self.triples_extracted is None:
            self.triples_extracted = []
        if self.metadata is None:
            self.metadata = {}

    def get_edit_text(self) -> str:
        """Get the combined edit text for KGGEN processing."""
        return f"""
Section: {self.section_title}
Edit Type: {self.edit_type.value}
Previous Text: {self.old_text}
New Text: {self.new_text}
Context Before: {self.context_before}
Context After: {self.context_after}
Affected Labels: {', '.join(self.affected_labels)}
""".strip()

    def get_full_context(self) -> str:
        """Get full context including surrounding text for extraction."""
        return f"{self.context_before}\n{self.new_text}\n{self.context_after}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "contract_id": str(self.contract_id),
            "edit_type": self.edit_type.value,
            "section_title": self.section_title,
            "section_start_char": self.section_start_char,
            "section_end_char": self.section_end_char,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "affected_labels": self.affected_labels,
            "edited_by": self.edited_by,
            "edit_reason": self.edit_reason,
            "timestamp": self.timestamp.isoformat(),
            "from_version": self.from_version,
            "to_version": self.to_version,
            "processed": self.processed,
            "entities_extracted": [str(e) for e in self.entities_extracted],
            "triples_extracted": [str(t) for t in self.triples_extracted],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContractEdit":
        """Create from dictionary representation."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            contract_id=UUID(data["contract_id"]) if isinstance(data.get("contract_id"), str) else data.get("contract_id", uuid4()),
            edit_type=EditType(data.get("edit_type", "general_modification")),
            section_title=data.get("section_title", ""),
            section_start_char=data.get("section_start_char", 0),
            section_end_char=data.get("section_end_char", 0),
            old_text=data.get("old_text", ""),
            new_text=data.get("new_text", ""),
            context_before=data.get("context_before", ""),
            context_after=data.get("context_after", ""),
            affected_labels=data.get("affected_labels", []),
            edited_by=data.get("edited_by", ""),
            edit_reason=data.get("edit_reason", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            from_version=data.get("from_version", 1),
            to_version=data.get("to_version", 2),
            processed=data.get("processed", False),
            entities_extracted=[UUID(e) if isinstance(e, str) else e for e in data.get("entities_extracted", [])],
            triples_extracted=[UUID(t) if isinstance(t, str) else t for t in data.get("triples_extracted", [])],
            metadata=data.get("metadata", {}),
        )
