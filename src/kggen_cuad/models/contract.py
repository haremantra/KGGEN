"""
Contract models for representing legal documents.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ContractStatus(str, Enum):
    """Processing status for a contract."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    AGGREGATING = "aggregating"
    AGGREGATED = "aggregated"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    FAILED = "failed"


class ContractSection(BaseModel):
    """A section or chunk of a contract document."""

    id: UUID = Field(default_factory=uuid4)
    contract_id: UUID
    title: str | None = None
    text: str
    start_page: int
    end_page: int
    start_char: int
    end_char: int
    section_type: str | None = None  # e.g., "header", "clause", "definition", "schedule"

    class Config:
        from_attributes = True


class Contract(BaseModel):
    """
    Represents a legal contract document.

    This is the primary model for contracts being processed through the pipeline.
    """

    id: UUID = Field(default_factory=uuid4)
    cuad_id: str = Field(..., description="CUAD dataset identifier")
    filename: str = Field(..., description="Original filename")
    contract_type: str | None = Field(
        default=None,
        description="Type of contract (e.g., License Agreement, Service Agreement)",
    )
    jurisdiction: str | None = Field(
        default=None, description="Governing law jurisdiction"
    )

    # Content
    raw_text: str | None = Field(default=None, description="Full extracted text")
    sections: list[ContractSection] = Field(
        default_factory=list, description="Document sections/chunks"
    )

    # Metadata
    page_count: int | None = None
    word_count: int | None = None
    parties: list[str] = Field(default_factory=list, description="Identified parties")

    # Processing status
    status: ContractStatus = Field(default=ContractStatus.PENDING)
    error_message: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    extracted_at: datetime | None = None
    aggregated_at: datetime | None = None
    resolved_at: datetime | None = None

    # CUAD annotations (if available)
    cuad_annotations: dict[str, Any] = Field(
        default_factory=dict, description="CUAD label annotations"
    )

    class Config:
        from_attributes = True
        use_enum_values = True

    def update_status(self, status: ContractStatus, error: str | None = None) -> None:
        """Update the contract processing status."""
        self.status = status
        self.updated_at = datetime.utcnow()
        if error:
            self.error_message = error

        # Set stage timestamps
        if status == ContractStatus.EXTRACTED:
            self.extracted_at = datetime.utcnow()
        elif status == ContractStatus.AGGREGATED:
            self.aggregated_at = datetime.utcnow()
        elif status == ContractStatus.RESOLVED:
            self.resolved_at = datetime.utcnow()

    @property
    def is_processed(self) -> bool:
        """Check if the contract has been fully processed."""
        return self.status == ContractStatus.RESOLVED

    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == ContractStatus.FAILED

    def compute_word_count(self) -> int:
        """Compute and set word count from raw text."""
        if self.raw_text:
            self.word_count = len(self.raw_text.split())
        return self.word_count or 0
