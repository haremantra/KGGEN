"""
API response model classes.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class ContractUploadResponse(BaseModel):
    """Response for contract upload."""
    contract_id: str
    filename: str
    page_count: int
    word_count: int
    contract_type: str | None = None
    parties: list[str] = Field(default_factory=list)
    status: str = "uploaded"


class ContractSummary(BaseModel):
    """Summary of a contract for list responses."""
    contract_id: str
    cuad_id: str
    filename: str
    contract_type: str | None = None
    parties: list[str] = Field(default_factory=list)
    status: str
    created_at: datetime
    version: int = 1


class ContractListResponse(BaseModel):
    """Response for contract list."""
    contracts: list[ContractSummary] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0


class PipelineStatusResponse(BaseModel):
    """Response for pipeline status."""
    pipeline_id: str
    status: str
    entities: int = 0
    triples: int = 0
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None


class ContractEditResponse(BaseModel):
    """Response for a single contract edit."""
    id: str
    contract_id: str
    edit_type: str
    section_title: str
    old_text: str
    new_text: str
    affected_labels: list[str] = Field(default_factory=list)
    edited_by: str = ""
    edit_reason: str = ""
    timestamp: datetime
    from_version: int
    to_version: int
    processed: bool = False
    entities_extracted: int = 0
    triples_extracted: int = 0


class ContractEditsListResponse(BaseModel):
    """Response for listing contract edits."""
    contract_id: str
    edits: list[ContractEditResponse] = Field(default_factory=list)
    total: int = 0
    unprocessed_count: int = 0


class EditExtractionRequest(BaseModel):
    """Request to process contract edits through KGGEN."""
    contract_id: str
    edit_ids: list[str] | None = None  # If None, process all unprocessed
    include_context: bool = True
    context_chars: int = 500


class EditExtractionResponse(BaseModel):
    """Response from processing contract edits."""
    contract_id: str
    edits_processed: int
    entities_extracted: int
    triples_extracted: int
    affected_labels: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
