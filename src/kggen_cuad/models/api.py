"""
API request and response models.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Query Models
# =============================================================================


class QueryRequest(BaseModel):
    """Request model for contract queries."""

    query: str = Field(..., description="Natural language query", min_length=1)
    contract_ids: list[UUID] | None = Field(
        default=None, description="Optional list of contract IDs to search"
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Optional filters"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    include_sources: bool = Field(
        default=True, description="Include source triples in response"
    )


class TripleResponse(BaseModel):
    """Triple in API response format."""

    id: UUID
    subject: str
    predicate: str
    object: str
    confidence: float
    contract_id: UUID | None = None
    cuad_label: str | None = None
    source_text: str | None = None


class QueryResponse(BaseModel):
    """Response model for contract queries."""

    query: str
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence")
    sources: list[TripleResponse] = Field(
        default_factory=list, description="Source triples"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Extraction Models
# =============================================================================


class ExtractionRequest(BaseModel):
    """Request model for contract extraction."""

    contract_ids: list[UUID] = Field(
        ..., description="Contract IDs to process", min_length=1
    )
    force: bool = Field(
        default=False, description="Force re-extraction even if already processed"
    )


class ExtractionResponse(BaseModel):
    """Response model for extraction requests."""

    job_id: UUID
    status: str
    contracts_queued: int
    message: str


class ExtractionStatusResponse(BaseModel):
    """Response model for extraction job status."""

    job_id: UUID
    status: str  # pending, running, completed, failed
    progress: float = Field(ge=0.0, le=1.0)
    contracts_total: int
    contracts_completed: int
    contracts_failed: int
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


# =============================================================================
# Graph Search Models
# =============================================================================


class GraphSearchRequest(BaseModel):
    """Request model for graph search."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    entity_types: list[str] | None = Field(
        default=None, description="Filter by entity types"
    )
    predicate_types: list[str] | None = Field(
        default=None, description="Filter by predicate types"
    )
    contract_ids: list[UUID] | None = Field(
        default=None, description="Filter by contract IDs"
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )


class EntityResponse(BaseModel):
    """Entity in API response format."""

    id: UUID
    name: str
    entity_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    contract_id: UUID | None = None
    is_canonical: bool = False


class GraphSearchResponse(BaseModel):
    """Response model for graph search."""

    query: str
    nodes: list[EntityResponse] = Field(default_factory=list)
    edges: list[TripleResponse] = Field(default_factory=list)
    total_nodes: int
    total_edges: int
    processing_time_ms: int


# =============================================================================
# Contract Models
# =============================================================================


class ContractUploadRequest(BaseModel):
    """Request model for contract upload."""

    cuad_id: str = Field(..., description="CUAD identifier")
    filename: str = Field(..., description="Original filename")
    contract_type: str | None = Field(default=None, description="Contract type")


class ContractResponse(BaseModel):
    """Response model for contract details."""

    id: UUID
    cuad_id: str
    filename: str
    contract_type: str | None
    jurisdiction: str | None
    status: str
    page_count: int | None
    word_count: int | None
    error_message: str | None
    created_at: datetime
    extracted_at: datetime | None
    resolved_at: datetime | None


class ContractListResponse(BaseModel):
    """Response model for contract list."""

    contracts: list[ContractResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Pipeline Models
# =============================================================================


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status."""

    status: str
    current_stage: str | None
    contracts_total: int
    contracts_extracted: int
    contracts_aggregated: int
    contracts_resolved: int
    contracts_failed: int
    graph_statistics: dict[str, Any] | None = None


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = "healthy"
    version: str
    environment: str
    services: dict[str, str] = Field(
        default_factory=dict, description="Status of dependent services"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    detail: str | None = None
    code: str | None = None
