"""
Contract edit tracking and retrieval routes.

Provides endpoints to:
- Record contract edits
- Retrieve last edits for KGGEN input
- Process edits through the extraction pipeline
"""

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from kggen_cuad.models.api import (
    ContractEditResponse,
    ContractEditsListResponse,
    EditExtractionRequest,
    EditExtractionResponse,
)
from kggen_cuad.services.contract_edit_service import get_contract_edit_service

logger = structlog.get_logger(__name__)
router = APIRouter()


class RecordEditRequest(BaseModel):
    """Request to record a contract edit."""
    contract_id: str
    old_text: str
    new_text: str
    section_title: str = ""
    section_start: int = 0
    section_end: int = 0
    edited_by: str = ""
    edit_reason: str = ""


class ComputeDiffRequest(BaseModel):
    """Request to compute diff between contract versions."""
    old_contract_text: str
    new_contract_text: str
    contract_id: str
    cuad_id: str = ""


class KGGENInputResponse(BaseModel):
    """Response containing edit data formatted for KGGEN input."""
    contract_id: str
    edits_count: int
    kggen_inputs: list[dict[str, Any]] = Field(default_factory=list)
    affected_labels: list[str] = Field(default_factory=list)


@router.get("/{contract_id}", response_model=ContractEditsListResponse)
async def get_contract_edits(
    contract_id: str,
    limit: int = Query(default=50, le=200),
    unprocessed_only: bool = Query(default=False),
) -> ContractEditsListResponse:
    """
    Get all edits for a contract.

    Returns edit history sorted by timestamp (most recent first).
    """
    service = get_contract_edit_service()

    try:
        contract_uuid = UUID(contract_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    edits = service.get_edits_for_contract(
        contract_id=contract_uuid,
        limit=limit,
        unprocessed_only=unprocessed_only,
    )

    unprocessed_count = service.get_unprocessed_count(contract_uuid)

    edit_responses = [
        ContractEditResponse(
            id=str(edit.id),
            contract_id=str(edit.contract_id),
            edit_type=edit.edit_type.value,
            section_title=edit.section_title,
            old_text=edit.old_text,
            new_text=edit.new_text,
            affected_labels=edit.affected_labels,
            edited_by=edit.edited_by,
            edit_reason=edit.edit_reason,
            timestamp=edit.timestamp,
            from_version=edit.from_version,
            to_version=edit.to_version,
            processed=edit.processed,
            entities_extracted=len(edit.entities_extracted),
            triples_extracted=len(edit.triples_extracted),
        )
        for edit in edits
    ]

    return ContractEditsListResponse(
        contract_id=contract_id,
        edits=edit_responses,
        total=len(edit_responses),
        unprocessed_count=unprocessed_count,
    )


@router.get("/{contract_id}/last", response_model=KGGENInputResponse)
async def get_last_edits_for_kggen(
    contract_id: str,
    count: int = Query(default=10, le=100),
    unprocessed_only: bool = Query(default=True),
    include_context: bool = Query(default=True),
) -> KGGENInputResponse:
    """
    Retrieve last code edits for contract as input to KGGEN for labeling.

    This is the primary endpoint for getting contract edits ready for
    KGGEN extraction and labeling. Returns edits formatted with:
    - Edit text and context
    - CUAD label hints for guided extraction
    - Section information for traceability

    Args:
        contract_id: The contract ID
        count: Number of recent edits to retrieve
        unprocessed_only: Only return edits not yet processed
        include_context: Include surrounding text context

    Returns:
        KGGENInputResponse with formatted edit data for pipeline input
    """
    service = get_contract_edit_service()

    try:
        contract_uuid = UUID(contract_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    # Get last edits
    edits = service.get_last_edits(
        contract_id=contract_uuid,
        count=count,
        unprocessed_only=unprocessed_only,
    )

    if not edits:
        return KGGENInputResponse(
            contract_id=contract_id,
            edits_count=0,
            kggen_inputs=[],
            affected_labels=[],
        )

    # Prepare for KGGEN input
    kggen_inputs = service.prepare_edits_for_kggen(
        edits=edits,
        include_context=include_context,
    )

    # Collect all affected labels
    all_labels = set()
    for edit in edits:
        all_labels.update(edit.affected_labels)

    logger.info(
        "last_edits_retrieved",
        contract_id=contract_id,
        edits_count=len(edits),
        affected_labels=list(all_labels),
    )

    return KGGENInputResponse(
        contract_id=contract_id,
        edits_count=len(edits),
        kggen_inputs=kggen_inputs,
        affected_labels=sorted(all_labels),
    )


@router.get("/{contract_id}/labels/{label}")
async def get_edits_by_label(
    contract_id: str,
    label: str,
    limit: int = Query(default=20, le=100),
) -> ContractEditsListResponse:
    """
    Get edits affecting a specific CUAD label category.

    Useful for targeted extraction of specific contract elements.

    Args:
        contract_id: The contract ID
        label: CUAD label category to filter by
        limit: Maximum edits to return
    """
    service = get_contract_edit_service()

    try:
        contract_uuid = UUID(contract_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    edits = service.get_edits_for_labeling(
        contract_id=contract_uuid,
        label_categories=[label],
    )

    edits = edits[:limit]

    edit_responses = [
        ContractEditResponse(
            id=str(edit.id),
            contract_id=str(edit.contract_id),
            edit_type=edit.edit_type.value,
            section_title=edit.section_title,
            old_text=edit.old_text,
            new_text=edit.new_text,
            affected_labels=edit.affected_labels,
            edited_by=edit.edited_by,
            edit_reason=edit.edit_reason,
            timestamp=edit.timestamp,
            from_version=edit.from_version,
            to_version=edit.to_version,
            processed=edit.processed,
            entities_extracted=len(edit.entities_extracted),
            triples_extracted=len(edit.triples_extracted),
        )
        for edit in edits
    ]

    return ContractEditsListResponse(
        contract_id=contract_id,
        edits=edit_responses,
        total=len(edit_responses),
        unprocessed_count=len([e for e in edits if not e.processed]),
    )


@router.post("/record")
async def record_edit(
    request: RecordEditRequest,
) -> ContractEditResponse:
    """
    Record a new contract edit.

    Use this when manual edits are made to a contract and need
    to be tracked for incremental KGGEN processing.
    """
    from kggen_cuad.models.contract import Contract

    service = get_contract_edit_service()

    try:
        contract_uuid = UUID(request.contract_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    # Create minimal contract object for edit recording
    contract = Contract(
        id=contract_uuid,
        cuad_id="",
        filename="",
        raw_text="",
        page_count=0,
        word_count=0,
    )

    edit = service.record_edit(
        contract=contract,
        old_text=request.old_text,
        new_text=request.new_text,
        section_title=request.section_title,
        section_start=request.section_start,
        section_end=request.section_end,
        edited_by=request.edited_by,
        edit_reason=request.edit_reason,
    )

    return ContractEditResponse(
        id=str(edit.id),
        contract_id=str(edit.contract_id),
        edit_type=edit.edit_type.value,
        section_title=edit.section_title,
        old_text=edit.old_text,
        new_text=edit.new_text,
        affected_labels=edit.affected_labels,
        edited_by=edit.edited_by,
        edit_reason=edit.edit_reason,
        timestamp=edit.timestamp,
        from_version=edit.from_version,
        to_version=edit.to_version,
        processed=edit.processed,
        entities_extracted=0,
        triples_extracted=0,
    )


@router.post("/diff")
async def compute_diff(
    request: ComputeDiffRequest,
) -> ContractEditsListResponse:
    """
    Compute diff between two contract versions and create edit records.

    Automatically detects changes and classifies them by CUAD labels.
    """
    from kggen_cuad.models.contract import Contract
    from uuid import uuid4

    service = get_contract_edit_service()

    try:
        contract_uuid = UUID(request.contract_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    # Create contract objects for diff computation
    old_contract = Contract(
        id=uuid4(),
        cuad_id=request.cuad_id,
        filename="old_version",
        raw_text=request.old_contract_text,
        page_count=1,
        word_count=len(request.old_contract_text.split()),
        version=1,
    )

    new_contract = Contract(
        id=contract_uuid,
        cuad_id=request.cuad_id,
        filename="new_version",
        raw_text=request.new_contract_text,
        page_count=1,
        word_count=len(request.new_contract_text.split()),
        version=2,
    )

    edits = service.compute_diff(old_contract, new_contract)

    edit_responses = [
        ContractEditResponse(
            id=str(edit.id),
            contract_id=str(edit.contract_id),
            edit_type=edit.edit_type.value,
            section_title=edit.section_title,
            old_text=edit.old_text,
            new_text=edit.new_text,
            affected_labels=edit.affected_labels,
            edited_by=edit.edited_by,
            edit_reason=edit.edit_reason,
            timestamp=edit.timestamp,
            from_version=edit.from_version,
            to_version=edit.to_version,
            processed=edit.processed,
            entities_extracted=0,
            triples_extracted=0,
        )
        for edit in edits
    ]

    return ContractEditsListResponse(
        contract_id=request.contract_id,
        edits=edit_responses,
        total=len(edit_responses),
        unprocessed_count=len(edits),
    )


@router.post("/process", response_model=EditExtractionResponse)
async def process_edits_through_kggen(
    request: EditExtractionRequest,
) -> EditExtractionResponse:
    """
    Process contract edits through the KGGEN extraction pipeline.

    This endpoint:
    1. Retrieves specified edits (or all unprocessed)
    2. Runs them through entity/relation extraction
    3. Marks edits as processed
    4. Returns extraction results

    Use this after getting edits via /last endpoint.
    """
    from kggen_cuad.pipeline.stage1_extraction import get_extraction_stage
    from kggen_cuad.models.contract import Contract, ContractSection

    service = get_contract_edit_service()
    extraction = get_extraction_stage()

    try:
        contract_uuid = UUID(request.contract_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    # Get edits to process
    if request.edit_ids:
        edit_uuids = [UUID(eid) for eid in request.edit_ids]
        edits = [
            service.get_edit(eid)
            for eid in edit_uuids
            if service.get_edit(eid) is not None
        ]
    else:
        edits = service.get_edits_for_contract(
            contract_id=contract_uuid,
            unprocessed_only=True,
        )

    if not edits:
        return EditExtractionResponse(
            contract_id=request.contract_id,
            edits_processed=0,
            entities_extracted=0,
            triples_extracted=0,
            affected_labels=[],
            errors=[],
        )

    total_entities = 0
    total_triples = 0
    all_labels = set()
    errors = []

    # Process each edit
    for edit in edits:
        try:
            # Get text for extraction
            text = edit.get_full_context() if request.include_context else edit.new_text

            # Create mini-contract for extraction
            mini_contract = Contract(
                id=edit.contract_id,
                cuad_id=f"edit_{edit.id}",
                filename="edit_extraction",
                raw_text=text,
                page_count=1,
                word_count=len(text.split()),
            )

            # Run extraction
            entities, triples = await extraction.process_contract(
                contract=mini_contract,
                chunk_size=len(text) + 100,  # Process as single chunk
                overlap=0,
            )

            # Update edit with extracted data
            entity_ids = [e.id for e in entities]
            triple_ids = [t.id for t in triples]

            # Add source edit ID to extracted items
            for entity in entities:
                entity.source_edit_id = edit.id
                entity.cuad_labels = edit.affected_labels

            for triple in triples:
                triple.source_edit_id = edit.id
                triple.cuad_labels = edit.affected_labels

            service.mark_edit_processed(
                edit_id=edit.id,
                entities_extracted=entity_ids,
                triples_extracted=triple_ids,
            )

            total_entities += len(entities)
            total_triples += len(triples)
            all_labels.update(edit.affected_labels)

        except Exception as e:
            error_msg = f"Edit {edit.id}: {str(e)}"
            errors.append(error_msg)
            logger.error("edit_processing_failed", edit_id=str(edit.id), error=str(e))

    logger.info(
        "edits_processed",
        contract_id=request.contract_id,
        edits_processed=len(edits),
        entities_extracted=total_entities,
        triples_extracted=total_triples,
    )

    return EditExtractionResponse(
        contract_id=request.contract_id,
        edits_processed=len(edits) - len(errors),
        entities_extracted=total_entities,
        triples_extracted=total_triples,
        affected_labels=sorted(all_labels),
        errors=errors,
    )


@router.delete("/{contract_id}")
async def clear_contract_edits(
    contract_id: str,
) -> dict[str, Any]:
    """
    Clear all edits for a contract.

    Use with caution - this removes edit history.
    """
    service = get_contract_edit_service()

    try:
        contract_uuid = UUID(contract_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contract ID format")

    count = service.clear_edits(contract_uuid)

    return {
        "contract_id": contract_id,
        "edits_cleared": count,
        "status": "cleared",
    }
