"""
Contract management routes.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

from kggen_cuad.config import get_settings
from kggen_cuad.models.api import ContractUploadResponse, ContractListResponse
from kggen_cuad.models.contract import Contract
from kggen_cuad.services.contract_loader import get_contract_loader

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=ContractUploadResponse)
async def upload_contract(
    file: UploadFile = File(...),
) -> ContractUploadResponse:
    """
    Upload a contract PDF for processing.
    """
    settings = get_settings()
    loader = get_contract_loader()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted",
        )

    # Save to temp location
    temp_path = Path(settings.temp_dir) / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        temp_path.write_bytes(content)

        # Load contract
        contract = loader.load_pdf(temp_path)

        # Detect type and extract parties
        loader.identify_contract_type(contract)
        loader.extract_parties(contract)

        logger.info(
            "contract_uploaded",
            contract_id=str(contract.id),
            filename=file.filename,
            pages=contract.page_count,
        )

        return ContractUploadResponse(
            contract_id=str(contract.id),
            filename=contract.filename,
            page_count=contract.page_count,
            word_count=contract.word_count,
            contract_type=contract.contract_type,
            parties=contract.parties,
            status="uploaded",
        )

    except Exception as e:
        logger.error("contract_upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


@router.post("/text", response_model=ContractUploadResponse)
async def upload_contract_text(
    text: str,
    cuad_id: str,
    filename: str = "text_input",
) -> ContractUploadResponse:
    """
    Upload contract as raw text.
    """
    loader = get_contract_loader()

    try:
        contract = loader.load_text(text, cuad_id, filename)
        loader.identify_contract_type(contract)
        loader.extract_parties(contract)

        return ContractUploadResponse(
            contract_id=str(contract.id),
            filename=filename,
            page_count=1,
            word_count=contract.word_count,
            contract_type=contract.contract_type,
            parties=contract.parties,
            status="uploaded",
        )

    except Exception as e:
        logger.error("contract_text_upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{contract_id}")
async def get_contract(
    contract_id: str,
) -> dict[str, Any]:
    """
    Get contract details by ID.
    """
    # In production, this would fetch from database
    # For now, return placeholder
    return {
        "contract_id": contract_id,
        "status": "not_found",
        "message": "Contract retrieval not yet implemented",
    }


@router.get("/")
async def list_contracts(
    limit: int = 20,
    offset: int = 0,
) -> ContractListResponse:
    """
    List all processed contracts.
    """
    # In production, this would fetch from database
    return ContractListResponse(
        contracts=[],
        total=0,
        limit=limit,
        offset=offset,
    )


@router.delete("/{contract_id}")
async def delete_contract(
    contract_id: str,
) -> dict[str, Any]:
    """
    Delete a contract and its extracted data.
    """
    from kggen_cuad.services.graph_service import get_graph_service

    graph = get_graph_service()

    try:
        count = await graph.delete_contract_data(UUID(contract_id))

        return {
            "contract_id": contract_id,
            "status": "deleted",
            "entities_removed": count,
        }

    except Exception as e:
        logger.error("contract_delete_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
