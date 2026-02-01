"""
Query answering routes.
"""

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from kggen_cuad.models.api import QueryRequest, QueryResponse
from kggen_cuad.services.query_service import get_query_service

logger = structlog.get_logger(__name__)
router = APIRouter()


class QueryRequestBody(BaseModel):
    """Query request body."""
    query: str
    contract_ids: list[str] | None = None
    include_sources: bool = True


@router.post("/", response_model=QueryResponse)
async def answer_query(
    request: QueryRequestBody,
) -> QueryResponse:
    """
    Answer a question about contracts using the knowledge graph.
    """
    query_service = get_query_service()

    try:
        contract_uuids = None
        if request.contract_ids:
            contract_uuids = [UUID(c) for c in request.contract_ids]

        response = await query_service.answer_query(
            query=request.query,
            contract_ids=contract_uuids,
            include_sources=request.include_sources,
        )

        return response

    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/licensing/{contract_id}", response_model=QueryResponse)
async def query_licensing(
    contract_id: str,
) -> QueryResponse:
    """
    Get licensing information from a contract.
    """
    query_service = get_query_service()

    try:
        response = await query_service.query_licensing(UUID(contract_id))
        return response

    except Exception as e:
        logger.error("query_licensing_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/obligations/{contract_id}", response_model=QueryResponse)
async def query_obligations(
    contract_id: str,
    party_name: str | None = None,
) -> QueryResponse:
    """
    Get obligations from a contract.
    """
    query_service = get_query_service()

    try:
        response = await query_service.query_obligations(
            UUID(contract_id),
            party_name=party_name,
        )
        return response

    except Exception as e:
        logger.error("query_obligations_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/restrictions/{contract_id}", response_model=QueryResponse)
async def query_restrictions(
    contract_id: str,
) -> QueryResponse:
    """
    Get restrictions from a contract.
    """
    query_service = get_query_service()

    try:
        response = await query_service.query_restrictions(UUID(contract_id))
        return response

    except Exception as e:
        logger.error("query_restrictions_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liability/{contract_id}", response_model=QueryResponse)
async def query_liability(
    contract_id: str,
) -> QueryResponse:
    """
    Get liability provisions from a contract.
    """
    query_service = get_query_service()

    try:
        response = await query_service.query_liability(UUID(contract_id))
        return response

    except Exception as e:
        logger.error("query_liability_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/termination/{contract_id}", response_model=QueryResponse)
async def query_termination(
    contract_id: str,
) -> QueryResponse:
    """
    Get termination conditions from a contract.
    """
    query_service = get_query_service()

    try:
        response = await query_service.query_termination(UUID(contract_id))
        return response

    except Exception as e:
        logger.error("query_termination_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governing-law/{contract_id}", response_model=QueryResponse)
async def query_governing_law(
    contract_id: str,
) -> QueryResponse:
    """
    Get governing law and jurisdiction from a contract.
    """
    query_service = get_query_service()

    try:
        response = await query_service.query_governing_law(UUID(contract_id))
        return response

    except Exception as e:
        logger.error("query_governing_law_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class CompareRequest(BaseModel):
    """Compare request body."""
    contract_ids: list[str]
    aspect: str = "all"


@router.post("/compare", response_model=QueryResponse)
async def compare_contracts(
    request: CompareRequest,
) -> QueryResponse:
    """
    Compare multiple contracts.
    """
    query_service = get_query_service()

    if len(request.contract_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 contract IDs required for comparison",
        )

    try:
        contract_uuids = [UUID(c) for c in request.contract_ids]

        response = await query_service.compare_contracts(
            contract_ids=contract_uuids,
            aspect=request.aspect,
        )

        return response

    except Exception as e:
        logger.error("compare_contracts_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_id}")
async def query_entity(
    entity_id: str,
    query: str = Query(..., description="Question about the entity"),
) -> QueryResponse:
    """
    Answer a question about a specific entity.
    """
    query_service = get_query_service()

    try:
        response = await query_service.query_entity(entity_id, query)
        return response

    except Exception as e:
        logger.error("query_entity_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_id}/summary")
async def get_entity_summary(
    entity_id: str,
) -> dict[str, str]:
    """
    Get a summary of an entity's relationships.
    """
    query_service = get_query_service()

    try:
        summary = await query_service.get_entity_summary(entity_id)
        return {"entity_id": entity_id, "summary": summary}

    except Exception as e:
        logger.error("get_entity_summary_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions/{contract_id}")
async def get_query_suggestions(
    contract_id: str,
) -> list[str]:
    """
    Get suggested queries for a contract.
    """
    query_service = get_query_service()

    try:
        suggestions = await query_service.suggest_queries(UUID(contract_id))
        return suggestions

    except Exception as e:
        logger.error("get_suggestions_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
