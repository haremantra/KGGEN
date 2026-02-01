"""
Knowledge graph routes.
"""

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query

from kggen_cuad.models.api import (
    EntityResponse,
    TripleResponse,
    SubGraphResponse,
    GraphStatsResponse,
)
from kggen_cuad.models.triple import EntityType, PredicateType
from kggen_cuad.services.graph_service import get_graph_service
from kggen_cuad.services.search_service import get_search_service

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_statistics() -> GraphStatsResponse:
    """
    Get knowledge graph statistics.
    """
    graph = get_graph_service()

    try:
        stats = await graph.get_statistics()

        return GraphStatsResponse(
            total_entities=stats.total_entities,
            total_triples=stats.total_triples,
            entities_by_type=stats.entities_by_type,
            triples_by_predicate=stats.triples_by_predicate,
            contracts_processed=stats.contracts_processed,
        )

    except Exception as e:
        logger.error("get_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
) -> EntityResponse:
    """
    Get entity by ID.
    """
    graph = get_graph_service()

    try:
        entity = await graph.get_entity(entity_id)

        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        return EntityResponse(
            id=str(entity.id),
            name=entity.name,
            entity_type=entity.entity_type.value,
            normalized_name=entity.normalized_name,
            properties=entity.properties,
            aliases=entity.aliases,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_entity_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities", response_model=list[EntityResponse])
async def list_entities(
    entity_type: str | None = None,
    limit: int = Query(default=50, le=500),
) -> list[EntityResponse]:
    """
    List entities, optionally filtered by type.
    """
    graph = get_graph_service()

    try:
        if entity_type:
            try:
                et = EntityType(entity_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid entity type: {entity_type}",
                )
            entities = await graph.get_entities_by_type(et, limit=limit)
        else:
            # Get all types
            entities = []
            for et in EntityType:
                type_entities = await graph.get_entities_by_type(et, limit=limit // 8)
                entities.extend(type_entities)

        return [
            EntityResponse(
                id=str(e.id),
                name=e.name,
                entity_type=e.entity_type.value,
                normalized_name=e.normalized_name,
                properties=e.properties,
            )
            for e in entities[:limit]
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_entities_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}/triples", response_model=list[TripleResponse])
async def get_entity_triples(
    entity_id: str,
    direction: str = "both",
) -> list[TripleResponse]:
    """
    Get triples for an entity.
    """
    graph = get_graph_service()

    try:
        triples = await graph.get_triples_for_entity(
            entity_id=entity_id,
            direction=direction,
        )

        return [
            TripleResponse(
                id=str(t.id),
                subject_id=str(t.subject_id),
                predicate=t.predicate.value,
                object_id=str(t.object_id),
                confidence=t.confidence,
                properties=t.properties,
            )
            for t in triples
        ]

    except Exception as e:
        logger.error("get_entity_triples_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}/neighborhood", response_model=SubGraphResponse)
async def get_entity_neighborhood(
    entity_id: str,
    depth: int = Query(default=2, le=5),
    max_nodes: int = Query(default=50, le=200),
) -> SubGraphResponse:
    """
    Get neighborhood subgraph around an entity.
    """
    graph = get_graph_service()

    try:
        subgraph = await graph.get_entity_neighborhood(
            entity_id=entity_id,
            depth=depth,
            max_nodes=max_nodes,
        )

        return SubGraphResponse(
            entities=[
                EntityResponse(
                    id=str(e.id),
                    name=e.name,
                    entity_type=e.entity_type.value,
                    normalized_name=e.normalized_name,
                )
                for e in subgraph.entities
            ],
            triples=[
                TripleResponse(
                    id=str(t.id),
                    subject_id=str(t.subject_id),
                    predicate=t.predicate.value,
                    object_id=str(t.object_id),
                    confidence=t.confidence,
                )
                for t in subgraph.triples
            ],
            center_entity_id=entity_id,
            depth=depth,
        )

    except Exception as e:
        logger.error("get_neighborhood_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/entities")
async def search_entities(
    query: str,
    entity_type: str | None = None,
    limit: int = Query(default=10, le=50),
) -> list[dict[str, Any]]:
    """
    Search for entities.
    """
    search = get_search_service()

    try:
        entity_types = [entity_type] if entity_type else None
        results = await search.search_entities(
            query=query,
            entity_types=entity_types,
            limit=limit,
        )

        return [
            {
                "id": str(entity.id),
                "name": entity.name,
                "entity_type": entity.entity_type.value,
                "score": score,
            }
            for entity, score in results
        ]

    except Exception as e:
        logger.error("search_entities_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/triples")
async def search_triples(
    query: str,
    predicate: str | None = None,
    limit: int = Query(default=10, le=50),
) -> list[dict[str, Any]]:
    """
    Search for triples.
    """
    search = get_search_service()

    try:
        predicate_types = [predicate] if predicate else None
        results = await search.search_triples(
            query=query,
            predicate_types=predicate_types,
            limit=limit,
        )

        return [
            {
                "id": str(triple.id),
                "subject_id": str(triple.subject_id),
                "predicate": triple.predicate.value,
                "object_id": str(triple.object_id),
                "score": score,
            }
            for triple, score in results
        ]

    except Exception as e:
        logger.error("search_triples_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paths")
async def find_paths(
    start_entity_id: str,
    end_entity_id: str,
    max_depth: int = Query(default=5, le=10),
) -> list[list[dict[str, Any]]]:
    """
    Find paths between two entities.
    """
    graph = get_graph_service()

    try:
        paths = await graph.get_path_between_entities(
            start_entity_id=start_entity_id,
            end_entity_id=end_entity_id,
            max_depth=max_depth,
        )

        return paths

    except Exception as e:
        logger.error("find_paths_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contracts/{contract_id}/subgraph", response_model=SubGraphResponse)
async def get_contract_subgraph(
    contract_id: str,
) -> SubGraphResponse:
    """
    Get all entities and triples from a specific contract.
    """
    graph = get_graph_service()

    try:
        subgraph = await graph.get_contract_subgraph(UUID(contract_id))

        return SubGraphResponse(
            entities=[
                EntityResponse(
                    id=str(e.id),
                    name=e.name,
                    entity_type=e.entity_type.value,
                    normalized_name=e.normalized_name,
                )
                for e in subgraph.entities
            ],
            triples=[
                TripleResponse(
                    id=str(t.id),
                    subject_id=str(t.subject_id),
                    predicate=t.predicate.value,
                    object_id=str(t.object_id),
                    confidence=t.confidence,
                )
                for t in subgraph.triples
            ],
            source_contract_id=contract_id,
        )

    except Exception as e:
        logger.error("get_contract_subgraph_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
