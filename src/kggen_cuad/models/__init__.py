"""
Pydantic models for KGGEN-CUAD.

This module contains all data models used throughout the application:
- Contract models for document representation
- Triple models for knowledge graph elements
- Graph models for knowledge graph containers
- API models for request/response schemas
"""

from kggen_cuad.models.contract import Contract, ContractSection, ContractStatus
from kggen_cuad.models.triple import (
    Entity,
    EntityType,
    Predicate,
    PredicateType,
    Triple,
)
from kggen_cuad.models.graph import KnowledgeGraph, SubGraph
from kggen_cuad.models.api import (
    QueryRequest,
    QueryResponse,
    ExtractionRequest,
    ExtractionResponse,
    GraphSearchRequest,
    GraphSearchResponse,
)

__all__ = [
    # Contract models
    "Contract",
    "ContractSection",
    "ContractStatus",
    # Triple models
    "Entity",
    "EntityType",
    "Predicate",
    "PredicateType",
    "Triple",
    # Graph models
    "KnowledgeGraph",
    "SubGraph",
    # API models
    "QueryRequest",
    "QueryResponse",
    "ExtractionRequest",
    "ExtractionResponse",
    "GraphSearchRequest",
    "GraphSearchResponse",
]
