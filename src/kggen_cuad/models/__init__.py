"""
KGGEN-CUAD model classes.
"""

from kggen_cuad.models.contract import (
    Contract,
    ContractSection,
    ContractStatus,
    ContractEdit,
    EditType,
)
from kggen_cuad.models.triple import (
    Entity,
    Triple,
    EntityType,
    PredicateType,
)
from kggen_cuad.models.graph import (
    KnowledgeGraph,
    GraphStatistics,
    SubGraph,
)
from kggen_cuad.models.api import (
    ContractUploadResponse,
    ContractListResponse,
    PipelineStatusResponse,
    ContractEditResponse,
    ContractEditsListResponse,
)

__all__ = [
    # Contract models
    "Contract",
    "ContractSection",
    "ContractStatus",
    "ContractEdit",
    "EditType",
    # Triple models
    "Entity",
    "Triple",
    "EntityType",
    "PredicateType",
    # Graph models
    "KnowledgeGraph",
    "GraphStatistics",
    "SubGraph",
    # API models
    "ContractUploadResponse",
    "ContractListResponse",
    "PipelineStatusResponse",
    "ContractEditResponse",
    "ContractEditsListResponse",
]
