"""Knowledge Graph Schema Models for KGGEN-CUAD.

Node Types (8): Party, IPAsset, Obligation, Restriction, LiabilityProvision,
                Temporal, Jurisdiction, ContractClause

Edge Types (10): LICENSES_TO, OWNS, ASSIGNS, HAS_OBLIGATION, SUBJECT_TO_RESTRICTION,
                 HAS_LIABILITY, GOVERNED_BY, CONTAINS_CLAUSE, EFFECTIVE_ON, TERMINATES_ON
"""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# Node Type Enums
class NodeType(str, Enum):
    """Knowledge graph node types."""
    PARTY = "Party"
    IP_ASSET = "IPAsset"
    OBLIGATION = "Obligation"
    RESTRICTION = "Restriction"
    LIABILITY_PROVISION = "LiabilityProvision"
    TEMPORAL = "Temporal"
    JURISDICTION = "Jurisdiction"
    CONTRACT_CLAUSE = "ContractClause"
    CONTRACT = "Contract"


class EdgeType(str, Enum):
    """Knowledge graph edge/relationship types."""
    LICENSES_TO = "LICENSES_TO"
    OWNS = "OWNS"
    ASSIGNS = "ASSIGNS"
    HAS_OBLIGATION = "HAS_OBLIGATION"
    SUBJECT_TO_RESTRICTION = "SUBJECT_TO_RESTRICTION"
    HAS_LIABILITY = "HAS_LIABILITY"
    GOVERNED_BY = "GOVERNED_BY"
    CONTAINS_CLAUSE = "CONTAINS_CLAUSE"
    EFFECTIVE_ON = "EFFECTIVE_ON"
    TERMINATES_ON = "TERMINATES_ON"


# Base Models
class KGNode(BaseModel):
    """Base knowledge graph node."""
    id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Node name/label")
    type: NodeType = Field(..., description="Node type")
    source_contract_id: str | None = Field(None, description="Source contract ID")
    cuad_label: str | None = Field(None, description="CUAD category label")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class KGEdge(BaseModel):
    """Knowledge graph edge/relationship."""
    id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    type: EdgeType = Field(..., description="Edge type")
    source_contract_id: str | None = Field(None, description="Source contract ID")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class Triple(BaseModel):
    """A subject-predicate-object triple extracted from a contract."""
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relationship/predicate")
    object: str = Field(..., description="Object entity")
    properties: dict[str, Any] = Field(default_factory=dict, description="Triple properties")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    source_text: str | None = Field(None, description="Source text span")


class ExtractionResult(BaseModel):
    """Result of extracting knowledge from a contract."""
    contract_id: str = Field(..., description="Contract identifier")
    contract_type: str | None = Field(None, description="Type of contract")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    llm_model: str = Field(..., description="LLM model used for extraction")
    entities: list[KGNode] = Field(default_factory=list, description="Extracted entities")
    triples: list[Triple] = Field(default_factory=list, description="Extracted triples")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Specialized Node Types
class Party(KGNode):
    """A party to a contract (company, individual, etc.)."""
    type: NodeType = NodeType.PARTY
    role: str | None = Field(None, description="Role in contract (Licensor, Licensee, etc.)")
    legal_entity_type: str | None = Field(None, description="Corporation, LLC, Individual, etc.")


class IPAsset(KGNode):
    """Intellectual property asset."""
    type: NodeType = NodeType.IP_ASSET
    ip_type: str | None = Field(None, description="Patent, Copyright, Trademark, Trade Secret, etc.")
    registration_number: str | None = Field(None, description="Registration/patent number if any")


class Obligation(KGNode):
    """A contractual obligation."""
    type: NodeType = NodeType.OBLIGATION
    obligor: str | None = Field(None, description="Party with the obligation")
    obligation_type: str | None = Field(None, description="provide, maintain, support, etc.")
    is_conditional: bool = Field(False, description="Whether obligation is conditional")


class Restriction(KGNode):
    """A contractual restriction."""
    type: NodeType = NodeType.RESTRICTION
    restriction_type: str | None = Field(None, description="non-compete, non-disclosure, etc.")
    duration: str | None = Field(None, description="Duration of restriction")
    scope: str | None = Field(None, description="Geographic or subject matter scope")


class LiabilityProvision(KGNode):
    """Liability-related provision."""
    type: NodeType = NodeType.LIABILITY_PROVISION
    provision_type: str | None = Field(None, description="cap, limitation, indemnification, etc.")
    amount: str | None = Field(None, description="Dollar amount or formula")
    exceptions: list[str] = Field(default_factory=list, description="Exceptions to the provision")


class Temporal(KGNode):
    """A temporal/date entity."""
    type: NodeType = NodeType.TEMPORAL
    date_value: datetime | None = Field(None, description="Parsed date value")
    temporal_type: str | None = Field(None, description="effective, expiration, renewal, etc.")


class Jurisdiction(KGNode):
    """Jurisdiction/governing law entity."""
    type: NodeType = NodeType.JURISDICTION
    jurisdiction_type: str | None = Field(None, description="state, country, etc.")
    legal_system: str = Field("common_law", description="common_law, civil_law, etc.")


class ContractClause(KGNode):
    """A contract clause."""
    type: NodeType = NodeType.CONTRACT_CLAUSE
    clause_type: str | None = Field(None, description="CUAD category")
    section_number: str | None = Field(None, description="Section number in contract")
    text: str | None = Field(None, description="Full clause text")
