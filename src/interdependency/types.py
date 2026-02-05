"""Data models for clause interdependency analysis.

Defines the types used throughout the interdependency module:
dependency types, clause nodes, edges, and report structures.
"""

from dataclasses import dataclass, field
from enum import Enum


class DependencyType(str, Enum):
    """Types of dependencies between contract clauses."""
    DEPENDS_ON = "DEPENDS_ON"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    REQUIRES = "REQUIRES"
    MITIGATES = "MITIGATES"
    RESTRICTS = "RESTRICTS"
    MODIFIES = "MODIFIES"


@dataclass
class ClauseNode:
    """A clause represented as a graph node."""
    label: str
    category: str
    text: str = ""
    confidence: float = 0.0
    present: bool = True

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "category": self.category,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "confidence": self.confidence,
            "present": self.present,
        }


@dataclass
class DependencyEdge:
    """A directed dependency between two clauses."""
    source_label: str
    target_label: str
    dependency_type: DependencyType
    strength: float = 1.0
    reason: str = ""
    bidirectional: bool = False
    detected_by: str = "rule"  # "rule" or "llm"
    content_finding: str = ""

    def to_dict(self) -> dict:
        return {
            "source_label": self.source_label,
            "target_label": self.target_label,
            "dependency_type": self.dependency_type.value,
            "strength": self.strength,
            "reason": self.reason,
            "bidirectional": self.bidirectional,
            "detected_by": self.detected_by,
            "content_finding": self.content_finding,
        }


@dataclass
class MissingRequirement:
    """A required clause that is missing from the contract."""
    required_by: str
    missing_label: str
    reason: str
    impact: str = ""
    severity: str = "MEDIUM"  # HIGH, MEDIUM, LOW

    def to_dict(self) -> dict:
        return {
            "required_by": self.required_by,
            "missing_label": self.missing_label,
            "reason": self.reason,
            "impact": self.impact,
            "severity": self.severity,
        }


@dataclass
class ClauseDependencyGraph:
    """The complete dependency graph for a contract."""
    contract_id: str
    nodes: list[ClauseNode] = field(default_factory=list)
    edges: list[DependencyEdge] = field(default_factory=list)
    contradiction_count: int = 0
    missing_requirements: list[MissingRequirement] = field(default_factory=list)
    max_impact_clause: str = ""

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "contradiction_count": self.contradiction_count,
            "missing_requirements": [m.to_dict() for m in self.missing_requirements],
            "max_impact_clause": self.max_impact_clause,
        }


@dataclass
class ImpactResult:
    """Result of impact analysis from a single clause."""
    source_label: str
    affected_clauses: list[dict] = field(default_factory=list)
    total_affected: int = 0
    max_depth: int = 0

    def to_dict(self) -> dict:
        return {
            "source_label": self.source_label,
            "affected_clauses": self.affected_clauses,
            "total_affected": self.total_affected,
            "max_depth": self.max_depth,
        }


@dataclass
class InterdependencyReport:
    """Full interdependency analysis report for a contract."""
    contract_id: str
    graph: ClauseDependencyGraph = field(default_factory=lambda: ClauseDependencyGraph(contract_id=""))
    contradictions: list[dict] = field(default_factory=list)
    missing_requirements: list[MissingRequirement] = field(default_factory=list)
    impact_rankings: list[dict] = field(default_factory=list)
    risk_amplifiers: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    risk_score_adjustment: int = 0
    cycles: list[list[str]] = field(default_factory=list)
    centrality_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "graph": self.graph.to_dict(),
            "contradictions": self.contradictions,
            "missing_requirements": [m.to_dict() for m in self.missing_requirements],
            "impact_rankings": self.impact_rankings,
            "risk_amplifiers": self.risk_amplifiers,
            "recommendations": self.recommendations,
            "risk_score_adjustment": self.risk_score_adjustment,
            "cycles": self.cycles,
            "centrality_scores": self.centrality_scores,
        }
