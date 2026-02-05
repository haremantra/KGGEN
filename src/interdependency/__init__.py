"""Clause Interdependency Analysis Module.

Detects, models, and analyzes dependencies between contract clauses
using static rules, graph algorithms, and optional LLM validation.
"""

from .types import (
    DependencyType,
    ClauseNode,
    DependencyEdge,
    MissingRequirement,
    ClauseDependencyGraph,
    ImpactResult,
    InterdependencyReport,
)
from .matrix import (
    DependencyRule,
    DEPENDENCY_RULES,
    get_rules_for_pair,
    get_rules_for_label,
    get_all_conflict_pairs,
    get_requires_rules,
    get_rules_requiring_llm,
)
from .detector import DependencyDetector
from .graph import DependencyGraphBuilder
from .analyzer import InterdependencyAnalyzer

__all__ = [
    # Types
    "DependencyType",
    "ClauseNode",
    "DependencyEdge",
    "MissingRequirement",
    "ClauseDependencyGraph",
    "ImpactResult",
    "InterdependencyReport",
    # Matrix
    "DependencyRule",
    "DEPENDENCY_RULES",
    "get_rules_for_pair",
    "get_rules_for_label",
    "get_all_conflict_pairs",
    "get_requires_rules",
    "get_rules_requiring_llm",
    # Classes
    "DependencyDetector",
    "DependencyGraphBuilder",
    "InterdependencyAnalyzer",
]
