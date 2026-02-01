"""
Pipeline stages for knowledge graph extraction.

Implements the 3-stage KGGen methodology:
1. Extraction - Extract entities and relations from contracts
2. Aggregation - Merge and normalize across contracts
3. Resolution - Entity clustering and canonicalization
"""

from kggen_cuad.pipeline.stage1_extraction import ExtractionStage
from kggen_cuad.pipeline.stage2_aggregation import AggregationStage
from kggen_cuad.pipeline.stage3_resolution import ResolutionStage
from kggen_cuad.pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "ExtractionStage",
    "AggregationStage",
    "ResolutionStage",
    "PipelineOrchestrator",
]
