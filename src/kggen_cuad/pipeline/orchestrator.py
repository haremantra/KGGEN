"""
Pipeline Orchestrator

Coordinates the 3-stage knowledge graph extraction pipeline.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.contract import Contract, ContractStatus
from kggen_cuad.models.triple import Entity, Triple
from kggen_cuad.models.graph import KnowledgeGraph, GraphStatistics
from kggen_cuad.pipeline.stage1_extraction import ExtractionStage, get_extraction_stage
from kggen_cuad.pipeline.stage2_aggregation import AggregationStage, get_aggregation_stage
from kggen_cuad.pipeline.stage3_resolution import ResolutionStage, get_resolution_stage
from kggen_cuad.services.contract_loader import ContractLoader, get_contract_loader
from kggen_cuad.services.graph_service import GraphService, get_graph_service
from kggen_cuad.services.search_service import SearchService, get_search_service
from kggen_cuad.storage.redis_cache import RedisCache, get_redis_cache

logger = structlog.get_logger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    LOADING = "loading"
    EXTRACTING = "extracting"
    AGGREGATING = "aggregating"
    RESOLVING = "resolving"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineResult:
    """Result of a pipeline execution."""

    def __init__(
        self,
        pipeline_id: UUID,
        status: PipelineStatus,
        knowledge_graph: KnowledgeGraph | None = None,
        statistics: GraphStatistics | None = None,
        error: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ):
        self.pipeline_id = pipeline_id
        self.status = status
        self.knowledge_graph = knowledge_graph
        self.statistics = statistics
        self.error = error
        self.started_at = started_at
        self.completed_at = completed_at

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": str(self.pipeline_id),
            "status": self.status.value,
            "entities": len(self.knowledge_graph.entities) if self.knowledge_graph else 0,
            "triples": len(self.knowledge_graph.triples) if self.knowledge_graph else 0,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


class PipelineOrchestrator:
    """
    Orchestrates the knowledge graph extraction pipeline.

    Coordinates:
    1. Contract loading
    2. Stage 1: Extraction
    3. Stage 2: Aggregation
    4. Stage 3: Resolution
    5. Graph persistence
    """

    def __init__(self):
        self.settings = get_settings()

        # Stages
        self._extraction: ExtractionStage | None = None
        self._aggregation: AggregationStage | None = None
        self._resolution: ResolutionStage | None = None

        # Services
        self._loader: ContractLoader | None = None
        self._graph: GraphService | None = None
        self._search: SearchService | None = None
        self._cache: RedisCache | None = None

        # Callbacks
        self._progress_callback: Callable[[PipelineStatus, float], None] | None = None

    @property
    def extraction(self) -> ExtractionStage:
        if self._extraction is None:
            self._extraction = get_extraction_stage()
        return self._extraction

    @property
    def aggregation(self) -> AggregationStage:
        if self._aggregation is None:
            self._aggregation = get_aggregation_stage()
        return self._aggregation

    @property
    def resolution(self) -> ResolutionStage:
        if self._resolution is None:
            self._resolution = get_resolution_stage()
        return self._resolution

    @property
    def loader(self) -> ContractLoader:
        if self._loader is None:
            self._loader = get_contract_loader()
        return self._loader

    @property
    def graph(self) -> GraphService:
        if self._graph is None:
            self._graph = get_graph_service()
        return self._graph

    @property
    def search(self) -> SearchService:
        if self._search is None:
            self._search = get_search_service()
        return self._search

    @property
    def cache(self) -> RedisCache:
        if self._cache is None:
            self._cache = get_redis_cache()
        return self._cache

    def set_progress_callback(
        self,
        callback: Callable[[PipelineStatus, float], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, status: PipelineStatus, progress: float) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            self._progress_callback(status, progress)

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    async def run(
        self,
        contract_paths: list[Path],
        skip_resolution: bool = False,
        persist_graph: bool = True,
    ) -> PipelineResult:
        """
        Run the full extraction pipeline.

        Args:
            contract_paths: Paths to contract PDFs
            skip_resolution: Skip Stage 3 (entity resolution)
            persist_graph: Whether to persist to Neo4j

        Returns:
            PipelineResult with the extracted knowledge graph
        """
        pipeline_id = uuid4()
        started_at = datetime.now()

        logger.info(
            "pipeline_started",
            pipeline_id=str(pipeline_id),
            contracts=len(contract_paths),
        )

        try:
            # Stage 0: Load contracts
            self._report_progress(PipelineStatus.LOADING, 0.0)
            contracts = await self._load_contracts(contract_paths)
            self._report_progress(PipelineStatus.LOADING, 1.0)

            if not contracts:
                return PipelineResult(
                    pipeline_id=pipeline_id,
                    status=PipelineStatus.FAILED,
                    error="No contracts loaded successfully",
                    started_at=started_at,
                    completed_at=datetime.now(),
                )

            # Stage 1: Extraction
            self._report_progress(PipelineStatus.EXTRACTING, 0.0)
            extractions = await self._run_extraction(contracts)
            self._report_progress(PipelineStatus.EXTRACTING, 1.0)

            # Stage 2: Aggregation
            self._report_progress(PipelineStatus.AGGREGATING, 0.0)
            aggregated_kg = await self._run_aggregation(extractions)
            self._report_progress(PipelineStatus.AGGREGATING, 1.0)

            # Stage 3: Resolution (optional)
            if skip_resolution:
                resolved_kg = aggregated_kg
            else:
                self._report_progress(PipelineStatus.RESOLVING, 0.0)
                resolved_kg = await self._run_resolution(aggregated_kg)
                self._report_progress(PipelineStatus.RESOLVING, 1.0)

            # Persist to graph database
            if persist_graph:
                self._report_progress(PipelineStatus.PERSISTING, 0.0)
                await self._persist_graph(resolved_kg)
                self._report_progress(PipelineStatus.PERSISTING, 1.0)

            # Calculate statistics
            statistics = self.aggregation.compute_statistics(resolved_kg)

            completed_at = datetime.now()

            logger.info(
                "pipeline_completed",
                pipeline_id=str(pipeline_id),
                entities=len(resolved_kg.entities),
                triples=len(resolved_kg.triples),
                duration_seconds=(completed_at - started_at).total_seconds(),
            )

            self._report_progress(PipelineStatus.COMPLETED, 1.0)

            return PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.COMPLETED,
                knowledge_graph=resolved_kg,
                statistics=statistics,
                started_at=started_at,
                completed_at=completed_at,
            )

        except Exception as e:
            logger.error(
                "pipeline_failed",
                pipeline_id=str(pipeline_id),
                error=str(e),
            )

            return PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )

    async def _load_contracts(
        self,
        contract_paths: list[Path],
    ) -> list[Contract]:
        """Load contracts from PDF files."""
        contracts = []

        for i, path in enumerate(contract_paths):
            try:
                contract = self.loader.load_pdf(path)
                contracts.append(contract)

                # Detect contract type
                self.loader.identify_contract_type(contract)

                # Extract parties
                self.loader.extract_parties(contract)

                logger.debug(
                    "contract_loaded",
                    path=str(path),
                    pages=contract.page_count,
                    words=contract.word_count,
                )

            except Exception as e:
                logger.error(
                    "contract_load_failed",
                    path=str(path),
                    error=str(e),
                )

            # Update progress
            progress = (i + 1) / len(contract_paths)
            self._report_progress(PipelineStatus.LOADING, progress)

        logger.info(
            "contracts_loaded",
            total=len(contract_paths),
            successful=len(contracts),
        )

        return contracts

    async def _run_extraction(
        self,
        contracts: list[Contract],
    ) -> dict[UUID, tuple[list[Entity], list[Triple]]]:
        """Run Stage 1: Extraction on all contracts."""
        extractions = await self.extraction.process_contracts_batch(contracts)

        total_entities = sum(len(e) for e, t in extractions.values())
        total_triples = sum(len(t) for e, t in extractions.values())

        logger.info(
            "extraction_stage_completed",
            contracts=len(contracts),
            entities=total_entities,
            triples=total_triples,
        )

        return extractions

    async def _run_aggregation(
        self,
        extractions: dict[UUID, tuple[list[Entity], list[Triple]]],
    ) -> KnowledgeGraph:
        """Run Stage 2: Aggregation."""
        aggregated = await self.aggregation.aggregate(extractions)

        logger.info(
            "aggregation_stage_completed",
            entities=len(aggregated.entities),
            triples=len(aggregated.triples),
        )

        return aggregated

    async def _run_resolution(
        self,
        kg: KnowledgeGraph,
    ) -> KnowledgeGraph:
        """Run Stage 3: Resolution."""
        resolved = await self.resolution.resolve(kg)

        logger.info(
            "resolution_stage_completed",
            entities=len(resolved.entities),
            triples=len(resolved.triples),
        )

        return resolved

    async def _persist_graph(
        self,
        kg: KnowledgeGraph,
    ) -> None:
        """Persist knowledge graph to Neo4j and Qdrant."""
        # Persist entities
        for entity in kg.entities:
            await self.graph.add_entity(entity)
            await self.search.index_entity(entity)

        # Persist triples
        entity_map = {str(e.id): e for e in kg.entities}

        for triple in kg.triples:
            await self.graph.add_triple(triple)

            # Get entity names for indexing
            subject = entity_map.get(str(triple.subject_id))
            obj = entity_map.get(str(triple.object_id))

            if subject and obj:
                await self.search.index_triple(
                    triple=triple,
                    subject_name=subject.name,
                    object_name=obj.name,
                )

        logger.info(
            "graph_persisted",
            entities=len(kg.entities),
            triples=len(kg.triples),
        )

    # =========================================================================
    # Single Contract Processing
    # =========================================================================

    async def process_single_contract(
        self,
        contract_path: Path,
        persist: bool = True,
    ) -> PipelineResult:
        """
        Process a single contract.

        Simplified pipeline for single contract processing.
        """
        return await self.run(
            contract_paths=[contract_path],
            skip_resolution=True,  # Skip resolution for single contracts
            persist_graph=persist,
        )

    async def process_contract_text(
        self,
        text: str,
        cuad_id: str,
        persist: bool = True,
    ) -> PipelineResult:
        """
        Process contract from raw text.

        Useful for testing or non-PDF sources.
        """
        pipeline_id = uuid4()
        started_at = datetime.now()

        try:
            # Create contract from text
            contract = self.loader.load_text(text, cuad_id)

            # Extract
            entities, triples = await self.extraction.process_contract(contract)

            # Build simple KG
            kg = KnowledgeGraph(
                id=uuid4(),
                entities=entities,
                triples=triples,
                metadata={
                    "source": "text",
                    "cuad_id": cuad_id,
                },
            )

            # Persist
            if persist:
                await self._persist_graph(kg)

            statistics = self.aggregation.compute_statistics(kg)

            return PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.COMPLETED,
                knowledge_graph=kg,
                statistics=statistics,
                started_at=started_at,
                completed_at=datetime.now(),
            )

        except Exception as e:
            return PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )

    # =========================================================================
    # Incremental Processing
    # =========================================================================

    async def add_contract(
        self,
        contract_path: Path,
        existing_kg_id: UUID | None = None,
    ) -> PipelineResult:
        """
        Add a contract to an existing knowledge graph.

        Runs extraction and merges with existing graph.
        """
        pipeline_id = uuid4()
        started_at = datetime.now()

        try:
            # Load contract
            contract = self.loader.load_pdf(contract_path)

            # Extract
            entities, triples = await self.extraction.process_contract(contract)

            # If existing KG, merge
            if existing_kg_id:
                # Get existing entities/triples and merge
                # This would require loading from database
                pass

            # Build KG
            kg = KnowledgeGraph(
                id=uuid4(),
                entities=entities,
                triples=triples,
                metadata={
                    "source_contract": str(contract.id),
                    "incremental": True,
                },
            )

            # Persist
            await self._persist_graph(kg)

            return PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.COMPLETED,
                knowledge_graph=kg,
                started_at=started_at,
                completed_at=datetime.now(),
            )

        except Exception as e:
            return PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )

    # =========================================================================
    # Status and Validation
    # =========================================================================

    async def get_pipeline_status(
        self,
        pipeline_id: UUID,
    ) -> PipelineStatus:
        """Get status of a running pipeline."""
        status = self.cache.get_contract_status(str(pipeline_id))
        if status:
            return PipelineStatus(status)
        return PipelineStatus.PENDING

    async def validate_result(
        self,
        result: PipelineResult,
    ) -> dict[str, Any]:
        """
        Validate pipeline result.

        Checks:
        - Entity coverage
        - Triple validity
        - Graph connectivity
        """
        if not result.knowledge_graph:
            return {"valid": False, "error": "No knowledge graph"}

        kg = result.knowledge_graph

        # Basic counts
        validation = {
            "valid": True,
            "entities": len(kg.entities),
            "triples": len(kg.triples),
            "warnings": [],
        }

        # Check entity coverage
        entity_ids = {str(e.id) for e in kg.entities}

        orphan_triples = 0
        for triple in kg.triples:
            if str(triple.subject_id) not in entity_ids:
                orphan_triples += 1
            if str(triple.object_id) not in entity_ids:
                orphan_triples += 1

        if orphan_triples > 0:
            validation["warnings"].append(
                f"{orphan_triples} triples reference missing entities"
            )

        # Check for isolated entities
        connected_entities = set()
        for triple in kg.triples:
            connected_entities.add(str(triple.subject_id))
            connected_entities.add(str(triple.object_id))

        isolated = len(entity_ids - connected_entities)
        if isolated > 0:
            validation["warnings"].append(
                f"{isolated} entities are not connected to any triples"
            )

        return validation


def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """Get pipeline orchestrator instance."""
    return PipelineOrchestrator()
