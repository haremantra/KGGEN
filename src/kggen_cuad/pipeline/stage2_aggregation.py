"""
Stage 2: Aggregation

Merges and normalizes entities and triples across contracts.
"""

from collections import defaultdict
from typing import Any
from uuid import UUID, uuid4

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.triple import Entity, Triple, EntityType, PredicateType
from kggen_cuad.models.graph import KnowledgeGraph, GraphStatistics
from kggen_cuad.services.embedding_service import get_embedding_service, EmbeddingService
from kggen_cuad.services.graph_service import get_graph_service, GraphService
from kggen_cuad.services.search_service import get_search_service, SearchService

logger = structlog.get_logger(__name__)


class AggregationStage:
    """
    Stage 2: Cross-Contract Aggregation.

    Merges entities and triples from multiple contracts:
    - Normalizes entity names and types
    - Deduplicates entities within and across contracts
    - Aggregates triples with confidence scoring
    """

    def __init__(self):
        self.settings = get_settings()
        self._embedding: EmbeddingService | None = None
        self._graph: GraphService | None = None
        self._search: SearchService | None = None

    @property
    def embedding(self) -> EmbeddingService:
        if self._embedding is None:
            self._embedding = get_embedding_service()
        return self._embedding

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

    async def aggregate(
        self,
        extractions: dict[UUID, tuple[list[Entity], list[Triple]]],
        similarity_threshold: float = 0.85,
    ) -> KnowledgeGraph:
        """
        Aggregate extractions from multiple contracts.

        Args:
            extractions: Dict mapping contract_id to (entities, triples)
            similarity_threshold: Threshold for entity matching

        Returns:
            Aggregated KnowledgeGraph
        """
        logger.info(
            "aggregation_started",
            contracts=len(extractions),
        )

        # Collect all entities and triples
        all_entities: list[Entity] = []
        all_triples: list[Triple] = []
        entity_sources: dict[UUID, list[UUID]] = defaultdict(list)  # entity_id -> contract_ids

        for contract_id, (entities, triples) in extractions.items():
            for entity in entities:
                all_entities.append(entity)
                entity_sources[entity.id].append(contract_id)
            all_triples.extend(triples)

        logger.info(
            "collected_extractions",
            total_entities=len(all_entities),
            total_triples=len(all_triples),
        )

        # Group entities by type
        entities_by_type: dict[EntityType, list[Entity]] = defaultdict(list)
        for entity in all_entities:
            entities_by_type[entity.entity_type].append(entity)

        # Deduplicate within each type
        merged_entities: list[Entity] = []
        entity_id_mapping: dict[UUID, UUID] = {}  # old_id -> new_id

        for entity_type, entities in entities_by_type.items():
            type_merged, type_mapping = await self._deduplicate_entities(
                entities=entities,
                threshold=similarity_threshold,
            )
            merged_entities.extend(type_merged)
            entity_id_mapping.update(type_mapping)

        # Remap triple entity IDs
        remapped_triples = self._remap_triples(all_triples, entity_id_mapping)

        # Deduplicate triples
        deduplicated_triples = self._deduplicate_triples(remapped_triples)

        # Calculate confidence scores
        for triple in deduplicated_triples:
            triple.confidence = self._calculate_triple_confidence(
                triple, entity_sources, len(extractions)
            )

        # Build knowledge graph
        kg = KnowledgeGraph(
            id=uuid4(),
            entities=merged_entities,
            triples=deduplicated_triples,
            metadata={
                "source_contracts": len(extractions),
                "original_entities": len(all_entities),
                "original_triples": len(all_triples),
                "merged_entities": len(merged_entities),
                "merged_triples": len(deduplicated_triples),
            },
        )

        logger.info(
            "aggregation_completed",
            merged_entities=len(merged_entities),
            merged_triples=len(deduplicated_triples),
            reduction_rate=1 - len(merged_entities) / max(len(all_entities), 1),
        )

        return kg

    async def _deduplicate_entities(
        self,
        entities: list[Entity],
        threshold: float = 0.85,
    ) -> tuple[list[Entity], dict[UUID, UUID]]:
        """
        Deduplicate entities using embedding similarity.

        Returns:
            Tuple of (merged_entities, id_mapping)
        """
        if not entities:
            return [], {}

        if len(entities) == 1:
            return entities, {}

        # Generate embeddings
        embeddings = [
            self.embedding.embed_entity(e.name, e.entity_type.value)
            for e in entities
        ]

        # Find similar pairs
        id_mapping: dict[UUID, UUID] = {}
        merged: list[Entity] = []
        processed: set[int] = set()

        for i, entity in enumerate(entities):
            if i in processed:
                continue

            # Find all similar entities
            similar_indices = [i]
            for j in range(i + 1, len(entities)):
                if j in processed:
                    continue

                similarity = self.embedding.compute_similarity(
                    embeddings[i], embeddings[j]
                )
                if similarity >= threshold:
                    similar_indices.append(j)

            # Merge similar entities
            if len(similar_indices) == 1:
                merged.append(entity)
            else:
                # Create merged entity
                merge_entities = [entities[idx] for idx in similar_indices]
                canonical = self._select_canonical_entity(merge_entities)

                merged.append(canonical)

                # Map all IDs to canonical
                for idx in similar_indices:
                    id_mapping[entities[idx].id] = canonical.id
                    processed.add(idx)

            processed.add(i)

        return merged, id_mapping

    def _select_canonical_entity(self, entities: list[Entity]) -> Entity:
        """Select the canonical form from a group of similar entities."""
        if len(entities) == 1:
            return entities[0]

        # Score each entity
        def score_entity(e: Entity) -> int:
            score = 0
            # Prefer longer, more descriptive names
            score += len(e.name)
            # Prefer names without articles
            if not e.name.lower().startswith(("the ", "a ", "an ")):
                score += 10
            # Prefer names with proper capitalization
            if e.name[0].isupper():
                score += 5
            # Prefer entities with more properties
            score += len(e.properties) * 2
            return score

        best = max(entities, key=score_entity)

        # Collect all aliases
        aliases = set()
        for e in entities:
            if e.id != best.id:
                aliases.add(e.name)
                if e.normalized_name and e.normalized_name != best.normalized_name:
                    aliases.add(e.normalized_name)

        # Update properties with aliases
        best.properties["aliases"] = list(aliases)

        return best

    def _remap_triples(
        self,
        triples: list[Triple],
        id_mapping: dict[UUID, UUID],
    ) -> list[Triple]:
        """Remap entity IDs in triples after deduplication."""
        remapped = []

        for triple in triples:
            new_subject_id = id_mapping.get(triple.subject_id, triple.subject_id)
            new_object_id = id_mapping.get(triple.object_id, triple.object_id)

            remapped.append(Triple(
                id=triple.id,
                subject_id=new_subject_id,
                predicate=triple.predicate,
                object_id=new_object_id,
                confidence=triple.confidence,
                properties=triple.properties,
                source_text=triple.source_text,
            ))

        return remapped

    def _deduplicate_triples(self, triples: list[Triple]) -> list[Triple]:
        """Deduplicate triples with the same subject-predicate-object."""
        seen: dict[str, Triple] = {}

        for triple in triples:
            key = f"{triple.subject_id}:{triple.predicate.value}:{triple.object_id}"

            if key in seen:
                # Merge: keep higher confidence
                existing = seen[key]
                if triple.confidence > existing.confidence:
                    seen[key] = triple
                else:
                    # Merge properties
                    for k, v in triple.properties.items():
                        if k not in existing.properties:
                            existing.properties[k] = v
            else:
                seen[key] = triple

        return list(seen.values())

    def _calculate_triple_confidence(
        self,
        triple: Triple,
        entity_sources: dict[UUID, list[UUID]],
        total_contracts: int,
    ) -> float:
        """
        Calculate confidence score for a triple.

        Higher if:
        - Source entities appear in multiple contracts
        - Base confidence is high
        """
        base_confidence = triple.confidence or 0.8

        # Check entity coverage
        subject_contracts = len(entity_sources.get(triple.subject_id, []))
        object_contracts = len(entity_sources.get(triple.object_id, []))

        # Boost for cross-contract validation
        coverage_boost = 0.0
        if total_contracts > 1:
            avg_coverage = (subject_contracts + object_contracts) / (2 * total_contracts)
            coverage_boost = avg_coverage * 0.2

        return min(1.0, base_confidence + coverage_boost)

    # =========================================================================
    # Normalization
    # =========================================================================

    async def normalize_entities(
        self,
        entities: list[Entity],
    ) -> list[Entity]:
        """Normalize entity names and properties."""
        normalized = []

        for entity in entities:
            # Normalize name
            name = entity.name.strip()

            # Remove common noise
            for noise in ['"', "'", ":", ";", ","]:
                name = name.strip(noise)

            # Normalize entity type-specific patterns
            if entity.entity_type == EntityType.PARTY:
                name = self._normalize_party_name(name)
            elif entity.entity_type == EntityType.TEMPORAL:
                name = self._normalize_temporal(name)

            normalized_entity = Entity(
                id=entity.id,
                name=name,
                entity_type=entity.entity_type,
                normalized_name=self._normalize_for_matching(name),
                properties=entity.properties,
                embedding=entity.embedding,
                source_text=entity.source_text,
            )
            normalized.append(normalized_entity)

        return normalized

    def _normalize_party_name(self, name: str) -> str:
        """Normalize party names."""
        # Common suffixes to standardize
        replacements = {
            " inc.": " Inc.",
            " inc": " Inc.",
            " llc": " LLC",
            " ltd": " Ltd.",
            " ltd.": " Ltd.",
            " corp.": " Corp.",
            " corp": " Corp.",
            " corporation": " Corporation",
            " company": " Company",
        }

        name_lower = name.lower()
        for old, new in replacements.items():
            if name_lower.endswith(old):
                name = name[:-len(old)] + new
                break

        return name

    def _normalize_temporal(self, name: str) -> str:
        """Normalize temporal references."""
        import re

        # Standardize date formats
        # MM/DD/YYYY -> YYYY-MM-DD
        date_pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        match = re.search(date_pattern, name)
        if match:
            month, day, year = match.groups()
            name = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        return name

    def _normalize_for_matching(self, name: str) -> str:
        """Normalize name for matching purposes."""
        normalized = name.lower().strip()

        # Remove articles
        for article in ["the ", "a ", "an "]:
            if normalized.startswith(article):
                normalized = normalized[len(article):]

        # Remove punctuation
        for char in [".", ",", "'", '"', "-", "(", ")"]:
            normalized = normalized.replace(char, " ")

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        return normalized

    # =========================================================================
    # Statistics
    # =========================================================================

    def compute_statistics(
        self,
        kg: KnowledgeGraph,
    ) -> GraphStatistics:
        """Compute statistics for a knowledge graph."""
        entities_by_type: dict[str, int] = defaultdict(int)
        for entity in kg.entities:
            entities_by_type[entity.entity_type.value] += 1

        triples_by_predicate: dict[str, int] = defaultdict(int)
        for triple in kg.triples:
            triples_by_predicate[triple.predicate.value] += 1

        return GraphStatistics(
            total_entities=len(kg.entities),
            total_triples=len(kg.triples),
            entities_by_type=dict(entities_by_type),
            triples_by_predicate=dict(triples_by_predicate),
            contracts_processed=kg.metadata.get("source_contracts", 0),
            average_triples_per_contract=len(kg.triples) / max(
                kg.metadata.get("source_contracts", 1), 1
            ),
        )


def get_aggregation_stage() -> AggregationStage:
    """Get aggregation stage instance."""
    return AggregationStage()
