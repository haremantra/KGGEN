"""
Stage 3: Entity Resolution

Clusters entities and selects canonical forms using S-BERT + k-means.
"""

from collections import defaultdict
from typing import Any
from uuid import UUID, uuid4

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.triple import Entity, Triple, EntityType
from kggen_cuad.models.graph import KnowledgeGraph
from kggen_cuad.services.embedding_service import get_embedding_service, EmbeddingService
from kggen_cuad.services.llm_service import get_llm_service, LLMService

logger = structlog.get_logger(__name__)


class ResolutionStage:
    """
    Stage 3: Entity Resolution.

    Implements the KGGen resolution methodology:
    - S-BERT embeddings for semantic similarity
    - k-means clustering (k=128 or sqrt(n))
    - LLM-based canonical form selection
    """

    def __init__(self):
        self.settings = get_settings()
        self._embedding: EmbeddingService | None = None
        self._llm: LLMService | None = None

        # Resolution parameters from KGGen
        self.default_k = 128
        self.min_cluster_size = 2

    @property
    def embedding(self) -> EmbeddingService:
        if self._embedding is None:
            self._embedding = get_embedding_service()
        return self._embedding

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service()
        return self._llm

    async def resolve(
        self,
        kg: KnowledgeGraph,
        use_llm: bool = True,
    ) -> KnowledgeGraph:
        """
        Resolve entities in a knowledge graph.

        Args:
            kg: Knowledge graph to resolve
            use_llm: Whether to use LLM for canonical selection

        Returns:
            Resolved knowledge graph
        """
        logger.info(
            "resolution_started",
            entities=len(kg.entities),
            triples=len(kg.triples),
        )

        # Group entities by type for type-aware resolution
        entities_by_type: dict[EntityType, list[Entity]] = defaultdict(list)
        for entity in kg.entities:
            entities_by_type[entity.entity_type].append(entity)

        # Resolve each type separately
        resolved_entities: list[Entity] = []
        entity_id_mapping: dict[UUID, UUID] = {}
        alias_mapping: dict[UUID, list[str]] = {}

        for entity_type, entities in entities_by_type.items():
            logger.debug(
                "resolving_type",
                type=entity_type.value,
                count=len(entities),
            )

            type_resolved, type_mapping, type_aliases = await self._resolve_entities(
                entities=entities,
                entity_type=entity_type,
                use_llm=use_llm,
            )

            resolved_entities.extend(type_resolved)
            entity_id_mapping.update(type_mapping)
            alias_mapping.update(type_aliases)

        # Remap triples
        resolved_triples = self._remap_triples(kg.triples, entity_id_mapping)

        # Deduplicate triples after remapping
        resolved_triples = self._deduplicate_triples(resolved_triples)

        # Update entity aliases
        for entity in resolved_entities:
            if entity.id in alias_mapping:
                entity.aliases = alias_mapping[entity.id]

        # Build resolved knowledge graph
        resolved_kg = KnowledgeGraph(
            id=uuid4(),
            entities=resolved_entities,
            triples=resolved_triples,
            metadata={
                **kg.metadata,
                "resolution_applied": True,
                "original_entities": len(kg.entities),
                "resolved_entities": len(resolved_entities),
                "original_triples": len(kg.triples),
                "resolved_triples": len(resolved_triples),
            },
        )

        logger.info(
            "resolution_completed",
            original_entities=len(kg.entities),
            resolved_entities=len(resolved_entities),
            reduction=1 - len(resolved_entities) / max(len(kg.entities), 1),
        )

        return resolved_kg

    async def _resolve_entities(
        self,
        entities: list[Entity],
        entity_type: EntityType,
        use_llm: bool = True,
    ) -> tuple[list[Entity], dict[UUID, UUID], dict[UUID, list[str]]]:
        """
        Resolve entities of a single type.

        Returns:
            Tuple of (resolved_entities, id_mapping, alias_mapping)
        """
        if len(entities) <= 1:
            return entities, {}, {}

        # Generate embeddings
        embeddings = [
            self.embedding.embed_entity(e.name, entity_type.value)
            for e in entities
        ]

        # Cluster entities
        n_clusters = self._calculate_cluster_count(len(entities))
        labels = self.embedding.cluster_embeddings(
            embeddings=embeddings,
            n_clusters=n_clusters,
        )

        # Group by cluster
        clusters: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        # Process each cluster
        resolved_entities: list[Entity] = []
        id_mapping: dict[UUID, UUID] = {}
        alias_mapping: dict[UUID, list[str]] = {}

        for cluster_id, indices in clusters.items():
            if len(indices) == 1:
                # Single entity in cluster - no resolution needed
                resolved_entities.append(entities[indices[0]])
            else:
                # Multiple entities - need resolution
                cluster_entities = [entities[i] for i in indices]

                # Select canonical form
                if use_llm:
                    canonical, aliases = await self._select_canonical_llm(
                        cluster_entities
                    )
                else:
                    canonical, aliases = self._select_canonical_heuristic(
                        cluster_entities
                    )

                resolved_entities.append(canonical)

                # Map all IDs to canonical
                for entity in cluster_entities:
                    if entity.id != canonical.id:
                        id_mapping[entity.id] = canonical.id

                alias_mapping[canonical.id] = aliases

        return resolved_entities, id_mapping, alias_mapping

    def _calculate_cluster_count(self, n: int) -> int:
        """Calculate optimal cluster count."""
        import math

        if n <= self.default_k:
            # Fewer entities than default k - use sqrt(n)
            return max(2, int(math.sqrt(n)))
        else:
            # Use default k or sqrt(n), whichever is smaller
            return min(self.default_k, max(2, int(math.sqrt(n))))

    async def _select_canonical_llm(
        self,
        entities: list[Entity],
    ) -> tuple[Entity, list[str]]:
        """Use LLM to select canonical form."""
        entity_names = [e.name for e in entities]

        canonical_name, aliases = await self.llm.select_canonical(entity_names)

        # Find or create canonical entity
        canonical = None
        for entity in entities:
            if entity.name == canonical_name:
                canonical = entity
                break

        if canonical is None:
            # LLM returned a new name - use first entity as base
            canonical = entities[0]
            canonical.name = canonical_name
            canonical.normalized_name = canonical_name.lower().strip()
            aliases = [e.name for e in entities if e.name != canonical_name]

        return canonical, aliases

    def _select_canonical_heuristic(
        self,
        entities: list[Entity],
    ) -> tuple[Entity, list[str]]:
        """Use heuristics to select canonical form."""
        def score_entity(e: Entity) -> int:
            score = 0

            # Prefer longer, more complete names
            score += len(e.name)

            # Prefer names without articles
            if not e.name.lower().startswith(("the ", "a ", "an ")):
                score += 20

            # Prefer proper capitalization
            if e.name[0].isupper():
                score += 10

            # Prefer names without special characters
            special_chars = sum(1 for c in e.name if c in "()[]{}\"'")
            score -= special_chars * 5

            # Prefer names with corporate suffixes (for parties)
            if e.entity_type == EntityType.PARTY:
                for suffix in [" Inc.", " LLC", " Ltd.", " Corp.", " Corporation"]:
                    if e.name.endswith(suffix):
                        score += 15
                        break

            return score

        # Sort by score
        sorted_entities = sorted(entities, key=score_entity, reverse=True)
        canonical = sorted_entities[0]

        # Collect aliases
        aliases = [e.name for e in sorted_entities[1:]]

        return canonical, aliases

    def _remap_triples(
        self,
        triples: list[Triple],
        id_mapping: dict[UUID, UUID],
    ) -> list[Triple]:
        """Remap entity IDs in triples."""
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
        """Deduplicate triples after remapping."""
        seen: dict[str, Triple] = {}

        for triple in triples:
            key = f"{triple.subject_id}:{triple.predicate.value}:{triple.object_id}"

            if key in seen:
                existing = seen[key]
                # Merge: average confidence
                new_confidence = (existing.confidence + triple.confidence) / 2
                existing.confidence = new_confidence
            else:
                seen[key] = triple

        return list(seen.values())

    # =========================================================================
    # Duplicate Detection
    # =========================================================================

    async def find_duplicates(
        self,
        entities: list[Entity],
        similarity_threshold: float = 0.8,
    ) -> list[list[Entity]]:
        """
        Find groups of duplicate entities.

        Returns list of groups, where each group contains duplicate entities.
        """
        if len(entities) <= 1:
            return []

        # Generate embeddings
        embeddings = [
            self.embedding.embed_entity(e.name, e.entity_type.value)
            for e in entities
        ]

        # Find similar pairs
        duplicate_groups: list[list[Entity]] = []
        processed: set[int] = set()

        for i in range(len(entities)):
            if i in processed:
                continue

            group = [entities[i]]
            processed.add(i)

            for j in range(i + 1, len(entities)):
                if j in processed:
                    continue

                similarity = self.embedding.compute_similarity(
                    embeddings[i], embeddings[j]
                )

                if similarity >= similarity_threshold:
                    group.append(entities[j])
                    processed.add(j)

            if len(group) >= self.min_cluster_size:
                duplicate_groups.append(group)

        return duplicate_groups

    async def validate_duplicates(
        self,
        duplicate_groups: list[list[Entity]],
    ) -> list[list[Entity]]:
        """
        Validate duplicate groups using LLM.

        Filters out false positives.
        """
        validated = []

        for group in duplicate_groups:
            names = [e.name for e in group]

            # Use LLM to identify actual duplicates
            llm_groups = await self.llm.identify_duplicates(names)

            if llm_groups:
                # LLM confirmed duplicates
                for llm_group in llm_groups:
                    matched_entities = [
                        e for e in group
                        if e.name in llm_group
                    ]
                    if len(matched_entities) >= 2:
                        validated.append(matched_entities)

        return validated

    # =========================================================================
    # Cross-Type Resolution
    # =========================================================================

    async def resolve_cross_type_references(
        self,
        kg: KnowledgeGraph,
    ) -> KnowledgeGraph:
        """
        Resolve references across entity types.

        For example: A Party mentioned in an Obligation might refer
        to the same Party entity.
        """
        # Build name index
        name_index: dict[str, list[Entity]] = defaultdict(list)
        for entity in kg.entities:
            normalized = entity.normalized_name or entity.name.lower()
            name_index[normalized].append(entity)

        # Find cross-references
        id_mapping: dict[UUID, UUID] = {}

        for normalized, entities in name_index.items():
            if len(entities) <= 1:
                continue

            # Check if entities are of different types but refer to same thing
            type_groups: dict[EntityType, list[Entity]] = defaultdict(list)
            for e in entities:
                type_groups[e.entity_type].append(e)

            # Party entities should be canonical
            if EntityType.PARTY in type_groups:
                canonical = type_groups[EntityType.PARTY][0]
                for entity_type, type_entities in type_groups.items():
                    if entity_type != EntityType.PARTY:
                        for e in type_entities:
                            # Don't merge, but add reference
                            e.properties["refers_to_party"] = str(canonical.id)

        return kg


def get_resolution_stage() -> ResolutionStage:
    """Get resolution stage instance."""
    return ResolutionStage()
