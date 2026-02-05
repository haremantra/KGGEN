"""Cross-Contract Aggregation — per-entity-type dedup thresholds.

Merges and normalizes entities and triples across multiple contracts,
using configurable per-type similarity thresholds.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..config import get_settings, DEFAULT_AGGREGATION_THRESHOLDS
from ..utils.embedding import get_embedding_service
from ..models.schema import KGNode, Triple, NodeType


DEFAULT_THRESHOLD = 0.80


@dataclass
class AggregationResult:
    """Result of cross-contract aggregation."""
    entities: list[KGNode]
    triples: list[Triple]
    id_mapping: dict[str, str] = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "merged_entities": len(self.entities),
            "merged_triples": len(self.triples),
            **self.statistics,
        }


class ContractAggregator:
    """Aggregates entities and triples across contracts with per-type dedup."""

    def __init__(self):
        self._settings = get_settings()
        self._embedding = None
        self._thresholds = dict(DEFAULT_AGGREGATION_THRESHOLDS)
        # Override with config if present
        self._thresholds.update(self._settings.aggregation_thresholds)

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = get_embedding_service()
        return self._embedding

    def _get_threshold(self, entity_type: str) -> float:
        """Get dedup threshold for an entity type."""
        return self._thresholds.get(entity_type, DEFAULT_THRESHOLD)

    def aggregate(
        self,
        extractions: dict[str, tuple[list[KGNode], list[Triple]]],
    ) -> AggregationResult:
        """Aggregate extractions from multiple contracts.

        Args:
            extractions: Dict mapping contract_id to (entities, triples)
        """
        all_entities: list[KGNode] = []
        all_triples: list[Triple] = []
        entity_sources: dict[str, list[str]] = defaultdict(list)

        for contract_id, (entities, triples) in extractions.items():
            for entity in entities:
                all_entities.append(entity)
                entity_sources[entity.id].append(contract_id)
            all_triples.extend(triples)

        original_entity_count = len(all_entities)
        original_triple_count = len(all_triples)

        # Group by type
        by_type: dict[str, list[KGNode]] = defaultdict(list)
        for entity in all_entities:
            by_type[entity.type.value].append(entity)

        # Dedup within each type using type-specific thresholds
        merged_entities: list[KGNode] = []
        id_mapping: dict[str, str] = {}

        for type_name, entities in by_type.items():
            threshold = self._get_threshold(type_name)
            type_merged, type_mapping = self._deduplicate_entities(entities, threshold)
            merged_entities.extend(type_merged)
            id_mapping.update(type_mapping)

        # Remap triples
        remapped_triples = self._remap_triples(all_triples, id_mapping)
        deduped_triples = self._deduplicate_triples(remapped_triples)

        # Confidence boosting for cross-contract entities
        total_contracts = len(extractions)
        for triple in deduped_triples:
            triple.confidence = self._boost_confidence(
                triple, entity_sources, total_contracts
            )

        stats = {
            "source_contracts": total_contracts,
            "original_entities": original_entity_count,
            "original_triples": original_triple_count,
            "reduction_rate": 1 - len(merged_entities) / max(original_entity_count, 1),
            "thresholds_used": {k: self._get_threshold(k) for k in by_type.keys()},
        }

        return AggregationResult(
            entities=merged_entities,
            triples=deduped_triples,
            id_mapping=id_mapping,
            statistics=stats,
        )

    def _deduplicate_entities(
        self,
        entities: list[KGNode],
        threshold: float,
    ) -> tuple[list[KGNode], dict[str, str]]:
        """Dedup entities using embedding similarity with given threshold."""
        if len(entities) <= 1:
            return entities, {}

        embeddings = [
            self.embedding.embed_entity(e.name, e.type.value) for e in entities
        ]

        id_mapping: dict[str, str] = {}
        merged: list[KGNode] = []
        processed: set[int] = set()

        for i in range(len(entities)):
            if i in processed:
                continue

            similar_indices = [i]
            for j in range(i + 1, len(entities)):
                if j in processed:
                    continue
                sim = self.embedding.compute_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    similar_indices.append(j)

            if len(similar_indices) == 1:
                merged.append(entities[i])
            else:
                group = [entities[idx] for idx in similar_indices]
                canonical = self._select_canonical(group)
                merged.append(canonical)
                for idx in similar_indices:
                    if entities[idx].id != canonical.id:
                        id_mapping[entities[idx].id] = canonical.id
                    processed.add(idx)

            processed.add(i)

        return merged, id_mapping

    def _select_canonical(self, entities: list[KGNode]) -> KGNode:
        """Select canonical entity from a group."""
        def score(e: KGNode) -> int:
            s = len(e.name)
            if not e.name.lower().startswith(("the ", "a ", "an ")):
                s += 10
            if e.name[0].isupper():
                s += 5
            s += len(e.properties) * 2
            return s

        best = max(entities, key=score)
        aliases = {e.name for e in entities if e.id != best.id}
        if aliases:
            best.properties["aliases"] = list(aliases)
        return best

    def _remap_triples(
        self, triples: list[Triple], id_mapping: dict[str, str]
    ) -> list[Triple]:
        """Remap entity references in triples."""
        remapped = []
        for t in triples:
            new_subject = id_mapping.get(t.subject, t.subject)
            new_object = id_mapping.get(t.object, t.object)
            remapped.append(Triple(
                subject=new_subject,
                predicate=t.predicate,
                object=new_object,
                properties=t.properties,
                confidence=t.confidence,
                source_text=t.source_text,
            ))
        return remapped

    def _deduplicate_triples(self, triples: list[Triple]) -> list[Triple]:
        """Dedup triples keeping highest confidence."""
        seen: dict[str, Triple] = {}
        for t in triples:
            key = f"{t.subject}:{t.predicate}:{t.object}"
            if key in seen:
                existing = seen[key]
                if t.confidence > existing.confidence:
                    seen[key] = t
                else:
                    for k, v in t.properties.items():
                        if k not in existing.properties:
                            existing.properties[k] = v
            else:
                seen[key] = t
        return list(seen.values())

    def _boost_confidence(
        self,
        triple: Triple,
        entity_sources: dict[str, list[str]],
        total_contracts: int,
    ) -> float:
        """Boost confidence for cross-contract validated triples."""
        base = triple.confidence or 0.7
        if total_contracts <= 1:
            return base

        subj_count = len(entity_sources.get(triple.subject, []))
        obj_count = len(entity_sources.get(triple.object, []))
        avg_coverage = (subj_count + obj_count) / (2 * total_contracts)
        boost = avg_coverage * 0.2

        return min(1.0, base + boost)

    def normalize_entities(self, entities: list[KGNode]) -> list[KGNode]:
        """Normalize entity names (party suffixes, date formats, etc.)."""
        normalized = []
        for entity in entities:
            name = entity.name.strip()
            for noise in ['"', "'", ":", ";"]:
                name = name.strip(noise)

            if entity.type == NodeType.PARTY:
                name = self._normalize_party_name(name)
            elif entity.type == NodeType.TEMPORAL:
                name = self._normalize_temporal(name)

            entity.name = name
            normalized.append(entity)
        return normalized

    def _normalize_party_name(self, name: str) -> str:
        """Standardize party name suffixes."""
        replacements = {
            " inc.": " Inc.", " inc": " Inc.", " llc": " LLC",
            " ltd": " Ltd.", " ltd.": " Ltd.", " corp.": " Corp.",
            " corp": " Corp.", " corporation": " Corporation",
            " company": " Company",
        }
        name_lower = name.lower()
        for old, new in replacements.items():
            if name_lower.endswith(old):
                name = name[:-len(old)] + new
                break
        return name

    def _normalize_temporal(self, name: str) -> str:
        """Standardize date formats to YYYY-MM-DD."""
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', name)
        if match:
            month, day, year = match.groups()
            name = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return name

    def compute_statistics(
        self, entities: list[KGNode], triples: list[Triple]
    ) -> dict:
        """Compute statistics for a set of entities and triples."""
        by_type: dict[str, int] = defaultdict(int)
        for e in entities:
            by_type[e.type.value] += 1

        by_predicate: dict[str, int] = defaultdict(int)
        for t in triples:
            by_predicate[t.predicate] += 1

        return {
            "total_entities": len(entities),
            "total_triples": len(triples),
            "entities_by_type": dict(by_type),
            "triples_by_predicate": dict(by_predicate),
        }
