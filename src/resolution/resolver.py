"""Entity Resolution — adaptive clustering + LLM canonical selection.

Resolves duplicate entities using S-BERT embeddings and k-means clustering
with silhouette score validation for optimal cluster count.
"""

# TODO: migrate to DBSCAN for automatic cluster count

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..config import get_settings
from ..utils.embedding import get_embedding_service
from ..utils.llm import get_llm_service
from ..models.schema import KGNode, Triple, NodeType


@dataclass
class ResolutionResult:
    """Result of entity resolution."""
    resolved_entities: list[KGNode]
    resolved_triples: list[Triple]
    id_mapping: dict[str, str] = field(default_factory=dict)  # old_id -> canonical_id
    alias_mapping: dict[str, list[str]] = field(default_factory=dict)  # canonical_id -> aliases
    original_count: int = 0
    resolved_count: int = 0

    def to_dict(self) -> dict:
        return {
            "original_count": self.original_count,
            "resolved_count": self.resolved_count,
            "reduction_rate": 1 - self.resolved_count / max(self.original_count, 1),
            "alias_groups": len(self.alias_mapping),
        }


class EntityResolver:
    """Resolves duplicate entities using embedding similarity and LLM validation."""

    def __init__(self, use_llm: bool = True):
        self._settings = get_settings()
        self._embedding = None
        self._llm = None
        self._use_llm = use_llm

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = get_embedding_service()
        return self._embedding

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm_service()
        return self._llm

    def resolve(
        self,
        entities: list[KGNode],
        triples: list[Triple],
    ) -> ResolutionResult:
        """Resolve duplicate entities and remap triples."""
        original_count = len(entities)

        # Group by type for type-aware resolution
        by_type: dict[NodeType, list[KGNode]] = defaultdict(list)
        for entity in entities:
            by_type[entity.type].append(entity)

        resolved_entities: list[KGNode] = []
        id_mapping: dict[str, str] = {}
        alias_mapping: dict[str, list[str]] = {}

        for node_type, type_entities in by_type.items():
            type_resolved, type_mapping, type_aliases = self._resolve_type(
                type_entities, node_type
            )
            resolved_entities.extend(type_resolved)
            id_mapping.update(type_mapping)
            alias_mapping.update(type_aliases)

        # Remap triples
        resolved_triples = self._remap_triples(triples, id_mapping)
        resolved_triples = self._dedup_triples(resolved_triples)

        return ResolutionResult(
            resolved_entities=resolved_entities,
            resolved_triples=resolved_triples,
            id_mapping=id_mapping,
            alias_mapping=alias_mapping,
            original_count=original_count,
            resolved_count=len(resolved_entities),
        )

    def _resolve_type(
        self,
        entities: list[KGNode],
        node_type: NodeType,
    ) -> tuple[list[KGNode], dict[str, str], dict[str, list[str]]]:
        """Resolve entities of a single type."""
        if len(entities) <= 1:
            return entities, {}, {}

        # Generate embeddings
        embeddings = [
            self.embedding.embed_entity(e.name, node_type.value) for e in entities
        ]

        n = len(entities)

        if n < 20:
            # Pairwise similarity for small sets
            clusters = self._pairwise_cluster(entities, embeddings)
        else:
            # Adaptive k-means with silhouette validation
            clusters = self._adaptive_kmeans_cluster(entities, embeddings)

        # Process clusters
        resolved: list[KGNode] = []
        id_map: dict[str, str] = {}
        alias_map: dict[str, list[str]] = {}

        for cluster in clusters:
            if len(cluster) == 1:
                resolved.append(cluster[0])
            else:
                canonical, aliases = self._select_canonical(cluster)
                resolved.append(canonical)
                alias_map[canonical.id] = aliases
                for entity in cluster:
                    if entity.id != canonical.id:
                        id_map[entity.id] = canonical.id

        return resolved, id_map, alias_map

    def _pairwise_cluster(
        self,
        entities: list[KGNode],
        embeddings: list[list[float]],
    ) -> list[list[KGNode]]:
        """Cluster via pairwise cosine similarity for small n (<20)."""
        threshold = self._settings.resolution_similarity_threshold
        n = len(entities)
        processed = set()
        clusters: list[list[KGNode]] = []

        for i in range(n):
            if i in processed:
                continue
            group = [entities[i]]
            processed.add(i)

            for j in range(i + 1, n):
                if j in processed:
                    continue
                sim = self.embedding.compute_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    group.append(entities[j])
                    processed.add(j)

            clusters.append(group)

        return clusters

    def _adaptive_kmeans_cluster(
        self,
        entities: list[KGNode],
        embeddings: list[list[float]],
    ) -> list[list[KGNode]]:
        """Adaptive k-means: try k-1, k, k+1 and pick best silhouette score."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = np.array(embeddings)
        n = len(entities)
        base_k = max(2, int(math.sqrt(n)))

        # Candidate k values
        candidates = [k for k in [base_k - 1, base_k, base_k + 1] if 2 <= k < n]

        best_labels = None
        best_score = -1.0

        for k in candidates:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        # Group entities by cluster label
        cluster_map: dict[int, list[KGNode]] = defaultdict(list)
        for idx, label in enumerate(best_labels):
            cluster_map[int(label)].append(entities[idx])

        return list(cluster_map.values())

    def _select_canonical(
        self, entities: list[KGNode]
    ) -> tuple[KGNode, list[str]]:
        """Select canonical entity from a cluster."""
        if self._use_llm:
            try:
                names = [e.name for e in entities]
                canonical_name, aliases = self.llm.select_canonical(names)
                # Find matching entity
                for e in entities:
                    if e.name == canonical_name:
                        return e, aliases
                # Fallback: use LLM's name on first entity
                entities[0].name = canonical_name
                return entities[0], [e.name for e in entities[1:]]
            except Exception:
                pass

        return self._heuristic_canonical(entities)

    def _heuristic_canonical(
        self, entities: list[KGNode]
    ) -> tuple[KGNode, list[str]]:
        """Heuristic canonical selection (fallback)."""
        def score(e: KGNode) -> int:
            s = len(e.name)
            if not e.name.lower().startswith(("the ", "a ", "an ")):
                s += 20
            if e.name[0].isupper():
                s += 10
            s -= sum(1 for c in e.name if c in "()[]{}\"'") * 5
            if e.type == NodeType.PARTY:
                for suffix in [" Inc.", " LLC", " Ltd.", " Corp.", " Corporation"]:
                    if e.name.endswith(suffix):
                        s += 15
                        break
            s += len(e.properties) * 2
            return s

        sorted_entities = sorted(entities, key=score, reverse=True)
        canonical = sorted_entities[0]
        aliases = [e.name for e in sorted_entities[1:]]
        return canonical, aliases

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

    def _dedup_triples(self, triples: list[Triple]) -> list[Triple]:
        """Deduplicate triples after remapping."""
        seen: dict[str, Triple] = {}
        for t in triples:
            key = f"{t.subject}:{t.predicate}:{t.object}"
            if key in seen:
                existing = seen[key]
                existing.confidence = (existing.confidence + t.confidence) / 2
            else:
                seen[key] = t
        return list(seen.values())

    def find_duplicates(
        self,
        entities: list[KGNode],
        similarity_threshold: float | None = None,
    ) -> list[list[KGNode]]:
        """Find groups of duplicate entities."""
        threshold = similarity_threshold or self._settings.resolution_similarity_threshold
        if len(entities) <= 1:
            return []

        embeddings = [
            self.embedding.embed_entity(e.name, e.type.value) for e in entities
        ]

        groups: list[list[KGNode]] = []
        processed: set[int] = set()

        for i in range(len(entities)):
            if i in processed:
                continue
            group = [entities[i]]
            processed.add(i)

            for j in range(i + 1, len(entities)):
                if j in processed:
                    continue
                sim = self.embedding.compute_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    group.append(entities[j])
                    processed.add(j)

            if len(group) >= 2:
                groups.append(group)

        return groups
