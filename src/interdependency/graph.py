"""Dependency graph builder and algorithm runner.

Builds a NetworkX directed graph from clause nodes and dependency edges,
then provides graph algorithms for analysis.
"""

from collections import deque

import networkx as nx

from .types import (
    DependencyType,
    ClauseNode,
    DependencyEdge,
    ClauseDependencyGraph,
    ImpactResult,
)


class DependencyGraphBuilder:
    """Builds and analyzes a clause dependency graph using NetworkX."""

    def __init__(self):
        self._graph: nx.DiGraph | None = None
        self._nodes: dict[str, ClauseNode] = {}
        self._edges: list[DependencyEdge] = []

    def build(
        self,
        nodes: list[ClauseNode],
        edges: list[DependencyEdge],
        contract_id: str,
    ) -> ClauseDependencyGraph:
        """Build the NetworkX graph from nodes and edges.

        Args:
            nodes: List of clause nodes (present and missing).
            edges: List of dependency edges.
            contract_id: Contract identifier.

        Returns:
            ClauseDependencyGraph with computed metadata.
        """
        self._graph = nx.DiGraph()
        self._nodes = {n.label: n for n in nodes}
        self._edges = edges

        # Add nodes
        for node in nodes:
            self._graph.add_node(
                node.label,
                category=node.category,
                present=node.present,
                confidence=node.confidence,
            )

        # Add edges
        for edge in edges:
            self._graph.add_edge(
                edge.source_label,
                edge.target_label,
                dependency_type=edge.dependency_type.value,
                strength=edge.strength,
                reason=edge.reason,
                bidirectional=edge.bidirectional,
            )
            # Add reverse edge for bidirectional
            if edge.bidirectional:
                self._graph.add_edge(
                    edge.target_label,
                    edge.source_label,
                    dependency_type=edge.dependency_type.value,
                    strength=edge.strength,
                    reason=edge.reason,
                    bidirectional=True,
                )

        contradictions = self.find_contradictions()

        return ClauseDependencyGraph(
            contract_id=contract_id,
            nodes=nodes,
            edges=edges,
            contradiction_count=len(contradictions),
            max_impact_clause=self._find_max_impact_clause(),
        )

    def impact_analysis(self, clause_label: str, max_hops: int = 3) -> ImpactResult:
        """BFS from a clause to find all affected clauses within max_hops.

        Args:
            clause_label: Starting clause label.
            max_hops: Maximum traversal depth.

        Returns:
            ImpactResult with affected clauses and depths.
        """
        if self._graph is None or clause_label not in self._graph:
            return ImpactResult(source_label=clause_label)

        affected = []
        visited = {clause_label}
        queue = deque([(clause_label, 0)])
        max_depth = 0

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for neighbor in self._graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_depth = depth + 1
                    max_depth = max(max_depth, next_depth)

                    edge_data = self._graph.edges[current, neighbor]
                    affected.append({
                        "label": neighbor,
                        "depth": next_depth,
                        "via": current,
                        "dependency_type": edge_data.get("dependency_type", ""),
                        "strength": edge_data.get("strength", 0),
                    })
                    queue.append((neighbor, next_depth))

        return ImpactResult(
            source_label=clause_label,
            affected_clauses=affected,
            total_affected=len(affected),
            max_depth=max_depth,
        )

    def find_contradictions(self) -> list[tuple[str, str, dict]]:
        """Find all CONFLICTS_WITH edges in the graph."""
        if self._graph is None:
            return []

        contradictions = []
        seen = set()

        for u, v, data in self._graph.edges(data=True):
            if data.get("dependency_type") == DependencyType.CONFLICTS_WITH.value:
                pair = tuple(sorted([u, v]))
                if pair not in seen:
                    seen.add(pair)
                    contradictions.append((u, v, data))

        return contradictions

    def find_cycles(self) -> list[list[str]]:
        """Find circular dependencies in the graph."""
        if self._graph is None:
            return []

        try:
            cycles = list(nx.simple_cycles(self._graph))
            # Filter out trivial bidirectional cycles (length 2)
            return [c for c in cycles if len(c) > 2]
        except nx.NetworkXError:
            return []

    def topological_sort(self) -> list[str]:
        """Get priority ordering of clauses (removes conflict edges first)."""
        if self._graph is None:
            return []

        # Create a copy without CONFLICTS_WITH edges for acyclic sort
        dag = self._graph.copy()
        edges_to_remove = [
            (u, v) for u, v, d in dag.edges(data=True)
            if d.get("dependency_type") == DependencyType.CONFLICTS_WITH.value
        ]
        dag.remove_edges_from(edges_to_remove)

        # Also remove any remaining cycles
        while True:
            try:
                return list(nx.topological_sort(dag))
            except nx.NetworkXUnfeasible:
                # Break a cycle by removing the weakest edge
                try:
                    cycle = nx.find_cycle(dag)
                    weakest = min(cycle, key=lambda e: dag.edges[e[0], e[1]].get("strength", 1))
                    dag.remove_edge(weakest[0], weakest[1])
                except nx.NetworkXError:
                    return list(dag.nodes)

    def connected_components(self) -> list[set[str]]:
        """Find independent clause groups."""
        if self._graph is None:
            return []
        return [set(c) for c in nx.weakly_connected_components(self._graph)]

    def centrality_scores(self) -> dict[str, float]:
        """Compute degree centrality for each clause."""
        if self._graph is None:
            return {}
        return nx.degree_centrality(self._graph)

    def betweenness_centrality(self) -> dict[str, float]:
        """Compute betweenness centrality for each clause."""
        if self._graph is None:
            return {}
        return nx.betweenness_centrality(self._graph)

    def get_layout(self) -> dict[str, tuple[float, float]]:
        """Compute spring layout positions for visualization."""
        if self._graph is None:
            return {}
        return nx.spring_layout(self._graph, k=2.0, iterations=50, seed=42)

    def _find_max_impact_clause(self) -> str:
        """Find the clause with the most outgoing dependencies."""
        if self._graph is None or len(self._graph) == 0:
            return ""

        max_label = ""
        max_out = 0

        for node in self._graph.nodes:
            out_degree = self._graph.out_degree(node)
            if out_degree > max_out:
                max_out = out_degree
                max_label = node

        return max_label
