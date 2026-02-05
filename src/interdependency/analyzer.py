"""Interdependency analysis orchestrator.

Coordinates detection, graph building, and algorithm execution
to produce a complete InterdependencyReport.
"""

from .types import (
    ClauseNode,
    DependencyType,
    InterdependencyReport,
)
from .detector import DependencyDetector
from .graph import DependencyGraphBuilder
from ..classification.cuad_labels import get_label_category
from ..pipeline import ContractAnalysis


class InterdependencyAnalyzer:
    """Orchestrates full clause interdependency analysis."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.detector = DependencyDetector(use_llm=use_llm)
        self.graph_builder = DependencyGraphBuilder()

    def analyze(self, analysis: ContractAnalysis) -> InterdependencyReport:
        """Run full interdependency analysis on a contract.

        Args:
            analysis: ContractAnalysis with classified/analyzed clauses.

        Returns:
            InterdependencyReport with graph, contradictions, missing requirements,
            impact rankings, risk amplifiers, and recommendations.
        """
        # Step 1: Build clause nodes
        nodes = self._build_nodes(analysis)

        # Step 2: Detect dependencies
        edges, missing = self.detector.detect(analysis)

        # Step 3: Build graph
        graph = self.graph_builder.build(nodes, edges, analysis.contract_id)
        graph.missing_requirements = missing

        # Step 4: Run algorithms
        contradictions = self._format_contradictions(
            self.graph_builder.find_contradictions()
        )
        cycles = self.graph_builder.find_cycles()
        centrality = self.graph_builder.centrality_scores()

        # Step 5: Compute impact rankings
        impact_rankings = self._compute_impact_rankings(analysis)

        # Step 6: Compute risk amplifiers
        risk_amplifiers = self._compute_risk_amplifiers(
            contradictions, missing, cycles
        )

        # Step 7: Compute risk score adjustment
        risk_adjustment = self._compute_risk_adjustment(
            contradictions, missing, cycles
        )

        # Step 8: Generate recommendations
        recommendations = self._generate_recommendations(
            contradictions, missing, cycles, centrality
        )

        return InterdependencyReport(
            contract_id=analysis.contract_id,
            graph=graph,
            contradictions=contradictions,
            missing_requirements=missing,
            impact_rankings=impact_rankings,
            risk_amplifiers=risk_amplifiers,
            recommendations=recommendations,
            risk_score_adjustment=risk_adjustment,
            cycles=cycles,
            centrality_scores=centrality,
        )

    def _build_nodes(self, analysis: ContractAnalysis) -> list[ClauseNode]:
        """Build ClauseNode list from analyzed clauses."""
        nodes = []
        for clause in analysis.analyzed_clauses:
            nodes.append(ClauseNode(
                label=clause.cuad_label,
                category=clause.category,
                text=clause.text,
                confidence=clause.label_confidence,
                present=True,
            ))
        return nodes

    def _format_contradictions(
        self, raw: list[tuple[str, str, dict]]
    ) -> list[dict]:
        """Format contradiction tuples into dicts."""
        return [
            {
                "clause_a": a,
                "clause_b": b,
                "reason": data.get("reason", ""),
                "strength": data.get("strength", 0),
            }
            for a, b, data in raw
        ]

    def _compute_impact_rankings(
        self, analysis: ContractAnalysis
    ) -> list[dict]:
        """Rank clauses by their impact (number of affected clauses)."""
        rankings = []

        for clause in analysis.analyzed_clauses:
            impact = self.graph_builder.impact_analysis(clause.cuad_label, max_hops=3)
            rankings.append({
                "label": clause.cuad_label,
                "total_affected": impact.total_affected,
                "max_depth": impact.max_depth,
                "affected_clauses": [c["label"] for c in impact.affected_clauses],
            })

        # Sort by total_affected descending
        rankings.sort(key=lambda r: r["total_affected"], reverse=True)
        return rankings

    def _compute_risk_amplifiers(
        self, contradictions: list[dict], missing: list, cycles: list
    ) -> list[dict]:
        """Identify factors that amplify overall contract risk."""
        amplifiers = []

        if contradictions:
            amplifiers.append({
                "type": "contradictions",
                "count": len(contradictions),
                "impact": "Contradicting clauses create legal ambiguity and litigation risk",
                "details": [
                    f"{c['clause_a']} conflicts with {c['clause_b']}"
                    for c in contradictions
                ],
            })

        high_missing = [m for m in missing if m.severity == "HIGH"]
        if high_missing:
            amplifiers.append({
                "type": "missing_critical_dependencies",
                "count": len(high_missing),
                "impact": "Missing required clauses may render existing provisions unenforceable",
                "details": [
                    f"{m.required_by} requires missing {m.missing_label}"
                    for m in high_missing
                ],
            })

        if cycles:
            amplifiers.append({
                "type": "circular_dependencies",
                "count": len(cycles),
                "impact": "Circular dependencies create interpretation ambiguity",
                "details": [" → ".join(c + [c[0]]) for c in cycles],
            })

        return amplifiers

    def _compute_risk_adjustment(
        self, contradictions: list[dict], missing: list, cycles: list
    ) -> int:
        """Compute risk score adjustment based on interdependency findings.

        Returns:
            Integer adjustment to add to the base risk score (capped at +40).
        """
        adjustment = 0

        # +10 per contradiction, scaled by strength
        for c in contradictions:
            adjustment += int(10 * c.get("strength", 1.0))

        # +8 per HIGH missing, +5 per MEDIUM missing
        for m in missing:
            if m.severity == "HIGH":
                adjustment += 8
            elif m.severity == "MEDIUM":
                adjustment += 5

        # +3 per non-trivial cycle
        adjustment += 3 * len(cycles)

        return min(adjustment, 40)

    def _generate_recommendations(
        self,
        contradictions: list[dict],
        missing: list,
        cycles: list,
        centrality: dict[str, float],
    ) -> list[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []

        # Contradiction recommendations
        for c in contradictions:
            recommendations.append(
                f"RESOLVE CONFLICT: '{c['clause_a']}' and '{c['clause_b']}' contain "
                f"contradictory provisions. Review and reconcile these clauses to eliminate ambiguity."
            )

        # Missing requirement recommendations
        high_missing = [m for m in missing if m.severity == "HIGH"]
        for m in high_missing:
            recommendations.append(
                f"ADD MISSING CLAUSE: '{m.missing_label}' is required by '{m.required_by}'. "
                f"{m.impact}"
            )

        medium_missing = [m for m in missing if m.severity == "MEDIUM"]
        if medium_missing:
            labels = ", ".join(f"'{m.missing_label}'" for m in medium_missing[:5])
            recommendations.append(
                f"CONSIDER ADDING: {labels} — these clauses would strengthen "
                f"existing provisions."
            )

        # Cycle recommendations
        for cycle in cycles:
            path = " → ".join(cycle + [cycle[0]])
            recommendations.append(
                f"REVIEW CIRCULAR DEPENDENCY: {path}. Circular dependencies can "
                f"create interpretation deadlocks."
            )

        # Centrality-based recommendations
        if centrality:
            top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            high_centrality = [label for label, score in top if score > 0.3]
            if high_centrality:
                labels = ", ".join(f"'{l}'" for l in high_centrality)
                recommendations.append(
                    f"HIGH-IMPACT CLAUSES: {labels} have high connectivity. "
                    f"Changes to these clauses will ripple through multiple provisions."
                )

        return recommendations
