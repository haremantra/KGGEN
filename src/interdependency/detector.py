"""Dependency detector for contract clauses.

Iterates static rules against classified clauses and optionally
uses LLM to validate content-dependent dependencies.
"""

import json

from anthropic import Anthropic

from .types import DependencyType, DependencyEdge, MissingRequirement
from .matrix import DEPENDENCY_RULES, get_requires_rules
from ..config import settings
from ..pipeline import ContractAnalysis


class DependencyDetector:
    """Detects dependencies between contract clauses using static rules and optional LLM."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm and settings.anthropic_api_key:
            self.client = Anthropic(api_key=settings.anthropic_api_key)
        else:
            self.client = None
            self.use_llm = False

    def detect(
        self, analysis: ContractAnalysis
    ) -> tuple[list[DependencyEdge], list[MissingRequirement]]:
        """Detect dependencies and missing requirements from a contract analysis.

        Args:
            analysis: ContractAnalysis with classified/analyzed clauses.

        Returns:
            Tuple of (detected edges, missing requirements).
        """
        # Build set of labels present in the contract
        present_labels: dict[str, str] = {}  # label -> clause text
        for clause in analysis.analyzed_clauses:
            present_labels[clause.cuad_label] = clause.text

        edges = self._detect_static(present_labels)
        missing = self._detect_missing(present_labels)

        # LLM validation for rules that need it
        if self.use_llm and self.client:
            edges = self._validate_with_llm(edges, present_labels)

        return edges, missing

    def _detect_static(self, present_labels: dict[str, str]) -> list[DependencyEdge]:
        """Apply all 73 static rules against present labels."""
        edges = []

        for rule in DEPENDENCY_RULES:
            source_present = rule.source_label in present_labels
            target_present = rule.target_label in present_labels

            if source_present and target_present:
                edges.append(DependencyEdge(
                    source_label=rule.source_label,
                    target_label=rule.target_label,
                    dependency_type=rule.dependency_type,
                    strength=rule.default_strength,
                    reason=rule.reason,
                    bidirectional=rule.bidirectional,
                    detected_by="rule",
                ))

            # For bidirectional rules, also check the reverse
            if rule.bidirectional and target_present and source_present:
                # Already covered above — bidirectional flag on the edge handles it
                pass

        return edges

    def _detect_missing(self, present_labels: dict[str, str]) -> list[MissingRequirement]:
        """Detect missing REQUIRES dependencies (source present, target absent)."""
        missing = []

        for rule in get_requires_rules():
            if rule.source_label in present_labels and rule.target_label not in present_labels:
                severity = "HIGH" if rule.default_strength >= 0.8 else "MEDIUM"
                missing.append(MissingRequirement(
                    required_by=rule.source_label,
                    missing_label=rule.target_label,
                    reason=rule.reason,
                    impact=f"'{rule.source_label}' expects '{rule.target_label}' to be present. "
                           f"Without it, the {rule.source_label.lower()} provision may be unenforceable "
                           f"or create ambiguity.",
                    severity=severity,
                ))

        return missing

    def _validate_with_llm(
        self, edges: list[DependencyEdge], present_labels: dict[str, str]
    ) -> list[DependencyEdge]:
        """Use LLM to validate edges that require content-level analysis."""
        validated = []

        for edge in edges:
            # Find the matching rule to check if LLM validation is needed
            needs_llm = any(
                r.source_label == edge.source_label
                and r.target_label == edge.target_label
                and r.requires_llm_validation
                for r in DEPENDENCY_RULES
            )

            if not needs_llm:
                validated.append(edge)
                continue

            # Get clause texts
            source_text = present_labels.get(edge.source_label, "")
            target_text = present_labels.get(edge.target_label, "")

            if not source_text or not target_text:
                validated.append(edge)
                continue

            # Ask LLM to validate
            result = self._llm_validate_edge(edge, source_text, target_text)
            if result:
                validated.append(result)

        return validated

    def _llm_validate_edge(
        self, edge: DependencyEdge, source_text: str, target_text: str
    ) -> DependencyEdge | None:
        """Ask LLM to validate a specific dependency edge."""
        prompt = f"""Analyze whether this dependency exists between two contract clauses.

DEPENDENCY TYPE: {edge.dependency_type.value}
RULE: {edge.reason}

CLAUSE A ({edge.source_label}):
{source_text[:500]}

CLAUSE B ({edge.target_label}):
{target_text[:500]}

Does this {edge.dependency_type.value} dependency actually exist based on the clause content?
Return JSON:
{{
    "exists": true/false,
    "strength": 0.0-1.0,
    "finding": "Brief explanation of the content-specific dependency or why it doesn't exist"
}}"""

        try:
            response = self.client.messages.create(
                model=settings.default_llm_model,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text
            if "```json" in result_text:
                start = result_text.find("```json") + 7
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()
            elif "```" in result_text:
                start = result_text.find("```") + 3
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()

            data = json.loads(result_text)

            if not data.get("exists", True):
                return None  # LLM says dependency doesn't exist

            edge.strength = data.get("strength", edge.strength)
            edge.content_finding = data.get("finding", "")
            edge.detected_by = "llm"
            return edge

        except Exception:
            # On LLM failure, keep the rule-based edge
            return edge
