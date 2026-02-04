"""Portfolio-level contract analysis.

Provides cross-contract analysis capabilities:
- Aggregate risk across multiple contracts
- Identify common and missing clauses
- Compare terms across vendors
- Gap analysis for missing protections
"""

from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from ..pipeline import ContractAnalysis, ContractAnalysisPipeline, AnalyzedClause
from ..risk.assessor import RiskAssessment, RiskAssessor, RiskFinding
from ..risk.rules import RiskSeverity, MISSING_CLAUSE_RISKS
from ..classification.cuad_labels import CUAD_LABELS, get_labels_by_category


@dataclass
class ContractSummary:
    """Summary of a single contract for portfolio view."""
    contract_id: str
    risk_score: int
    risk_level: str
    labels_found: list[str]
    critical_findings: int
    high_findings: int
    missing_protections: list[str]

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "labels_found": self.labels_found,
            "critical_findings": self.critical_findings,
            "high_findings": self.high_findings,
            "missing_protections": self.missing_protections,
        }


@dataclass
class PortfolioRiskSummary:
    """Aggregated risk summary across portfolio."""
    total_contracts: int
    average_risk_score: float
    contracts_by_risk_level: dict[str, int]
    most_common_risks: list[tuple[str, int]]  # (label, count)
    most_common_gaps: list[tuple[str, int]]  # (missing label, count)
    highest_risk_contracts: list[str]

    def to_dict(self) -> dict:
        return {
            "total_contracts": self.total_contracts,
            "average_risk_score": round(self.average_risk_score, 1),
            "contracts_by_risk_level": self.contracts_by_risk_level,
            "most_common_risks": [
                {"label": label, "count": count}
                for label, count in self.most_common_risks
            ],
            "most_common_gaps": [
                {"label": label, "count": count}
                for label, count in self.most_common_gaps
            ],
            "highest_risk_contracts": self.highest_risk_contracts,
        }


@dataclass
class ClauseComparison:
    """Comparison of a specific clause across contracts."""
    label: str
    category: str
    contracts_with_clause: list[str]
    contracts_without_clause: list[str]
    extracted_values: dict[str, list[dict]]  # contract_id -> extracted values
    coverage_percentage: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "category": self.category,
            "contracts_with_clause": self.contracts_with_clause,
            "contracts_without_clause": self.contracts_without_clause,
            "extracted_values": self.extracted_values,
            "coverage_percentage": round(self.coverage_percentage, 1),
        }


@dataclass
class ContractComparison:
    """Side-by-side comparison of two contracts."""
    contract_a: str
    contract_b: str
    shared_clauses: list[str]
    only_in_a: list[str]
    only_in_b: list[str]
    risk_comparison: dict
    clause_differences: list[dict]

    def to_dict(self) -> dict:
        return {
            "contract_a": self.contract_a,
            "contract_b": self.contract_b,
            "shared_clauses": self.shared_clauses,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "risk_comparison": self.risk_comparison,
            "clause_differences": self.clause_differences,
        }


@dataclass
class GapAnalysis:
    """Analysis of missing protections across portfolio."""
    standard_protections: list[str]  # Expected labels
    gap_by_contract: dict[str, list[str]]  # contract_id -> missing labels
    gap_frequency: list[tuple[str, int, float]]  # (label, count, percentage)
    priority_recommendations: list[dict]

    def to_dict(self) -> dict:
        return {
            "standard_protections": self.standard_protections,
            "gap_by_contract": self.gap_by_contract,
            "gap_frequency": [
                {"label": label, "missing_count": count, "missing_percentage": round(pct, 1)}
                for label, count, pct in self.gap_frequency
            ],
            "priority_recommendations": self.priority_recommendations,
        }


@dataclass
class PortfolioAnalysis:
    """Complete portfolio analysis results."""
    contracts: list[ContractSummary]
    risk_summary: PortfolioRiskSummary
    clause_coverage: dict[str, ClauseComparison]
    gap_analysis: GapAnalysis

    def to_dict(self) -> dict:
        return {
            "contracts": [c.to_dict() for c in self.contracts],
            "risk_summary": self.risk_summary.to_dict(),
            "clause_coverage": {k: v.to_dict() for k, v in self.clause_coverage.items()},
            "gap_analysis": self.gap_analysis.to_dict(),
        }


# Standard protections to check for in gap analysis
STANDARD_PROTECTIONS = [
    "Cap On Liability",
    "Governing Law",
    "Termination For Convenience",
    "Notice Period To Terminate Renewal",
    "Warranty Duration",
    "Post-Termination Services",
    "Source Code Escrow",
    "Insurance",
    "Anti-Assignment",
    "Change Of Control",
]


class PortfolioAnalyzer:
    """Analyzer for portfolio-level contract analysis."""

    def __init__(self, use_llm: bool = True):
        """Initialize the portfolio analyzer.

        Args:
            use_llm: Whether to use LLM for complex analyses
        """
        self.pipeline = ContractAnalysisPipeline()
        self.risk_assessor = RiskAssessor(use_llm=use_llm)
        self._analyses: dict[str, ContractAnalysis] = {}
        self._risk_assessments: dict[str, RiskAssessment] = {}

    def add_analysis(
        self,
        analysis: ContractAnalysis,
        risk_assessment: RiskAssessment | None = None
    ):
        """Add a pre-computed analysis to the portfolio.

        Args:
            analysis: ContractAnalysis from the pipeline
            risk_assessment: Optional pre-computed risk assessment
        """
        self._analyses[analysis.contract_id] = analysis
        if risk_assessment:
            self._risk_assessments[analysis.contract_id] = risk_assessment
        else:
            self._risk_assessments[analysis.contract_id] = self.risk_assessor.assess(analysis)

    def analyze_folder(
        self,
        folder_path: str | Path,
        limit: int | None = None,
        file_pattern: str = "*.pdf"
    ) -> PortfolioAnalysis:
        """Analyze all contracts in a folder.

        Args:
            folder_path: Path to folder containing contracts
            limit: Maximum number of contracts to analyze
            file_pattern: Glob pattern for contract files

        Returns:
            PortfolioAnalysis with aggregated results
        """
        from ..utils.pdf_reader import extract_text_from_pdf

        folder = Path(folder_path)
        files = list(folder.glob(file_pattern))

        if limit:
            files = files[:limit]

        print(f"Analyzing {len(files)} contracts from {folder}...")

        # Initialize pipeline once
        self.pipeline.initialize()

        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")

            try:
                # Read contract
                if file_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                # Run analysis
                analysis = self.pipeline.analyze(text, contract_id=file_path.stem)
                risk = self.risk_assessor.assess(analysis)

                self._analyses[analysis.contract_id] = analysis
                self._risk_assessments[analysis.contract_id] = risk

                print(f"  Risk Score: {risk.overall_risk_score}/100 ({risk.risk_level})")

            except Exception as e:
                print(f"  Error analyzing {file_path.name}: {e}")

        return self.get_portfolio_analysis()

    def get_portfolio_analysis(self) -> PortfolioAnalysis:
        """Generate portfolio analysis from loaded contracts.

        Returns:
            PortfolioAnalysis with aggregated results
        """
        if not self._analyses:
            raise ValueError("No contracts loaded. Use add_analysis() or analyze_folder() first.")

        # Build contract summaries
        contract_summaries = []
        for contract_id, analysis in self._analyses.items():
            risk = self._risk_assessments[contract_id]

            labels_found = [c.cuad_label for c in analysis.analyzed_clauses]
            critical = sum(1 for f in risk.findings if f.severity == RiskSeverity.CRITICAL)
            high = sum(1 for f in risk.findings if f.severity == RiskSeverity.HIGH)
            missing = [f.label for f in risk.missing_clause_risks]

            contract_summaries.append(ContractSummary(
                contract_id=contract_id,
                risk_score=risk.overall_risk_score,
                risk_level=risk.risk_level,
                labels_found=labels_found,
                critical_findings=critical,
                high_findings=high,
                missing_protections=missing,
            ))

        # Build risk summary
        risk_summary = self._build_risk_summary(contract_summaries)

        # Build clause coverage
        clause_coverage = self._build_clause_coverage()

        # Build gap analysis
        gap_analysis = self._build_gap_analysis(contract_summaries)

        return PortfolioAnalysis(
            contracts=contract_summaries,
            risk_summary=risk_summary,
            clause_coverage=clause_coverage,
            gap_analysis=gap_analysis,
        )

    def compare_contracts(
        self,
        contract_a: str,
        contract_b: str
    ) -> ContractComparison:
        """Compare two contracts side by side.

        Args:
            contract_a: ID of first contract
            contract_b: ID of second contract

        Returns:
            ContractComparison with differences
        """
        if contract_a not in self._analyses or contract_b not in self._analyses:
            raise ValueError("Both contracts must be loaded first")

        analysis_a = self._analyses[contract_a]
        analysis_b = self._analyses[contract_b]
        risk_a = self._risk_assessments[contract_a]
        risk_b = self._risk_assessments[contract_b]

        labels_a = set(c.cuad_label for c in analysis_a.analyzed_clauses)
        labels_b = set(c.cuad_label for c in analysis_b.analyzed_clauses)

        shared = list(labels_a & labels_b)
        only_a = list(labels_a - labels_b)
        only_b = list(labels_b - labels_a)

        # Compare risk
        risk_comparison = {
            contract_a: {"score": risk_a.overall_risk_score, "level": risk_a.risk_level},
            contract_b: {"score": risk_b.overall_risk_score, "level": risk_b.risk_level},
            "difference": risk_a.overall_risk_score - risk_b.overall_risk_score,
        }

        # Get clause differences for shared clauses
        clause_differences = []
        for label in shared:
            clause_a = next((c for c in analysis_a.analyzed_clauses if c.cuad_label == label), None)
            clause_b = next((c for c in analysis_b.analyzed_clauses if c.cuad_label == label), None)

            if clause_a and clause_b:
                # Extract key values for comparison
                values_a = {v.field: v.value for v in clause_a.extracted_values}
                values_b = {v.field: v.value for v in clause_b.extracted_values}

                if values_a or values_b:
                    clause_differences.append({
                        "label": label,
                        contract_a: values_a,
                        contract_b: values_b,
                    })

        return ContractComparison(
            contract_a=contract_a,
            contract_b=contract_b,
            shared_clauses=shared,
            only_in_a=only_a,
            only_in_b=only_b,
            risk_comparison=risk_comparison,
            clause_differences=clause_differences,
        )

    def _build_risk_summary(
        self,
        summaries: list[ContractSummary]
    ) -> PortfolioRiskSummary:
        """Build aggregated risk summary."""

        total = len(summaries)
        avg_score = sum(s.risk_score for s in summaries) / total if total > 0 else 0

        # Count by risk level
        by_level = defaultdict(int)
        for s in summaries:
            by_level[s.risk_level] += 1

        # Most common risks
        risk_counts = defaultdict(int)
        for s in summaries:
            for label in s.labels_found:
                risk_counts[label] += 1
        most_common = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Most common gaps
        gap_counts = defaultdict(int)
        for s in summaries:
            for label in s.missing_protections:
                gap_counts[label] += 1
        most_gaps = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Highest risk contracts
        sorted_by_risk = sorted(summaries, key=lambda x: x.risk_score, reverse=True)
        highest_risk = [s.contract_id for s in sorted_by_risk[:5]]

        return PortfolioRiskSummary(
            total_contracts=total,
            average_risk_score=avg_score,
            contracts_by_risk_level=dict(by_level),
            most_common_risks=most_common,
            most_common_gaps=most_gaps,
            highest_risk_contracts=highest_risk,
        )

    def _build_clause_coverage(self) -> dict[str, ClauseComparison]:
        """Build clause coverage analysis across portfolio."""

        all_labels = list(CUAD_LABELS.keys())
        coverage = {}

        for label in all_labels:
            with_clause = []
            without_clause = []
            values_by_contract = {}

            for contract_id, analysis in self._analyses.items():
                clause = next(
                    (c for c in analysis.analyzed_clauses if c.cuad_label == label),
                    None
                )

                if clause:
                    with_clause.append(contract_id)
                    if clause.extracted_values:
                        values_by_contract[contract_id] = [
                            {"field": v.field, "value": v.value}
                            for v in clause.extracted_values
                        ]
                else:
                    without_clause.append(contract_id)

            total = len(self._analyses)
            pct = (len(with_clause) / total * 100) if total > 0 else 0

            from ..classification.cuad_labels import get_label_category
            category = get_label_category(label)

            coverage[label] = ClauseComparison(
                label=label,
                category=category,
                contracts_with_clause=with_clause,
                contracts_without_clause=without_clause,
                extracted_values=values_by_contract,
                coverage_percentage=pct,
            )

        return coverage

    def _build_gap_analysis(
        self,
        summaries: list[ContractSummary]
    ) -> GapAnalysis:
        """Build gap analysis for missing protections."""

        total = len(summaries)
        gap_by_contract = {}
        gap_counts = defaultdict(int)

        for summary in summaries:
            missing = [
                label for label in STANDARD_PROTECTIONS
                if label not in summary.labels_found
            ]
            gap_by_contract[summary.contract_id] = missing

            for label in missing:
                gap_counts[label] += 1

        # Sort by frequency
        gap_frequency = [
            (label, count, (count / total * 100) if total > 0 else 0)
            for label, count in sorted(
                gap_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # Build priority recommendations
        recommendations = []
        for label, count, pct in gap_frequency[:5]:
            if pct >= 50:
                rule = MISSING_CLAUSE_RISKS.get(label)
                recommendations.append({
                    "label": label,
                    "missing_percentage": round(pct, 1),
                    "priority": "HIGH" if pct >= 75 else "MEDIUM",
                    "recommendation": rule.recommendation if rule else f"Add {label} clause",
                    "affected_contracts": [
                        cid for cid, gaps in gap_by_contract.items()
                        if label in gaps
                    ],
                })

        return GapAnalysis(
            standard_protections=STANDARD_PROTECTIONS,
            gap_by_contract=gap_by_contract,
            gap_frequency=gap_frequency,
            priority_recommendations=recommendations,
        )


def analyze_portfolio(
    folder_path: str | Path,
    limit: int | None = None,
    use_llm: bool = True
) -> PortfolioAnalysis:
    """Convenience function to analyze a portfolio of contracts.

    Args:
        folder_path: Path to folder containing contract PDFs
        limit: Maximum number of contracts to analyze
        use_llm: Whether to use LLM for complex analyses

    Returns:
        PortfolioAnalysis with complete results
    """
    analyzer = PortfolioAnalyzer(use_llm=use_llm)
    return analyzer.analyze_folder(folder_path, limit=limit)
