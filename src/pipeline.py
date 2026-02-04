"""Integrated Contract Analysis Pipeline.

Combines clause classification with knowledge graph extraction:
1. Classify clauses → identify CUAD categories present
2. Extract structured data → entities, relationships, values from high-confidence matches
3. Assess risks → rule-based + LLM hybrid risk scoring
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from anthropic import Anthropic

from .classification.classifier import ClauseClassifier, ClassifiedClause, ClauseLabel
from .classification.cuad_labels import CUAD_LABELS
from .config import settings


@dataclass
class ExtractedValue:
    """A specific value extracted from a clause."""
    field: str
    value: str
    confidence: float


@dataclass
class AnalyzedClause:
    """A clause with classification and extracted data."""
    text: str
    cuad_label: str
    label_confidence: float
    category: str
    extracted_values: list[ExtractedValue] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)


@dataclass
class ContractAnalysis:
    """Complete analysis of a contract."""
    contract_id: str
    total_clauses: int
    analyzed_clauses: list[AnalyzedClause]
    summary: dict


# Extraction prompts for each CUAD category
EXTRACTION_PROMPTS = {
    "Parties": "Extract: party names, roles (licensor/licensee/vendor/customer), entity types (corporation/LLC/individual)",
    "Agreement Date": "Extract: the specific date the agreement was signed or executed",
    "Effective Date": "Extract: the specific date the contract becomes effective",
    "Expiration Date": "Extract: the specific expiration/end date or term length",
    "Renewal Term": "Extract: renewal period length, automatic vs manual renewal, number of renewals allowed",
    "Notice Period To Terminate Renewal": "Extract: number of days notice required, who must give notice",
    "Governing Law": "Extract: state/country governing the contract, jurisdiction for disputes",
    "License Grant": "Extract: type (exclusive/non-exclusive), scope, permitted uses, restrictions",
    "Irrevocable Or Perpetual License": "Extract: whether license is perpetual/irrevocable, any conditions",
    "Anti-Assignment": "Extract: assignment restrictions, consent requirements, exceptions",
    "Non-Compete": "Extract: duration, geographic scope, restricted activities",
    "Non-Disparagement": "Extract: scope of restriction, duration, consequences",
    "No-Solicit Of Employees": "Extract: duration, which employees covered, exceptions",
    "No-Solicit Of Customers": "Extract: duration, which customers covered, exceptions",
    "Exclusivity": "Extract: exclusive rights granted, scope, duration, territory",
    "Change Of Control": "Extract: what triggers change of control, consequences, notification requirements",
    "Cap On Liability": "Extract: liability cap amount/formula, what's covered, exceptions to cap",
    "Uncapped Liability": "Extract: what liabilities are uncapped, conditions",
    "Liquidated Damages": "Extract: damage amounts, triggering events, calculation method",
    "Revenue/Profit Sharing": "Extract: percentage splits, calculation method, payment timing",
    "Minimum Commitment": "Extract: minimum amounts, time periods, consequences of not meeting",
    "Audit Rights": "Extract: who can audit, frequency, notice required, scope",
    "Insurance": "Extract: coverage types required, minimum amounts, certificate requirements",
    "Warranty Duration": "Extract: warranty period length, what's covered, remedies",
    "Post-Termination Services": "Extract: services required, duration, pricing",
    "Termination For Convenience": "Extract: notice period, any penalties, conditions",
    "IP Ownership Assignment": "Extract: what IP is assigned, to whom, when assignment occurs",
    "Joint IP Ownership": "Extract: how joint ownership works, usage rights, commercialization",
    "Source Code Escrow": "Extract: escrow agent, release conditions, update requirements",
    "Third Party Beneficiary": "Extract: who the beneficiaries are, what rights they have",
    "Most Favored Nation": "Extract: what terms are covered, comparison mechanism",
    "Rofr/Rofo/Rofn": "Extract: type of right, what it applies to, exercise period",
}


class ContractAnalysisPipeline:
    """Integrated pipeline for contract analysis."""

    def __init__(
        self,
        classification_threshold: float = 0.40,
        extraction_threshold: float = 0.45,
    ):
        """Initialize the pipeline.

        Args:
            classification_threshold: Min confidence to include a classification
            extraction_threshold: Min confidence to run extraction on a clause
        """
        self.classification_threshold = classification_threshold
        self.extraction_threshold = extraction_threshold
        self.classifier = ClauseClassifier(use_qdrant=False)
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self._initialized = False

    def initialize(self):
        """Initialize the classifier."""
        if not self._initialized:
            print("Initializing classifier...")
            self.classifier.initialize()
            self._initialized = True

    def analyze(self, contract_text: str, contract_id: str = "unknown") -> ContractAnalysis:
        """Run full analysis pipeline on a contract.

        Args:
            contract_text: Full contract text
            contract_id: Identifier for the contract

        Returns:
            ContractAnalysis with classifications and extractions
        """
        self.initialize()

        # Step 1: Classify all clauses
        print("Step 1: Classifying clauses...")
        classified = self.classifier.classify_contract(
            contract_text,
            top_k=3,
            threshold=self.classification_threshold
        )
        print(f"  Found {len(classified)} clauses with classifications")

        # Step 2: Extract structured data from high-confidence matches
        print("Step 2: Extracting structured data from high-confidence clauses...")
        analyzed_clauses = []

        # Group by best label to avoid duplicate extractions
        seen_labels = {}
        for clause in classified:
            if not clause.labels:
                continue

            best_label = clause.labels[0]

            # Only process if above extraction threshold
            if best_label.confidence < self.extraction_threshold:
                continue

            # Keep best match per label
            if best_label.label not in seen_labels or best_label.confidence > seen_labels[best_label.label][0]:
                seen_labels[best_label.label] = (best_label.confidence, clause, best_label)

        # Extract from best matches
        extraction_count = 0
        for label_name, (conf, clause, label_info) in seen_labels.items():
            print(f"  Extracting from [{label_name}] ({conf:.1%})...")

            extracted = self._extract_from_clause(
                clause.text,
                label_name,
                label_info.description
            )

            analyzed_clauses.append(AnalyzedClause(
                text=clause.text[:500] + "..." if len(clause.text) > 500 else clause.text,
                cuad_label=label_name,
                label_confidence=conf,
                category=label_info.category,
                extracted_values=extracted.get("values", []),
                entities=extracted.get("entities", []),
                relationships=extracted.get("relationships", []),
            ))
            extraction_count += 1

        print(f"  Extracted from {extraction_count} clauses")

        # Step 3: Build summary
        summary = self._build_summary(analyzed_clauses)

        return ContractAnalysis(
            contract_id=contract_id,
            total_clauses=len(classified),
            analyzed_clauses=analyzed_clauses,
            summary=summary,
        )

    def _extract_from_clause(self, clause_text: str, label: str, description: str) -> dict:
        """Extract structured data from a specific clause."""

        extraction_prompt = EXTRACTION_PROMPTS.get(
            label,
            f"Extract: key terms, values, entities, and obligations related to {label}"
        )

        prompt = f"""Analyze this contract clause labeled as "{label}" ({description}).

CLAUSE TEXT:
{clause_text}

TASK: {extraction_prompt}

Return a JSON object with:
{{
  "values": [
    {{"field": "field_name", "value": "extracted_value", "confidence": 0.0-1.0}}
  ],
  "entities": ["entity1", "entity2"],
  "relationships": [
    {{"subject": "entity1", "predicate": "relationship", "object": "entity2"}}
  ]
}}

Be precise. Only extract what is explicitly stated. Use confidence scores to indicate certainty."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text

            # Parse JSON from response
            if "```json" in result_text:
                start = result_text.find("```json") + 7
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()
            elif "```" in result_text:
                start = result_text.find("```") + 3
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()

            data = json.loads(result_text)

            # Convert to ExtractedValue objects
            values = [
                ExtractedValue(
                    field=v.get("field", ""),
                    value=v.get("value", ""),
                    confidence=v.get("confidence", 0.5)
                )
                for v in data.get("values", [])
            ]

            return {
                "values": values,
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
            }

        except Exception as e:
            print(f"    Extraction error: {e}")
            return {"values": [], "entities": [], "relationships": []}

    def _build_summary(self, analyzed_clauses: list[AnalyzedClause]) -> dict:
        """Build a summary of the analysis."""

        by_category = {}
        key_findings = []

        for clause in analyzed_clauses:
            # Group by category
            if clause.category not in by_category:
                by_category[clause.category] = []
            by_category[clause.category].append({
                "label": clause.cuad_label,
                "confidence": clause.label_confidence,
                "extracted_count": len(clause.extracted_values),
            })

            # Collect key findings (high confidence extractions)
            for val in clause.extracted_values:
                if val.confidence >= 0.7:
                    key_findings.append({
                        "label": clause.cuad_label,
                        "field": val.field,
                        "value": val.value,
                    })

        return {
            "labels_found": len(analyzed_clauses),
            "by_category": by_category,
            "key_findings": key_findings[:20],  # Top 20
        }


def analyze_contract_file(
    file_path: str,
    output_path: str = None,
    include_risk: bool = True,
    use_llm_risk: bool = True,
) -> tuple["ContractAnalysis", "RiskAssessment | None"]:
    """Analyze a contract file and optionally save results.

    Args:
        file_path: Path to contract (PDF or text)
        output_path: Optional path to save JSON results
        include_risk: Whether to run risk assessment
        use_llm_risk: Whether to use LLM for complex risk analysis

    Returns:
        Tuple of (ContractAnalysis, RiskAssessment or None)
    """
    from .utils.pdf_reader import extract_text_from_pdf

    path = Path(file_path)

    # Read file
    if path.suffix.lower() == '.pdf':
        contract_text = extract_text_from_pdf(path)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            contract_text = f.read()

    # Run analysis
    pipeline = ContractAnalysisPipeline()
    result = pipeline.analyze(contract_text, contract_id=path.stem)

    # Run risk assessment
    risk_assessment = None
    if include_risk:
        from .risk.assessor import RiskAssessor
        assessor = RiskAssessor(use_llm=use_llm_risk)
        risk_assessment = assessor.assess(result)
        print(f"\nRisk Assessment: {risk_assessment.overall_risk_score}/100 ({risk_assessment.risk_level})")
        print(f"  {risk_assessment.summary}")

    # Save if requested
    if output_path:
        output_data = {
            "contract_id": result.contract_id,
            "total_clauses": result.total_clauses,
            "summary": result.summary,
            "analyzed_clauses": [
                {
                    "cuad_label": c.cuad_label,
                    "confidence": c.label_confidence,
                    "category": c.category,
                    "text_preview": c.text[:200],
                    "extracted_values": [
                        {"field": v.field, "value": v.value, "confidence": v.confidence}
                        for v in c.extracted_values
                    ],
                    "entities": c.entities,
                    "relationships": c.relationships,
                }
                for c in result.analyzed_clauses
            ],
        }

        # Add risk assessment to output
        if risk_assessment:
            output_data["risk_assessment"] = risk_assessment.to_dict()

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_path}")

    return result, risk_assessment
