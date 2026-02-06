"""Hybrid risk assessment engine.

Combines rule-based scoring for known patterns with LLM assessment
for complex or ambiguous clauses.
"""

import re
import json
from dataclasses import dataclass, field
from anthropic import Anthropic


# Patterns indicating mutual/bilateral language in liability clauses
MUTUAL_LIABILITY_PATTERNS = [
    r'\beither\s+party\b',
    r'\bneither\s+party\b',
    r'\bboth\s+parties\b',
    r'\beach\s+party\b',
    r'\bno\s+party\b',
    r'\bthe\s+parties\b',
    r'\bmutual(?:ly)?\b',
]


def _has_mutual_language(text: str) -> bool:
    """Check if clause text contains mutual/bilateral language."""
    if not text:
        return False
    text_lower = text.lower()
    for pattern in MUTUAL_LIABILITY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

from .rules import (
    RiskSeverity,
    RiskRule,
    RISK_RULES,
    MISSING_CLAUSE_RISKS,
    get_rule_for_label,
    get_missing_clause_rule,
    get_labels_requiring_llm,
)
from ..config import settings
from ..pipeline import ContractAnalysis, AnalyzedClause


@dataclass
class RiskFinding:
    """A specific risk finding from the assessment."""
    label: str
    severity: RiskSeverity
    reason: str
    recommendation: str
    clause_text: str | None = None
    confidence: float = 1.0
    source: str = "rule"  # "rule" or "llm"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "severity": self.severity.value,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "clause_text": self.clause_text[:200] + "..." if self.clause_text and len(self.clause_text) > 200 else self.clause_text,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment for a contract."""
    contract_id: str
    overall_risk_score: int  # 0-100, higher = more risk
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    findings: list[RiskFinding] = field(default_factory=list)
    missing_clause_risks: list[RiskFinding] = field(default_factory=list)
    llm_analysis: str | None = None
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level,
            "findings": [f.to_dict() for f in self.findings],
            "missing_clause_risks": [f.to_dict() for f in self.missing_clause_risks],
            "llm_analysis": self.llm_analysis,
            "summary": self.summary,
        }


# Severity weights for risk score calculation
SEVERITY_WEIGHTS = {
    RiskSeverity.CRITICAL: 25,
    RiskSeverity.HIGH: 15,
    RiskSeverity.MEDIUM: 8,
    RiskSeverity.LOW: 3,
    RiskSeverity.INFO: 0,
}


class RiskAssessor:
    """Hybrid risk assessor combining rules and LLM analysis."""

    def __init__(self, use_llm: bool = True):
        """Initialize the risk assessor.

        Args:
            use_llm: Whether to use LLM for complex clause analysis
        """
        self.use_llm = use_llm
        if use_llm:
            self.client = Anthropic(api_key=settings.anthropic_api_key)

    def assess(self, analysis: ContractAnalysis) -> RiskAssessment:
        """Assess risks in a contract analysis.

        Args:
            analysis: ContractAnalysis from the pipeline

        Returns:
            RiskAssessment with findings and score
        """
        findings = []

        # Get labels present in the contract
        present_labels = set()

        # Step 1: Rule-based assessment of present clauses
        for clause in analysis.analyzed_clauses:
            label = clause.cuad_label
            present_labels.add(label)

            rule = get_rule_for_label(label)
            if not rule:
                continue

            # Check for mutual language in liability-related clauses
            severity = rule.severity
            reason = rule.reason
            liability_labels = {"Uncapped Liability", "Cap On Liability", "Liquidated Damages"}

            if label in liability_labels and _has_mutual_language(clause.text):
                # Mutual exclusion of damages is standard - downgrade severity
                if label == "Uncapped Liability":
                    severity = RiskSeverity.INFO
                    reason = "Mutual exclusion of consequential damages (standard provision)"
                elif label == "Liquidated Damages":
                    severity = RiskSeverity.MEDIUM
                    reason = "Mutual liquidated damages provision"

            # Add finding based on rule
            finding = RiskFinding(
                label=label,
                severity=severity,
                reason=reason,
                recommendation=rule.recommendation,
                clause_text=clause.text,
                confidence=clause.label_confidence,
                source="rule",
            )
            findings.append(finding)

            # Check if this clause needs LLM analysis
            if self.use_llm and rule.requires_llm and clause.label_confidence >= 0.5:
                llm_finding = self._llm_assess_clause(clause, rule)
                if llm_finding:
                    # Replace or augment the rule finding
                    findings[-1] = llm_finding

        # Step 2: Identify missing critical clauses
        missing_risks = self._assess_missing_clauses(present_labels)

        # Step 3: Calculate overall risk score
        risk_score = self._calculate_risk_score(findings, missing_risks)
        risk_level = self._score_to_level(risk_score)

        # Step 4: Optional LLM summary for high-risk contracts
        llm_analysis = None
        if self.use_llm and risk_score >= 50:
            llm_analysis = self._generate_llm_summary(analysis, findings, missing_risks)

        # Build summary
        summary = self._build_summary(findings, missing_risks, risk_score)

        return RiskAssessment(
            contract_id=analysis.contract_id,
            overall_risk_score=risk_score,
            risk_level=risk_level,
            findings=findings,
            missing_clause_risks=missing_risks,
            llm_analysis=llm_analysis,
            summary=summary,
        )

    def _llm_assess_clause(self, clause: AnalyzedClause, rule: RiskRule) -> RiskFinding | None:
        """Use LLM to assess a complex clause."""
        prompt = f"""Analyze this contract clause for legal risk.

CLAUSE TYPE: {clause.cuad_label}
CLAUSE TEXT:
{clause.text}

KNOWN RISK CONTEXT:
- Default severity: {rule.severity.value}
- Typical concern: {rule.reason}

IMPORTANT CONSIDERATIONS:
- Check if language is MUTUAL (applies to both parties equally) vs ONE-SIDED
- Mutual exclusion of consequential damages is standard practice (lower risk)
- One-sided liability limitations or uncapped indemnification are high risk
- Look for phrases like "either party", "neither party", "each party" indicating mutuality

Provide a risk assessment:
1. Is this mutual or one-sided? (affects severity)
2. Actual severity for THIS specific clause (CRITICAL/HIGH/MEDIUM/LOW/INFO)
3. Specific risks identified in this language
4. Recommended negotiation points

Return JSON:
{{
    "severity": "HIGH",
    "is_mutual": true,
    "specific_reason": "why this particular clause is risky",
    "recommendation": "specific negotiation advice",
    "confidence": 0.85
}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text

            # Parse JSON
            if "```json" in result_text:
                start = result_text.find("```json") + 7
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()
            elif "```" in result_text:
                start = result_text.find("```") + 3
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()

            data = json.loads(result_text)

            return RiskFinding(
                label=clause.cuad_label,
                severity=RiskSeverity(data.get("severity", rule.severity.value)),
                reason=data.get("specific_reason", rule.reason),
                recommendation=data.get("recommendation", rule.recommendation),
                clause_text=clause.text,
                confidence=data.get("confidence", 0.8),
                source="llm",
            )

        except Exception as e:
            print(f"LLM assessment failed for {clause.cuad_label}: {e}")
            return None

    def _assess_missing_clauses(self, present_labels: set[str]) -> list[RiskFinding]:
        """Identify risks from missing important clauses."""
        missing_risks = []

        for label, rule in MISSING_CLAUSE_RISKS.items():
            if label not in present_labels:
                missing_risks.append(RiskFinding(
                    label=label,
                    severity=rule.severity,
                    reason=f"MISSING: {rule.reason}",
                    recommendation=rule.recommendation,
                    clause_text=None,
                    confidence=1.0,
                    source="rule",
                ))

        return missing_risks

    def _calculate_risk_score(
        self,
        findings: list[RiskFinding],
        missing_risks: list[RiskFinding]
    ) -> int:
        """Calculate overall risk score (0-100)."""
        total_weight = 0

        # Weight present clause risks
        for finding in findings:
            weight = SEVERITY_WEIGHTS.get(finding.severity, 0)
            total_weight += weight * finding.confidence

        # Weight missing clause risks (slightly lower impact)
        for finding in missing_risks:
            weight = SEVERITY_WEIGHTS.get(finding.severity, 0) * 0.75
            total_weight += weight

        # Normalize to 0-100 (cap at 100)
        # Assume max reasonable weight is ~150 for a very risky contract
        score = min(100, int((total_weight / 150) * 100))

        return score

    def _score_to_level(self, score: int) -> str:
        """Convert numeric score to risk level."""
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_llm_summary(
        self,
        analysis: ContractAnalysis,
        findings: list[RiskFinding],
        missing_risks: list[RiskFinding]
    ) -> str:
        """Generate LLM summary for high-risk contracts."""

        # Build context
        high_severity = [f for f in findings if f.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)]
        missing_high = [f for f in missing_risks if f.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)]

        prompt = f"""As a legal analyst, summarize the key risks in this contract.

CONTRACT: {analysis.contract_id}

HIGH-SEVERITY FINDINGS:
{json.dumps([f.to_dict() for f in high_severity[:5]], indent=2)}

MISSING CRITICAL CLAUSES:
{json.dumps([f.to_dict() for f in missing_high], indent=2)}

KEY EXTRACTED TERMS:
{json.dumps(analysis.summary.get('key_findings', [])[:10], indent=2)}

Provide a 2-3 sentence executive summary of:
1. The most critical risk(s) in this contract
2. Priority items to negotiate before signing

Be specific and actionable. Focus on business impact."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"LLM summary generation failed: {e}")
            return None

    def _build_summary(
        self,
        findings: list[RiskFinding],
        missing_risks: list[RiskFinding],
        risk_score: int
    ) -> str:
        """Build a text summary of the risk assessment."""

        critical_count = sum(1 for f in findings if f.severity == RiskSeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == RiskSeverity.HIGH)
        missing_critical = sum(1 for f in missing_risks if f.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH))

        parts = [f"Risk Score: {risk_score}/100"]

        if critical_count > 0:
            parts.append(f"{critical_count} CRITICAL issue(s)")
        if high_count > 0:
            parts.append(f"{high_count} HIGH risk issue(s)")
        if missing_critical > 0:
            parts.append(f"{missing_critical} missing protection(s)")

        return " | ".join(parts)


def assess_contract(analysis: ContractAnalysis, use_llm: bool = True) -> RiskAssessment:
    """Convenience function to assess a contract analysis.

    Args:
        analysis: ContractAnalysis from the pipeline
        use_llm: Whether to use LLM for complex clauses

    Returns:
        RiskAssessment with findings and score
    """
    assessor = RiskAssessor(use_llm=use_llm)
    return assessor.assess(analysis)
