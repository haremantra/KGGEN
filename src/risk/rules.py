"""Risk rules for CUAD contract clause categories.

Defines risk severity levels and assessment rules for known contract patterns.
"""

from enum import Enum
from dataclasses import dataclass


class RiskSeverity(str, Enum):
    """Risk severity levels."""
    CRITICAL = "CRITICAL"  # Requires immediate attention
    HIGH = "HIGH"          # Significant risk exposure
    MEDIUM = "MEDIUM"      # Moderate concern
    LOW = "LOW"            # Minor issue
    INFO = "INFO"          # Informational only


@dataclass
class RiskRule:
    """A risk assessment rule for a CUAD label."""
    severity: RiskSeverity
    reason: str
    recommendation: str
    requires_llm: bool = False  # True if clause needs LLM analysis


# Risk rules for CUAD categories
# Format: label -> (severity when PRESENT, severity when MISSING, rule details)
RISK_RULES = {
    # === GENERAL INFORMATION ===
    "Document Name": RiskRule(
        severity=RiskSeverity.INFO,
        reason="Document identification",
        recommendation="Ensure document name matches intended agreement type",
    ),
    "Parties": RiskRule(
        severity=RiskSeverity.INFO,
        reason="Party identification",
        recommendation="Verify all parties are correctly identified with proper legal names",
    ),
    "Agreement Date": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Date tracking",
        recommendation="Confirm date aligns with negotiation timeline",
    ),
    "Effective Date": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Contract timing impacts rights and obligations",
        recommendation="Verify effective date timing works for implementation needs",
    ),
    "Expiration Date": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Contract duration affects planning and budgeting",
        recommendation="Ensure term length aligns with business needs",
    ),
    "Renewal Term": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Auto-renewal may lock you into unfavorable terms",
        recommendation="Review renewal terms and set calendar reminders for termination deadlines",
    ),
    "Notice Period To Terminate Renewal": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Missing termination window leads to unwanted renewals",
        recommendation="Calendar the notice deadline well in advance",
    ),
    "Governing Law": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Jurisdiction affects enforceability and dispute costs",
        recommendation="Prefer home jurisdiction or neutral venue",
    ),
    "License Grant": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Core rights definition - must cover intended use",
        recommendation="Verify license scope covers all planned use cases",
        requires_llm=True,
    ),
    "Irrevocable Or Perpetual License": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Perpetual rights provide long-term security",
        recommendation="Confirm if perpetual license survives termination",
    ),

    # === RESTRICTIVE COVENANTS ===
    "Anti-Assignment": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Restricts M&A flexibility and corporate restructuring",
        recommendation="Negotiate carve-out for affiliates and change of control",
    ),
    "Non-Compete": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="May restrict business operations and growth",
        recommendation="Narrow scope, duration, and geographic limitations",
        requires_llm=True,
    ),
    "Non-Disparagement": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Standard provision with limited business impact",
        recommendation="Ensure carve-out for legally required disclosures",
    ),
    "No-Solicit Of Employees": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Limits hiring flexibility",
        recommendation="Negotiate reasonable duration (1-2 years) and scope",
    ),
    "No-Solicit Of Customers": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="May limit business development opportunities",
        recommendation="Clarify which customers are covered and for how long",
    ),
    "Exclusivity": RiskRule(
        severity=RiskSeverity.CRITICAL,
        reason="Locks out alternative vendors/partners",
        recommendation="Avoid exclusivity or limit scope/duration significantly",
        requires_llm=True,
    ),
    "Change Of Control": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="May trigger termination rights during M&A",
        recommendation="Negotiate for continuity through change of control events",
    ),
    "Covenant Not To Sue": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Waives legal remedies for potential claims",
        recommendation="Carefully scope what claims are waived",
        requires_llm=True,
    ),
    "Competitive Restriction Exception": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Provides flexibility in competitive restrictions",
        recommendation="Verify exceptions cover needed scenarios",
    ),
    "Non-Transferable License": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Limits license portability",
        recommendation="Request affiliate/subsidiary transfer rights",
    ),
    "Volume Restriction": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="May limit scalability",
        recommendation="Ensure volume limits accommodate growth projections",
    ),

    # === REVENUE RISKS ===
    "Cap On Liability": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Limits recourse for damages",
        recommendation="Negotiate cap at minimum 12 months fees or higher",
        requires_llm=True,
    ),
    "Uncapped Liability": RiskRule(
        severity=RiskSeverity.CRITICAL,
        reason="Unlimited financial exposure",
        recommendation="Never accept uncapped liability - negotiate a reasonable cap",
        requires_llm=True,  # LLM checks if mutual exclusion vs one-sided liability
    ),
    "Liquidated Damages": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Pre-set penalties may exceed actual damages",
        recommendation="Ensure amounts are proportional to potential harm",
        requires_llm=True,
    ),
    "Revenue/Profit Sharing": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Ongoing financial obligations based on performance",
        recommendation="Clearly define calculation methodology and audit rights",
        requires_llm=True,
    ),
    "Minimum Commitment": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Guaranteed payment regardless of usage",
        recommendation="Ensure minimums align with realistic usage projections",
    ),
    "Audit Rights": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Third party access to records creates operational burden",
        recommendation="Limit audit frequency and require reasonable notice",
    ),
    "Insurance": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Additional cost and compliance obligation",
        recommendation="Verify insurance requirements are commercially reasonable",
    ),
    "Warranty Duration": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Defines protection period for defects",
        recommendation="Seek warranty period of at least 12 months",
    ),
    "Post-Termination Services": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Ongoing obligations after relationship ends",
        recommendation="Ensure transition assistance is adequate for migration",
    ),
    "Termination For Convenience": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Allows counterparty to exit without cause",
        recommendation="Request mutual termination rights and adequate notice period",
    ),

    # === INTELLECTUAL PROPERTY ===
    "IP Ownership Assignment": RiskRule(
        severity=RiskSeverity.CRITICAL,
        reason="Transfers ownership of intellectual property",
        recommendation="Carefully scope what IP is assigned - retain rights to pre-existing IP",
        requires_llm=True,
    ),
    "Joint IP Ownership": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Shared ownership creates commercialization complexity",
        recommendation="Clearly define exploitation rights and revenue sharing",
        requires_llm=True,
    ),
    "Source Code Escrow": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Provides protection if vendor fails",
        recommendation="Verify escrow release conditions and update frequency",
    ),
    "Affiliate License-Licensor": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Licensor affiliate extension",
        recommendation="Verify affiliate scope definition",
    ),
    "Affiliate License-Licensee": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Extends license to your affiliates",
        recommendation="Ensure affiliate definition covers all subsidiaries",
    ),
    "Unlimited/All-You-Can-Eat-License": RiskRule(
        severity=RiskSeverity.LOW,
        reason="Provides usage flexibility without volume concerns",
        recommendation="Confirm no hidden usage restrictions",
    ),

    # === SPECIAL PROVISIONS ===
    "Third Party Beneficiary": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Creates rights for non-parties",
        recommendation="Review who has third party rights and limit if possible",
    ),
    "Most Favored Nation": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="Pricing/terms must match best offered to others",
        recommendation="As a customer, this is favorable - ensure enforcement mechanism",
    ),
    "Rofr/Rofo/Rofn": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Provides opportunity to match competing offers",
        recommendation="Verify timing and process for exercising rights",
        requires_llm=True,
    ),
}


# Missing clause risk definitions
MISSING_CLAUSE_RISKS = {
    "Cap On Liability": RiskRule(
        severity=RiskSeverity.CRITICAL,
        reason="No liability cap means unlimited exposure",
        recommendation="Negotiate a liability cap before signing",
    ),
    "Source Code Escrow": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="No access to source code if vendor fails (software contracts)",
        recommendation="Request source code escrow for critical software dependencies",
    ),
    "Termination For Convenience": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="No exit option without breach",
        recommendation="Negotiate mutual termination for convenience with notice period",
    ),
    "Notice Period To Terminate Renewal": RiskRule(
        severity=RiskSeverity.HIGH,
        reason="May be auto-locked into renewals",
        recommendation="Add notice period clause for terminating auto-renewals",
    ),
    "Governing Law": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="Jurisdiction uncertainty in disputes",
        recommendation="Specify governing law favorable to your location",
    ),
    "Warranty Duration": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="No defined warranty period",
        recommendation="Add explicit warranty period of at least 12 months",
    ),
    "Post-Termination Services": RiskRule(
        severity=RiskSeverity.MEDIUM,
        reason="No transition assistance guaranteed",
        recommendation="Add transition services clause for orderly migration",
    ),
    "Insurance": RiskRule(
        severity=RiskSeverity.LOW,
        reason="No insurance requirements for counterparty",
        recommendation="Consider requiring proof of insurance for critical vendors",
    ),
}


def get_rule_for_label(label: str) -> RiskRule | None:
    """Get the risk rule for a CUAD label."""
    return RISK_RULES.get(label)


def get_missing_clause_rule(label: str) -> RiskRule | None:
    """Get the risk rule for when a clause is missing."""
    return MISSING_CLAUSE_RISKS.get(label)


def get_high_risk_labels() -> list[str]:
    """Get labels that are HIGH or CRITICAL severity."""
    return [
        label for label, rule in RISK_RULES.items()
        if rule.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)
    ]


def get_labels_requiring_llm() -> list[str]:
    """Get labels that need LLM analysis for proper risk assessment."""
    return [
        label for label, rule in RISK_RULES.items()
        if rule.requires_llm
    ]
