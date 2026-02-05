"""Static dependency rules between CUAD clause label pairs.

73 rules encoding legal domain knowledge about how contract clauses
interact, conflict, require, mitigate, restrict, and modify each other.
"""

from dataclasses import dataclass

from .types import DependencyType


@dataclass
class DependencyRule:
    """A static rule defining a dependency between two CUAD labels."""
    source_label: str
    target_label: str
    dependency_type: DependencyType
    default_strength: float
    reason: str
    bidirectional: bool = False
    requires_llm_validation: bool = False


# === 73 Static Dependency Rules ===

DEPENDENCY_RULES: list[DependencyRule] = [
    # --- REQUIRES (22 rules) ---
    DependencyRule(
        "Renewal Term", "Notice Period To Terminate Renewal", DependencyType.REQUIRES, 0.95,
        "Renewal terms require a notice period to allow termination of auto-renewal",
    ),
    DependencyRule(
        "Renewal Term", "Expiration Date", DependencyType.REQUIRES, 0.9,
        "Renewal terms presuppose an initial expiration date to renew from",
    ),
    DependencyRule(
        "Revenue/Profit Sharing", "Audit Rights", DependencyType.REQUIRES, 0.9,
        "Revenue sharing arrangements require audit rights to verify calculations",
    ),
    DependencyRule(
        "Exclusivity", "Expiration Date", DependencyType.REQUIRES, 0.85,
        "Exclusive arrangements should have a defined duration",
    ),
    DependencyRule(
        "Non-Compete", "Expiration Date", DependencyType.REQUIRES, 0.85,
        "Non-compete restrictions should have a defined end date for enforceability",
    ),
    DependencyRule(
        "Non-Compete", "Governing Law", DependencyType.REQUIRES, 0.8,
        "Non-compete enforceability varies by jurisdiction; governing law is critical",
    ),
    DependencyRule(
        "Liquidated Damages", "Cap On Liability", DependencyType.REQUIRES, 0.75,
        "Liquidated damages should be bounded by a liability cap",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Termination For Convenience", "Notice Period To Terminate Renewal", DependencyType.REQUIRES, 0.7,
        "Termination for convenience typically requires a notice period",
    ),
    DependencyRule(
        "IP Ownership Assignment", "Parties", DependencyType.REQUIRES, 0.95,
        "IP assignment must clearly identify assignor and assignee parties",
    ),
    DependencyRule(
        "License Grant", "Parties", DependencyType.REQUIRES, 0.95,
        "License grant requires identified licensor and licensee",
    ),
    DependencyRule(
        "Change Of Control", "Anti-Assignment", DependencyType.REQUIRES, 0.7,
        "Change of control provisions should align with assignment restrictions",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Minimum Commitment", "Expiration Date", DependencyType.REQUIRES, 0.8,
        "Minimum commitments need a defined period over which they apply",
    ),
    DependencyRule(
        "Minimum Commitment", "Termination For Convenience", DependencyType.REQUIRES, 0.65,
        "Minimum commitments should address early termination scenarios",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Post-Termination Services", "Expiration Date", DependencyType.REQUIRES, 0.85,
        "Post-termination services presuppose a termination/expiration event",
    ),
    DependencyRule(
        "Warranty Duration", "Effective Date", DependencyType.REQUIRES, 0.8,
        "Warranty duration needs a start date reference point",
    ),
    DependencyRule(
        "No-Solicit Of Employees", "Expiration Date", DependencyType.REQUIRES, 0.75,
        "Employee non-solicitation should have a defined duration",
    ),
    DependencyRule(
        "No-Solicit Of Customers", "Expiration Date", DependencyType.REQUIRES, 0.75,
        "Customer non-solicitation should have a defined duration",
    ),
    DependencyRule(
        "Joint IP Ownership", "Parties", DependencyType.REQUIRES, 0.9,
        "Joint IP ownership must identify all co-owners",
    ),
    DependencyRule(
        "Source Code Escrow", "License Grant", DependencyType.REQUIRES, 0.85,
        "Source code escrow provisions require an underlying software license",
    ),
    DependencyRule(
        "Most Favored Nation", "Audit Rights", DependencyType.REQUIRES, 0.7,
        "MFN clauses need audit/verification mechanism to ensure compliance",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Insurance", "Cap On Liability", DependencyType.REQUIRES, 0.6,
        "Insurance requirements should align with liability cap structure",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Rofr/Rofo/Rofn", "Notice Period To Terminate Renewal", DependencyType.REQUIRES, 0.65,
        "Right of first refusal requires a notice mechanism to exercise the right",
    ),

    # --- CONFLICTS_WITH (8 rules) ---
    DependencyRule(
        "Cap On Liability", "Uncapped Liability", DependencyType.CONFLICTS_WITH, 0.9,
        "A liability cap directly contradicts uncapped liability provisions",
        bidirectional=True,
        requires_llm_validation=True,
    ),
    DependencyRule(
        "License Grant", "IP Ownership Assignment", DependencyType.CONFLICTS_WITH, 0.7,
        "Licensing IP and assigning it to the same party may conflict",
        bidirectional=True,
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Exclusivity", "Non-Transferable License", DependencyType.CONFLICTS_WITH, 0.6,
        "Exclusive rights may conflict with non-transferability if sublicensing is needed",
        bidirectional=True,
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Volume Restriction", "Unlimited/All-You-Can-Eat-License", DependencyType.CONFLICTS_WITH, 0.95,
        "Volume restrictions directly contradict unlimited license grants",
        bidirectional=True,
    ),
    DependencyRule(
        "Termination For Convenience", "Irrevocable Or Perpetual License", DependencyType.CONFLICTS_WITH, 0.75,
        "Termination for convenience may conflict with irrevocable/perpetual rights",
        bidirectional=True,
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Non-Compete", "Competitive Restriction Exception", DependencyType.CONFLICTS_WITH, 0.6,
        "Broad non-compete may conflict with competitive restriction exceptions",
        bidirectional=True,
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Anti-Assignment", "Affiliate License-Licensee", DependencyType.CONFLICTS_WITH, 0.5,
        "Anti-assignment restrictions may conflict with affiliate license extensions",
        bidirectional=True,
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Exclusivity", "Affiliate License-Licensor", DependencyType.CONFLICTS_WITH, 0.55,
        "Exclusivity granted to licensee may conflict with licensor's affiliate licensing",
        bidirectional=True,
        requires_llm_validation=True,
    ),

    # --- MITIGATES (9 rules) ---
    DependencyRule(
        "Insurance", "Uncapped Liability", DependencyType.MITIGATES, 0.8,
        "Insurance coverage helps mitigate exposure from uncapped liability",
    ),
    DependencyRule(
        "Cap On Liability", "Liquidated Damages", DependencyType.MITIGATES, 0.75,
        "Liability cap limits total exposure from liquidated damages",
    ),
    DependencyRule(
        "Irrevocable Or Perpetual License", "Termination For Convenience", DependencyType.MITIGATES, 0.7,
        "Perpetual license survives termination, mitigating convenience termination risk",
    ),
    DependencyRule(
        "Source Code Escrow", "Termination For Convenience", DependencyType.MITIGATES, 0.75,
        "Escrow ensures continued access to source code after termination",
    ),
    DependencyRule(
        "Post-Termination Services", "Termination For Convenience", DependencyType.MITIGATES, 0.7,
        "Post-termination services soften the impact of convenience termination",
    ),
    DependencyRule(
        "Competitive Restriction Exception", "Non-Compete", DependencyType.MITIGATES, 0.65,
        "Exceptions to competitive restrictions reduce the burden of non-compete",
    ),
    DependencyRule(
        "Warranty Duration", "Uncapped Liability", DependencyType.MITIGATES, 0.5,
        "Defined warranty period limits the timeframe for uncapped warranty claims",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Audit Rights", "Revenue/Profit Sharing", DependencyType.MITIGATES, 0.7,
        "Audit rights mitigate risk of revenue misreporting in sharing arrangements",
    ),
    DependencyRule(
        "Insurance", "Liquidated Damages", DependencyType.MITIGATES, 0.65,
        "Insurance coverage can offset liquidated damages exposure",
    ),

    # --- RESTRICTS (10 rules) ---
    DependencyRule(
        "Anti-Assignment", "License Grant", DependencyType.RESTRICTS, 0.8,
        "Anti-assignment limits the transferability of the license grant",
    ),
    DependencyRule(
        "Governing Law", "Non-Compete", DependencyType.RESTRICTS, 0.7,
        "Governing law jurisdiction may restrict non-compete enforceability",
    ),
    DependencyRule(
        "Volume Restriction", "License Grant", DependencyType.RESTRICTS, 0.85,
        "Volume restrictions narrow the scope of the license grant",
    ),
    DependencyRule(
        "Non-Transferable License", "License Grant", DependencyType.RESTRICTS, 0.9,
        "Non-transferability restricts how the license can be shared or assigned",
    ),
    DependencyRule(
        "Non-Compete", "License Grant", DependencyType.RESTRICTS, 0.6,
        "Non-compete may restrict how licensed technology can be used commercially",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Anti-Assignment", "Change Of Control", DependencyType.RESTRICTS, 0.75,
        "Anti-assignment provisions restrict change of control scenarios",
    ),
    DependencyRule(
        "Governing Law", "Exclusivity", DependencyType.RESTRICTS, 0.55,
        "Antitrust laws in the governing jurisdiction may restrict exclusivity",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Cap On Liability", "Revenue/Profit Sharing", DependencyType.RESTRICTS, 0.5,
        "Liability cap may restrict remedies for revenue sharing disputes",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Covenant Not To Sue", "Uncapped Liability", DependencyType.RESTRICTS, 0.7,
        "Covenant not to sue restricts the ability to pursue uncapped liability claims",
    ),
    DependencyRule(
        "Expiration Date", "Non-Compete", DependencyType.RESTRICTS, 0.6,
        "Contract expiration may limit the enforceable duration of non-compete",
        requires_llm_validation=True,
    ),

    # --- MODIFIES (12 rules) ---
    DependencyRule(
        "Liquidated Damages", "Cap On Liability", DependencyType.MODIFIES, 0.8,
        "Liquidated damages provisions may adjust how the liability cap is calculated",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Renewal Term", "Expiration Date", DependencyType.MODIFIES, 0.9,
        "Renewal extends the effective expiration date of the contract",
    ),
    DependencyRule(
        "Affiliate License-Licensee", "License Grant", DependencyType.MODIFIES, 0.75,
        "Affiliate licensing expands the scope of the original license grant",
    ),
    DependencyRule(
        "Affiliate License-Licensor", "License Grant", DependencyType.MODIFIES, 0.7,
        "Licensor affiliate licensing may modify the exclusivity of the license",
    ),
    DependencyRule(
        "Change Of Control", "Renewal Term", DependencyType.MODIFIES, 0.65,
        "Change of control may trigger or prevent automatic renewal",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Most Favored Nation", "Revenue/Profit Sharing", DependencyType.MODIFIES, 0.7,
        "MFN clause may modify revenue sharing terms to match better deals",
    ),
    DependencyRule(
        "Most Favored Nation", "Minimum Commitment", DependencyType.MODIFIES, 0.65,
        "MFN clause may adjust minimum commitment based on terms offered to others",
    ),
    DependencyRule(
        "Competitive Restriction Exception", "Exclusivity", DependencyType.MODIFIES, 0.6,
        "Competitive exceptions carve out portions of the exclusivity grant",
    ),
    DependencyRule(
        "Post-Termination Services", "Expiration Date", DependencyType.MODIFIES, 0.7,
        "Post-termination obligations effectively extend certain duties past expiration",
    ),
    DependencyRule(
        "Third Party Beneficiary", "License Grant", DependencyType.MODIFIES, 0.55,
        "Third party beneficiary rights may extend license benefits beyond named parties",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Rofr/Rofo/Rofn", "Anti-Assignment", DependencyType.MODIFIES, 0.6,
        "Right of first refusal modifies the assignment restriction with a pre-emptive right",
    ),
    DependencyRule(
        "Change Of Control", "Termination For Convenience", DependencyType.MODIFIES, 0.7,
        "Change of control may trigger or modify termination rights",
        requires_llm_validation=True,
    ),

    # --- DEPENDS_ON (12 rules) ---
    DependencyRule(
        "Non-Compete", "No-Solicit Of Employees", DependencyType.DEPENDS_ON, 0.7,
        "Non-compete provisions are often paired with and reinforce employee non-solicitation",
    ),
    DependencyRule(
        "Non-Compete", "No-Solicit Of Customers", DependencyType.DEPENDS_ON, 0.7,
        "Non-compete provisions are often paired with customer non-solicitation",
    ),
    DependencyRule(
        "No-Solicit Of Employees", "No-Solicit Of Customers", DependencyType.DEPENDS_ON, 0.6,
        "Employee and customer non-solicitation provisions typically appear together",
        bidirectional=True,
    ),
    DependencyRule(
        "License Grant", "Parties", DependencyType.DEPENDS_ON, 0.95,
        "License grant depends on clearly identified parties",
    ),
    DependencyRule(
        "Expiration Date", "Effective Date", DependencyType.DEPENDS_ON, 0.9,
        "Expiration date is computed relative to the effective date",
    ),
    DependencyRule(
        "Renewal Term", "Agreement Date", DependencyType.DEPENDS_ON, 0.6,
        "Renewal term timing may reference the original agreement date",
    ),
    DependencyRule(
        "Cap On Liability", "License Grant", DependencyType.DEPENDS_ON, 0.5,
        "Liability cap scope often depends on the licensed rights granted",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Warranty Duration", "Agreement Date", DependencyType.DEPENDS_ON, 0.65,
        "Warranty duration often starts from agreement or delivery date",
    ),
    DependencyRule(
        "Exclusivity", "License Grant", DependencyType.DEPENDS_ON, 0.85,
        "Exclusivity terms define the exclusive scope of the license grant",
    ),
    DependencyRule(
        "IP Ownership Assignment", "License Grant", DependencyType.DEPENDS_ON, 0.6,
        "IP assignment and licensing terms must be consistent",
        requires_llm_validation=True,
    ),
    DependencyRule(
        "Non-Disparagement", "Governing Law", DependencyType.DEPENDS_ON, 0.5,
        "Non-disparagement enforceability depends on governing jurisdiction",
    ),
    DependencyRule(
        "Minimum Commitment", "Revenue/Profit Sharing", DependencyType.DEPENDS_ON, 0.6,
        "Minimum commitment amounts often relate to revenue sharing thresholds",
        requires_llm_validation=True,
    ),
]


# === Helper Functions ===

def get_rules_for_pair(source: str, target: str) -> list[DependencyRule]:
    """Get all rules between a specific source and target label pair."""
    rules = []
    for rule in DEPENDENCY_RULES:
        if rule.source_label == source and rule.target_label == target:
            rules.append(rule)
        elif rule.bidirectional and rule.source_label == target and rule.target_label == source:
            rules.append(rule)
    return rules


def get_rules_for_label(label: str) -> list[DependencyRule]:
    """Get all rules where a label appears as source or target."""
    return [
        rule for rule in DEPENDENCY_RULES
        if rule.source_label == label or rule.target_label == label
    ]


def get_all_conflict_pairs() -> list[tuple[str, str]]:
    """Get all label pairs that have CONFLICTS_WITH rules."""
    return [
        (rule.source_label, rule.target_label)
        for rule in DEPENDENCY_RULES
        if rule.dependency_type == DependencyType.CONFLICTS_WITH
    ]


def get_requires_rules() -> list[DependencyRule]:
    """Get all REQUIRES rules (useful for completeness checking)."""
    return [
        rule for rule in DEPENDENCY_RULES
        if rule.dependency_type == DependencyType.REQUIRES
    ]


def get_rules_requiring_llm() -> list[DependencyRule]:
    """Get all rules that require LLM validation for accurate detection."""
    return [
        rule for rule in DEPENDENCY_RULES
        if rule.requires_llm_validation
    ]
