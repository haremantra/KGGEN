"""CUAD label definitions with descriptions and example patterns."""

CUAD_LABELS = {
    # General Information
    "Document Name": {
        "description": "Name of the contract document",
        "category": "general_information",
        "patterns": [
            "This Agreement",
            "This License Agreement",
            "This Master Services Agreement",
            "This Software License Agreement",
        ]
    },
    "Parties": {
        "description": "Names of parties to the contract, including their legal entity type and role",
        "category": "general_information",
        "patterns": [
            "between Company A and Company B",
            "by and between",
            "the parties hereto",
            "Licensor and Licensee",
        ]
    },
    "Agreement Date": {
        "description": "Date when agreement was signed or executed",
        "category": "general_information",
        "patterns": [
            "dated as of",
            "entered into as of",
            "executed on",
            "effective as of the date first written above",
        ]
    },
    "Effective Date": {
        "description": "Date when contract becomes effective and binding",
        "category": "general_information",
        "patterns": [
            "effective date",
            "shall become effective",
            "commencing on",
            "takes effect on",
        ]
    },
    "Expiration Date": {
        "description": "Date when contract expires or terminates",
        "category": "general_information",
        "patterns": [
            "shall expire on",
            "termination date",
            "until",
            "for a period of",
            "initial term",
        ]
    },
    "Renewal Term": {
        "description": "Automatic or optional renewal term after initial term expires",
        "category": "general_information",
        "patterns": [
            "shall automatically renew",
            "successive renewal terms",
            "renewal period",
            "extend for additional",
        ]
    },
    "Notice Period To Terminate Renewal": {
        "description": "Notice period required to terminate automatic renewal",
        "category": "general_information",
        "patterns": [
            "days prior written notice",
            "notice of non-renewal",
            "terminate upon notice",
            "written notice of termination",
        ]
    },
    "Governing Law": {
        "description": "State or country's law governing contract interpretation and disputes",
        "category": "general_information",
        "patterns": [
            "governed by the laws of",
            "construed in accordance with",
            "subject to the laws of",
            "jurisdiction of",
        ]
    },
    "License Grant": {
        "description": "Grant of license rights from licensor to licensee",
        "category": "general_information",
        "patterns": [
            "hereby grants",
            "license to use",
            "right and license",
            "non-exclusive license",
            "exclusive license",
        ]
    },
    "Irrevocable Or Perpetual License": {
        "description": "License grant that is irrevocable or perpetual in duration",
        "category": "general_information",
        "patterns": [
            "irrevocable license",
            "perpetual license",
            "in perpetuity",
            "forever",
            "irrevocable right",
        ]
    },

    # Restrictive Covenants
    "Anti-Assignment": {
        "description": "Consent or notice required if contract is assigned to third party",
        "category": "restrictive_covenants",
        "patterns": [
            "shall not assign",
            "may not assign without consent",
            "assignment requires approval",
            "non-assignable",
        ]
    },
    "Non-Compete": {
        "description": "Restriction on competing with counterparty during or after agreement",
        "category": "restrictive_covenants",
        "patterns": [
            "shall not compete",
            "non-competition",
            "competitive activities",
            "refrain from competing",
        ]
    },
    "Non-Disparagement": {
        "description": "Requirement not to make negative statements about counterparty",
        "category": "restrictive_covenants",
        "patterns": [
            "shall not disparage",
            "non-disparagement",
            "negative statements",
            "detrimental comments",
        ]
    },
    "No-Solicit Of Employees": {
        "description": "Restriction on soliciting or hiring counterparty's employees",
        "category": "restrictive_covenants",
        "patterns": [
            "shall not solicit employees",
            "non-solicitation of employees",
            "hiring restriction",
            "recruit employees",
        ]
    },
    "No-Solicit Of Customers": {
        "description": "Restriction on soliciting counterparty's customers",
        "category": "restrictive_covenants",
        "patterns": [
            "shall not solicit customers",
            "non-solicitation of customers",
            "customer restrictions",
            "solicit business from",
        ]
    },
    "Exclusivity": {
        "description": "Exclusive dealing or relationship requirements",
        "category": "restrictive_covenants",
        "patterns": [
            "exclusive",
            "sole and exclusive",
            "exclusively",
            "exclusive right",
            "exclusive provider",
        ]
    },
    "Change Of Control": {
        "description": "Provisions triggered by merger, acquisition, or change of ownership",
        "category": "restrictive_covenants",
        "patterns": [
            "change of control",
            "change in ownership",
            "merger or acquisition",
            "sale of assets",
            "controlling interest",
        ]
    },
    "Covenant Not To Sue": {
        "description": "Agreement not to sue counterparty for certain claims",
        "category": "restrictive_covenants",
        "patterns": [
            "covenant not to sue",
            "waiver of claims",
            "release of claims",
            "shall not bring any action",
        ]
    },
    "Competitive Restriction Exception": {
        "description": "Exceptions or carve-outs to competitive restrictions",
        "category": "restrictive_covenants",
        "patterns": [
            "except for",
            "notwithstanding the foregoing",
            "shall not apply to",
            "exception to non-compete",
        ]
    },
    "Non-Transferable License": {
        "description": "License that cannot be transferred to third parties",
        "category": "restrictive_covenants",
        "patterns": [
            "non-transferable",
            "may not transfer",
            "personal license",
            "cannot sublicense",
        ]
    },
    "Volume Restriction": {
        "description": "Restrictions on volume, quantity, or usage limits",
        "category": "restrictive_covenants",
        "patterns": [
            "volume limitations",
            "quantity restrictions",
            "usage limits",
            "maximum units",
        ]
    },

    # Revenue Risks
    "Cap On Liability": {
        "description": "Maximum liability amount or limitation on damages",
        "category": "revenue_risks",
        "patterns": [
            "liability shall not exceed",
            "maximum liability",
            "cap on damages",
            "limitation of liability",
            "aggregate liability",
        ]
    },
    "Uncapped Liability": {
        "description": "Liability that is not subject to any cap or limitation",
        "category": "revenue_risks",
        "patterns": [
            "unlimited liability",
            "shall not be subject to any limitation",
            "notwithstanding any limitation",
            "uncapped",
        ]
    },
    "Liquidated Damages": {
        "description": "Pre-determined damages amount for specific breaches",
        "category": "revenue_risks",
        "patterns": [
            "liquidated damages",
            "stipulated damages",
            "agreed damages",
            "per day penalty",
        ]
    },
    "Revenue/Profit Sharing": {
        "description": "Revenue sharing or profit sharing arrangements",
        "category": "revenue_risks",
        "patterns": [
            "revenue sharing",
            "profit sharing",
            "share of revenue",
            "percentage of profits",
            "royalty",
        ]
    },
    "Minimum Commitment": {
        "description": "Minimum purchase, usage, or payment commitments",
        "category": "revenue_risks",
        "patterns": [
            "minimum commitment",
            "minimum purchase",
            "minimum order",
            "guaranteed minimum",
        ]
    },
    "Audit Rights": {
        "description": "Right to audit counterparty's books, records, or compliance",
        "category": "revenue_risks",
        "patterns": [
            "audit rights",
            "right to audit",
            "inspect books and records",
            "examination of records",
        ]
    },
    "Insurance": {
        "description": "Insurance requirements and coverage obligations",
        "category": "revenue_risks",
        "patterns": [
            "shall maintain insurance",
            "insurance coverage",
            "proof of insurance",
            "certificate of insurance",
        ]
    },
    "Warranty Duration": {
        "description": "Duration or period of warranties provided",
        "category": "revenue_risks",
        "patterns": [
            "warranty period",
            "warranty duration",
            "warrants for a period of",
            "warranty expires",
        ]
    },
    "Post-Termination Services": {
        "description": "Services or obligations that continue after contract termination",
        "category": "revenue_risks",
        "patterns": [
            "post-termination",
            "following termination",
            "survive termination",
            "transition services",
            "wind-down",
        ]
    },
    "Termination For Convenience": {
        "description": "Right to terminate contract without cause or breach",
        "category": "revenue_risks",
        "patterns": [
            "terminate for convenience",
            "terminate without cause",
            "terminate at any time",
            "terminate for any reason",
        ]
    },

    # Intellectual Property
    "IP Ownership Assignment": {
        "description": "Assignment or transfer of intellectual property ownership",
        "category": "intellectual_property",
        "patterns": [
            "assigns all right, title and interest",
            "transfer of ownership",
            "work for hire",
            "all IP shall belong to",
            "assigns intellectual property",
        ]
    },
    "Joint IP Ownership": {
        "description": "Joint or shared ownership of intellectual property",
        "category": "intellectual_property",
        "patterns": [
            "jointly owned",
            "joint ownership",
            "co-owned",
            "shared ownership",
        ]
    },
    "Source Code Escrow": {
        "description": "Source code escrow arrangements for software",
        "category": "intellectual_property",
        "patterns": [
            "source code escrow",
            "escrow agent",
            "escrow arrangement",
            "deposit source code",
        ]
    },
    "Affiliate License-Licensor": {
        "description": "License extends to affiliates of the licensor",
        "category": "intellectual_property",
        "patterns": [
            "licensor's affiliates",
            "affiliates of licensor",
            "licensor and its affiliates",
        ]
    },
    "Affiliate License-Licensee": {
        "description": "License extends to affiliates of the licensee",
        "category": "intellectual_property",
        "patterns": [
            "licensee's affiliates",
            "affiliates of licensee",
            "licensee and its affiliates",
            "subsidiary license",
        ]
    },
    "Unlimited/All-You-Can-Eat-License": {
        "description": "Unlimited usage license without volume or user restrictions",
        "category": "intellectual_property",
        "patterns": [
            "unlimited license",
            "unlimited users",
            "enterprise-wide",
            "site license",
            "unlimited usage",
        ]
    },

    # Special Provisions
    "Third Party Beneficiary": {
        "description": "Third parties who have rights or benefits under the contract",
        "category": "special_provisions",
        "patterns": [
            "third party beneficiary",
            "no third party rights",
            "intended beneficiary",
        ]
    },
    "Most Favored Nation": {
        "description": "Most favored nation or most favored customer clause",
        "category": "special_provisions",
        "patterns": [
            "most favored nation",
            "most favored customer",
            "MFN",
            "best pricing",
        ]
    },
    "Rofr/Rofo/Rofn": {
        "description": "Right of first refusal, first offer, or first negotiation",
        "category": "special_provisions",
        "patterns": [
            "right of first refusal",
            "right of first offer",
            "right of first negotiation",
            "ROFR",
            "ROFO",
        ]
    },
}


def get_all_labels() -> list[str]:
    """Get list of all CUAD label names."""
    return list(CUAD_LABELS.keys())


def get_label_description(label: str) -> str:
    """Get description for a label."""
    return CUAD_LABELS.get(label, {}).get("description", "")


def get_label_patterns(label: str) -> list[str]:
    """Get example patterns for a label."""
    return CUAD_LABELS.get(label, {}).get("patterns", [])


def get_label_category(label: str) -> str:
    """Get category for a label."""
    return CUAD_LABELS.get(label, {}).get("category", "")


def get_labels_by_category(category: str) -> list[str]:
    """Get all labels in a category."""
    return [
        label for label, info in CUAD_LABELS.items()
        if info.get("category") == category
    ]
