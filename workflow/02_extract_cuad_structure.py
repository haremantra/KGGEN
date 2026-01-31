#!/usr/bin/env python3
"""
Extract and structure CUAD dataset components for PRD development.
This script analyzes the CUAD paper and extracts dataset structure and label categories.
"""

import json
from pathlib import Path

# CUAD Dataset Analysis
cuad_analysis = {
    "dataset_name": "CUAD: Contract Understanding Atticus Dataset",
    "purpose": "Expert-annotated NLP dataset for legal contract review",
    "organization": "The Atticus Project",
    "publication": "NeurIPS 2021 Track on Datasets and Benchmarks",
    "availability": "atticusprojectai.org/cuad, github.com/TheAtticusProject/cuad/",

    "dataset_statistics": {
        "total_contracts": 510,
        "total_annotations": 13101,
        "label_categories": 41,
        "contract_types": 25,
        "pages_per_contract": {
            "range": "few pages to over 100 pages",
            "variation": "Wide variation in length"
        },
        "annotation_density": "~10% of each contract highlighted (0.25% per label on average)",
        "estimated_value": "$2+ million (conservative estimate based on expert time)"
    },

    "annotation_process": {
        "annotators": "Dozens of law student annotators supervised by experienced lawyers",
        "training_duration": "70-100 hours per annotator",
        "documentation": "Over 100 pages of rules and annotation standards",
        "quality_control": "Each annotation verified by 3 additional annotators",
        "review_intensity": "9,283 pages reviewed at least 4 times, 5-10 minutes per page",
        "billing_rate_assumption": "$500 per hour (typical for large law firms)"
    },

    "contract_types": [
        {"name": "Affiliate Agreement", "count": 10},
        {"name": "Agency Agreement", "count": 13},
        {"name": "Collaboration Agreement", "count": 26},
        {"name": "Co-Branding Agreement", "count": 22},
        {"name": "Consulting Agreement", "count": 11},
        {"name": "Development Agreement", "count": 29},
        {"name": "Distributor Agreement", "count": 32},
        {"name": "Endorsement Agreement", "count": 24},
        {"name": "Franchise Agreement", "count": 15},
        {"name": "Hosting Agreement", "count": 20},
        {"name": "IP Agreement", "count": 17},
        {"name": "Joint Venture Agreement", "count": 23},
        {"name": "License Agreement", "count": 33},
        {"name": "Maintenance Agreement", "count": 34},
        {"name": "Manufacturing Agreement", "count": 17},
        {"name": "Marketing Agreement", "count": 20},
        {"name": "Non-Compete Agreement", "count": 10},
        {"name": "Outsourcing Agreement", "count": 15},
        {"name": "Promotion Agreement", "count": 12},
        {"name": "Reseller Agreement", "count": 18},
        {"name": "Service Agreement", "count": 42},
        {"name": "Sponsorship Agreement", "count": 14},
        {"name": "Supply Agreement", "count": 23},
        {"name": "Strategic Alliance Agreement", "count": 18},
        {"name": "Transportation Agreement", "count": 8}
    ],

    "label_categories": {
        "general_information": [
            {"name": "Document Name", "description": "Name of the contract document"},
            {"name": "Parties", "description": "Names of parties to the contract"},
            {"name": "Agreement Date", "description": "Date when agreement was signed"},
            {"name": "Effective Date", "description": "Date when contract becomes effective"},
            {"name": "Expiration Date", "description": "Date when contract expires"},
            {"name": "Renewal Term", "description": "Renewal term after initial term expires"},
            {"name": "Notice Period To Terminate Renewal", "description": "Notice period required to terminate renewal"},
            {"name": "Governing Law", "description": "State/country's law governing contract interpretation"},
            {"name": "License Grant", "description": "License grant provisions"},
            {"name": "Irrevocable Or Perpetual License", "description": "License grant that is irrevocable or perpetual"}
        ],

        "restrictive_covenants": [
            {"name": "Anti-Assignment", "description": "Consent or notice required if contract assigned to third party"},
            {"name": "Non-Compete", "description": "Restriction on competing with counterparty"},
            {"name": "Non-Disparagement", "description": "Requirement not to disparage counterparty"},
            {"name": "No-Solicit Of Employees", "description": "Restriction on soliciting employees"},
            {"name": "No-Solicit Of Customers", "description": "Restriction on soliciting customers"},
            {"name": "Exclusivity", "description": "Exclusive relationship requirements"},
            {"name": "Change Of Control", "description": "Provisions triggered by change of control"},
            {"name": "Covenant Not To Sue", "description": "Agreement not to sue counterparty"},
            {"name": "Competitive Restriction Exception", "description": "Exceptions to competitive restrictions"},
            {"name": "Non-Transferable License", "description": "License that cannot be transferred"},
            {"name": "Volume Restriction", "description": "Restrictions on volume or quantity"}
        ],

        "revenue_risks": [
            {"name": "Cap On Liability", "description": "Maximum liability amount"},
            {"name": "Uncapped Liability", "description": "Uncapped liability upon breach"},
            {"name": "Liquidated Damages", "description": "Pre-determined damages for breach"},
            {"name": "Revenue/Profit Sharing", "description": "Revenue or profit sharing arrangements"},
            {"name": "Minimum Commitment", "description": "Minimum purchase or usage commitments"},
            {"name": "Audit Rights", "description": "Right to audit counterparty"},
            {"name": "Insurance", "description": "Insurance requirements"},
            {"name": "Warranty Duration", "description": "Duration of warranties"},
            {"name": "Post-Termination Services", "description": "Services required after termination"},
            {"name": "Termination For Convenience", "description": "Right to terminate without cause"}
        ],

        "intellectual_property": [
            {"name": "IP Ownership Assignment", "description": "Assignment of IP ownership"},
            {"name": "Joint IP Ownership", "description": "Joint ownership of IP"},
            {"name": "Source Code Escrow", "description": "Source code escrow arrangements"},
            {"name": "Affiliate License-Licensor", "description": "License to affiliates of licensor"},
            {"name": "Affiliate License-Licensee", "description": "License to affiliates of licensee"},
            {"name": "Unlimited/All-You-Can-Eat-License", "description": "Unlimited usage license"}
        ],

        "special_provisions": [
            {"name": "Third Party Beneficiary", "description": "Third parties who benefit from contract"},
            {"name": "Most Favored Nation", "description": "Most favored nation clause"},
            {"name": "Rofr/Rofo/Rofn", "description": "Right of first refusal/offer/negotiation"}
        ]
    },

    "technology_relevant_categories": {
        "description": "Subset of 41 labels most relevant to technology agreements",
        "categories": [
            "IP Ownership Assignment",
            "Joint IP Ownership",
            "Source Code Escrow",
            "License Grant",
            "Irrevocable Or Perpetual License",
            "Non-Transferable License",
            "Unlimited/All-You-Can-Eat-License",
            "Affiliate License-Licensor",
            "Affiliate License-Licensee",
            "Development Agreement",
            "Hosting Agreement",
            "Service Agreement",
            "Maintenance Agreement",
            "Cap On Liability",
            "Uncapped Liability",
            "Warranty Duration",
            "Non-Compete",
            "Exclusivity",
            "Change Of Control",
            "Audit Rights",
            "Anti-Assignment"
        ]
    },

    "task_definition": {
        "type": "Extractive span identification",
        "structure": "Similar to SQuAD 2.0 (allows no-answer)",
        "input": "Contract text + label category description",
        "output": "Start and end tokens identifying relevant text spans",
        "challenge": "Finding needles in haystack (0.25% relevant per label)",
        "levels_of_work": {
            "contract_analysis": "Find relevant clauses, identify their content, track locations (automatable)",
            "counseling": "Assess risk, provide business advice (requires experienced lawyers)"
        }
    },

    "data_source": {
        "source": "EDGAR (Electronic Data Gathering, Analysis, and Retrieval system)",
        "provider": "U.S. Securities and Exchange Commission (SEC)",
        "access": "Free and open to the public",
        "characteristics": "More complicated and heavily negotiated than general contracts",
        "advantage": "Contains large sample of rare/difficult clauses"
    },

    "baseline_performance": {
        "best_model": "DeBERTa-xlarge",
        "aupr": "47.8%",
        "precision_at_80_recall": "44.0%",
        "precision_at_90_recall": "17.8%",
        "finding": "Performance nascent but improving; substantial room for improvement"
    },

    "value_proposition": {
        "law_firms": "~50% of time spent on contract review",
        "cost": "$500-$900/hour billing rates, hundreds of thousands per transaction",
        "automation_benefit": "Reduce drudgery, increase access to legal support for small businesses and individuals"
    }
}

# Technology agreement specific ontology
tech_agreement_ontology = {
    "domain": "Technology Agreements",
    "scope": "AI software for lawyers - contract analysis tool",
    "target_users": [
        "Engineering team building AI legal software",
        "Senior practicing lawyer in technology (team member)",
        "Law firms reviewing technology contracts",
        "Companies entering technology agreements"
    ],

    "core_entity_types": {
        "parties": ["Licensor", "Licensee", "Developer", "Client", "Vendor", "Service Provider"],
        "intellectual_property": ["Source Code", "Patents", "Trademarks", "Trade Secrets", "Copyrights", "Derivatives"],
        "deliverables": ["Software", "Documentation", "APIs", "SDKs", "Updates", "Maintenance"],
        "financial": ["License Fees", "Maintenance Fees", "Royalties", "Milestones", "Penalties"],
        "timeframes": ["Term", "Renewal Period", "Notice Period", "Warranty Period", "Transition Period"]
    },

    "core_relationship_types": {
        "ownership": ["owns", "assigns", "retains", "jointly_owns", "licenses"],
        "obligations": ["must_provide", "must_maintain", "must_support", "must_update", "must_escrow"],
        "restrictions": ["cannot_compete", "cannot_disclose", "cannot_assign", "cannot_sublicense"],
        "rights": ["can_terminate", "can_audit", "can_modify", "has_right_to"],
        "liability": ["is_liable_for", "is_not_liable_for", "indemnifies", "warrants"]
    },

    "common_law_considerations": {
        "jurisdiction": "Common law systems (US, UK, Commonwealth)",
        "key_principles": [
            "Freedom of contract",
            "Meeting of minds (mutual assent)",
            "Consideration",
            "Reasonableness standard",
            "Good faith and fair dealing"
        ],
        "interpretation_rules": [
            "Plain meaning rule",
            "Contra proferentem (against the drafter)",
            "Business efficacy test",
            "Contextual interpretation"
        ]
    }
}

# Save structured analyses
output_path_cuad = Path("/app/sandbox/session_20260112_140312_4731309a153b/data/cuad_dataset_analysis.json")
output_path_tech = Path("/app/sandbox/session_20260112_140312_4731309a153b/data/tech_agreement_ontology.json")

with open(output_path_cuad, 'w') as f:
    json.dump(cuad_analysis, f, indent=2)

with open(output_path_tech, 'w') as f:
    json.dump(tech_agreement_ontology, f, indent=2)

print(f"✓ CUAD dataset analysis saved to: {output_path_cuad}")
print(f"✓ Technology agreement ontology saved to: {output_path_tech}")
print(f"✓ Documented {len(cuad_analysis['contract_types'])} contract types")
print(f"✓ Extracted {cuad_analysis['dataset_statistics']['label_categories']} label categories")
print(f"✓ Identified {len(tech_agreement_ontology['core_entity_types'])} core entity types")
print(f"✓ Defined {len(tech_agreement_ontology['core_relationship_types'])} relationship types")
