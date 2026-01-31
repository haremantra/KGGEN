#!/usr/bin/env python3
"""
Map KGGen methodology to CUAD contract domain for PRD development.
This script creates the technical architecture mapping between KGGen pipeline and CUAD contracts.
"""

import json
from pathlib import Path

# Load previous analyses
data_dir = Path("/app/sandbox/session_20260112_140312_4731309a153b/data")

with open(data_dir / "kggen_methodology_analysis.json") as f:
    kggen = json.load(f)

with open(data_dir / "cuad_dataset_analysis.json") as f:
    cuad = json.load(f)

with open(data_dir / "tech_agreement_ontology.json") as f:
    tech_ontology = json.load(f)

# Create comprehensive mapping
kggen_cuad_mapping = {
    "product_name": "CUAD Knowledge Graph Generator for Legal Contract Analysis",
    "product_vision": "Apply KGGen methodology to extract structured knowledge graphs from CUAD contracts, enabling LLMs to provide context-aware legal analysis for technology agreements under common law",

    "architecture_mapping": {
        "overview": "Adapt KGGen's 3-stage pipeline to process legal contracts from CUAD dataset",

        "stage_1_adaptation": {
            "original": "Entity and Relation Extraction from general text",
            "adapted": "Legal Entity and Contract Clause Extraction",
            "input": "CUAD contract text (510 contracts, 25 types, focus on technology agreements)",
            "process": {
                "step_1": {
                    "task": "Extract legal entities from contract",
                    "entities_to_extract": [
                        "Parties (Licensor, Licensee, etc.)",
                        "Dates (Effective, Expiration, Renewal)",
                        "Jurisdictions (Governing Law)",
                        "IP Assets (Patents, Source Code, Trade Secrets)",
                        "Financial Terms (Fees, Royalties, Caps)",
                        "Obligations and Rights"
                    ],
                    "llm_model": "Claude Sonnet 3.5 (preferred, 73% MINE-1) or GPT-4o (66% MINE-1)",
                    "framework": "DSPy signatures for structured output"
                },
                "step_2": {
                    "task": "Extract contractual relationships",
                    "relations_to_extract": [
                        "Party-to-Party (licensor-licenses_to-licensee)",
                        "Party-to-IP (party-owns/assigns/retains-ip_asset)",
                        "Party-to-Obligation (party-must_provide/maintain-deliverable)",
                        "Party-to-Restriction (party-cannot_compete/disclose-scope)",
                        "Party-to-Liability (party-is_liable_for/warrants-scope)",
                        "Clause-to-Condition (clause-triggers_on-event)",
                        "Term-to-Timeframe (agreement-has_term-duration)"
                    ],
                    "cuad_labels_mapping": {
                        "ownership_relations": ["IP Ownership Assignment", "Joint IP Ownership"],
                        "license_relations": ["License Grant", "Irrevocable Or Perpetual License", "Non-Transferable License"],
                        "restriction_relations": ["Non-Compete", "Exclusivity", "Anti-Assignment", "No-Solicit Of Employees"],
                        "liability_relations": ["Cap On Liability", "Uncapped Liability", "Liquidated Damages"],
                        "temporal_relations": ["Effective Date", "Expiration Date", "Renewal Term", "Warranty Duration"]
                    }
                }
            },
            "output": "Subject-Predicate-Object triples representing contract structure",
            "example_triples": [
                "(TechCorp, licenses_to, ClientCo)",
                "(TechCorp, assigns_ip_ownership, ClientCo)",
                "(ClientCo, has_cap_on_liability, $1M)",
                "(Agreement, governed_by_law_of, California)",
                "(License, is_irrevocable, True)",
                "(TechCorp, cannot_solicit_employees_of, ClientCo)"
            ]
        },

        "stage_2_adaptation": {
            "original": "Aggregation across all source texts",
            "adapted": "Cross-Contract Aggregation",
            "input": "Knowledge graphs from individual contracts",
            "process": {
                "aggregation_scope": "Aggregate across all contracts of same type (e.g., all License Agreements)",
                "normalization": "Normalize to lowercase, standardize legal terminology",
                "deduplication": "Remove exact duplicate triples across contracts",
                "clustering_dimension": "Group by contract type, jurisdiction, industry vertical"
            },
            "output": "Unified knowledge graph representing patterns across contract corpus",
            "benefits": [
                "Identify common clauses across technology agreements",
                "Detect anomalous or unusual contract terms",
                "Build ontology of standard technology contract provisions",
                "Enable cross-contract comparison and analysis"
            ]
        },

        "stage_3_adaptation": {
            "original": "Entity and Edge Resolution for general text",
            "adapted": "Legal Entity and Clause Resolution",
            "input": "Aggregated knowledge graph with potential duplicates",
            "process": {
                "entity_resolution": {
                    "challenge": "Legal entities may have multiple representations",
                    "examples": [
                        "['Software', 'the Software', 'Licensed Software', 'the Program'] -> 'Licensed Software'",
                        "['Licensor', 'Company', 'ABC Corp', 'the Provider'] -> 'ABC Corp (Licensor)'",
                        "['source code', 'Source Code', 'source code files', 'code base'] -> 'Source Code'"
                    ],
                    "method": {
                        "clustering": "S-BERT embeddings + k-means (k=128)",
                        "retrieval": "Top-16 similar using BM25 + semantic similarity",
                        "deduplication": "LLM identifies synonyms considering legal context",
                        "canonicalization": "Select most legally precise term as canonical form"
                    },
                    "legal_considerations": [
                        "Preserve legal distinctions (e.g., 'exclusive license' vs 'non-exclusive license')",
                        "Respect defined terms in contracts",
                        "Maintain jurisdiction-specific terminology",
                        "Do not conflate legally distinct concepts"
                    ]
                },
                "edge_resolution": {
                    "challenge": "Contract obligations expressed in varied language",
                    "examples": [
                        "['must provide', 'shall provide', 'will provide', 'obligated to provide'] -> 'must_provide'",
                        "['owns', 'has ownership of', 'possesses all rights to'] -> 'owns'",
                        "['cannot compete', 'may not compete', 'restricted from competing'] -> 'cannot_compete'"
                    ],
                    "method": "Same clustering + LLM approach as entity resolution",
                    "output": "Canonical set of contract relationship types"
                }
            },
            "output": "Dense, de-duplicated knowledge graph with canonical entities and relations",
            "quality_target": "98% valid triples (matching KGGen benchmark)"
        }
    },

    "knowledge_graph_schema": {
        "graph_type": "Property Graph (nodes with properties, labeled edges)",
        "storage_format": "Neo4j, NetworkX, or RDF triple store",

        "node_types": {
            "Party": {
                "properties": ["name", "role", "type", "jurisdiction"],
                "examples": ["ABC Corp (Licensor)", "XYZ Inc (Licensee)"]
            },
            "IPAsset": {
                "properties": ["name", "type", "description", "registration_number"],
                "examples": ["Source Code", "Patent US12345", "Trademark 'TechBrand'"],
                "types": ["patent", "trademark", "copyright", "trade_secret", "source_code"]
            },
            "Obligation": {
                "properties": ["description", "party", "scope", "timeframe"],
                "examples": ["Provide Maintenance", "Deliver Updates", "Pay License Fees"]
            },
            "Restriction": {
                "properties": ["type", "scope", "duration", "exceptions"],
                "examples": ["Non-Compete", "No-Solicit", "Anti-Assignment"],
                "cuad_labels": ["Non-Compete", "Exclusivity", "Anti-Assignment", "No-Solicit Of Employees", "No-Solicit Of Customers"]
            },
            "LiabilityProvision": {
                "properties": ["type", "cap_amount", "scope", "exceptions"],
                "examples": ["Cap On Liability $1M", "Uncapped for IP Infringement"],
                "cuad_labels": ["Cap On Liability", "Uncapped Liability", "Liquidated Damages"]
            },
            "Temporal": {
                "properties": ["date", "duration", "type"],
                "examples": ["Effective Date: 2024-01-01", "Term: 3 years", "Notice Period: 90 days"],
                "cuad_labels": ["Effective Date", "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal"]
            },
            "Jurisdiction": {
                "properties": ["state", "country", "legal_system"],
                "examples": ["California", "New York", "Delaware"],
                "cuad_label": "Governing Law"
            },
            "ContractClause": {
                "properties": ["clause_type", "text", "cuad_label", "page_number"],
                "examples": ["IP Assignment Clause", "Exclusivity Clause"],
                "cuad_labels": "All 41 CUAD labels"
            }
        },

        "edge_types": {
            "LICENSES_TO": {"from": "Party", "to": "Party", "properties": ["license_type", "scope"]},
            "OWNS": {"from": "Party", "to": "IPAsset", "properties": ["ownership_type"]},
            "ASSIGNS": {"from": "Party", "to": "IPAsset", "properties": ["assignment_scope", "conditions"]},
            "HAS_OBLIGATION": {"from": "Party", "to": "Obligation", "properties": ["priority"]},
            "SUBJECT_TO_RESTRICTION": {"from": "Party", "to": "Restriction", "properties": ["exceptions"]},
            "HAS_LIABILITY": {"from": "Party", "to": "LiabilityProvision", "properties": []},
            "GOVERNED_BY": {"from": "Contract", "to": "Jurisdiction", "properties": []},
            "CONTAINS_CLAUSE": {"from": "Contract", "to": "ContractClause", "properties": ["page_number"]},
            "EFFECTIVE_ON": {"from": "Contract", "to": "Temporal", "properties": ["event_type"]},
            "TERMINATES_ON": {"from": "Contract", "to": "Temporal", "properties": ["conditions"]}
        }
    },

    "llm_integration": {
        "purpose": "Provide structured context to LLM for contract analysis queries",
        "retrieval_mechanism": {
            "query_type": "Natural language question about contract",
            "example_queries": [
                "What IP rights does the licensee receive?",
                "Are there any non-compete restrictions?",
                "What is the liability cap?",
                "When does this agreement expire?",
                "What are the termination conditions?"
            ],
            "retrieval_process": {
                "step_1": "Embed query using all-MiniLM-L6-v2",
                "step_2": "Retrieve top-10 relevant triples using BM25 + cosine similarity",
                "step_3": "Expand to 2-hop neighbors for multi-hop reasoning",
                "step_4": "Extract subgraph with 20-30 triples"
            },
            "context_provision": {
                "format": "Structured triples + original contract text chunks",
                "template": "Based on the knowledge graph: [triples]. Original contract text: [chunks]. Answer: [question]",
                "llm_model": "Claude Sonnet 3.5, GPT-4o, or equivalent"
            }
        },
        "use_cases": [
            "Contract Q&A: Answer specific questions about contract terms",
            "Risk Analysis: Identify unusual or risky clauses",
            "Comparison: Compare terms across multiple contracts",
            "Compliance: Check contract against company standards",
            "Due Diligence: Rapid contract review for M&A"
        ]
    },

    "common_law_considerations": {
        "legal_system": "Common Law (US, UK, Commonwealth jurisdictions)",
        "interpretation_principles": [
            "Plain meaning rule: Terms interpreted by ordinary meaning",
            "Contra proferentem: Ambiguity construed against drafter",
            "Business efficacy: Interpret to give commercial effect",
            "Contextual interpretation: Consider entire contract"
        ],
        "ontology_alignment": {
            "defined_terms": "Preserve contract-specific defined terms",
            "standard_clauses": "Map to common law standard form provisions",
            "jurisdictional_variations": "Tag nodes with applicable jurisdiction",
            "precedent_linking": "Optional: Link clauses to case law precedents"
        },
        "key_legal_concepts": {
            "offer_acceptance": "Formation of contract",
            "consideration": "Exchange of value",
            "capacity": "Legal ability to contract",
            "intention": "Intention to create legal relations",
            "certainty": "Terms must be certain and complete"
        }
    }
}

# Save mapping
output_path = data_dir / "kggen_cuad_mapping.json"
with open(output_path, 'w') as f:
    json.dump(kggen_cuad_mapping, f, indent=2)

print(f"✓ KGGen-CUAD mapping saved to: {output_path}")
print(f"✓ Defined {len(kggen_cuad_mapping['knowledge_graph_schema']['node_types'])} node types")
print(f"✓ Defined {len(kggen_cuad_mapping['knowledge_graph_schema']['edge_types'])} edge types")
print(f"✓ Mapped {len(kggen_cuad_mapping['architecture_mapping'])} architecture stages")
print(f"✓ Specified {len(kggen_cuad_mapping['llm_integration']['use_cases'])} use cases")
print(f"✓ Documented {len(kggen_cuad_mapping['common_law_considerations']['interpretation_principles'])} legal principles")
