#!/usr/bin/env python3
"""
Generate comprehensive PRD structure for CUAD Knowledge Graph Generator.
This script creates the complete Product Requirements Document structure.
"""

import json
from pathlib import Path
from datetime import datetime

# Load all analyses
data_dir = Path("/app/sandbox/session_20260112_140312_4731309a153b/data")

with open(data_dir / "kggen_methodology_analysis.json") as f:
    kggen = json.load(f)

with open(data_dir / "cuad_dataset_analysis.json") as f:
    cuad = json.load(f)

with open(data_dir / "tech_agreement_ontology.json") as f:
    tech_ontology = json.load(f)

with open(data_dir / "kggen_cuad_mapping.json") as f:
    mapping = json.load(f)

# Generate comprehensive PRD structure
prd_structure = {
    "document_metadata": {
        "title": "Product Requirements Document: CUAD Knowledge Graph Generator for Legal AI",
        "subtitle": "Applying KGGen Methodology to Technology Contract Analysis",
        "version": "1.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "Draft",
        "classification": "Technical Product Specification",
        "target_audience": [
            "Engineering team building AI legal software",
            "Senior technology lawyer (technical team member)",
            "Data scientists working on legal NLP",
            "Product managers for legal tech solutions"
        ]
    },

    "executive_summary": {
        "problem_statement": "Legal contract review is expensive ($500-900/hour), time-consuming (50% of law firm time), and inaccessible to small businesses. Current AI solutions lack structured understanding of contract relationships and common law context.",

        "solution": "Build a knowledge graph extraction system that applies KGGen methodology to CUAD contract dataset, creating structured ontologies of technology agreements that enable context-aware LLM analysis under common law principles.",

        "key_benefits": [
            "Structured contract understanding: Convert unstructured contracts into queryable knowledge graphs",
            "Context-aware AI: Provide LLMs with precise contract relationships, not just text chunks",
            "Common law alignment: Ontology respects legal interpretation principles and jurisdictional nuances",
            "Scalable analysis: Process 510+ contracts with 41 clause types automatically",
            "Technology focus: Specialized for tech agreements (IP, licensing, SaaS, development)",
            "High accuracy: Target 98% valid triples (KGGen benchmark)",
            "Engineer-lawyer collaboration: Technical product designed with legal practitioner input"
        ],

        "success_metrics": {
            "technical": [
                "Extract knowledge graphs from 510 CUAD contracts with 98% triple validity",
                "Achieve 65%+ MINE-1 score (information retention)",
                "Process contracts at scale: 100+ contracts/hour",
                "Support multi-hop reasoning with 2-hop subgraph expansion"
            ],
            "business": [
                "Reduce contract review time by 50%",
                "Enable automated clause identification for 41+ categories",
                "Provide instant answers to common contract queries",
                "Scale legal support to small businesses and individuals"
            ],
            "user": [
                "Engineers can build reliable legal AI features",
                "Lawyers can validate AI outputs with structured data",
                "Rapid prototyping of legal analysis tools",
                "Interpretable AI with explainable knowledge graph reasoning"
            ]
        }
    },

    "product_vision": {
        "primary_goal": "Create a production-ready knowledge graph extraction system for legal contracts that enables next-generation AI-powered contract analysis tools",

        "target_use_cases": [
            {
                "name": "Automated Contract Q&A",
                "description": "Answer specific questions about contract terms",
                "example": "User: 'What IP rights does the licensee receive?' System retrieves IP-related nodes and generates answer",
                "value": "Instant answers without manual review"
            },
            {
                "name": "Risk Analysis Dashboard",
                "description": "Identify unusual or risky clauses across contract portfolio",
                "example": "Flag contracts with uncapped liability, missing IP assignment, or aggressive non-compete",
                "value": "Proactive risk management"
            },
            {
                "name": "Contract Comparison",
                "description": "Compare terms across multiple contracts or against standards",
                "example": "Compare license terms across 50 vendor agreements",
                "value": "Negotiation leverage and consistency"
            },
            {
                "name": "Compliance Verification",
                "description": "Check contracts against company policies or legal requirements",
                "example": "Verify all contracts have required insurance, audit rights, and liability caps",
                "value": "Regulatory compliance"
            },
            {
                "name": "Due Diligence Acceleration",
                "description": "Rapid contract review for M&A, financing, or audits",
                "example": "Extract all change-of-control provisions from 200 contracts in minutes",
                "value": "Deal velocity"
            }
        ],

        "technology_agreement_focus": {
            "rationale": "Technology contracts have unique characteristics requiring specialized ontology",
            "key_aspects": [
                "IP ownership and licensing (patents, source code, trademarks)",
                "SaaS and software licensing terms",
                "Development and customization agreements",
                "Open source compliance and derivative works",
                "API access and integration rights",
                "Data ownership and processing",
                "Service level agreements and uptime guarantees",
                "Maintenance and support obligations",
                "Escrow and succession planning"
            ],
            "common_law_context": "Contracts interpreted under US/UK/Commonwealth common law principles with emphasis on freedom of contract, plain meaning, and business efficacy"
        }
    },

    "technical_architecture": {
        "overview": "3-stage pipeline adapting KGGen methodology for legal contracts",

        "system_diagram": {
            "flow": "CUAD Contracts → Stage 1: Extraction → Stage 2: Aggregation → Stage 3: Resolution → Knowledge Graph → LLM Context Provider → User Application",
            "components": [
                "Input: CUAD dataset (510 contracts, 25 types)",
                "Extraction Engine: LLM-based entity and relation extraction",
                "Aggregation Service: Cross-contract normalization and deduplication",
                "Resolution Engine: Entity and edge clustering with LLM canonicalization",
                "Graph Database: Neo4j or equivalent property graph store",
                "Query Interface: BM25 + semantic search for retrieval",
                "LLM Integration: Context provider for Claude/GPT-4",
                "API Layer: REST/GraphQL for application integration"
            ]
        },

        "stage_1_extraction": {
            "input": "Contract text from CUAD (PDF or text format)",
            "preprocessing": [
                "OCR if needed (contracts are PDFs)",
                "Section identification (parties, terms, schedules)",
                "CUAD label alignment (map text to 41 categories)"
            ],
            "extraction_process": {
                "llm_model": "Claude Sonnet 3.5 (primary, 73% MINE-1) with GPT-4o fallback (66% MINE-1)",
                "framework": "DSPy for structured LLM orchestration",
                "prompt_engineering": {
                    "step_1_entities": {
                        "instruction": "Extract legal entities from contract",
                        "output_schema": "List of entities with types (Party, IPAsset, Obligation, etc.)",
                        "constraints": [
                            "Preserve legal distinctions (exclusive vs non-exclusive)",
                            "Respect contract defined terms (capitalized terms)",
                            "Identify all parties and their roles",
                            "Extract all dates with context (effective, expiration, notice)"
                        ]
                    },
                    "step_2_relations": {
                        "instruction": "Extract contractual relationships between entities",
                        "output_schema": "(subject, predicate, object) triples",
                        "relation_types": [
                            "licenses_to", "owns", "assigns", "retains",
                            "must_provide", "must_maintain", "must_support",
                            "cannot_compete", "cannot_disclose", "cannot_assign",
                            "is_liable_for", "is_not_liable_for", "indemnifies",
                            "governed_by", "terminates_on", "renews_after"
                        ],
                        "constraints": [
                            "Link relations to specific parties",
                            "Capture conditions and exceptions",
                            "Preserve temporal aspects",
                            "Note jurisdiction-specific terms"
                        ]
                    }
                },
                "cuad_integration": {
                    "method": "Use CUAD labels as hints for extraction",
                    "process": "For each contract, identify clauses by CUAD category, extract entities/relations from those clauses",
                    "benefit": "Leverage expert annotations to guide extraction",
                    "labels_to_relations": {
                        "IP_Ownership_Assignment": ["assigns", "owns"],
                        "License_Grant": ["licenses_to", "has_right_to"],
                        "Non_Compete": ["cannot_compete", "restricted_from"],
                        "Cap_On_Liability": ["has_liability_cap", "limited_to"],
                        "Governing_Law": ["governed_by"],
                        "Effective_Date": ["effective_on"],
                        "Expiration_Date": ["terminates_on"],
                        "Renewal_Term": ["renews_after", "extends_for"]
                    }
                }
            },
            "output": {
                "format": "JSON array of (subject, predicate, object, properties) tuples",
                "example": [
                    {"subject": "ABC Corp", "predicate": "licenses_to", "object": "XYZ Inc", "properties": {"license_type": "exclusive", "scope": "worldwide"}},
                    {"subject": "ABC Corp", "predicate": "assigns", "object": "Source Code", "properties": {"assignment_type": "full", "effective": "upon_payment"}},
                    {"subject": "Agreement", "predicate": "governed_by", "object": "California", "properties": {"legal_system": "common_law"}}
                ],
                "metadata": {
                    "contract_id": "CUAD_contract_123",
                    "contract_type": "License Agreement",
                    "extraction_timestamp": "2024-01-01T00:00:00Z",
                    "llm_model": "claude-sonnet-3.5",
                    "confidence_scores": "per-triple confidence from LLM"
                }
            },
            "performance_targets": {
                "throughput": "1-2 contracts per minute",
                "accuracy": "95%+ entity extraction accuracy",
                "coverage": "Extract from all 41 CUAD label categories",
                "consistency": "Maintain consistency between entities and relations within contract"
            }
        },

        "stage_2_aggregation": {
            "input": "Knowledge graphs from individual contracts (510 graphs)",
            "aggregation_dimensions": [
                "By contract type: All License Agreements, All Development Agreements, etc.",
                "By jurisdiction: All California contracts, All Delaware contracts, etc.",
                "By time period: Contracts from 2020-2023",
                "Global: All 510 contracts combined"
            ],
            "normalization": {
                "text_normalization": "Convert to lowercase, remove punctuation",
                "legal_term_normalization": {
                    "method": "Maintain legal register while normalizing synonyms",
                    "examples": [
                        "'shall provide' -> 'must_provide'",
                        "'is obligated to deliver' -> 'must_provide'",
                        "'has exclusive rights to' -> 'exclusively_owns'"
                    ],
                    "constraints": "Preserve legal significance (SHALL vs MAY vs SHOULD)"
                },
                "entity_normalization": {
                    "parties": "Normalize party names but preserve legal entity distinctions",
                    "ip_assets": "Standardize IP asset type terminology",
                    "temporal": "Normalize date formats to ISO 8601",
                    "financial": "Normalize currency and amount formats"
                }
            },
            "deduplication": {
                "exact_matches": "Remove identical triples across contracts",
                "near_matches": "Flag for Stage 3 resolution",
                "statistics_tracking": {
                    "unique_entities": "Count distinct entities across corpus",
                    "unique_relations": "Count distinct relation types",
                    "triple_frequency": "Track how often each triple pattern appears",
                    "contract_coverage": "Which contracts contain each triple pattern"
                }
            },
            "output": {
                "unified_graph": "Single knowledge graph with all triples",
                "statistics": {
                    "total_triples": "Estimated 50,000-100,000 triples from 510 contracts",
                    "unique_entities": "Estimated 10,000-20,000 unique entities before resolution",
                    "unique_relations": "Estimated 500-1000 unique relation types before resolution",
                    "contracts_processed": 510
                },
                "metadata": {
                    "aggregation_date": "timestamp",
                    "source_contracts": "list of contract IDs",
                    "aggregation_dimensions": "by_type, by_jurisdiction, global"
                }
            },
            "performance_targets": {
                "processing_time": "Aggregate 510 contracts in <10 minutes",
                "memory_efficiency": "Process in batches if needed",
                "storage": "Optimized graph representation (compressed)"
            }
        },

        "stage_3_resolution": {
            "input": "Aggregated knowledge graph with potential duplicates/synonyms",
            "purpose": "Reduce sparsity by merging equivalent entities and relations",

            "entity_resolution": {
                "challenge": "Legal entities have multiple representations across contracts",
                "method": {
                    "step_1_clustering": {
                        "algorithm": "k-means clustering",
                        "embedding_model": "S-BERT (Sentence-BERT)",
                        "cluster_size": 128,
                        "parallelization": "Process clusters in parallel"
                    },
                    "step_2_retrieval": {
                        "method": "Fused retrieval: BM25 + semantic embedding",
                        "top_k": 16,
                        "purpose": "Find most similar entities within cluster"
                    },
                    "step_3_deduplication": {
                        "llm_task": "Identify exact duplicates",
                        "llm_model": "Claude Sonnet 3.5 or GPT-4o",
                        "considerations": [
                            "Tense variations (provide/provides/provided)",
                            "Plurality (license/licenses)",
                            "Case (Source Code/source code/SOURCE CODE)",
                            "Abbreviations (IP/Intellectual Property)",
                            "Shorthand (non-compete/covenant not to compete)"
                        ],
                        "legal_constraints": [
                            "Do NOT merge legally distinct terms (exclusive vs non-exclusive)",
                            "Do NOT merge jurisdiction-specific terms",
                            "Preserve defined terms from contracts",
                            "Respect semantic differences in legal context"
                        ]
                    },
                    "step_4_canonicalization": {
                        "llm_task": "Select canonical representative",
                        "criteria": [
                            "Most legally precise",
                            "Most commonly used in legal contracts",
                            "Clearest meaning",
                            "Matches CUAD terminology when possible"
                        ],
                        "alias_tracking": "Maintain mapping of all variants to canonical form (like Wikidata)"
                    },
                    "step_5_iteration": {
                        "process": "Remove processed entities, repeat until cluster empty",
                        "optimization": "Process independent clusters in parallel"
                    }
                },
                "examples": [
                    {
                        "variants": ["Software", "the Software", "Licensed Software", "the Program", "the Application"],
                        "canonical": "Licensed Software",
                        "aliases": ["Software", "the Program", "the Application"]
                    },
                    {
                        "variants": ["source code", "Source Code", "source code files", "code base", "source code repository"],
                        "canonical": "Source Code",
                        "aliases": ["code base", "source code repository"]
                    },
                    {
                        "variants": ["Licensor", "Company", "ABC Corp", "the Provider", "Vendor"],
                        "canonical": "ABC Corp (Licensor)",
                        "aliases": ["Company", "Provider", "Vendor"]
                    }
                ]
            },

            "edge_resolution": {
                "challenge": "Contract obligations expressed in varied language",
                "method": "Same clustering + LLM approach as entity resolution",
                "examples": [
                    {
                        "variants": ["must provide", "shall provide", "will provide", "obligated to provide", "required to provide"],
                        "canonical": "must_provide"
                    },
                    {
                        "variants": ["owns", "has ownership of", "possesses all rights to", "holds exclusive rights to"],
                        "canonical": "owns"
                    },
                    {
                        "variants": ["cannot compete", "may not compete", "restricted from competing", "prohibited from competing"],
                        "canonical": "cannot_compete"
                    },
                    {
                        "variants": ["governed by law of", "subject to laws of", "interpreted under law of"],
                        "canonical": "governed_by"
                    }
                ],
                "legal_preservation": [
                    "Distinguish SHALL (mandatory) vs MAY (optional) vs SHOULD (recommended)",
                    "Preserve conditional relationships (if...then)",
                    "Maintain temporal sequence (before, after, during)",
                    "Respect scope modifiers (exclusive, non-exclusive, limited, unlimited)"
                ]
            },

            "output": {
                "resolved_graph": "Dense knowledge graph with canonical entities and relations",
                "resolution_statistics": {
                    "entities_before": "e.g., 15,000",
                    "entities_after": "e.g., 8,000 (47% reduction)",
                    "relations_before": "e.g., 800",
                    "relations_after": "e.g., 150 (81% reduction)",
                    "alias_mappings": "Dictionary of variant -> canonical"
                },
                "quality_metrics": {
                    "triple_validity": "Target: 98% (manual validation on sample)",
                    "cluster_quality": "Silhouette score for clustering",
                    "canonicalization_consistency": "Inter-rater agreement on canonical selection"
                }
            },

            "performance_targets": {
                "processing_time": "Resolve 50,000 triples in <1 hour",
                "llm_calls": "Optimize to minimize LLM API calls (batch processing)",
                "accuracy": "98% valid triples after resolution"
            }
        },

        "knowledge_graph_storage": {
            "database_options": [
                {
                    "name": "Neo4j",
                    "type": "Property Graph Database",
                    "pros": ["Native graph storage", "Cypher query language", "Excellent visualization", "Strong community"],
                    "cons": ["Commercial license for production", "Memory intensive"],
                    "use_case": "Production deployment with complex queries"
                },
                {
                    "name": "NetworkX + PostgreSQL",
                    "type": "Hybrid (graph library + relational DB)",
                    "pros": ["Python native", "Flexible", "Open source", "Good for prototyping"],
                    "cons": ["Not optimized for large graphs", "Query performance"],
                    "use_case": "Development and prototyping"
                },
                {
                    "name": "RDF Triple Store (e.g., GraphDB, Virtuoso)",
                    "type": "RDF/OWL semantic database",
                    "pros": ["Standards-based (RDF, OWL, SPARQL)", "Reasoning capabilities", "Ontology support"],
                    "cons": ["Steeper learning curve", "Different paradigm"],
                    "use_case": "If need semantic web standards or reasoning"
                }
            ],
            "recommended": "Neo4j for production (scalability, maturity) with NetworkX for development",

            "schema_implementation": {
                "node_labels": ["Party", "IPAsset", "Obligation", "Restriction", "LiabilityProvision", "Temporal", "Jurisdiction", "ContractClause", "Contract"],
                "relationship_types": ["LICENSES_TO", "OWNS", "ASSIGNS", "HAS_OBLIGATION", "SUBJECT_TO_RESTRICTION", "HAS_LIABILITY", "GOVERNED_BY", "CONTAINS_CLAUSE", "EFFECTIVE_ON", "TERMINATES_ON"],
                "properties": {
                    "all_nodes": ["id", "name", "type", "source_contract_id", "cuad_label", "confidence_score"],
                    "all_edges": ["id", "source", "target", "type", "properties", "source_contract_id", "confidence_score"]
                }
            },

            "indexing": {
                "full_text_search": "Index all node names and properties for fast text search",
                "semantic_search": "Store embeddings for all nodes (using S-BERT or all-MiniLM-L6-v2)",
                "cuad_label_index": "Index by CUAD category for fast filtering",
                "contract_id_index": "Index by source contract for provenance tracking"
            }
        },

        "llm_integration_layer": {
            "purpose": "Provide structured context to LLM for contract analysis queries",

            "query_processing": {
                "input": "Natural language question about contracts",
                "example_queries": [
                    "What IP rights does the licensee receive in contracts with ABC Corp?",
                    "Show me all non-compete restrictions longer than 2 years",
                    "What is the liability cap in the development agreement with XYZ Inc?",
                    "When does the license agreement expire?",
                    "Which contracts lack source code escrow provisions?"
                ],

                "retrieval_pipeline": {
                    "step_1_embed_query": {
                        "model": "all-MiniLM-L6-v2 (same as KGGen evaluation)",
                        "output": "Query embedding vector"
                    },
                    "step_2_initial_retrieval": {
                        "method": "Hybrid search: BM25 (keyword) + cosine similarity (semantic)",
                        "weighting": "Equal weight (0.5 BM25, 0.5 semantic) as per KGGen",
                        "top_k": 10,
                        "search_space": "All nodes and edges in knowledge graph",
                        "output": "Top-10 most relevant triples"
                    },
                    "step_3_subgraph_expansion": {
                        "method": "Expand to 2-hop neighbors",
                        "rationale": "Enable multi-hop reasoning (e.g., Party -> License -> IP Asset)",
                        "expansion_size": "+10 triples (total ~20 triples)",
                        "pruning": "Remove low-relevance nodes outside main subgraph"
                    },
                    "step_4_context_enrichment": {
                        "add_original_text": "Retrieve original contract text chunks for retrieved triples",
                        "add_cuad_labels": "Include CUAD label category for each triple",
                        "add_metadata": "Contract type, jurisdiction, dates, parties"
                    }
                },

                "context_formatting": {
                    "template": {
                        "structure": [
                            "# Contract Knowledge Graph Context",
                            "## Relevant Entities and Relationships",
                            "[List of triples in structured format]",
                            "## Original Contract Text",
                            "[Relevant text chunks from source contracts]",
                            "## Query",
                            "[User question]",
                            "## Instructions",
                            "Answer the query based on the knowledge graph and contract text. Be precise and cite specific clauses."
                        ]
                    },
                    "triple_format": "- (Subject) --[Relationship]--> (Object) [Properties: ...]",
                    "example": "- (ABC Corp) --[licenses_to]--> (XYZ Inc) [license_type: exclusive, scope: worldwide]"
                },

                "llm_generation": {
                    "model": "Claude Sonnet 3.5 (primary) or GPT-4o",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                    "system_prompt": "You are a legal contract analysis assistant. Provide accurate, precise answers based on the knowledge graph and contract text. Always cite specific clauses and explain your reasoning. If information is not in the context, say so.",
                    "output": "Natural language answer with citations"
                }
            },

            "api_interface": {
                "endpoints": [
                    {
                        "path": "/api/v1/query",
                        "method": "POST",
                        "input": {"query": "string", "contract_ids": ["optional", "list"], "filters": "optional"},
                        "output": {"answer": "string", "sources": ["list of triples"], "confidence": "float"}
                    },
                    {
                        "path": "/api/v1/graph/search",
                        "method": "POST",
                        "input": {"query": "string", "top_k": "int"},
                        "output": {"subgraph": "JSON graph", "triples": ["list"]}
                    },
                    {
                        "path": "/api/v1/graph/entity/{entity_id}",
                        "method": "GET",
                        "output": {"entity": "object", "relationships": ["list"], "contracts": ["list"]}
                    },
                    {
                        "path": "/api/v1/contracts/{contract_id}/graph",
                        "method": "GET",
                        "output": {"contract_id": "string", "graph": "JSON", "statistics": "object"}
                    }
                ],
                "authentication": "API key based",
                "rate_limiting": "100 requests/minute per key"
            }
        }
    },

    "data_requirements": {
        "input_data": {
            "source": "CUAD Dataset",
            "url": "atticusprojectai.org/cuad",
            "format": "PDF contracts + JSON annotations",
            "size": "510 contracts, 13,101 annotations, 41 label categories",
            "license": "Open source (research and commercial use)",
            "download": "github.com/TheAtticusProject/cuad/"
        },

        "preprocessing": {
            "pdf_to_text": "Extract text from PDFs (use pdfplumber or PyPDF2)",
            "section_identification": "Identify contract sections (parties, terms, exhibits)",
            "annotation_alignment": "Map CUAD annotations to extracted text",
            "quality_checks": [
                "Verify all 510 contracts can be processed",
                "Check annotation coverage (all 41 categories present)",
                "Validate text extraction quality"
            ]
        },

        "technology_agreement_filtering": {
            "rationale": "Focus on technology-relevant contract types",
            "filter_criteria": "Select contracts related to software, IP, licensing, development, services",
            "included_types": [
                "License Agreement (33 contracts)",
                "Development Agreement (29 contracts)",
                "IP Agreement (17 contracts)",
                "Service Agreement (42 contracts)",
                "Hosting Agreement (20 contracts)",
                "Maintenance Agreement (34 contracts)",
                "Consulting Agreement (11 contracts)"
            ],
            "estimated_subset": "~200 technology-focused contracts from CUAD"
        },

        "storage_requirements": {
            "raw_data": "~500 MB (PDF contracts)",
            "extracted_text": "~100 MB (text format)",
            "knowledge_graph": "~1-2 GB (Neo4j database with 50K-100K triples)",
            "embeddings": "~500 MB (S-BERT embeddings for all nodes)",
            "total": "~3-4 GB for complete system"
        }
    },

    "technical_specifications": {
        "development_stack": {
            "primary_language": "Python 3.11+",
            "llm_orchestration": "DSPy (Declarative Self-improving Python)",
            "llm_providers": {
                "primary": "Claude Sonnet 3.5 (Anthropic)",
                "fallback": "GPT-4o (OpenAI)",
                "alternative": "Gemini 2.0 Flash (Google)"
            },
            "embedding_models": {
                "clustering": "S-BERT (Sentence-BERT)",
                "retrieval": "all-MiniLM-L6-v2 (SentenceTransformers)",
                "storage": "FAISS for vector similarity search"
            },
            "graph_database": "Neo4j 5.x (primary) or NetworkX + PostgreSQL (development)",
            "search_engine": "Elasticsearch with BM25 or custom implementation",
            "web_framework": "FastAPI (REST API)",
            "background_jobs": "Celery + Redis (for async processing)",
            "monitoring": "Prometheus + Grafana"
        },

        "key_libraries": {
            "llm_interaction": ["dspy-ai", "anthropic", "openai"],
            "nlp": ["sentence-transformers", "spacy", "transformers"],
            "graph": ["neo4j", "networkx", "py2neo"],
            "search": ["rank-bm25", "elasticsearch-py", "faiss-cpu"],
            "pdf_processing": ["pdfplumber", "PyPDF2", "pdf2image"],
            "data_processing": ["pandas", "numpy", "pydantic"],
            "testing": ["pytest", "pytest-asyncio", "hypothesis"]
        },

        "infrastructure": {
            "compute": {
                "extraction": "GPU optional (for large-scale processing), CPU sufficient for small batches",
                "resolution": "CPU-intensive (clustering), can parallelize",
                "query": "Low latency (<500ms), CPU sufficient"
            },
            "storage": {
                "contracts": "S3 or local file system",
                "knowledge_graph": "Neo4j instance (4-8 GB RAM recommended)",
                "embeddings": "FAISS index (in-memory or mmap)",
                "cache": "Redis for query caching"
            },
            "deployment": {
                "options": ["Docker containers", "Kubernetes", "Cloud VM"],
                "scaling": "Horizontal (multiple workers for extraction/resolution)",
                "monitoring": "Health checks, performance metrics, error tracking"
            }
        },

        "performance_targets": {
            "extraction": "1-2 contracts/minute per worker",
            "aggregation": "510 contracts in <10 minutes",
            "resolution": "50K triples in <1 hour",
            "query_latency": "<500ms for simple queries, <2s for complex",
            "throughput": "100+ queries/second",
            "availability": "99.9% uptime for production"
        },

        "quality_assurance": {
            "unit_tests": "90%+ code coverage",
            "integration_tests": "End-to-end pipeline testing",
            "validation": {
                "triple_validity": "Manual review of 100 random triples, target 98% valid",
                "mine_1_score": "Evaluate on MINE-1 benchmark, target 65%+",
                "entity_extraction": "Compare against CUAD annotations, target 95%+ accuracy",
                "resolution_quality": "Manual review of entity clusters, target 95%+ correct"
            },
            "continuous_monitoring": {
                "metrics": ["Extraction errors", "LLM API failures", "Query latency", "User satisfaction"],
                "alerting": "Slack/email for critical failures",
                "logging": "Structured logging for all operations"
            }
        }
    },

    "implementation_roadmap": {
        "phase_1_prototype": {
            "duration": "4-6 weeks",
            "goals": "Validate KGGen on CUAD, build minimal viable pipeline",
            "milestones": [
                "Week 1: Set up development environment, download CUAD, explore data",
                "Week 2: Implement Stage 1 extraction for 10 sample contracts",
                "Week 3: Implement Stage 2 aggregation and Stage 3 resolution (basic)",
                "Week 4: Build simple query interface, test on sample questions",
                "Week 5: Evaluate on MINE-1 benchmark subset, measure quality",
                "Week 6: Iterate on prompts and resolution logic based on evaluation"
            ],
            "deliverables": [
                "Working extraction pipeline for CUAD contracts",
                "Knowledge graph with 10-50 contracts",
                "Basic query interface (CLI or simple web UI)",
                "Quality metrics report (triple validity, MINE-1 score)"
            ],
            "success_criteria": [
                "Extract knowledge graphs from 50+ contracts",
                "Achieve 90%+ triple validity",
                "Answer 10 sample queries correctly",
                "Demonstrate value to stakeholders"
            ]
        },

        "phase_2_scale_and_optimize": {
            "duration": "6-8 weeks",
            "goals": "Process full CUAD dataset, optimize performance, enhance quality",
            "milestones": [
                "Week 1-2: Scale extraction to all 510 CUAD contracts",
                "Week 3-4: Optimize resolution algorithm (parallelization, caching)",
                "Week 5: Implement advanced query features (filters, aggregations)",
                "Week 6: Build REST API with FastAPI",
                "Week 7: Conduct comprehensive evaluation (MINE-1, MINE-2, quality review)",
                "Week 8: Performance tuning and bug fixes"
            ],
            "deliverables": [
                "Complete knowledge graph (510 contracts, 50K-100K triples)",
                "Production-grade REST API",
                "Comprehensive evaluation report",
                "Performance benchmarks and optimization recommendations"
            ],
            "success_criteria": [
                "Process all 510 contracts successfully",
                "Achieve 98% triple validity",
                "MINE-1 score 65%+",
                "Query latency <500ms for 90% of queries",
                "API throughput 100+ requests/second"
            ]
        },

        "phase_3_product_integration": {
            "duration": "4-6 weeks",
            "goals": "Integrate with user-facing application, deploy to production",
            "milestones": [
                "Week 1: Design user interface for contract analysis tool",
                "Week 2: Implement contract Q&A feature with LLM integration",
                "Week 3: Build risk analysis dashboard",
                "Week 4: Implement contract comparison and compliance features",
                "Week 5: User testing with legal practitioners",
                "Week 6: Production deployment and monitoring setup"
            ],
            "deliverables": [
                "Web application for contract analysis",
                "User documentation and tutorials",
                "Deployment guide for production",
                "Monitoring and alerting system"
            ],
            "success_criteria": [
                "Positive user feedback from legal practitioners",
                "Demonstrate 50% time savings in contract review",
                "Successful production deployment",
                "99.9% uptime in first month"
            ]
        },

        "phase_4_enhancement_and_expansion": {
            "duration": "Ongoing",
            "goals": "Expand ontology, add features, improve accuracy",
            "areas": [
                "Ontology expansion: Add more contract types, clause categories",
                "Common law reasoning: Incorporate case law and legal precedents",
                "Multi-jurisdiction support: Extend beyond US to UK, Commonwealth",
                "Advanced analytics: Trend analysis, risk scoring, negotiation insights",
                "Integration: Connect to document management systems, CRMs",
                "Model improvement: Fine-tune LLMs on legal contracts",
                "User feedback: Iterate based on real-world usage"
            ]
        }
    },

    "risks_and_mitigations": {
        "technical_risks": [
            {
                "risk": "LLM hallucination: Extracting incorrect entities/relations",
                "impact": "High - reduces trust in system",
                "likelihood": "Medium",
                "mitigation": [
                    "Use high-quality models (Claude Sonnet 3.5, 73% MINE-1)",
                    "Strong prompt constraints to prevent incorrect grouping",
                    "Confidence scores for all extractions",
                    "Human review of low-confidence triples",
                    "Continuous evaluation and prompt refinement"
                ]
            },
            {
                "risk": "Resolution errors: Merging legally distinct entities",
                "impact": "High - legal errors",
                "likelihood": "Medium",
                "mitigation": [
                    "Conservative resolution approach",
                    "LLM prompted with legal constraints",
                    "Manual review of resolution examples",
                    "Maintain aliases for transparency",
                    "Lawyer validation of ontology"
                ]
            },
            {
                "risk": "Scalability: System too slow for large contract portfolios",
                "impact": "Medium - limits use cases",
                "likelihood": "Low",
                "mitigation": [
                    "Parallel processing of contracts",
                    "Incremental graph updates",
                    "Caching of frequent queries",
                    "Optimize resolution algorithm",
                    "Use efficient graph database"
                ]
            },
            {
                "risk": "LLM API costs: High cost for large-scale processing",
                "impact": "Medium - budget constraints",
                "likelihood": "Medium",
                "mitigation": [
                    "Batch processing to reduce API calls",
                    "Cache LLM outputs",
                    "Use cheaper models for non-critical tasks",
                    "Optimize prompts to reduce token usage",
                    "Consider self-hosted models for extraction"
                ]
            }
        ],

        "legal_risks": [
            {
                "risk": "Incorrect legal advice: System provides wrong interpretation",
                "impact": "Critical - liability",
                "likelihood": "Medium",
                "mitigation": [
                    "Disclaimer: System is assistive tool, not legal advice",
                    "Lawyer review of outputs",
                    "Confidence scores and uncertainty quantification",
                    "Explain reasoning with citations",
                    "Insurance for E&O (Errors and Omissions)"
                ]
            },
            {
                "risk": "Confidentiality breach: Exposure of contract information",
                "impact": "Critical - legal/financial",
                "likelihood": "Low",
                "mitigation": [
                    "Secure storage with encryption",
                    "Access control and authentication",
                    "Audit logs for all data access",
                    "Compliance with data protection regulations",
                    "Self-hosted option for sensitive contracts"
                ]
            },
            {
                "risk": "Jurisdictional issues: System not applicable outside common law",
                "impact": "Medium - limits market",
                "likelihood": "High",
                "mitigation": [
                    "Clear scope: Focus on US/UK/Commonwealth common law",
                    "Jurisdiction tagging in knowledge graph",
                    "Future expansion to civil law systems",
                    "Partner with local legal experts"
                ]
            }
        ],

        "business_risks": [
            {
                "risk": "User adoption: Lawyers reluctant to use AI tools",
                "impact": "High - product failure",
                "likelihood": "Medium",
                "mitigation": [
                    "Involve lawyers in design and testing",
                    "Emphasize augmentation, not replacement",
                    "Transparent, explainable system",
                    "Strong accuracy and reliability",
                    "Education and training materials"
                ]
            },
            {
                "risk": "Competition: Other legal AI tools in market",
                "impact": "Medium - market share",
                "likelihood": "High",
                "mitigation": [
                    "Unique value: Structured knowledge graph, not just text search",
                    "Common law specialization",
                    "Technology contract focus",
                    "Open source components to build community",
                    "Partnerships with law firms and legal tech companies"
                ]
            }
        ]
    },

    "success_metrics": {
        "technical_metrics": [
            {"metric": "Triple validity", "target": "98%", "measurement": "Manual review of 100 random triples"},
            {"metric": "MINE-1 score", "target": "65%+", "measurement": "Run KGGen MINE-1 benchmark"},
            {"metric": "Entity extraction accuracy", "target": "95%+", "measurement": "Compare against CUAD annotations"},
            {"metric": "Query latency", "target": "<500ms", "measurement": "P95 latency for sample queries"},
            {"metric": "Extraction throughput", "target": "1-2 contracts/min", "measurement": "Avg contracts processed per minute"},
            {"metric": "Graph density", "target": "Reduce unique relations by 80%", "measurement": "Before/after resolution comparison"}
        ],

        "business_metrics": [
            {"metric": "Time savings", "target": "50% reduction in contract review time", "measurement": "User studies with lawyers"},
            {"metric": "Query accuracy", "target": "90%+ correct answers", "measurement": "Human evaluation of sample queries"},
            {"metric": "User satisfaction", "target": "4/5+ rating", "measurement": "User surveys"},
            {"metric": "Contract coverage", "target": "Process 510 CUAD contracts", "measurement": "System logs"},
            {"metric": "Feature usage", "target": "All 5 use cases demonstrated", "measurement": "Case studies"}
        ],

        "legal_metrics": [
            {"metric": "Lawyer validation", "target": "95%+ agreement on outputs", "measurement": "Lawyer review of system outputs"},
            {"metric": "Error rate", "target": "<2% critical errors", "measurement": "Track errors reported by users"},
            {"metric": "Compliance", "target": "100% with data protection regulations", "measurement": "Compliance audit"}
        ]
    },

    "appendices": {
        "appendix_a_cuad_labels": {
            "description": "Complete list of 41 CUAD label categories",
            "source": "CUAD paper supplementary materials",
            "categories": [
                "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
                "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law", "Most Favored Nation",
                "Non-Compete", "Exclusivity", "No-Solicit Of Customers", "No-Solicit Of Employees",
                "Non-Disparagement", "Termination For Convenience", "Rofr/Rofo/Rofn", "Change Of Control",
                "Anti-Assignment", "Revenue/Profit Sharing", "Cap On Liability", "Uncapped Liability",
                "Liquidated Damages", "Warranty Duration", "Insurance", "Covenant Not To Sue",
                "Third Party Beneficiary", "Irrevocable Or Perpetual License", "Source Code Escrow",
                "Post-Termination Services", "Audit Rights", "Volume Restriction", "IP Ownership Assignment",
                "Joint IP Ownership", "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
                "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License", "Minimum Commitment",
                "Competitive Restriction Exception", "Price Restrictions"
            ]
        },

        "appendix_b_common_law_principles": {
            "description": "Key common law contract interpretation principles",
            "principles": [
                {
                    "name": "Plain Meaning Rule (Literal Interpretation)",
                    "description": "Terms interpreted by their ordinary, plain meaning as understood by reasonable person",
                    "application": "Canonical entity names should use clear, plain legal terminology"
                },
                {
                    "name": "Contra Proferentem (Against the Drafter)",
                    "description": "Ambiguous terms construed against party who drafted the contract",
                    "application": "Flag ambiguous terms in knowledge graph for human review"
                },
                {
                    "name": "Business Efficacy Test",
                    "description": "Interpret contract to give commercial effect and make business sense",
                    "application": "Relations should reflect business relationships, not just textual patterns"
                },
                {
                    "name": "Entire Agreement Clause",
                    "description": "Contract is complete expression of parties' agreement, excludes external evidence",
                    "application": "Extract from full contract text, don't rely on external sources"
                },
                {
                    "name": "Contextual Interpretation (Purposive Approach)",
                    "description": "Consider contract as whole, commercial context, and parties' intentions",
                    "application": "Use full contract context for entity/relation extraction, not isolated clauses"
                }
            ]
        },

        "appendix_c_example_queries": {
            "description": "Example natural language queries and expected knowledge graph responses",
            "examples": [
                {
                    "query": "What IP rights does the licensee receive in the software license agreement?",
                    "relevant_triples": [
                        "(ABC Corp, licenses_to, XYZ Inc)",
                        "(License, is_non_exclusive, True)",
                        "(License, scope, worldwide)",
                        "(License, includes_right_to, use)",
                        "(License, includes_right_to, modify)",
                        "(XYZ Inc, cannot_transfer_license_to, third_party)"
                    ],
                    "answer": "The licensee (XYZ Inc) receives a non-exclusive, worldwide license to use and modify the software. However, the license is non-transferable and cannot be sublicensed to third parties."
                },
                {
                    "query": "Are there any non-compete restrictions, and if so, what is the scope?",
                    "relevant_triples": [
                        "(ABC Corp, subject_to_restriction, Non-Compete)",
                        "(Non-Compete, scope, same_market)",
                        "(Non-Compete, duration, 2_years)",
                        "(Non-Compete, geography, California)",
                        "(Non-Compete, effective_from, termination_date)"
                    ],
                    "answer": "Yes, ABC Corp is subject to a non-compete restriction. The restriction prohibits competing in the same market within California for 2 years following termination of the agreement."
                },
                {
                    "query": "What is the liability cap in this agreement?",
                    "relevant_triples": [
                        "(Agreement, has_liability_provision, Cap On Liability)",
                        "(Cap On Liability, amount, $1,000,000)",
                        "(Cap On Liability, scope, direct_damages)",
                        "(Cap On Liability, exceptions, [IP_infringement, willful_misconduct])"
                    ],
                    "answer": "The agreement includes a liability cap of $1,000,000 for direct damages. However, this cap does not apply to IP infringement claims or willful misconduct, which remain uncapped."
                }
            ]
        },

        "appendix_d_technology_stack": {
            "description": "Detailed technology stack with versions and alternatives",
            "components": [
                {
                    "category": "LLM Providers",
                    "primary": "Anthropic Claude Sonnet 3.5 (claude-3-5-sonnet-20241022)",
                    "alternatives": ["OpenAI GPT-4o (gpt-4o-2024-11-20)", "Google Gemini 2.0 Flash"],
                    "rationale": "Claude Sonnet 3.5 achieves 73% MINE-1 score, best for legal reasoning"
                },
                {
                    "category": "LLM Orchestration",
                    "primary": "DSPy (dspy-ai >= 2.0)",
                    "alternatives": ["LangChain", "Raw API calls"],
                    "rationale": "DSPy provides structured output, signatures, and easy multi-model support"
                },
                {
                    "category": "Embedding Models",
                    "primary": "all-MiniLM-L6-v2 (sentence-transformers)",
                    "alternatives": ["OpenAI text-embedding-3-small", "Cohere embed-english-v3.0"],
                    "rationale": "Matches KGGen benchmark, fast, good quality, open source"
                },
                {
                    "category": "Graph Database",
                    "primary": "Neo4j 5.x Community Edition",
                    "alternatives": ["ArangoDB", "Amazon Neptune", "NetworkX + PostgreSQL"],
                    "rationale": "Industry standard, Cypher query language, excellent tooling"
                },
                {
                    "category": "Search Engine",
                    "primary": "rank-bm25 (Python library) + FAISS",
                    "alternatives": ["Elasticsearch", "Meilisearch"],
                    "rationale": "Lightweight, no infrastructure overhead, sufficient for prototype"
                },
                {
                    "category": "Web Framework",
                    "primary": "FastAPI 0.109+",
                    "alternatives": ["Flask", "Django REST Framework"],
                    "rationale": "Fast, async support, automatic API documentation, modern"
                },
                {
                    "category": "Background Jobs",
                    "primary": "Celery + Redis",
                    "alternatives": ["RQ (Redis Queue)", "Dramatiq"],
                    "rationale": "Mature, scalable, good for long-running extraction tasks"
                }
            ]
        }
    }
}

# Save PRD structure
output_path = Path("/app/sandbox/session_20260112_140312_4731309a153b/data/prd_structure.json")
with open(output_path, 'w') as f:
    json.dump(prd_structure, f, indent=2)

print(f"✓ Comprehensive PRD structure saved to: {output_path}")
print(f"✓ Generated {len(prd_structure)} main sections")
print(f"✓ Technical architecture: {len(prd_structure['technical_architecture'])} components")
print(f"✓ Implementation roadmap: {len(prd_structure['implementation_roadmap'])} phases")
print(f"✓ Success metrics: {len(prd_structure['success_metrics']['technical_metrics'])} technical + {len(prd_structure['success_metrics']['business_metrics'])} business")
print(f"✓ Risks documented: {len(prd_structure['risks_and_mitigations']['technical_risks'])} technical + {len(prd_structure['risks_and_mitigations']['legal_risks'])} legal")
print(f"✓ File size: ~{output_path.stat().st_size / 1024:.1f} KB")
