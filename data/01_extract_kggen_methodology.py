#!/usr/bin/env python3
"""
Extract and structure KGGen methodology components for PRD development.
This script analyzes the KGGen paper and extracts key technical components.
"""

import json
from pathlib import Path

# KGGen Methodology Analysis
kggen_analysis = {
    "system_name": "KGGen: Knowledge Graph Generator",
    "purpose": "Extract high-quality knowledge graphs from plain text using language models",
    "authors": "Stanford University, University of Toronto, FAR AI",
    "publication": "NeurIPS 2025",

    "core_innovation": {
        "title": "Multi-stage extraction with entity resolution",
        "description": "Unlike other KG generators, KGGen clusters and de-duplicates related entities to reduce sparsity in extracted KGs",
        "key_differentiator": "Entity and edge resolution prevents formation of sparse KGs with unique, non-reusable nodes and edges"
    },

    "pipeline_stages": {
        "stage_1": {
            "name": "Entity and Relation Extraction",
            "description": "Takes unstructured text as input and produces initial knowledge graph as extracted triples",
            "implementation": {
                "model": "Google Gemini 2.0 Flash (also works with Claude Sonnet 3.5, GPT-4o)",
                "framework": "DSPy signatures for structured output",
                "steps": [
                    "Extract list of entities from source text",
                    "Given entities, extract subject-predicate-object relations"
                ],
                "rationale": "2-step approach ensures consistency between entities"
            },
            "output": "Initial knowledge graph with (subject, predicate, object) triples"
        },

        "stage_2": {
            "name": "Aggregation",
            "description": "Collect all unique entities and edges across all source graphs",
            "implementation": {
                "method": "Combine into single graph",
                "normalization": "All entities and edges normalized to lowercase",
                "llm_required": False
            },
            "purpose": "Reduce redundancy in the KG"
        },

        "stage_3": {
            "name": "Entity and Edge Resolution",
            "description": "Merge nodes and edges representing the same real-world entity or concept",
            "implementation": {
                "approach": "Two-stage: embedding-based clustering + LLM-based de-duplication",
                "clustering": {
                    "embedding_model": "S-BERT",
                    "clustering_algorithm": "k-means",
                    "cluster_size": 128
                },
                "deduplication": {
                    "retrieval": "Top-k (k=16) most semantically similar items using fused BM25 and semantic embedding",
                    "llm_task": "Identify exact duplicates considering tense, plurality, case, abbreviations, shorthand",
                    "canonicalization": "LLM selects canonical representative (like Wikidata aliases)",
                    "iteration": "Remove processed items and repeat until cluster empty"
                }
            },
            "scalability": "Process semantic clusters in parallel for large KGs",
            "example": "Consolidates 'Olympic Winter Games', 'Winter Olympics', 'winter Olympic games' into single representation"
        }
    },

    "technical_specifications": {
        "prompting_strategy": {
            "constraint_type": "Strong constraints via prompting",
            "purpose": "Prevent incorrect grouping of similar but distinct entities",
            "examples": "Avoid conflating 'Type 1 diabetes' and 'Type 2 diabetes', 'hypertension' and 'stress', 'MRI' and 'CT scan'"
        },

        "embedding_models": {
            "semantic_embedding": "S-BERT (Sentence-BERT)",
            "retrieval": "all-MiniLM-L6-v2 from SentenceTransformers",
            "similarity_scoring": "Fused BM25 + cosine similarity"
        },

        "llm_models_tested": {
            "claude_sonnet_35": {"mine1_score": "73%", "performance": "Highest"},
            "gpt_4o": {"mine1_score": "66%", "performance": "High"},
            "gemini_2_flash": {"mine1_score": "44%", "performance": "Baseline"}
        }
    },

    "benchmarks": {
        "mine_1": {
            "name": "MINE-1: Knowledge Retention",
            "purpose": "Measure fraction of information captured from articles",
            "dataset": "100 articles, 15 facts each (mean 592 words)",
            "evaluation": "Binary scoring: fact can be inferred from KG subgraph (1) or not (0)",
            "kggen_performance": "66.07% average (Claude: 73%, GPT-4o: 66%, Gemini: 44%)",
            "comparison": "Outperforms GraphRAG (47.80%) and OpenIE (29.84%)"
        },

        "mine_2": {
            "name": "MINE-2: KG-Assisted RAG",
            "purpose": "Measure downstream RAG performance on multi-million token datasets",
            "dataset": "WikiQA: 20,400 questions from 1,995 Wikipedia articles",
            "evaluation": "LLM-as-Judge determines if answer contains correct response",
            "kggen_performance": "Comparable to GraphRAG"
        },

        "semeval_2010": {
            "dataset": "100 randomly selected sentences with manually labeled entities",
            "entity_extraction_accuracy": "96% (96/100)",
            "advantage": "Extracts more detailed entity descriptions than human annotations"
        }
    },

    "quality_advantages": {
        "triple_validity": "98% valid triples (vs GraphRAG 0%, OpenIE 55%)",
        "node_quality": "Informative, coherent, captures critical relationships",
        "edge_quality": "Concise relation types that generalize easily",
        "scalability": "Better scaling with respect to text source size",
        "sparsity_reduction": "Effective de-duplication reduces graph sparsity"
    },

    "implementation_details": {
        "code_availability": "Open-source at https://github.com/stair-lab/kg-gen/",
        "python_library": "Available as Python package",
        "framework": "DSPy for LLM orchestration",
        "parallelization": "Cluster processing can be parallelized"
    }
}

# Save structured analysis
output_path = Path("/app/sandbox/session_20260112_140312_4731309a153b/data/kggen_methodology_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(kggen_analysis, f, indent=2)

print(f"✓ KGGen methodology analysis saved to: {output_path}")
print(f"✓ Extracted {len(kggen_analysis['pipeline_stages'])} pipeline stages")
print(f"✓ Documented {len(kggen_analysis['benchmarks'])} benchmarks")
print(f"✓ Analyzed {len(kggen_analysis['technical_specifications']['llm_models_tested'])} LLM models")
