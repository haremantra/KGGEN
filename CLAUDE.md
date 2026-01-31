# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KGGEN-CUAD is a Knowledge Graph Generator that applies the KGGen methodology to the CUAD (Contract Understanding Atticus Dataset) for legal contract analysis. The system extracts structured knowledge graphs from legal contracts to enable context-aware LLM analysis, specializing in technology agreements within common law jurisdictions.

**Current Status:** Planning/PRD phase - workflow scripts and analysis outputs exist, implementation not yet started.

## Project Structure

```
KGGEN/
├── workflow/                    # Analysis and planning scripts
│   ├── 01_extract_kggen_methodology.py
│   ├── 02_extract_cuad_structure.py
│   ├── 03_map_kggen_to_cuad.py
│   ├── 04_generate_prd_structure.py
│   └── 05_create_architecture_diagram.py
├── data/                        # Structured analysis outputs (JSON)
│   ├── kggen_methodology_analysis.json
│   ├── cuad_dataset_analysis.json
│   ├── tech_agreement_ontology.json
│   ├── kggen_cuad_mapping.json
│   └── prd_structure.json       # Complete PRD structure (48KB)
├── figures/                     # Architecture diagrams
├── converted_md/                # Converted markdown from source PDFs
├── COMMERCIALIZATION/           # Business planning documents
└── writing_outputs/             # Generated documentation
```

## Key Reference Documents

- **README.md** - Comprehensive project documentation and analysis summary
- **data/prd_structure.json** - Complete PRD with technical specifications
- **data/kggen_cuad_mapping.json** - Architecture mapping of KGGen to CUAD domain
- **figures/system_architecture_diagram.png** - Visual system architecture

## Architecture Overview

The system implements a 3-stage pipeline:

1. **Extraction** - LLM-based entity and relation extraction from contracts using DSPy
2. **Aggregation** - Cross-contract normalization and deduplication
3. **Resolution** - Entity clustering and canonicalization using S-BERT + k-means

**Target Stack (not yet implemented):**
- Python 3.11+, DSPy, FastAPI, Celery
- Claude Sonnet 3.5 / GPT-4o for LLM
- Neo4j (graph), PostgreSQL, Qdrant (vectors), Elasticsearch, Redis
- React frontend with TypeScript

## Development Commands

```bash
# Run workflow scripts (Python 3.12 required per pyproject.toml)
python workflow/01_extract_kggen_methodology.py
python workflow/02_extract_cuad_structure.py
python workflow/03_map_kggen_to_cuad.py
python workflow/04_generate_prd_structure.py
python workflow/05_create_architecture_diagram.py
```

## Knowledge Graph Schema

**Node Types (8):** Party, IPAsset, Obligation, Restriction, LiabilityProvision, Temporal, Jurisdiction, ContractClause

**Edge Types (10):** LICENSES_TO, OWNS, ASSIGNS, HAS_OBLIGATION, SUBJECT_TO_RESTRICTION, HAS_LIABILITY, GOVERNED_BY, CONTAINS_CLAUSE, EFFECTIVE_ON, TERMINATES_ON

## Performance Targets

| Metric | Target |
|--------|--------|
| Triple Validity | 98% |
| MINE-1 Score | 65%+ |
| Entity Extraction Accuracy | 95%+ |
| Query Latency (P95) | <500ms |
| Extraction Throughput | 1-2 contracts/min |

## CUAD Dataset

- 510 contracts, 13,101 annotations, 41 label categories
- ~200 contracts relevant to technology agreements
- Categories: General Information, Restrictive Covenants, Revenue Risks, Intellectual Property, Special Provisions

## Implementation Notes

When implementing the planned `src/` directory structure:
- Follow the 8-phase dependency order in data/prd_structure.json
- Use code templates from the PRD for service classes, routes, and tests
- Apply cross-cutting concerns: telemetry, logging, correlation IDs, auth, rate limiting, caching, circuit breakers
- Run validation at each checkpoint gate before proceeding to next phase
