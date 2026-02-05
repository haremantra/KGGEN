# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KGGEN-CUAD is a Knowledge Graph Generator and Contract Analysis system that applies the KGGen methodology to the CUAD (Contract Understanding Atticus Dataset) for legal contract analysis. The system extracts structured knowledge graphs from legal contracts, performs risk assessment, and enables portfolio-level analysis for technology agreements within common law jurisdictions.

**Current Status:** Full merge complete — Risk assessment, portfolio analysis, clause interdependency, entity resolution, hybrid search, RAG query, REST API, and web dashboard (7 pages).

## Project Structure

```
KGGEN/
├── src/
│   ├── api/                    # FastAPI REST API
│   │   ├── app.py              # Application factory with CORS, lifespan
│   │   └── routes.py           # Contract & portfolio endpoints
│   ├── classification/         # CUAD clause classification
│   │   ├── classifier.py       # Semantic similarity classifier
│   │   └── cuad_labels.py      # 41 CUAD label definitions
│   ├── extraction/             # Entity/relationship extraction
│   │   └── extractor.py        # LLM-based KG extraction
│   ├── risk/                   # Risk assessment engine
│   │   ├── rules.py            # Risk rules for all CUAD categories
│   │   └── assessor.py         # Hybrid rule + LLM risk scoring
│   ├── interdependency/         # Clause interdependency analysis
│   │   ├── types.py             # DependencyType, ClauseNode, DependencyEdge, etc.
│   │   ├── matrix.py            # 73 static dependency rules between CUAD labels
│   │   ├── detector.py          # Rule matching + LLM validation
│   │   ├── graph.py             # NetworkX graph builder + algorithms
│   │   └── analyzer.py          # Orchestrator producing InterdependencyReport
│   ├── resolution/             # Entity resolution
│   │   ├── resolver.py         # Adaptive k-means + LLM canonical selection
│   │   └── __init__.py         # Bridge: analysis_to_entities_triples()
│   ├── aggregation/            # Cross-contract aggregation
│   │   ├── aggregator.py       # Per-entity-type dedup thresholds
│   │   └── __init__.py
│   ├── search/                 # Hybrid search
│   │   ├── service.py          # BM25 + semantic with RRF fusion
│   │   └── backends.py         # InMemory (size guards) + Qdrant backends
│   ├── query/                  # RAG query service
│   │   └── service.py          # Context retrieval → LLM answer generation
│   ├── portfolio/              # Portfolio-level analysis
│   │   └── analyzer.py         # Cross-contract comparison & gaps
│   ├── utils/                  # Utilities
│   │   ├── pdf_reader.py       # PDF text extraction
│   │   ├── neo4j_store.py      # Graph database storage
│   │   ├── embedding.py        # S-BERT embedding service with caching
│   │   └── llm.py              # LLM service with retry + fallback
│   ├── config.py               # Pydantic settings from .env
│   ├── pipeline.py             # Integrated analysis pipeline
│   └── main.py                 # CLI entry point
├── scripts/
│   ├── ralph.py                # RALPH loop task orchestrator
│   └── ralph_manifest.yaml     # 13-task merge manifest
├── streamlit_app.py            # Web dashboard (7 pages)
├── docker-compose.yml          # Full stack: API, Streamlit, Neo4j, etc.
├── Dockerfile                  # Python 3.11 container
├── pyproject.toml              # Dependencies
├── data/                       # Sample contracts & analysis outputs
└── workflow/                   # Original PRD generation scripts
```

## Key Commands

```bash
# Start API server
python -m src.main serve --port 8000 --reload

# Start Streamlit dashboard
streamlit run streamlit_app.py

# Analyze a contract
python -m src.main analyze <pdf_path> -o results.json

# Risk assessment only
python -m src.main risks <pdf_path> --no-llm

# Portfolio analysis
python -m src.main portfolio <folder> --limit 10 -o portfolio.json

# Clause interdependency analysis
python -m src.main dependencies <pdf_path> --no-llm

# Compare two contracts
python -m src.main compare contract1.pdf contract2.pdf

# Entity resolution
python -m src.main resolve <pdf_path> --no-llm

# Search knowledge graph
python -m src.main search "liability cap"

# Query (RAG-based Q&A)
python -m src.main query "What are the licensing terms?"

# RALPH merge tracker
python scripts/ralph.py --report

# Docker deployment
docker-compose up -d
```

## Architecture Patterns

### Risk Assessment (Hybrid Approach)
- **Rule-based**: Fast, deterministic scoring for known patterns (41 rules)
- **LLM-based**: Claude analysis for complex/ambiguous clauses (flagged with `requires_llm=True`)
- Risk score 0-100 maps to levels: LOW (0-24), MEDIUM (25-49), HIGH (50-74), CRITICAL (75-100)

### API Design
- FastAPI with async endpoints
- In-memory storage for demo (replace with PostgreSQL for production)
- Lazy-loaded pipeline and assessor singletons
- CORS configured for Streamlit frontend

### Pipeline Flow
1. **Classification**: Semantic similarity against 41 CUAD label embeddings
2. **Extraction**: LLM extracts entities, values, relationships from high-confidence clauses
3. **Risk Assessment**: Rule matching + optional LLM analysis
4. **Interdependency**: 73 static rules + optional LLM validation → NetworkX graph → impact/contradiction/cycle analysis → risk score adjustment (capped +40)
5. **Resolution** (optional): S-BERT embeddings → adaptive k-means with silhouette validation → LLM canonical selection
6. **Search** (optional): Index entities/triples → hybrid BM25 + semantic search → RRF fusion
7. **Query** (optional): RAG pipeline — search context retrieval → LLM answer generation
8. **Portfolio**: Aggregation, gap analysis, cross-contract comparison

## Code Conventions

- Pydantic models for API request/response schemas
- Dataclasses for internal data structures (ContractAnalysis, RiskAssessment)
- Type hints throughout
- `to_dict()` methods for JSON serialization
- Error handling with HTTPException in routes

## Environment Variables

Required:
- `ANTHROPIC_API_KEY` - For Claude LLM features

Optional (for full stack):
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `POSTGRES_*` settings
- `REDIS_URL`
- `QDRANT_HOST`, `QDRANT_PORT`

## Testing

```bash
# API health check
curl http://localhost:8000/health

# Upload and analyze
curl -X POST http://localhost:8000/api/contracts/upload \
  -F "file=@contract.pdf"

# Get risks
curl http://localhost:8000/api/contracts/{id}/risks

# Get dependencies
curl http://localhost:8000/api/contracts/{id}/dependencies

# Get contradictions
curl http://localhost:8000/api/contracts/{id}/contradictions

# Get completeness (missing requirements)
curl http://localhost:8000/api/contracts/{id}/completeness

# Impact analysis for a clause
curl http://localhost:8000/api/contracts/{id}/impact/License%20Grant?max_hops=3

# Search entities
curl -X POST http://localhost:8000/api/search/entities \
  -H "Content-Type: application/json" -d '{"query": "liability cap"}'

# Search triples
curl -X POST http://localhost:8000/api/search/triples \
  -H "Content-Type: application/json" -d '{"query": "license grant"}'

# Query (RAG)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" -d '{"question": "What are the licensing terms?"}'
```

## Key Files to Understand

1. **src/risk/rules.py** - Risk rule definitions for all 41 CUAD categories
2. **src/risk/assessor.py** - Hybrid scoring engine with LLM integration
3. **src/portfolio/analyzer.py** - Cross-contract analysis logic
4. **src/api/routes.py** - All REST endpoints
5. **streamlit_app.py** - Dashboard UI implementation
6. **src/pipeline.py** - Core analysis pipeline
7. **src/interdependency/matrix.py** - 73 dependency rules between CUAD label pairs
8. **src/interdependency/analyzer.py** - Interdependency analysis orchestrator

## Performance Notes

- Classifier initialization loads sentence-transformers model (~2-3s)
- LLM calls add ~2-5s per clause for risk analysis
- Use `--no-llm` flag for faster rule-only assessment
- Portfolio analysis scales linearly with contract count
