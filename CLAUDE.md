# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KGGEN-CUAD is a Knowledge Graph Generator and Contract Analysis system that applies the KGGen methodology to the CUAD (Contract Understanding Atticus Dataset) for legal contract analysis. The system extracts structured knowledge graphs from legal contracts, performs risk assessment, and enables portfolio-level analysis for technology agreements within common law jurisdictions.

**Current Status:** MVP Complete - Risk assessment, portfolio analysis, REST API, and web dashboard implemented.

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
│   ├── portfolio/              # Portfolio-level analysis
│   │   └── analyzer.py         # Cross-contract comparison & gaps
│   ├── utils/                  # Utilities
│   │   ├── pdf_reader.py       # PDF text extraction
│   │   └── neo4j_store.py      # Graph database storage
│   ├── config.py               # Pydantic settings from .env
│   ├── pipeline.py             # Integrated analysis pipeline
│   └── main.py                 # CLI entry point
├── streamlit_app.py            # Web dashboard (5 pages)
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

# Compare two contracts
python -m src.main compare contract1.pdf contract2.pdf

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
4. **Portfolio**: Aggregation, gap analysis, cross-contract comparison

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
```

## Key Files to Understand

1. **src/risk/rules.py** - Risk rule definitions for all 41 CUAD categories
2. **src/risk/assessor.py** - Hybrid scoring engine with LLM integration
3. **src/portfolio/analyzer.py** - Cross-contract analysis logic
4. **src/api/routes.py** - All REST endpoints
5. **streamlit_app.py** - Dashboard UI implementation
6. **src/pipeline.py** - Core analysis pipeline

## Performance Notes

- Classifier initialization loads sentence-transformers model (~2-3s)
- LLM calls add ~2-5s per clause for risk analysis
- Use `--no-llm` flag for faster rule-only assessment
- Portfolio analysis scales linearly with contract count
