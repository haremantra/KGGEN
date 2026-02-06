# KGGEN-CUAD: Contract Analysis & Portfolio Risk Management

A knowledge graph-powered contract analysis system that combines clause classification, entity extraction, and risk assessment for legal contracts. Built on the CUAD (Contract Understanding Atticus Dataset) methodology with 41 legal clause categories.

> **[Why I Built This](WHY_I_BUILT_THIS.md)** - Read about the motivation behind this project

## Features

- **Clause Classification**: Fine-tuned sentence-transformer achieving 61% accuracy on CUAD labels (up from 29% baseline) with calibrated confidence scores
- **Risk Assessment**: Hybrid rule-based + LLM analysis for comprehensive risk scoring
- **Clause Interdependency**: 73 dependency rules detecting contradictions, missing requirements, and cascading impacts
- **Entity Resolution**: Semantic clustering with adaptive k-means to unify entities across contracts
- **Hybrid Search**: BM25 + semantic vector search with reciprocal rank fusion
- **RAG Query**: Natural language Q&A over your contract portfolio
- **Portfolio Analysis**: Cross-contract comparison, gap analysis, and aggregate risk metrics
- **REST API**: FastAPI backend with full contract lifecycle management
- **Web Dashboard**: Interactive 7-page Streamlit UI for visual analysis
- **CLI Tools**: Command-line interface for batch processing and automation

## Quick Start

### Prerequisites

- Python 3.11+
- Anthropic API key (for Claude LLM features)
- Docker (optional, for full stack deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/haremantra/KGGEN.git
cd KGGEN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Basic Usage

```bash
# Analyze a single contract
python -m src.main analyze contract.pdf -o results.json

# Get risk assessment
python -m src.main risks contract.pdf

# Analyze portfolio of contracts
python -m src.main portfolio ./contracts/ --limit 10

# Compare two contracts
python -m src.main compare contract1.pdf contract2.pdf

# Start API server
python -m src.main serve --port 8000

# Start web dashboard (in another terminal)
streamlit run streamlit_app.py
```

## Architecture

```
KGGEN/
├── src/
│   ├── api/                    # FastAPI REST API
│   │   ├── app.py              # Application factory with CORS
│   │   └── routes.py           # Contract, portfolio, search & query endpoints
│   ├── classification/         # CUAD clause classification
│   │   ├── classifier.py       # Fine-tuned classifier with Platt scaling
│   │   └── cuad_labels.py      # 41 CUAD label definitions
│   ├── extraction/             # Entity/relationship extraction
│   │   └── extractor.py        # LLM-based KG extraction
│   ├── risk/                   # Risk assessment engine
│   │   ├── rules.py            # Risk rules for CUAD categories
│   │   └── assessor.py         # Hybrid rule + LLM risk scoring
│   ├── interdependency/        # Clause interdependency analysis
│   │   ├── types.py            # DependencyType, ClauseNode, DependencyEdge
│   │   ├── matrix.py           # 73 static dependency rules
│   │   ├── detector.py         # Rule matching + LLM validation
│   │   ├── graph.py            # NetworkX graph builder + algorithms
│   │   └── analyzer.py         # Orchestrator producing InterdependencyReport
│   ├── resolution/             # Entity resolution
│   │   └── resolver.py         # Adaptive k-means + LLM canonical selection
│   ├── aggregation/            # Cross-contract aggregation
│   │   └── aggregator.py       # Per-entity-type dedup thresholds
│   ├── search/                 # Hybrid search
│   │   ├── service.py          # BM25 + semantic with RRF fusion
│   │   └── backends.py         # InMemory + Qdrant backends
│   ├── query/                  # RAG query service
│   │   └── service.py          # Context retrieval → LLM answer
│   ├── portfolio/              # Portfolio-level analysis
│   │   └── analyzer.py         # Cross-contract comparison & gaps
│   ├── utils/                  # Utilities
│   │   ├── pdf_reader.py       # PDF text extraction
│   │   ├── neo4j_store.py      # Graph database storage
│   │   ├── embedding.py        # S-BERT embedding service
│   │   └── llm.py              # LLM service with retry + fallback
│   ├── config.py               # Pydantic settings from .env
│   ├── pipeline.py             # Integrated analysis pipeline
│   └── main.py                 # CLI entry point
├── scripts/
│   ├── ralph.py                # RALPH task orchestrator
│   ├── finetune_classifier.py  # Fine-tune on CUAD dataset
│   ├── calibrate_classifier.py # Platt scaling calibration
│   └── benchmark_classifier.py # Accuracy benchmarking
├── models/                     # Fine-tuned models (git-ignored)
│   └── cuad-MiniLM-L6-v2-finetuned/
├── streamlit_app.py            # Web dashboard (7 pages)
├── docker-compose.yml          # Full stack deployment
├── Dockerfile                  # Container image
└── pyproject.toml              # Dependencies
```

## Risk Assessment

The system uses a hybrid approach combining deterministic rules with LLM analysis:

### Risk Severity Levels

| Level | Score Impact | Description |
|-------|--------------|-------------|
| CRITICAL | 25 pts | Requires immediate attention (e.g., uncapped liability) |
| HIGH | 15 pts | Significant risk exposure (e.g., broad non-compete) |
| MEDIUM | 8 pts | Moderate concern (e.g., auto-renewal terms) |
| LOW | 3 pts | Minor issue (e.g., standard restrictions) |
| INFO | 0 pts | Informational only |

### Risk Score Calculation

- **0-24**: LOW risk
- **25-49**: MEDIUM risk
- **50-74**: HIGH risk
- **75-100**: CRITICAL risk

### Key Risk Categories

**Critical Risks (Always Flag)**
- Uncapped liability clauses
- IP ownership assignment without compensation
- Broad exclusivity provisions

**High Risks (Requires Review)**
- License grant restrictions
- Non-compete clauses
- Liability caps below industry standard
- Change of control provisions
- Termination for convenience (one-sided)

**Missing Protection Risks**
- No liability cap
- No source code escrow (software contracts)
- No termination for convenience
- No governing law specified

## Clause Interdependency Analysis

Contracts are interconnected systems where modifying one clause can cascade through the document. The interdependency module maps these relationships:

### Dependency Types

| Type | Description | Example |
|------|-------------|---------|
| REQUIRES | Clause A needs clause B to be enforceable | Indemnification requires Insurance |
| MODIFIES | Clause A changes the effect of clause B | Cap on Liability modifies Indemnification |
| CONFLICTS | Clauses may contradict each other | Exclusivity vs. Non-Exclusive License |
| SUPERSEDES | Clause A overrides clause B | Specific provision supersedes general |
| TRIGGERS | Clause A activates clause B | Termination triggers Post-Termination Services |
| REFERENCES | Clause A mentions clause B | License Grant references IP Ownership |

### Analysis Outputs

- **Contradiction Detection**: Finds clauses with conflicting terms
- **Missing Requirements**: Identifies required companion clauses that are absent
- **Impact Analysis**: Maps which clauses are affected when one is modified
- **Risk Adjustment**: Adds up to +40 points to risk score based on structural issues

## API Reference

### Contract Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/contracts/upload` | Upload and analyze a contract |
| GET | `/api/contracts` | List all contracts |
| GET | `/api/contracts/{id}` | Get contract analysis |
| GET | `/api/contracts/{id}/risks` | Get risk assessment |
| GET | `/api/contracts/{id}/dependencies` | Get clause dependency graph |
| GET | `/api/contracts/{id}/contradictions` | Get detected contradictions |
| GET | `/api/contracts/{id}/completeness` | Get missing requirements |
| GET | `/api/contracts/{id}/impact/{label}` | Impact analysis for a clause |
| DELETE | `/api/contracts/{id}` | Delete a contract |
| POST | `/api/contracts/bulk-delete` | Delete multiple or all contracts |

### Portfolio Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/portfolio/analyze` | Analyze multiple contracts |
| GET | `/api/portfolio/risks` | Portfolio risk summary |
| GET | `/api/portfolio/gaps` | Missing clause analysis |
| POST | `/api/portfolio/compare` | Compare two contracts |
| GET | `/api/portfolio/clause/{label}` | Clause coverage analysis |

### Search & Query Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search/entities` | Search entities in knowledge graph |
| POST | `/api/search/triples` | Search relationships/triples |
| POST | `/api/query` | RAG-based natural language Q&A |
| GET | `/api/contracts/{id}/qa?q=...` | Q&A for specific contract |
| POST | `/api/query/compare` | Compare contracts on specific aspect |

### Example API Usage

```bash
# Upload a contract
curl -X POST "http://localhost:8000/api/contracts/upload" \
  -F "file=@contract.pdf" \
  -F "analyze=true"

# Get risk assessment
curl "http://localhost:8000/api/contracts/{contract_id}/risks"

# Compare contracts
curl -X POST "http://localhost:8000/api/portfolio/compare" \
  -H "Content-Type: application/json" \
  -d '{"contract_a": "id1", "contract_b": "id2"}'
```

## CLI Commands

```bash
# Extract knowledge graph entities
python -m src.main extract <pdf_path> [-o output.json] [-s]

# Classify contract clauses
python -m src.main classify <pdf_path> [-t threshold] [-o output.json]

# Full analysis with risk assessment
python -m src.main analyze <pdf_path> [-o output.json] [--no-llm]

# Detailed risk assessment
python -m src.main risks <pdf_path> [-o output.json] [--no-llm]

# Portfolio analysis
python -m src.main portfolio <folder> [-l limit] [-p pattern] [-o output.json]

# Compare two contracts
python -m src.main compare <contract1> <contract2> [-o output.json]

# Database operations
python -m src.main init-db    # Initialize Neo4j schema
python -m src.main stats      # Show database statistics

# Start servers
python -m src.main serve [--host 0.0.0.0] [--port 8000] [--reload]
```

## Web Dashboard

The Streamlit dashboard provides visual analysis across 7 pages:

1. **Upload**: Drag-and-drop contract upload with batch processing and contract management (delete individual or all)
2. **Portfolio**: Risk distribution charts, heatmaps, highest-risk contracts
3. **Analysis**: Individual contract deep-dive with clause-by-clause findings
4. **Compare**: Side-by-side contract comparison
5. **Gaps**: Missing protection checker with recommendations
6. **Dependencies**: Clause interdependency graph visualization
7. **Search**: Natural language search and Q&A across portfolio

Access at: http://localhost:8501

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - Neo4j: http://localhost:7474
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
# - Qdrant: http://localhost:6333

# View logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Stop all services
docker-compose down
```

## CUAD Label Categories

The system classifies contracts using 41 CUAD labels across 5 categories:

### General Information (10 labels)
- Document Name, Parties, Agreement Date, Effective Date, Expiration Date
- Renewal Term, Notice Period To Terminate Renewal, Governing Law
- License Grant, Irrevocable Or Perpetual License

### Restrictive Covenants (11 labels)
- Anti-Assignment, Non-Compete, Non-Disparagement
- No-Solicit Of Employees, No-Solicit Of Customers
- Exclusivity, Change Of Control, Covenant Not To Sue
- Competitive Restriction Exception, Non-Transferable License, Volume Restriction

### Revenue Risks (10 labels)
- Cap On Liability, Uncapped Liability, Liquidated Damages
- Revenue/Profit Sharing, Minimum Commitment, Audit Rights
- Insurance, Warranty Duration, Post-Termination Services
- Termination For Convenience

### Intellectual Property (6 labels)
- IP Ownership Assignment, Joint IP Ownership, Source Code Escrow
- Affiliate License-Licensor, Affiliate License-Licensee
- Unlimited/All-You-Can-Eat-License

### Special Provisions (4 labels)
- Third Party Beneficiary, Most Favored Nation, Rofr/Rofo/Rofn

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Neo4j (optional - for graph storage)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=kggen_password

# PostgreSQL (optional - for metadata)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=kggen
POSTGRES_PASSWORD=kggen_password
POSTGRES_DB=kggen

# Redis (optional - for caching)
REDIS_URL=redis://localhost:6379/0

# Qdrant (optional - for vector search)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Application
LOG_LEVEL=INFO
DEBUG=false
DEFAULT_LLM_MODEL=claude-sonnet-4-20250514
```

## Performance

### Classifier Metrics (Fine-tuned Model)

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Accuracy | 29.4% | 61.0% | +31.6 pp |
| Macro F1 | - | 53.2% | - |
| Weighted F1 | - | 62.2% | - |
| ECE (calibrated) | 4.5% | 4.4% | Excellent calibration |

*Trained on 8,257 CUAD clause pairs using MultipleNegativesRankingLoss*

### System Performance

| Metric | Value | Description |
|--------|-------|-------------|
| Extraction Throughput | 1-2 contracts/min | Full pipeline processing |
| API Latency (P95) | <500ms | Non-analysis endpoints |
| Analysis Time | 30-120s | Per contract (depends on size) |
| Test Coverage | 52% | 251 tests across unit and integration |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (251 tests, 52% coverage)
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing

# Run RALPH task orchestrator
python scripts/ralph.py --manifest scripts/ralph_tests_manifest.yaml --report

# Fine-tune classifier on CUAD dataset
python scripts/prepare_cuad_dataset.py
python scripts/finetune_classifier.py
python scripts/calibrate_classifier.py finetuned
python scripts/benchmark_classifier.py finetuned

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## Knowledge Graph Schema

### Node Types
- **Party**: Contract parties (licensor, licensee, vendor, customer)
- **IPAsset**: Intellectual property (software, patents, trademarks)
- **Obligation**: Contractual obligations and duties
- **Restriction**: Limitations and restrictions
- **LiabilityProvision**: Liability terms and caps
- **Temporal**: Dates and time periods
- **Jurisdiction**: Governing law and venue
- **ContractClause**: Individual contract clauses

### Edge Types
- LICENSES_TO, OWNS, ASSIGNS
- HAS_OBLIGATION, SUBJECT_TO_RESTRICTION
- HAS_LIABILITY, GOVERNED_BY
- CONTAINS_CLAUSE, EFFECTIVE_ON, TERMINATES_ON

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CUAD Dataset**: Contract Understanding Atticus Dataset by The Atticus Project
- **KGGen Methodology**: Knowledge Graph Generation research
- **Anthropic Claude**: LLM-powered analysis and risk assessment

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/haremantra/KGGEN/issues)
- Documentation: See `/docs` folder for detailed guides
