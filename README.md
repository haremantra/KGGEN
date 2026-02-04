# KGGEN-CUAD: Contract Analysis & Portfolio Risk Management

A knowledge graph-powered contract analysis system that combines clause classification, entity extraction, and risk assessment for legal contracts. Built on the CUAD (Contract Understanding Atticus Dataset) methodology with 41 legal clause categories.

## Features

- **Clause Classification**: Automatically identify 41 CUAD legal clause types using semantic similarity
- **Risk Assessment**: Hybrid rule-based + LLM analysis for comprehensive risk scoring
- **Portfolio Analysis**: Cross-contract comparison, gap analysis, and aggregate risk metrics
- **REST API**: FastAPI backend with full contract lifecycle management
- **Web Dashboard**: Interactive Streamlit UI for visual analysis
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
│   │   ├── app.py              # Application factory
│   │   └── routes.py           # API endpoints
│   ├── classification/         # CUAD clause classification
│   │   ├── classifier.py       # Semantic similarity classifier
│   │   └── cuad_labels.py      # 41 CUAD label definitions
│   ├── extraction/             # Entity/relationship extraction
│   │   └── extractor.py        # LLM-based extraction
│   ├── risk/                   # Risk assessment engine
│   │   ├── rules.py            # Risk rules for CUAD categories
│   │   └── assessor.py         # Hybrid risk scoring
│   ├── portfolio/              # Portfolio-level analysis
│   │   └── analyzer.py         # Cross-contract analysis
│   ├── utils/                  # Utilities
│   │   ├── pdf_reader.py       # PDF text extraction
│   │   └── neo4j_store.py      # Graph database storage
│   ├── config.py               # Configuration management
│   ├── pipeline.py             # Integrated analysis pipeline
│   └── main.py                 # CLI entry point
├── streamlit_app.py            # Web dashboard
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

## API Reference

### Contract Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/contracts/upload` | Upload and analyze a contract |
| GET | `/api/contracts` | List all contracts |
| GET | `/api/contracts/{id}` | Get contract analysis |
| GET | `/api/contracts/{id}/risks` | Get risk assessment |
| DELETE | `/api/contracts/{id}` | Delete a contract |

### Portfolio Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/portfolio/analyze` | Analyze multiple contracts |
| GET | `/api/portfolio/risks` | Portfolio risk summary |
| GET | `/api/portfolio/gaps` | Missing clause analysis |
| POST | `/api/portfolio/compare` | Compare two contracts |
| GET | `/api/portfolio/clause/{label}` | Clause coverage analysis |

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

The Streamlit dashboard provides visual analysis across 5 pages:

1. **Upload**: Drag-and-drop contract upload with batch processing
2. **Portfolio**: Risk distribution charts, heatmaps, highest-risk contracts
3. **Analysis**: Individual contract deep-dive with clause-by-clause findings
4. **Compare**: Side-by-side contract comparison
5. **Gaps**: Missing protection checker with recommendations

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

| Metric | Target | Description |
|--------|--------|-------------|
| Classification Accuracy | 85%+ | CUAD label identification |
| Risk Score Correlation | 90%+ | Agreement with legal expert review |
| Extraction Throughput | 1-2 contracts/min | Full pipeline processing |
| API Latency (P95) | <500ms | Non-analysis endpoints |
| Analysis Time | 30-120s | Per contract (depends on size) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

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
