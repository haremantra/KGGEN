"""API routes for contract analysis."""

import json
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..pipeline import ContractAnalysisPipeline, ContractAnalysis
from ..risk.assessor import RiskAssessor, RiskAssessment
from ..portfolio.analyzer import PortfolioAnalyzer, PortfolioAnalysis
from ..interdependency.analyzer import InterdependencyAnalyzer
from ..interdependency.types import InterdependencyReport


router = APIRouter(tags=["contracts"])

# In-memory storage for demo (replace with database in production)
_contract_store: dict[str, dict] = {}
_analysis_store: dict[str, ContractAnalysis] = {}
_risk_store: dict[str, RiskAssessment] = {}
_interdependency_store: dict[str, InterdependencyReport] = {}

# Lazy-loaded pipeline and assessor
_pipeline: ContractAnalysisPipeline | None = None
_risk_assessor: RiskAssessor | None = None
_portfolio_analyzer: PortfolioAnalyzer | None = None
_interdependency_analyzer: InterdependencyAnalyzer | None = None


def get_pipeline() -> ContractAnalysisPipeline:
    """Get or create the analysis pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ContractAnalysisPipeline()
        _pipeline.initialize()
    return _pipeline


def get_risk_assessor() -> RiskAssessor:
    """Get or create the risk assessor."""
    global _risk_assessor
    if _risk_assessor is None:
        _risk_assessor = RiskAssessor(use_llm=True)
    return _risk_assessor


def get_portfolio_analyzer() -> PortfolioAnalyzer:
    """Get or create the portfolio analyzer."""
    global _portfolio_analyzer
    if _portfolio_analyzer is None:
        _portfolio_analyzer = PortfolioAnalyzer(use_llm=True)
    return _portfolio_analyzer


def get_interdependency_analyzer() -> InterdependencyAnalyzer:
    """Get or create the interdependency analyzer."""
    global _interdependency_analyzer
    if _interdependency_analyzer is None:
        _interdependency_analyzer = InterdependencyAnalyzer(use_llm=True)
    return _interdependency_analyzer


# === Pydantic Models ===

class ContractUploadResponse(BaseModel):
    """Response after uploading a contract."""
    contract_id: str
    filename: str
    status: str
    message: str


class ContractSummaryResponse(BaseModel):
    """Summary of a contract."""
    contract_id: str
    filename: str
    status: str
    total_clauses: int
    labels_found: int
    risk_score: Optional[int] = None
    risk_level: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Full contract analysis response."""
    contract_id: str
    total_clauses: int
    summary: dict
    analyzed_clauses: list[dict]


class RiskResponse(BaseModel):
    """Risk assessment response."""
    contract_id: str
    overall_risk_score: int
    risk_level: str
    findings: list[dict]
    missing_clause_risks: list[dict]
    summary: str
    llm_analysis: Optional[str] = None


class PortfolioAnalyzeRequest(BaseModel):
    """Request to analyze multiple contracts."""
    contract_ids: list[str] = Field(..., description="List of contract IDs to analyze")


class CompareRequest(BaseModel):
    """Request to compare two contracts."""
    contract_a: str
    contract_b: str


# === Contract Endpoints ===

@router.post("/contracts/upload", response_model=ContractUploadResponse)
async def upload_contract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analyze: bool = Query(True, description="Analyze immediately after upload"),
):
    """Upload a contract for analysis.

    Accepts PDF or text files. Returns a contract ID for future reference.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ['.pdf', '.txt', '.md']:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use PDF or text files.")

    # Generate contract ID
    contract_id = str(uuid.uuid4())[:8]

    # Read file content
    content = await file.read()

    # Store contract metadata
    _contract_store[contract_id] = {
        "filename": file.filename,
        "status": "uploaded",
        "size": len(content),
    }

    if analyze:
        # Analyze synchronously for now (could be async with Celery)
        try:
            # Save to temp file for PDF processing
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Extract text
            if suffix == '.pdf':
                from ..utils.pdf_reader import extract_text_from_pdf
                text = extract_text_from_pdf(Path(tmp_path))
            else:
                text = content.decode('utf-8', errors='ignore')

            # Run analysis
            pipeline = get_pipeline()
            analysis = pipeline.analyze(text, contract_id=contract_id)

            # Run risk assessment
            assessor = get_risk_assessor()
            risk = assessor.assess(analysis)

            # Run interdependency analysis
            dep_analyzer = get_interdependency_analyzer()
            dep_report = dep_analyzer.analyze(analysis)

            # Adjust risk score based on interdependency findings
            if dep_report.risk_score_adjustment > 0:
                adjusted = min(100, risk.overall_risk_score + dep_report.risk_score_adjustment)
                risk.overall_risk_score = adjusted
                if adjusted >= 75:
                    risk.risk_level = "CRITICAL"
                elif adjusted >= 50:
                    risk.risk_level = "HIGH"
                elif adjusted >= 25:
                    risk.risk_level = "MEDIUM"
                else:
                    risk.risk_level = "LOW"

            # Store results
            _analysis_store[contract_id] = analysis
            _risk_store[contract_id] = risk
            _interdependency_store[contract_id] = dep_report
            _contract_store[contract_id]["status"] = "analyzed"

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            _contract_store[contract_id]["status"] = "error"
            _contract_store[contract_id]["error"] = str(e)
            raise HTTPException(500, f"Analysis failed: {str(e)}")

    return ContractUploadResponse(
        contract_id=contract_id,
        filename=file.filename,
        status=_contract_store[contract_id]["status"],
        message="Contract uploaded and analyzed" if analyze else "Contract uploaded",
    )


@router.get("/contracts", response_model=list[ContractSummaryResponse])
async def list_contracts():
    """List all uploaded contracts with their status."""
    results = []

    for contract_id, meta in _contract_store.items():
        summary = ContractSummaryResponse(
            contract_id=contract_id,
            filename=meta["filename"],
            status=meta["status"],
            total_clauses=0,
            labels_found=0,
        )

        # Add analysis info if available
        if contract_id in _analysis_store:
            analysis = _analysis_store[contract_id]
            summary.total_clauses = analysis.total_clauses
            summary.labels_found = len(analysis.analyzed_clauses)

        if contract_id in _risk_store:
            risk = _risk_store[contract_id]
            summary.risk_score = risk.overall_risk_score
            summary.risk_level = risk.risk_level

        results.append(summary)

    return results


@router.get("/contracts/{contract_id}", response_model=AnalysisResponse)
async def get_contract(contract_id: str):
    """Get full analysis for a contract."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    if contract_id not in _analysis_store:
        raise HTTPException(400, f"Contract {contract_id} has not been analyzed")

    analysis = _analysis_store[contract_id]

    return AnalysisResponse(
        contract_id=analysis.contract_id,
        total_clauses=analysis.total_clauses,
        summary=analysis.summary,
        analyzed_clauses=[
            {
                "cuad_label": c.cuad_label,
                "confidence": c.label_confidence,
                "category": c.category,
                "text_preview": c.text[:200] if c.text else "",
                "extracted_values": [
                    {"field": v.field, "value": v.value, "confidence": v.confidence}
                    for v in c.extracted_values
                ],
                "entities": c.entities,
            }
            for c in analysis.analyzed_clauses
        ],
    )


@router.get("/contracts/{contract_id}/risks", response_model=RiskResponse)
async def get_contract_risks(contract_id: str):
    """Get risk assessment for a contract."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    if contract_id not in _risk_store:
        raise HTTPException(400, f"Contract {contract_id} has not been risk-assessed")

    risk = _risk_store[contract_id]

    return RiskResponse(
        contract_id=risk.contract_id,
        overall_risk_score=risk.overall_risk_score,
        risk_level=risk.risk_level,
        findings=[f.to_dict() for f in risk.findings],
        missing_clause_risks=[f.to_dict() for f in risk.missing_clause_risks],
        summary=risk.summary,
        llm_analysis=risk.llm_analysis,
    )


class BulkDeleteRequest(BaseModel):
    """Request to delete multiple contracts."""
    contract_ids: list[str] | None = Field(None, description="Specific IDs to delete, or null for all")
    delete_all: bool = Field(False, description="Set true to delete all contracts")


@router.post("/contracts/bulk-delete")
async def bulk_delete_contracts(request: BulkDeleteRequest):
    """Delete multiple contracts or all contracts."""
    if request.delete_all:
        count = len(_contract_store)
        if count == 0:
            raise HTTPException(400, "No contracts to delete")
        _contract_store.clear()
        _analysis_store.clear()
        _risk_store.clear()
        _interdependency_store.clear()
        return {"status": "deleted", "count": count, "message": f"Deleted {count} contract(s)"}

    if not request.contract_ids:
        raise HTTPException(400, "Provide contract_ids or set delete_all=true")

    deleted = 0
    for cid in request.contract_ids:
        if cid in _contract_store:
            del _contract_store[cid]
            _analysis_store.pop(cid, None)
            _risk_store.pop(cid, None)
            _interdependency_store.pop(cid, None)
            deleted += 1

    return {"status": "deleted", "count": deleted, "message": f"Deleted {deleted} contract(s)"}


@router.delete("/contracts/{contract_id}")
async def delete_contract(contract_id: str):
    """Delete a contract and its analysis."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    del _contract_store[contract_id]
    _analysis_store.pop(contract_id, None)
    _risk_store.pop(contract_id, None)
    _interdependency_store.pop(contract_id, None)

    return {"status": "deleted", "contract_id": contract_id}


# === Portfolio Endpoints ===

@router.post("/portfolio/analyze")
async def analyze_portfolio(request: PortfolioAnalyzeRequest):
    """Analyze multiple contracts as a portfolio.

    Pass a list of contract IDs that have already been uploaded and analyzed.
    """
    # Validate all contracts exist
    for cid in request.contract_ids:
        if cid not in _analysis_store:
            raise HTTPException(400, f"Contract {cid} not found or not analyzed")

    # Build portfolio analyzer with existing analyses
    analyzer = get_portfolio_analyzer()

    for cid in request.contract_ids:
        analyzer.add_analysis(
            _analysis_store[cid],
            _risk_store.get(cid)
        )

    # Get portfolio analysis
    portfolio = analyzer.get_portfolio_analysis()

    return portfolio.to_dict()


@router.get("/portfolio/risks")
async def get_portfolio_risks():
    """Get aggregated risk summary across all analyzed contracts."""
    if not _risk_store:
        raise HTTPException(400, "No contracts have been analyzed")

    # Build portfolio
    analyzer = get_portfolio_analyzer()
    for cid, analysis in _analysis_store.items():
        analyzer.add_analysis(analysis, _risk_store.get(cid))

    portfolio = analyzer.get_portfolio_analysis()

    return portfolio.risk_summary.to_dict()


@router.get("/portfolio/gaps")
async def get_portfolio_gaps():
    """Get gap analysis showing missing protections across portfolio."""
    if not _analysis_store:
        raise HTTPException(400, "No contracts have been analyzed")

    analyzer = get_portfolio_analyzer()
    for cid, analysis in _analysis_store.items():
        analyzer.add_analysis(analysis, _risk_store.get(cid))

    portfolio = analyzer.get_portfolio_analysis()

    return portfolio.gap_analysis.to_dict()


@router.post("/portfolio/compare")
async def compare_contracts(request: CompareRequest):
    """Compare two contracts side by side."""
    for cid in [request.contract_a, request.contract_b]:
        if cid not in _analysis_store:
            raise HTTPException(400, f"Contract {cid} not found or not analyzed")

    analyzer = get_portfolio_analyzer()
    analyzer.add_analysis(_analysis_store[request.contract_a], _risk_store.get(request.contract_a))
    analyzer.add_analysis(_analysis_store[request.contract_b], _risk_store.get(request.contract_b))

    comparison = analyzer.compare_contracts(request.contract_a, request.contract_b)

    return comparison.to_dict()


@router.get("/portfolio/clause/{label}")
async def get_clause_coverage(label: str):
    """Get coverage analysis for a specific clause type across portfolio."""
    if not _analysis_store:
        raise HTTPException(400, "No contracts have been analyzed")

    analyzer = get_portfolio_analyzer()
    for cid, analysis in _analysis_store.items():
        analyzer.add_analysis(analysis, _risk_store.get(cid))

    portfolio = analyzer.get_portfolio_analysis()

    if label not in portfolio.clause_coverage:
        raise HTTPException(404, f"Clause label '{label}' not recognized")

    return portfolio.clause_coverage[label].to_dict()


# === Interdependency Endpoints ===

class DependencyGraphResponse(BaseModel):
    """Full dependency graph response."""
    contract_id: str
    nodes: list[dict]
    edges: list[dict]
    contradiction_count: int
    missing_requirements: list[dict]
    max_impact_clause: str
    recommendations: list[str]
    risk_score_adjustment: int


class ImpactResponse(BaseModel):
    """Impact analysis response for a specific clause."""
    source_label: str
    affected_clauses: list[dict]
    total_affected: int
    max_depth: int


class ContradictionResponse(BaseModel):
    """Contradiction list response."""
    contract_id: str
    contradictions: list[dict]
    count: int


class CompletenessResponse(BaseModel):
    """Missing requirements / completeness response."""
    contract_id: str
    missing_requirements: list[dict]
    count: int


@router.get("/contracts/{contract_id}/dependencies", response_model=DependencyGraphResponse)
async def get_dependencies(contract_id: str):
    """Get full dependency graph for a contract."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    if contract_id not in _interdependency_store:
        raise HTTPException(400, f"Contract {contract_id} has no interdependency analysis")

    report = _interdependency_store[contract_id]
    graph = report.graph

    return DependencyGraphResponse(
        contract_id=contract_id,
        nodes=[n.to_dict() for n in graph.nodes],
        edges=[e.to_dict() for e in graph.edges],
        contradiction_count=graph.contradiction_count,
        missing_requirements=[m.to_dict() for m in graph.missing_requirements],
        max_impact_clause=graph.max_impact_clause,
        recommendations=report.recommendations,
        risk_score_adjustment=report.risk_score_adjustment,
    )


@router.get("/contracts/{contract_id}/impact/{label}", response_model=ImpactResponse)
async def get_impact(
    contract_id: str,
    label: str,
    max_hops: int = Query(3, ge=1, le=10, description="Maximum traversal depth"),
):
    """Get impact analysis for a specific clause label."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    if contract_id not in _interdependency_store:
        raise HTTPException(400, f"Contract {contract_id} has no interdependency analysis")

    # Re-run impact analysis with the requested max_hops
    analyzer = get_interdependency_analyzer()
    impact = analyzer.graph_builder.impact_analysis(label, max_hops=max_hops)

    return ImpactResponse(
        source_label=impact.source_label,
        affected_clauses=impact.affected_clauses,
        total_affected=impact.total_affected,
        max_depth=impact.max_depth,
    )


@router.get("/contracts/{contract_id}/contradictions", response_model=ContradictionResponse)
async def get_contradictions(contract_id: str):
    """Get all contradictions found in a contract."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    if contract_id not in _interdependency_store:
        raise HTTPException(400, f"Contract {contract_id} has no interdependency analysis")

    report = _interdependency_store[contract_id]

    return ContradictionResponse(
        contract_id=contract_id,
        contradictions=report.contradictions,
        count=len(report.contradictions),
    )


@router.get("/contracts/{contract_id}/completeness", response_model=CompletenessResponse)
async def get_completeness(contract_id: str):
    """Get missing required clauses for a contract."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    if contract_id not in _interdependency_store:
        raise HTTPException(400, f"Contract {contract_id} has no interdependency analysis")

    report = _interdependency_store[contract_id]

    return CompletenessResponse(
        contract_id=contract_id,
        missing_requirements=[m.to_dict() for m in report.missing_requirements],
        count=len(report.missing_requirements),
    )


# === Search & Query Endpoints ===

class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Max results")
    kind: Optional[str] = Field(None, description="'entity' or 'triple'")


class QueryRequest(BaseModel):
    """Query request for RAG-based Q&A."""
    question: str = Field(..., description="Question to answer")


class CompareQueryRequest(BaseModel):
    """Request to compare contracts via query."""
    aspect: str = Field("all", description="Aspect to compare: licensing, obligations, restrictions, liability, all")


@router.post("/search/entities")
async def search_entities(request: SearchRequest):
    """Search for entities in the knowledge graph."""
    from ..search.service import get_search_service
    search = get_search_service()
    results = search.search_entities(request.query, limit=request.limit)
    return {
        "query": request.query,
        "results": [{"payload": p, "score": s} for p, s in results],
        "count": len(results),
    }


@router.post("/search/triples")
async def search_triples(request: SearchRequest):
    """Search for triples in the knowledge graph."""
    from ..search.service import get_search_service
    search = get_search_service()
    results = search.search_triples(request.query, limit=request.limit)
    return {
        "query": request.query,
        "results": [{"payload": p, "score": s} for p, s in results],
        "count": len(results),
    }


@router.post("/query")
async def answer_query(request: QueryRequest):
    """Answer a question about contracts using RAG."""
    from ..query.service import get_query_service
    query_svc = get_query_service()
    response = query_svc.query(request.question)
    return response.to_dict()


@router.get("/contracts/{contract_id}/qa")
async def contract_qa(
    contract_id: str,
    q: str = Query(..., description="Question about this contract"),
):
    """Answer a question about a specific contract."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    from ..query.service import get_query_service
    query_svc = get_query_service()
    response = query_svc.query(q)
    return response.to_dict()


@router.post("/query/compare")
async def compare_via_query(request: CompareQueryRequest):
    """Compare contracts on a specific aspect using RAG."""
    from ..query.service import get_query_service
    query_svc = get_query_service()
    response = query_svc.compare(request.aspect)
    return response.to_dict()
