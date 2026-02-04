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


router = APIRouter(tags=["contracts"])

# In-memory storage for demo (replace with database in production)
_contract_store: dict[str, dict] = {}
_analysis_store: dict[str, ContractAnalysis] = {}
_risk_store: dict[str, RiskAssessment] = {}

# Lazy-loaded pipeline and assessor
_pipeline: ContractAnalysisPipeline | None = None
_risk_assessor: RiskAssessor | None = None
_portfolio_analyzer: PortfolioAnalyzer | None = None


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

            # Store results
            _analysis_store[contract_id] = analysis
            _risk_store[contract_id] = risk
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


@router.delete("/contracts/{contract_id}")
async def delete_contract(contract_id: str):
    """Delete a contract and its analysis."""
    if contract_id not in _contract_store:
        raise HTTPException(404, f"Contract {contract_id} not found")

    del _contract_store[contract_id]
    _analysis_store.pop(contract_id, None)
    _risk_store.pop(contract_id, None)

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
