"""Shared pytest fixtures and mocks for KGGEN-CUAD test suite."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.models.schema import KGNode, Triple, NodeType, ExtractionResult
from src.pipeline import ContractAnalysis, AnalyzedClause, ExtractedValue
from src.interdependency.types import (
    ClauseNode, DependencyEdge, DependencyType, MissingRequirement,
)
from src.risk.assessor import RiskAssessment, RiskFinding
from src.risk.rules import RiskSeverity


# ---------------------------------------------------------------------------
# Singleton cache clearing (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_singletons():
    """Clear all @lru_cache singletons between tests."""
    from src.utils.embedding import get_embedding_service
    from src.utils.llm import get_llm_service
    from src.search.service import get_search_service
    from src.query.service import get_query_service

    get_embedding_service.cache_clear()
    get_llm_service.cache_clear()
    get_search_service.cache_clear()
    get_query_service.cache_clear()
    yield


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_nodes():
    """Six diverse KGNode instances."""
    return [
        KGNode(id="p1", name="ACME SOFTWARE INC.", type=NodeType.PARTY,
               source_contract_id="c1", cuad_label="Parties", confidence_score=0.95),
        KGNode(id="p2", name="BETA TECHNOLOGIES LLC", type=NodeType.PARTY,
               source_contract_id="c1", cuad_label="Parties", confidence_score=0.92),
        KGNode(id="o1", name="Provide maintenance services", type=NodeType.OBLIGATION,
               source_contract_id="c1", cuad_label="Post-Termination Services", confidence_score=0.80),
        KGNode(id="r1", name="Non-compete restriction", type=NodeType.RESTRICTION,
               source_contract_id="c1", cuad_label="Non-Compete", confidence_score=0.85),
        KGNode(id="t1", name="January 1, 2024", type=NodeType.TEMPORAL,
               source_contract_id="c1", cuad_label="Effective Date", confidence_score=0.90),
        KGNode(id="ip1", name="Licensed Software", type=NodeType.IP_ASSET,
               source_contract_id="c1", cuad_label="License Grant", confidence_score=0.88),
    ]


@pytest.fixture
def sample_triples():
    """Five triples linking sample entities."""
    return [
        Triple(subject="ACME SOFTWARE INC.", predicate="LICENSES_TO",
               object="BETA TECHNOLOGIES LLC", confidence=0.9),
        Triple(subject="ACME SOFTWARE INC.", predicate="OWNS",
               object="Licensed Software", confidence=0.95),
        Triple(subject="BETA TECHNOLOGIES LLC", predicate="HAS_OBLIGATION",
               object="Provide maintenance services", confidence=0.8),
        Triple(subject="BETA TECHNOLOGIES LLC", predicate="SUBJECT_TO_RESTRICTION",
               object="Non-compete restriction", confidence=0.85),
        Triple(subject="Licensed Software", predicate="EFFECTIVE_ON",
               object="January 1, 2024", confidence=0.9),
    ]


@pytest.fixture
def sample_contract_analysis():
    """ContractAnalysis with 5 AnalyzedClauses."""
    return ContractAnalysis(
        contract_id="test-contract-001",
        total_clauses=10,
        analyzed_clauses=[
            AnalyzedClause(
                text="Licensor grants Licensee a non-exclusive license to use the Software.",
                cuad_label="License Grant",
                label_confidence=0.87,
                category="general_information",
                extracted_values=[
                    ExtractedValue(field="license_type", value="non-exclusive", confidence=0.9),
                ],
                entities=["ACME SOFTWARE INC.", "BETA TECHNOLOGIES LLC"],
                relationships=[{"subject": "ACME", "predicate": "LICENSES_TO", "object": "BETA"}],
            ),
            AnalyzedClause(
                text="Total liability shall not exceed $100,000.",
                cuad_label="Cap On Liability",
                label_confidence=0.82,
                category="revenue_risks",
                extracted_values=[
                    ExtractedValue(field="cap_amount", value="$100,000", confidence=0.95),
                ],
                entities=["Liability Cap"],
                relationships=[],
            ),
            AnalyzedClause(
                text="This Agreement shall be governed by the laws of Delaware.",
                cuad_label="Governing Law",
                label_confidence=0.91,
                category="general_information",
                extracted_values=[
                    ExtractedValue(field="jurisdiction", value="Delaware", confidence=0.95),
                ],
                entities=["Delaware"],
                relationships=[],
            ),
            AnalyzedClause(
                text="Licensee shall not compete with Licensor for 2 years.",
                cuad_label="Non-Compete",
                label_confidence=0.78,
                category="restrictive_covenants",
                extracted_values=[
                    ExtractedValue(field="duration", value="2 years", confidence=0.85),
                ],
                entities=["Non-Compete Clause"],
                relationships=[],
            ),
            AnalyzedClause(
                text="This Agreement expires on December 31, 2025.",
                cuad_label="Expiration Date",
                label_confidence=0.88,
                category="general_information",
                extracted_values=[
                    ExtractedValue(field="expiration_date", value="December 31, 2025", confidence=0.9),
                ],
                entities=["December 31, 2025"],
                relationships=[],
            ),
        ],
        summary={
            "labels_found": 5,
            "by_category": {
                "general_information": [
                    {"label": "License Grant", "confidence": 0.87, "extracted_count": 1},
                    {"label": "Governing Law", "confidence": 0.91, "extracted_count": 1},
                    {"label": "Expiration Date", "confidence": 0.88, "extracted_count": 1},
                ],
                "revenue_risks": [
                    {"label": "Cap On Liability", "confidence": 0.82, "extracted_count": 1},
                ],
                "restrictive_covenants": [
                    {"label": "Non-Compete", "confidence": 0.78, "extracted_count": 1},
                ],
            },
            "key_findings": [
                {"label": "License Grant", "field": "license_type", "value": "non-exclusive"},
                {"label": "Cap On Liability", "field": "cap_amount", "value": "$100,000"},
                {"label": "Governing Law", "field": "jurisdiction", "value": "Delaware"},
            ],
        },
    )


@pytest.fixture
def sample_clause_nodes():
    """Eight ClauseNode instances for interdependency testing."""
    return [
        ClauseNode(label="License Grant", category="general_information",
                   text="Non-exclusive license granted.", confidence=0.87),
        ClauseNode(label="Cap On Liability", category="revenue_risks",
                   text="Liability capped at $100K.", confidence=0.82),
        ClauseNode(label="Governing Law", category="general_information",
                   text="Governed by Delaware law.", confidence=0.91),
        ClauseNode(label="Non-Compete", category="restrictive_covenants",
                   text="2-year non-compete.", confidence=0.78),
        ClauseNode(label="Expiration Date", category="general_information",
                   text="Expires Dec 31, 2025.", confidence=0.88),
        ClauseNode(label="Renewal Term", category="general_information",
                   text="Auto-renews for 1 year.", confidence=0.75),
        ClauseNode(label="Termination For Convenience", category="revenue_risks",
                   text="Either party may terminate with 30 days notice.", confidence=0.80),
        ClauseNode(label="IP Ownership Assignment", category="intellectual_property",
                   text="All IP assigned to Licensor.", confidence=0.85),
    ]


@pytest.fixture
def sample_dependency_edges():
    """Ten DependencyEdge instances with mixed types."""
    return [
        DependencyEdge(source_label="Renewal Term", target_label="Expiration Date",
                       dependency_type=DependencyType.REQUIRES, strength=0.9,
                       reason="Renewal needs expiration date to renew from"),
        DependencyEdge(source_label="Non-Compete", target_label="License Grant",
                       dependency_type=DependencyType.RESTRICTS, strength=0.8,
                       reason="Non-compete restricts license usage scope"),
        DependencyEdge(source_label="Cap On Liability", target_label="Termination For Convenience",
                       dependency_type=DependencyType.MITIGATES, strength=0.7,
                       reason="Liability cap mitigates termination risk"),
        DependencyEdge(source_label="License Grant", target_label="IP Ownership Assignment",
                       dependency_type=DependencyType.CONFLICTS_WITH, strength=0.85,
                       reason="License grant may conflict with IP assignment"),
        DependencyEdge(source_label="Governing Law", target_label="Non-Compete",
                       dependency_type=DependencyType.MODIFIES, strength=0.6,
                       reason="Jurisdiction affects non-compete enforceability"),
        DependencyEdge(source_label="Expiration Date", target_label="Termination For Convenience",
                       dependency_type=DependencyType.DEPENDS_ON, strength=0.7,
                       reason="Expiration interacts with termination rights"),
        DependencyEdge(source_label="License Grant", target_label="Governing Law",
                       dependency_type=DependencyType.REQUIRES, strength=0.5,
                       reason="License needs governing law for enforcement"),
        DependencyEdge(source_label="IP Ownership Assignment", target_label="License Grant",
                       dependency_type=DependencyType.CONFLICTS_WITH, strength=0.85,
                       reason="IP assignment may conflict with license grant",
                       bidirectional=True),
        DependencyEdge(source_label="Renewal Term", target_label="Termination For Convenience",
                       dependency_type=DependencyType.DEPENDS_ON, strength=0.65,
                       reason="Renewal depends on termination terms"),
        DependencyEdge(source_label="Non-Compete", target_label="Expiration Date",
                       dependency_type=DependencyType.REQUIRES, strength=0.75,
                       reason="Non-compete duration tied to contract expiration"),
    ]


@pytest.fixture
def sample_risk_assessment():
    """RiskAssessment with findings."""
    return RiskAssessment(
        contract_id="test-contract-001",
        overall_risk_score=45,
        risk_level="MEDIUM",
        findings=[
            RiskFinding(
                label="Non-Compete", severity=RiskSeverity.HIGH,
                reason="May restrict business operations",
                recommendation="Narrow scope and duration",
                clause_text="2-year non-compete.", confidence=0.78,
            ),
            RiskFinding(
                label="Cap On Liability", severity=RiskSeverity.HIGH,
                reason="Limits recourse for damages",
                recommendation="Negotiate higher cap",
                clause_text="$100K cap.", confidence=0.82,
            ),
        ],
        missing_clause_risks=[
            RiskFinding(
                label="Source Code Escrow", severity=RiskSeverity.HIGH,
                reason="MISSING: No access to source code if vendor fails",
                recommendation="Request source code escrow",
            ),
        ],
        summary="Risk Score: 45/100 | 2 HIGH risk issue(s) | 1 missing protection(s)",
    )


# ---------------------------------------------------------------------------
# Mock service factories
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_service(monkeypatch):
    """Mock EmbeddingService avoiding S-BERT model load."""
    mock = MagicMock()
    rng = np.random.RandomState(42)

    def _embed(text, use_cache=True):
        seed = hash(text) % 2**31
        return rng.RandomState(seed).randn(384).tolist()

    def _embed_batch(texts, use_cache=True, batch_size=32):
        return [_embed(t) for t in texts]

    def _compute_similarity(emb1, emb2):
        a, b = np.array(emb1), np.array(emb2)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _find_similar(query_emb, candidates, top_k=10, threshold=0.0):
        results = []
        for i, c in enumerate(candidates):
            score = _compute_similarity(query_emb, c)
            if score >= threshold:
                results.append((i, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    mock.embed = MagicMock(side_effect=_embed)
    mock.embed_batch = MagicMock(side_effect=_embed_batch)
    mock.embed_entity = MagicMock(side_effect=lambda n, t: _embed(f"{n}:{t}"))
    mock.embed_triple = MagicMock(side_effect=lambda s, p, o: _embed(f"{s}:{p}:{o}"))
    mock.compute_similarity = MagicMock(side_effect=_compute_similarity)
    mock.find_similar = MagicMock(side_effect=_find_similar)
    mock.cluster_embeddings = MagicMock(return_value=[0, 0, 1, 1, 2])

    monkeypatch.setattr("src.utils.embedding.get_embedding_service", lambda: mock)
    return mock


@pytest.fixture
def mock_llm_service(monkeypatch):
    """Mock LLMService avoiding API calls."""
    mock = MagicMock()
    mock.generate = MagicMock(return_value=("Mocked LLM response", "mock-model"))
    mock.extract_entities = MagicMock(return_value=[
        {"name": "ACME Corp", "type": "Party", "confidence": "HIGH"},
    ])
    mock.extract_relations = MagicMock(return_value=[
        {"subject": "ACME Corp", "predicate": "LICENSES_TO", "object": "Beta LLC", "confidence": "HIGH"},
    ])
    mock.identify_duplicates = MagicMock(return_value=[
        ["ACME Corp", "ACME Corporation", "the Company"],
    ])
    mock.select_canonical = MagicMock(return_value=("ACME Corporation", ["ACME Corp", "the Company"]))
    mock.answer_query = MagicMock(return_value=("The license is non-exclusive.", 0.9))

    monkeypatch.setattr("src.utils.llm.get_llm_service", lambda: mock)
    return mock


@pytest.fixture
def minimal_contract_text():
    """Minimal contract for fast tests."""
    return """SOFTWARE LICENSE AGREEMENT

This Agreement is entered into as of January 1, 2024 between
ACME SOFTWARE INC. ("Licensor") and BETA TECHNOLOGIES LLC ("Licensee").

1. LICENSE GRANT
Licensor grants Licensee a non-exclusive, non-transferable license to use the Software
for internal business purposes only.

2. INTELLECTUAL PROPERTY
All intellectual property rights in the Software remain with Licensor.

3. RESTRICTIONS
Licensee shall not reverse engineer, decompile, or disassemble the Software.

4. NON-COMPETE
Licensee shall not develop competing software products for a period of 2 years.

5. CAP ON LIABILITY
Licensor's total liability under this Agreement shall not exceed $100,000.

6. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

7. TERMINATION
Either party may terminate this Agreement with 30 days written notice.

8. EXPIRATION
This Agreement expires on December 31, 2025.

9. RENEWAL
This Agreement will automatically renew for successive one-year terms.

10. INSURANCE
Licensee shall maintain commercial general liability insurance of at least $1,000,000.
"""
