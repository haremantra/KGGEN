"""Tests for src/models/schema.py — Pydantic models and enums."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models.schema import (
    NodeType, EdgeType, KGNode, KGEdge, Triple, ExtractionResult,
    Party, Obligation, Restriction, LiabilityProvision, Temporal,
    Jurisdiction, ContractClause, IPAsset,
)


class TestEnums:

    def test_node_type_values(self):
        assert len(NodeType) == 9

    def test_node_type_includes_party(self):
        assert NodeType.PARTY.value == "Party"

    def test_node_type_includes_ip_asset(self):
        assert NodeType.IP_ASSET.value == "IPAsset"

    def test_edge_type_values(self):
        assert len(EdgeType) == 10

    def test_edge_type_includes_licenses_to(self):
        assert EdgeType.LICENSES_TO.value == "LICENSES_TO"


class TestKGNode:

    def test_creation(self):
        node = KGNode(id="n1", name="Test", type=NodeType.PARTY)
        assert node.name == "Test"
        assert node.type == NodeType.PARTY

    def test_default_confidence(self):
        node = KGNode(id="n1", name="Test", type=NodeType.PARTY)
        assert node.confidence_score == 1.0

    def test_confidence_upper_bound(self):
        with pytest.raises(ValidationError):
            KGNode(id="n1", name="Test", type=NodeType.PARTY, confidence_score=1.5)

    def test_confidence_lower_bound(self):
        with pytest.raises(ValidationError):
            KGNode(id="n1", name="Test", type=NodeType.PARTY, confidence_score=-0.1)

    def test_optional_fields(self):
        node = KGNode(id="n1", name="Test", type=NodeType.PARTY)
        assert node.source_contract_id is None
        assert node.cuad_label is None
        assert node.properties == {}


class TestTriple:

    def test_creation(self):
        t = Triple(subject="A", predicate="LICENSES_TO", object="B")
        assert t.subject == "A"
        assert t.object == "B"

    def test_default_confidence(self):
        t = Triple(subject="A", predicate="P", object="B")
        assert t.confidence == 1.0

    def test_with_source_text(self):
        t = Triple(subject="A", predicate="P", object="B", source_text="clause text")
        assert t.source_text == "clause text"


class TestSpecializedNodes:

    def test_party(self):
        p = Party(id="p1", name="ACME", type=NodeType.PARTY, role="Licensor",
                  legal_entity_type="Corporation")
        assert p.role == "Licensor"
        assert p.legal_entity_type == "Corporation"

    def test_obligation(self):
        o = Obligation(id="o1", name="Provide support", type=NodeType.OBLIGATION,
                       obligor="ACME", obligation_type="service", is_conditional=True)
        assert o.is_conditional is True
        assert o.obligor == "ACME"

    def test_restriction(self):
        r = Restriction(id="r1", name="Non-compete", type=NodeType.RESTRICTION,
                        restriction_type="non-compete", duration="2 years", scope="North America")
        assert r.duration == "2 years"
        assert r.scope == "North America"

    def test_liability_provision(self):
        lp = LiabilityProvision(id="l1", name="Liability Cap", type=NodeType.LIABILITY_PROVISION,
                                provision_type="cap", amount="$100,000",
                                exceptions=["willful misconduct", "IP infringement"])
        assert lp.amount == "$100,000"
        assert len(lp.exceptions) == 2

    def test_temporal(self):
        t = Temporal(id="t1", name="Jan 2024", type=NodeType.TEMPORAL,
                     temporal_type="effective_date")
        assert t.temporal_type == "effective_date"

    def test_jurisdiction(self):
        j = Jurisdiction(id="j1", name="Delaware", type=NodeType.JURISDICTION,
                         jurisdiction_type="governing_law")
        assert j.legal_system == "common_law"  # default

    def test_contract_clause(self):
        cc = ContractClause(id="cc1", name="Section 1", type=NodeType.CONTRACT_CLAUSE,
                            clause_type="license", section_number="1.1",
                            text="Full clause text here.")
        assert cc.section_number == "1.1"

    def test_ip_asset(self):
        ip = IPAsset(id="ip1", name="Software", type=NodeType.IP_ASSET,
                     ip_type="software", registration_number="REG-123")
        assert ip.ip_type == "software"


class TestExtractionResult:

    def test_creation(self):
        er = ExtractionResult(contract_id="c1", llm_model="test-model")
        assert er.contract_id == "c1"
        assert isinstance(er.extraction_timestamp, datetime)

    def test_defaults(self):
        er = ExtractionResult(contract_id="c1", llm_model="test-model")
        assert er.entities == []
        assert er.triples == []
        assert er.metadata == {}
