"""Stage 3: Entity and relation resolution."""

from ..pipeline import ContractAnalysis, AnalyzedClause
from ..models.schema import KGNode, Triple, NodeType


def analysis_to_entities_triples(
    analysis: ContractAnalysis,
) -> tuple[list[KGNode], list[Triple]]:
    """Bridge function: convert ContractAnalysis to (entities, triples) for resolution.

    Maps AnalyzedClause data to KGNode entities and Triple relationships
    so the EntityResolver can process them.
    """
    entities: list[KGNode] = []
    triples: list[Triple] = []

    for clause in analysis.analyzed_clauses:
        # Create entities from extracted entity names
        for entity_name in clause.entities:
            entities.append(KGNode(
                id=f"{analysis.contract_id}:{entity_name}",
                name=entity_name,
                type=NodeType.CONTRACT_CLAUSE,
                source_contract_id=analysis.contract_id,
                cuad_label=clause.cuad_label,
                confidence_score=clause.label_confidence,
            ))

        # Create triples from extracted relationships
        for rel in clause.relationships:
            triples.append(Triple(
                subject=rel.get("subject", ""),
                predicate=rel.get("predicate", ""),
                object=rel.get("object", ""),
                confidence=clause.label_confidence,
                source_text=clause.text[:200] if clause.text else None,
            ))

    return entities, triples
