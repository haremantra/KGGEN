"""
Stage 1: Extraction

Extracts entities and relations from contract text using LLMs.
"""

from typing import Any
from uuid import UUID, uuid4

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.contract import Contract, ContractSection
from kggen_cuad.models.triple import Entity, Triple, EntityType, PredicateType
from kggen_cuad.services.llm_service import get_llm_service, LLMService
from kggen_cuad.services.contract_loader import get_contract_loader, ContractLoader

logger = structlog.get_logger(__name__)


class ExtractionStage:
    """
    Stage 1: Entity and Relation Extraction.

    Processes contract text to extract:
    - Entities (parties, IP assets, obligations, etc.)
    - Relations (licensing, ownership, obligations, etc.)
    """

    def __init__(self):
        self.settings = get_settings()
        self._llm: LLMService | None = None
        self._loader: ContractLoader | None = None

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service()
        return self._llm

    @property
    def loader(self) -> ContractLoader:
        if self._loader is None:
            self._loader = get_contract_loader()
        return self._loader

    async def process_contract(
        self,
        contract: Contract,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> tuple[list[Entity], list[Triple]]:
        """
        Extract entities and relations from a contract.

        Args:
            contract: The contract to process
            chunk_size: Size of text chunks for processing
            overlap: Overlap between chunks

        Returns:
            Tuple of (entities, triples)
        """
        logger.info(
            "extraction_started",
            contract_id=str(contract.id),
            word_count=contract.word_count,
        )

        # Chunk the contract for processing
        chunks = self.loader.chunk_contract(
            contract=contract,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_entities: list[Entity] = []
        all_triples: list[Triple] = []
        entity_name_map: dict[str, Entity] = {}

        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.debug(
                "processing_chunk",
                chunk=i + 1,
                total=len(chunks),
            )

            # Extract entities from chunk
            chunk_entities = await self._extract_entities(chunk.text, contract.id)

            # Deduplicate entities within contract
            for entity in chunk_entities:
                key = f"{entity.normalized_name}:{entity.entity_type.value}"
                if key not in entity_name_map:
                    entity_name_map[key] = entity
                    all_entities.append(entity)

            # Extract relations from chunk
            chunk_triples = await self._extract_relations(
                chunk.text,
                list(entity_name_map.values()),
                contract.id,
            )
            all_triples.extend(chunk_triples)

        # Post-process: resolve entity references in triples
        all_triples = self._resolve_triple_entities(all_triples, entity_name_map)

        logger.info(
            "extraction_completed",
            contract_id=str(contract.id),
            entities=len(all_entities),
            triples=len(all_triples),
        )

        return all_entities, all_triples

    async def _extract_entities(
        self,
        text: str,
        contract_id: UUID,
    ) -> list[Entity]:
        """Extract entities from a text chunk."""
        raw_entities = await self.llm.extract_entities(text)

        entities = []
        for raw in raw_entities:
            try:
                entity_type = self._parse_entity_type(raw.get("type", ""))
                if entity_type is None:
                    continue

                entity = Entity(
                    id=uuid4(),
                    name=raw.get("name", ""),
                    entity_type=entity_type,
                    normalized_name=self._normalize_name(raw.get("name", "")),
                    properties=raw.get("properties", {}),
                    source_text=text[:500],
                )
                entities.append(entity)
            except Exception as e:
                logger.warning("entity_parse_failed", raw=raw, error=str(e))
                continue

        return entities

    async def _extract_relations(
        self,
        text: str,
        entities: list[Entity],
        contract_id: UUID,
    ) -> list[Triple]:
        """Extract relations from a text chunk."""
        raw_relations = await self.llm.extract_relations(
            text,
            [{"name": e.name, "type": e.entity_type.value} for e in entities],
        )

        triples = []
        for raw in raw_relations:
            try:
                predicate = self._parse_predicate_type(raw.get("predicate", ""))
                if predicate is None:
                    continue

                # Find subject and object entities
                subject_name = raw.get("subject", "")
                object_name = raw.get("object", "")

                subject_id = self._find_entity_id(subject_name, entities)
                object_id = self._find_entity_id(object_name, entities)

                if subject_id is None or object_id is None:
                    # Create placeholder IDs for later resolution
                    subject_id = subject_id or uuid4()
                    object_id = object_id or uuid4()

                triple = Triple(
                    id=uuid4(),
                    subject_id=subject_id,
                    predicate=predicate,
                    object_id=object_id,
                    confidence=raw.get("confidence", 0.8),
                    properties=raw.get("properties", {}),
                    source_text=text[:500],
                )
                triples.append(triple)
            except Exception as e:
                logger.warning("relation_parse_failed", raw=raw, error=str(e))
                continue

        return triples

    def _parse_entity_type(self, type_str: str) -> EntityType | None:
        """Parse entity type string to enum."""
        type_map = {
            "party": EntityType.PARTY,
            "ipasset": EntityType.IP_ASSET,
            "ip_asset": EntityType.IP_ASSET,
            "obligation": EntityType.OBLIGATION,
            "restriction": EntityType.RESTRICTION,
            "liabilityprovision": EntityType.LIABILITY_PROVISION,
            "liability_provision": EntityType.LIABILITY_PROVISION,
            "temporal": EntityType.TEMPORAL,
            "jurisdiction": EntityType.JURISDICTION,
            "contractclause": EntityType.CONTRACT_CLAUSE,
            "contract_clause": EntityType.CONTRACT_CLAUSE,
        }
        return type_map.get(type_str.lower().replace(" ", ""))

    def _parse_predicate_type(self, pred_str: str) -> PredicateType | None:
        """Parse predicate string to enum."""
        pred_map = {
            "licenses_to": PredicateType.LICENSES_TO,
            "owns": PredicateType.OWNS,
            "assigns": PredicateType.ASSIGNS,
            "has_obligation": PredicateType.HAS_OBLIGATION,
            "subject_to_restriction": PredicateType.SUBJECT_TO_RESTRICTION,
            "has_liability": PredicateType.HAS_LIABILITY,
            "governed_by": PredicateType.GOVERNED_BY,
            "contains_clause": PredicateType.CONTAINS_CLAUSE,
            "effective_on": PredicateType.EFFECTIVE_ON,
            "terminates_on": PredicateType.TERMINATES_ON,
        }
        return pred_map.get(pred_str.lower().replace(" ", "_"))

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        # Remove common articles and clean whitespace
        normalized = name.strip()
        for prefix in ["the ", "a ", "an "]:
            if normalized.lower().startswith(prefix):
                normalized = normalized[len(prefix):]
        return normalized.strip()

    def _find_entity_id(
        self,
        name: str,
        entities: list[Entity],
    ) -> UUID | None:
        """Find entity ID by name."""
        normalized = self._normalize_name(name)

        # Exact match
        for entity in entities:
            if entity.normalized_name == normalized:
                return entity.id
            if entity.name.lower() == name.lower():
                return entity.id

        # Fuzzy match (substring)
        for entity in entities:
            if normalized.lower() in entity.normalized_name.lower():
                return entity.id
            if entity.normalized_name.lower() in normalized.lower():
                return entity.id

        return None

    def _resolve_triple_entities(
        self,
        triples: list[Triple],
        entity_map: dict[str, Entity],
    ) -> list[Triple]:
        """Resolve entity references in triples."""
        # Build ID lookup
        id_map: dict[str, UUID] = {}
        for key, entity in entity_map.items():
            id_map[entity.name.lower()] = entity.id
            id_map[entity.normalized_name.lower()] = entity.id

        resolved = []
        for triple in triples:
            # Triples with valid UUIDs pass through
            resolved.append(triple)

        return resolved

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def process_contracts_batch(
        self,
        contracts: list[Contract],
        max_concurrent: int = 3,
    ) -> dict[UUID, tuple[list[Entity], list[Triple]]]:
        """
        Process multiple contracts in batch.

        Returns dict mapping contract_id to (entities, triples).
        """
        import asyncio

        results: dict[UUID, tuple[list[Entity], list[Triple]]] = {}

        # Process in batches
        for i in range(0, len(contracts), max_concurrent):
            batch = contracts[i:i + max_concurrent]

            tasks = [self.process_contract(c) for c in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for contract, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "contract_extraction_failed",
                        contract_id=str(contract.id),
                        error=str(result),
                    )
                    results[contract.id] = ([], [])
                else:
                    results[contract.id] = result

        logger.info(
            "batch_extraction_completed",
            contracts=len(contracts),
            successful=sum(1 for r in results.values() if r[0] or r[1]),
        )

        return results

    # =========================================================================
    # Entity Type Detection
    # =========================================================================

    async def detect_contract_type(self, contract: Contract) -> str | None:
        """Detect the type of contract from text."""
        return self.loader.identify_contract_type(contract)

    async def extract_parties(self, contract: Contract) -> list[str]:
        """Extract party names from contract."""
        return self.loader.extract_parties(contract)


def get_extraction_stage() -> ExtractionStage:
    """Get extraction stage instance."""
    return ExtractionStage()
