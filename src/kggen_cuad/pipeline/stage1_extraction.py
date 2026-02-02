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

    # =========================================================================
    # Contract Edit Processing for KGGEN Labeling
    # =========================================================================

    async def process_edit(
        self,
        edit_data: dict,
        include_context: bool = True,
    ) -> tuple[list[Entity], list[Triple]]:
        """
        Process a single contract edit for KGGEN extraction.

        This method provides label-guided extraction based on CUAD categories
        affected by the edit.

        Args:
            edit_data: Prepared edit data from ContractEditService.prepare_edits_for_kggen()
            include_context: Whether to use full context text

        Returns:
            Tuple of (entities, triples) extracted from the edit
        """
        edit_id = edit_data.get("edit_id", "")
        contract_id = UUID(edit_data.get("contract_id", str(uuid4())))
        text = edit_data.get("text", "")
        affected_labels = edit_data.get("affected_labels", [])
        label_hints = edit_data.get("cuad_label_hints", "")

        logger.info(
            "edit_extraction_started",
            edit_id=edit_id,
            affected_labels=affected_labels,
        )

        # Extract entities with label guidance
        entities = await self._extract_entities_with_labels(
            text=text,
            contract_id=contract_id,
            affected_labels=affected_labels,
            label_hints=label_hints,
        )

        # Extract relations
        triples = await self._extract_relations(
            text=text,
            entities=entities,
            contract_id=contract_id,
        )

        # Tag extracted items with edit source
        edit_uuid = UUID(edit_id) if edit_id else uuid4()
        for entity in entities:
            entity.source_edit_id = edit_uuid
            entity.cuad_labels = affected_labels

        for triple in triples:
            triple.source_edit_id = edit_uuid
            triple.cuad_labels = affected_labels

        logger.info(
            "edit_extraction_completed",
            edit_id=edit_id,
            entities=len(entities),
            triples=len(triples),
        )

        return entities, triples

    async def _extract_entities_with_labels(
        self,
        text: str,
        contract_id: UUID,
        affected_labels: list[str],
        label_hints: str,
    ) -> list[Entity]:
        """
        Extract entities with CUAD label guidance.

        Uses the affected labels to focus extraction on relevant entity types.
        """
        # Map CUAD labels to expected entity types
        expected_types = self._labels_to_entity_types(affected_labels)

        raw_entities = await self.llm.extract_entities(text)

        entities = []
        for raw in raw_entities:
            try:
                entity_type = self._parse_entity_type(raw.get("type", ""))
                if entity_type is None:
                    continue

                # Boost confidence for entities matching expected types
                confidence = raw.get("confidence", 0.8)
                if entity_type in expected_types:
                    confidence = min(confidence + 0.1, 1.0)

                entity = Entity(
                    id=uuid4(),
                    name=raw.get("name", ""),
                    entity_type=entity_type,
                    normalized_name=self._normalize_name(raw.get("name", "")),
                    properties=raw.get("properties", {}),
                    source_text=text[:500],
                    source_contract_id=contract_id,
                    confidence=confidence,
                    cuad_labels=affected_labels,
                )
                entities.append(entity)
            except Exception as e:
                logger.warning("entity_parse_failed", raw=raw, error=str(e))
                continue

        return entities

    def _labels_to_entity_types(self, labels: list[str]) -> set[EntityType]:
        """Map CUAD labels to expected entity types."""
        type_map = {
            "Parties": {EntityType.PARTY},
            "IP Ownership": {EntityType.IP_ASSET, EntityType.PARTY},
            "IP Assignment": {EntityType.IP_ASSET, EntityType.PARTY},
            "License Grant": {EntityType.IP_ASSET, EntityType.PARTY},
            "Cap on Liability": {EntityType.LIABILITY_PROVISION},
            "Liquidated Damages": {EntityType.LIABILITY_PROVISION},
            "Uncapped Liability": {EntityType.LIABILITY_PROVISION},
            "Effective Date": {EntityType.TEMPORAL},
            "Expiration Date": {EntityType.TEMPORAL},
            "Renewal Term": {EntityType.TEMPORAL},
            "Non-Compete": {EntityType.RESTRICTION},
            "Exclusivity": {EntityType.RESTRICTION},
            "Anti-Assignment": {EntityType.RESTRICTION},
            "Governing Law": {EntityType.JURISDICTION},
            "Revenue Commitment": {EntityType.OBLIGATION},
            "Audit Rights": {EntityType.OBLIGATION},
            "Indemnification": {EntityType.OBLIGATION},
        }

        expected = set()
        for label in labels:
            if label in type_map:
                expected.update(type_map[label])

        return expected if expected else set(EntityType)

    async def process_edits_batch(
        self,
        edits_data: list[dict],
        max_concurrent: int = 3,
    ) -> dict[str, tuple[list[Entity], list[Triple]]]:
        """
        Process multiple contract edits in batch.

        Args:
            edits_data: List of prepared edit data from ContractEditService
            max_concurrent: Max concurrent extractions

        Returns:
            Dict mapping edit_id to (entities, triples)
        """
        import asyncio

        results: dict[str, tuple[list[Entity], list[Triple]]] = {}

        # Process in batches
        for i in range(0, len(edits_data), max_concurrent):
            batch = edits_data[i:i + max_concurrent]

            tasks = [self.process_edit(edit) for edit in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for edit_data, result in zip(batch, batch_results):
                edit_id = edit_data.get("edit_id", "")
                if isinstance(result, Exception):
                    logger.error(
                        "edit_extraction_failed",
                        edit_id=edit_id,
                        error=str(result),
                    )
                    results[edit_id] = ([], [])
                else:
                    results[edit_id] = result

        logger.info(
            "batch_edit_extraction_completed",
            edits=len(edits_data),
            successful=sum(1 for r in results.values() if r[0] or r[1]),
        )

        return results

    async def extract_from_last_edits(
        self,
        contract_id: UUID,
        count: int = 10,
        unprocessed_only: bool = True,
    ) -> tuple[list[Entity], list[Triple], list[str]]:
        """
        Convenience method to retrieve and process last contract edits.

        Combines edit retrieval and extraction in one call.

        Args:
            contract_id: The contract ID
            count: Number of recent edits to process
            unprocessed_only: Only process unprocessed edits

        Returns:
            Tuple of (all_entities, all_triples, edit_ids_processed)
        """
        from kggen_cuad.services.contract_edit_service import get_contract_edit_service

        edit_service = get_contract_edit_service()

        # Get last edits
        edits = edit_service.get_last_edits(
            contract_id=contract_id,
            count=count,
            unprocessed_only=unprocessed_only,
        )

        if not edits:
            logger.info("no_edits_to_process", contract_id=str(contract_id))
            return [], [], []

        # Prepare for extraction
        edits_data = edit_service.prepare_edits_for_kggen(edits, include_context=True)

        # Process all edits
        results = await self.process_edits_batch(edits_data)

        # Aggregate results
        all_entities: list[Entity] = []
        all_triples: list[Triple] = []
        edit_ids_processed: list[str] = []

        for edit_id, (entities, triples) in results.items():
            all_entities.extend(entities)
            all_triples.extend(triples)
            edit_ids_processed.append(edit_id)

            # Mark edit as processed
            edit_service.mark_edit_processed(
                edit_id=UUID(edit_id),
                entities_extracted=[e.id for e in entities],
                triples_extracted=[t.id for t in triples],
            )

        logger.info(
            "last_edits_extracted",
            contract_id=str(contract_id),
            edits_processed=len(edit_ids_processed),
            entities=len(all_entities),
            triples=len(all_triples),
        )

        return all_entities, all_triples, edit_ids_processed


def get_extraction_stage() -> ExtractionStage:
    """Get extraction stage instance."""
    return ExtractionStage()
