"""Contract Knowledge Graph Extractor.

Stage 1: LLM-based entity and relation extraction from legal contracts.
Supports text chunking, within-doc dedup, and categorical confidence calibration.
"""

import json
import re
import uuid
from ..config import settings
from ..models.schema import (
    ExtractionResult,
    KGNode,
    Triple,
    NodeType,
)

# Categorical confidence mapping — LLMs are poorly calibrated at numeric floats
CONFIDENCE_MAP = {"LOW": 0.4, "MEDIUM": 0.7, "HIGH": 0.9}
DEFAULT_CONFIDENCE = 0.7


EXTRACTION_SYSTEM_PROMPT = """You are a legal contract analysis expert. Your task is to extract structured knowledge from legal contracts in the form of entities and relationships (triples).

## Entity Types to Extract:
1. **Party** - Companies, individuals, or entities that are parties to the contract
2. **IPAsset** - Intellectual property (software, patents, trademarks, copyrights, source code)
3. **Obligation** - Things a party must do (provide, maintain, support, deliver)
4. **Restriction** - Things a party cannot do (non-compete, non-disclosure, non-solicitation)
5. **LiabilityProvision** - Liability caps, limitations, indemnifications
6. **Temporal** - Dates (effective date, expiration, renewal periods)
7. **Jurisdiction** - Governing law, venue, legal jurisdiction
8. **ContractClause** - Specific contract clauses/sections

## Relationship Types to Extract:
- LICENSES_TO, OWNS, ASSIGNS, HAS_OBLIGATION, SUBJECT_TO_RESTRICTION
- HAS_LIABILITY, GOVERNED_BY, CONTAINS_CLAUSE, EFFECTIVE_ON, TERMINATES_ON

## Output Format:
Return a JSON object with two arrays:
1. "entities": Each with id, name, type, properties, and confidence (LOW, MEDIUM, or HIGH)
2. "triples": Each with subject, predicate, object, properties, and confidence (LOW, MEDIUM, or HIGH)

## Confidence Levels:
- HIGH: Explicitly stated, unambiguous
- MEDIUM: Clearly implied or inferable from context
- LOW: Possibly present but uncertain

## Guidelines:
- Extract ALL parties mentioned, including their roles
- Preserve legal distinctions (exclusive vs non-exclusive)
- Respect contract defined terms (capitalized terms)
- Extract dates with their context
- Capture conditions and exceptions
- Be precise - only extract what is explicitly stated"""


def _normalize_name(name: str) -> str:
    """Normalize entity name for dedup matching."""
    normalized = name.strip()
    for prefix in ["the ", "a ", "an "]:
        if normalized.lower().startswith(prefix):
            normalized = normalized[len(prefix):]
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _map_confidence(conf_value) -> float:
    """Map categorical or numeric confidence to float."""
    if isinstance(conf_value, str):
        return CONFIDENCE_MAP.get(conf_value.upper(), DEFAULT_CONFIDENCE)
    if isinstance(conf_value, (int, float)):
        return float(conf_value)
    return DEFAULT_CONFIDENCE


def _chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at a sentence boundary
        if end < len(text):
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class ContractExtractor:
    """Extracts knowledge graphs from legal contracts using LLM."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.default_llm_model
        from anthropic import Anthropic
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def extract(
        self,
        contract_text: str,
        contract_id: str | None = None,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> ExtractionResult:
        """Extract entities and relations from a contract with chunking and dedup."""
        contract_id = contract_id or str(uuid.uuid4())

        chunks = _chunk_text(contract_text, chunk_size, overlap)

        all_entities: list[KGNode] = []
        all_triples: list[Triple] = []
        entity_dedup: dict[str, KGNode] = {}  # normalized_name:type -> entity

        for i, chunk in enumerate(chunks):
            print(f"  Extracting chunk {i+1}/{len(chunks)}...")
            chunk_data = self._extract_chunk(chunk, contract_id)

            # Dedup entities within contract
            for entity in chunk_data.get("entities", []):
                key = f"{_normalize_name(entity.name)}:{entity.type.value}"
                if key not in entity_dedup:
                    entity_dedup[key] = entity
                    all_entities.append(entity)

            all_triples.extend(chunk_data.get("triples", []))

        return ExtractionResult(
            contract_id=contract_id,
            llm_model=self.model,
            entities=all_entities,
            triples=all_triples,
            metadata={
                "chunks_processed": len(chunks),
                "entities_before_dedup": len(all_entities) + (len(entity_dedup) - len(all_entities)),
            },
        )

    def _extract_chunk(self, chunk_text: str, contract_id: str) -> dict:
        """Extract from a single chunk."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=settings.extraction_temperature,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract entities and relationships from this contract section:

---CONTRACT SECTION---
{chunk_text}
---END---

Return JSON with "entities" and "triples" arrays.
For confidence, use exactly one of: LOW, MEDIUM, or HIGH.""",
                    }
                ],
            )

            response_text = response.content[0].text
            data = self._parse_json_response(response_text)

            entities = self._parse_entities(data.get("entities", []), contract_id)
            triples = self._parse_triples(data.get("triples", []))

            return {"entities": entities, "triples": triples}
        except Exception as e:
            print(f"    Chunk extraction error: {e}")
            return {"entities": [], "triples": []}

    def extract_batch(
        self,
        contracts: list[tuple[str, str]],
    ) -> dict[str, ExtractionResult]:
        """Extract from multiple contracts. Input: list of (text, contract_id) tuples."""
        results = {}
        for text, cid in contracts:
            print(f"Extracting: {cid}")
            results[cid] = self.extract(text, cid)
        return results

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"entities": [], "triples": []}

    def _parse_entities(self, raw_entities: list, contract_id: str) -> list[KGNode]:
        """Convert raw entity dicts to KGNode objects."""
        entities = []
        for raw in raw_entities:
            try:
                node_type = self._map_entity_type(raw.get("type", ""))
                entity = KGNode(
                    id=raw.get("id", str(uuid.uuid4())),
                    name=raw.get("name", "Unknown"),
                    type=node_type,
                    source_contract_id=contract_id,
                    cuad_label=raw.get("cuad_label"),
                    confidence_score=_map_confidence(raw.get("confidence")),
                    properties=raw.get("properties", {}),
                )
                entities.append(entity)
            except Exception:
                continue
        return entities

    def _map_entity_type(self, type_str: str) -> NodeType:
        """Map string type to NodeType enum."""
        type_mapping = {
            "party": NodeType.PARTY,
            "ipasset": NodeType.IP_ASSET,
            "ip_asset": NodeType.IP_ASSET,
            "obligation": NodeType.OBLIGATION,
            "restriction": NodeType.RESTRICTION,
            "liabilityprovision": NodeType.LIABILITY_PROVISION,
            "liability_provision": NodeType.LIABILITY_PROVISION,
            "liability": NodeType.LIABILITY_PROVISION,
            "temporal": NodeType.TEMPORAL,
            "date": NodeType.TEMPORAL,
            "jurisdiction": NodeType.JURISDICTION,
            "contractclause": NodeType.CONTRACT_CLAUSE,
            "contract_clause": NodeType.CONTRACT_CLAUSE,
            "clause": NodeType.CONTRACT_CLAUSE,
            "contract": NodeType.CONTRACT,
        }
        return type_mapping.get(type_str.lower(), NodeType.CONTRACT_CLAUSE)

    def _parse_triples(self, raw_triples: list) -> list[Triple]:
        """Convert raw triple dicts to Triple objects."""
        triples = []
        for raw in raw_triples:
            try:
                triple = Triple(
                    subject=raw.get("subject", ""),
                    predicate=raw.get("predicate", ""),
                    object=raw.get("object", ""),
                    properties=raw.get("properties", {}),
                    confidence=_map_confidence(raw.get("confidence")),
                    source_text=raw.get("source_text"),
                )
                if triple.subject and triple.predicate and triple.object:
                    triples.append(triple)
            except Exception:
                continue
        return triples
