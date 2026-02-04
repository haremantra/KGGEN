"""Contract Knowledge Graph Extractor.

Stage 1: LLM-based entity and relation extraction from legal contracts.
"""

import json
import uuid
from anthropic import Anthropic
from ..config import settings
from ..models.schema import (
    ExtractionResult,
    KGNode,
    Triple,
    NodeType,
)


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
- licenses_to: Party licenses IP to another party
- owns: Party owns an asset
- assigns: Party assigns rights to another
- has_obligation: Party has an obligation
- subject_to_restriction: Party is subject to a restriction
- has_liability: Party has liability provisions
- governed_by: Contract is governed by jurisdiction
- contains_clause: Contract contains a clause
- effective_on: Something becomes effective on a date
- terminates_on: Something terminates on a date

## Output Format:
Return a JSON object with two arrays:
1. "entities": List of extracted entities with id, name, type, and properties
2. "triples": List of (subject, predicate, object, properties) tuples

## Guidelines:
- Extract ALL parties mentioned, including their roles (Licensor, Licensee, etc.)
- Preserve legal distinctions (exclusive vs non-exclusive, limited vs unlimited)
- Respect contract defined terms (capitalized terms have special meaning)
- Extract dates with their context (effective, expiration, notice periods)
- Capture conditions and exceptions
- Note jurisdiction-specific terms
- Be precise - only extract what is explicitly stated in the contract"""


class ContractExtractor:
    """Extracts knowledge graphs from legal contracts using LLM."""

    def __init__(self, model: str | None = None):
        """Initialize the extractor.

        Args:
            model: LLM model to use. Defaults to settings.default_llm_model.
        """
        self.model = model or settings.default_llm_model
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def extract(self, contract_text: str, contract_id: str | None = None) -> ExtractionResult:
        """Extract entities and relations from a contract.

        Args:
            contract_text: The full text of the contract.
            contract_id: Optional identifier for the contract.

        Returns:
            ExtractionResult containing extracted entities and triples.
        """
        contract_id = contract_id or str(uuid.uuid4())

        # Call LLM for extraction
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=settings.extraction_temperature,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract entities and relationships from this contract:

---CONTRACT START---
{contract_text}
---CONTRACT END---

Return your response as a JSON object with "entities" and "triples" arrays."""
                }
            ]
        )

        # Parse the response
        response_text = response.content[0].text

        # Try to extract JSON from the response
        extraction_data = self._parse_json_response(response_text)

        # Convert to typed models
        entities = self._parse_entities(extraction_data.get("entities", []), contract_id)
        triples = self._parse_triples(extraction_data.get("triples", []))

        return ExtractionResult(
            contract_id=contract_id,
            llm_model=self.model,
            entities=entities,
            triples=triples,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks first
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
            # Try to find any JSON object in the text
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
                    confidence_score=raw.get("confidence", 1.0),
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
                    confidence=raw.get("confidence", 1.0),
                    source_text=raw.get("source_text"),
                )
                if triple.subject and triple.predicate and triple.object:
                    triples.append(triple)
            except Exception:
                continue
        return triples
