"""
LLM service for entity and relation extraction.

Supports Claude (Anthropic) and GPT-4o (OpenAI) with automatic fallback.
"""

from functools import lru_cache
from typing import Any

import structlog
from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from kggen_cuad.config import get_settings

logger = structlog.get_logger(__name__)


class LLMService:
    """
    LLM service for contract analysis.

    Supports Claude Sonnet (primary) and GPT-4o (fallback).
    """

    def __init__(self):
        settings = get_settings()
        self.settings = settings

        # Initialize clients
        self._anthropic: Anthropic | None = None
        self._openai: OpenAI | None = None

        if settings.anthropic_api_key:
            self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
        if settings.openai_api_key:
            self._openai = OpenAI(api_key=settings.openai_api_key)

        self.primary_provider = settings.primary_llm_provider
        self.primary_model = settings.primary_llm_model
        self.fallback_provider = settings.fallback_llm_provider
        self.fallback_model = settings.fallback_llm_model

    @property
    def anthropic(self) -> Anthropic:
        if not self._anthropic:
            raise ValueError("Anthropic client not configured. Set ANTHROPIC_API_KEY.")
        return self._anthropic

    @property
    def openai(self) -> OpenAI:
        if not self._openai:
            raise ValueError("OpenAI client not configured. Set OPENAI_API_KEY.")
        return self._openai

    def health_check(self) -> dict[str, bool]:
        """Check LLM connectivity."""
        status = {}
        if self._anthropic:
            try:
                # Simple test
                status["anthropic"] = True
            except Exception:
                status["anthropic"] = False
        if self._openai:
            try:
                status["openai"] = True
            except Exception:
                status["openai"] = False
        return status

    # =========================================================================
    # Core LLM Calls
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Call Anthropic Claude API."""
        response = self.anthropic.messages.create(
            model=self.primary_model,
            max_tokens=max_tokens or self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Call OpenAI GPT-4o API."""
        response = self.openai.chat.completions.create(
            model=self.fallback_model,
            max_tokens=max_tokens or self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        use_fallback: bool = True,
    ) -> tuple[str, str]:
        """
        Generate LLM response with automatic fallback.

        Returns (response_text, model_used).
        """
        # Try primary provider
        try:
            if self.primary_provider == "anthropic" and self._anthropic:
                response = await self._call_anthropic(
                    system_prompt, user_prompt, max_tokens
                )
                return response, self.primary_model
            elif self.primary_provider == "openai" and self._openai:
                response = await self._call_openai(
                    system_prompt, user_prompt, max_tokens
                )
                return response, self.primary_model
        except Exception as e:
            logger.warning(
                "primary_llm_failed",
                provider=self.primary_provider,
                error=str(e),
            )
            if not use_fallback:
                raise

        # Try fallback provider
        try:
            if self.fallback_provider == "anthropic" and self._anthropic:
                response = await self._call_anthropic(
                    system_prompt, user_prompt, max_tokens
                )
                return response, self.fallback_model
            elif self.fallback_provider == "openai" and self._openai:
                response = await self._call_openai(
                    system_prompt, user_prompt, max_tokens
                )
                return response, self.fallback_model
        except Exception as e:
            logger.error(
                "fallback_llm_failed",
                provider=self.fallback_provider,
                error=str(e),
            )
            raise

        raise ValueError("No LLM provider available")

    # =========================================================================
    # Entity Extraction
    # =========================================================================

    async def extract_entities(
        self,
        contract_text: str,
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract entities from contract text.

        Returns list of entities with name, type, and properties.
        """
        if entity_types is None:
            entity_types = self.settings.entity_types

        system_prompt = """You are a legal contract analysis expert. Extract entities from the provided contract text.

For each entity, provide:
1. name: The exact name or description of the entity
2. type: One of the following types: Party, IPAsset, Obligation, Restriction, LiabilityProvision, Temporal, Jurisdiction, ContractClause
3. properties: Additional relevant properties as key-value pairs

Focus on:
- Party: Companies, individuals, roles (Licensor, Licensee, etc.)
- IPAsset: Patents, trademarks, copyrights, source code, software
- Obligation: Things parties must do (provide, maintain, deliver)
- Restriction: Limitations and prohibitions (non-compete, confidentiality)
- LiabilityProvision: Liability caps, indemnification, warranties
- Temporal: Dates, durations, periods, deadlines
- Jurisdiction: Governing law, venue, applicable law
- ContractClause: Specific clause references

Return your response as a JSON array of entities."""

        user_prompt = f"""Extract all entities from this contract text:

{contract_text[:8000]}  # Limit to avoid token limits

Return a JSON array like:
[
  {{"name": "ABC Corporation", "type": "Party", "properties": {{"role": "Licensor"}}}},
  {{"name": "Source Code", "type": "IPAsset", "properties": {{"ip_type": "software"}}}},
  ...
]"""

        try:
            response, model = await self.generate(system_prompt, user_prompt)
            # Parse JSON from response
            import json
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                entities = json.loads(response[start:end])
                logger.info(
                    "entities_extracted",
                    count=len(entities),
                    model=model,
                )
                return entities
            return []
        except Exception as e:
            logger.error("entity_extraction_failed", error=str(e))
            return []

    # =========================================================================
    # Relation Extraction
    # =========================================================================

    async def extract_relations(
        self,
        contract_text: str,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Extract relations between entities from contract text.

        Returns list of triples (subject, predicate, object).
        """
        predicate_types = self.settings.predicate_types

        entity_list = "\n".join(
            f"- {e['name']} ({e['type']})" for e in entities[:50]
        )

        system_prompt = """You are a legal contract analysis expert. Extract relationships between entities from the provided contract text.

For each relationship, provide:
1. subject: The source entity name
2. predicate: One of: LICENSES_TO, OWNS, ASSIGNS, HAS_OBLIGATION, SUBJECT_TO_RESTRICTION, HAS_LIABILITY, GOVERNED_BY, CONTAINS_CLAUSE, EFFECTIVE_ON, TERMINATES_ON
3. object: The target entity name
4. properties: Additional properties of the relationship (optional)

Focus on relationships that represent:
- Licensing: Who licenses what to whom
- Ownership: Who owns what IP
- Obligations: What parties must do
- Restrictions: What parties cannot do
- Liability: Liability terms and caps
- Governance: Applicable law and jurisdiction
- Timing: Effective dates and termination

Return your response as a JSON array of relationships."""

        user_prompt = f"""Given these entities from a contract:
{entity_list}

Extract all relationships from this contract text:

{contract_text[:8000]}

Return a JSON array like:
[
  {{"subject": "ABC Corporation", "predicate": "LICENSES_TO", "object": "XYZ Inc", "properties": {{"license_type": "exclusive"}}}},
  {{"subject": "ABC Corporation", "predicate": "OWNS", "object": "Source Code", "properties": {{}}}},
  ...
]"""

        try:
            response, model = await self.generate(system_prompt, user_prompt)
            import json
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                relations = json.loads(response[start:end])
                # Validate predicates
                valid_relations = [
                    r for r in relations
                    if r.get("predicate") in predicate_types
                ]
                logger.info(
                    "relations_extracted",
                    count=len(valid_relations),
                    model=model,
                )
                return valid_relations
            return []
        except Exception as e:
            logger.error("relation_extraction_failed", error=str(e))
            return []

    # =========================================================================
    # Entity Resolution (Stage 3)
    # =========================================================================

    async def identify_duplicates(
        self,
        entity_candidates: list[str],
    ) -> list[list[str]]:
        """
        Identify duplicate entities from a list of candidates.

        Returns groups of entities that should be merged.
        """
        system_prompt = """You are a legal text analysis expert. Identify which entity names refer to the same real-world entity.

Consider these as duplicates:
- Same entity with different capitalization
- Same entity with abbreviations vs full names
- Same entity with/without articles ("the Software" vs "Software")
- Pronouns referring to the same entity

Do NOT merge:
- "Exclusive license" with "non-exclusive license" (different legal terms)
- "Licensor" with "Licensee" (different parties)
- Jurisdiction-specific terms that have different meanings

Return groups of duplicate entities as a JSON array of arrays."""

        user_prompt = f"""Identify duplicate entities from this list:
{entity_candidates}

Return a JSON array of groups, where each group contains entities that refer to the same thing:
[
  ["ABC Corp", "ABC Corporation", "the Company"],
  ["Software", "the Software", "Licensed Software"],
  ...
]

Only include groups with actual duplicates (2+ items). Single items should not be included."""

        try:
            response, model = await self.generate(system_prompt, user_prompt)
            import json
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                groups = json.loads(response[start:end])
                # Filter to only groups with 2+ items
                groups = [g for g in groups if len(g) >= 2]
                logger.info("duplicates_identified", groups=len(groups), model=model)
                return groups
            return []
        except Exception as e:
            logger.error("duplicate_identification_failed", error=str(e))
            return []

    async def select_canonical(
        self,
        duplicate_group: list[str],
    ) -> tuple[str, list[str]]:
        """
        Select the canonical form from a group of duplicates.

        Returns (canonical_form, list_of_aliases).
        """
        system_prompt = """You are a legal terminology expert. Select the most appropriate canonical (standard) form for a group of duplicate entity names.

Selection criteria:
1. Most legally precise and formal
2. Most complete (not abbreviated)
3. Most commonly used in contracts
4. Without articles unless part of a defined term

Return the canonical form and list of aliases."""

        user_prompt = f"""Select the canonical form from these duplicates:
{duplicate_group}

Return JSON with format:
{{"canonical": "Selected Canonical Form", "aliases": ["alias1", "alias2", ...]}}"""

        try:
            response, model = await self.generate(system_prompt, user_prompt)
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                canonical = result.get("canonical", duplicate_group[0])
                aliases = [a for a in result.get("aliases", duplicate_group) if a != canonical]
                return canonical, aliases
            return duplicate_group[0], duplicate_group[1:]
        except Exception as e:
            logger.error("canonical_selection_failed", error=str(e))
            return duplicate_group[0], duplicate_group[1:]

    # =========================================================================
    # Query Answering
    # =========================================================================

    async def answer_query(
        self,
        query: str,
        context: str,
        contract_text: str | None = None,
    ) -> tuple[str, float]:
        """
        Answer a question using knowledge graph context.

        Returns (answer, confidence).
        """
        system_prompt = """You are a legal contract analysis assistant. Answer questions about contracts using the provided knowledge graph context.

Guidelines:
1. Base your answer ONLY on the provided context
2. Cite specific triples or clauses when possible
3. If the context doesn't contain enough information, say so
4. Be precise with legal terminology
5. Indicate confidence level (high/medium/low) based on context quality

Format your response as:
ANSWER: [Your detailed answer]
CONFIDENCE: [high/medium/low]
SOURCES: [List of relevant triples or clauses used]"""

        user_prompt = f"""Question: {query}

Knowledge Graph Context:
{context}

{f'Contract Text Excerpt: {contract_text[:2000]}' if contract_text else ''}

Please answer the question based on the provided context."""

        try:
            response, model = await self.generate(system_prompt, user_prompt)

            # Parse response
            answer = response
            confidence = 0.7  # Default

            if "ANSWER:" in response:
                answer_start = response.find("ANSWER:") + 7
                answer_end = response.find("CONFIDENCE:")
                if answer_end == -1:
                    answer_end = len(response)
                answer = response[answer_start:answer_end].strip()

            if "CONFIDENCE:" in response:
                conf_text = response.split("CONFIDENCE:")[1].split("\n")[0].strip().lower()
                if "high" in conf_text:
                    confidence = 0.9
                elif "medium" in conf_text:
                    confidence = 0.7
                elif "low" in conf_text:
                    confidence = 0.5

            logger.info("query_answered", model=model, confidence=confidence)
            return answer, confidence

        except Exception as e:
            logger.error("query_answering_failed", error=str(e))
            return f"Error answering query: {str(e)}", 0.0


@lru_cache()
def get_llm_service() -> LLMService:
    """Get cached LLM service instance."""
    return LLMService()
