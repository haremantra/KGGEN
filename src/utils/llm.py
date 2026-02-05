"""LLM service for entity/relation extraction and query answering.

Supports Anthropic Claude (primary) and OpenAI (fallback) with tenacity retry.
"""

import json
from functools import lru_cache
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings


class LLMService:
    """LLM service with primary + fallback providers and retry logic."""

    def __init__(self):
        self._settings = get_settings()
        self._anthropic = None
        self._openai = None

        if self._settings.anthropic_api_key:
            from anthropic import Anthropic
            self._anthropic = Anthropic(api_key=self._settings.anthropic_api_key)

        if self._settings.openai_api_key:
            from openai import OpenAI
            self._openai = OpenAI(api_key=self._settings.openai_api_key)

        self.primary_provider = self._settings.primary_llm_provider
        self.primary_model = self._settings.primary_llm_model
        self.fallback_provider = self._settings.fallback_llm_provider
        self.fallback_model = self._settings.fallback_llm_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_anthropic(
        self, system_prompt: str, user_prompt: str, max_tokens: int | None = None
    ) -> str:
        if self._anthropic is None:
            raise ValueError("Anthropic client not configured. Set ANTHROPIC_API_KEY.")
        response = self._anthropic.messages.create(
            model=self.primary_model if self.primary_provider == "anthropic" else self.fallback_model,
            max_tokens=max_tokens or self._settings.llm_max_tokens,
            temperature=self._settings.extraction_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_openai(
        self, system_prompt: str, user_prompt: str, max_tokens: int | None = None
    ) -> str:
        if self._openai is None:
            raise ValueError("OpenAI client not configured. Set OPENAI_API_KEY.")
        response = self._openai.chat.completions.create(
            model=self.fallback_model if self.fallback_provider == "openai" else self.primary_model,
            max_tokens=max_tokens or self._settings.llm_max_tokens,
            temperature=self._settings.extraction_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        use_fallback: bool = True,
    ) -> tuple[str, str]:
        """Generate LLM response with automatic fallback. Returns (text, model_used)."""
        # Try primary
        try:
            if self.primary_provider == "anthropic" and self._anthropic:
                text = self._call_anthropic(system_prompt, user_prompt, max_tokens)
                return text, self.primary_model
            elif self.primary_provider == "openai" and self._openai:
                text = self._call_openai(system_prompt, user_prompt, max_tokens)
                return text, self.primary_model
        except Exception:
            if not use_fallback:
                raise

        # Try fallback
        if self.fallback_provider == "anthropic" and self._anthropic:
            text = self._call_anthropic(system_prompt, user_prompt, max_tokens)
            return text, self.fallback_model
        elif self.fallback_provider == "openai" and self._openai:
            text = self._call_openai(system_prompt, user_prompt, max_tokens)
            return text, self.fallback_model

        raise ValueError("No LLM provider available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    def _parse_json(self, text: str) -> Any:
        """Extract JSON from LLM response text."""
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
            # Try to find JSON array or object
            for opener, closer in [("[", "]"), ("{", "}")]:
                start = text.find(opener)
                end = text.rfind(closer) + 1
                if start >= 0 and end > start:
                    try:
                        return json.loads(text[start:end])
                    except json.JSONDecodeError:
                        continue
            return None

    def extract_entities(self, contract_text: str) -> list[dict[str, Any]]:
        """Extract entities from contract text."""
        system_prompt = """You are a legal contract analysis expert. Extract entities from the provided contract text.

For each entity, provide:
1. name: The exact name or description
2. type: One of: Party, IPAsset, Obligation, Restriction, LiabilityProvision, Temporal, Jurisdiction, ContractClause
3. properties: Additional relevant properties as key-value pairs
4. confidence: LOW, MEDIUM, or HIGH

Return a JSON array of entities."""

        user_prompt = f"""Extract all entities from this contract text:

{contract_text[:8000]}

Return a JSON array like:
[
  {{"name": "ABC Corporation", "type": "Party", "properties": {{"role": "Licensor"}}, "confidence": "HIGH"}},
  {{"name": "Source Code", "type": "IPAsset", "properties": {{"ip_type": "software"}}, "confidence": "MEDIUM"}}
]"""

        try:
            response, _ = self.generate(system_prompt, user_prompt)
            entities = self._parse_json(response)
            return entities if isinstance(entities, list) else []
        except Exception:
            return []

    def extract_relations(
        self, contract_text: str, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract relations between entities."""
        entity_list = "\n".join(f"- {e['name']} ({e['type']})" for e in entities[:50])

        system_prompt = """You are a legal contract analysis expert. Extract relationships between entities.

For each relationship, provide:
1. subject: Source entity name
2. predicate: One of: LICENSES_TO, OWNS, ASSIGNS, HAS_OBLIGATION, SUBJECT_TO_RESTRICTION, HAS_LIABILITY, GOVERNED_BY, CONTAINS_CLAUSE, EFFECTIVE_ON, TERMINATES_ON
3. object: Target entity name
4. confidence: LOW, MEDIUM, or HIGH

Return a JSON array."""

        user_prompt = f"""Given these entities:
{entity_list}

Extract relationships from this text:

{contract_text[:8000]}

Return a JSON array like:
[
  {{"subject": "ABC Corp", "predicate": "LICENSES_TO", "object": "XYZ Inc", "confidence": "HIGH"}}
]"""

        try:
            response, _ = self.generate(system_prompt, user_prompt)
            relations = self._parse_json(response)
            return relations if isinstance(relations, list) else []
        except Exception:
            return []

    def identify_duplicates(self, entity_names: list[str]) -> list[list[str]]:
        """Identify duplicate entities from a list of candidates."""
        system_prompt = """You are a legal text analysis expert. Identify which entity names refer to the same real-world entity.

Consider as duplicates: same entity with different capitalization, abbreviations vs full names, with/without articles.
Do NOT merge: different legal terms, different parties (Licensor vs Licensee).

Return groups of duplicate entities as a JSON array of arrays."""

        user_prompt = f"""Identify duplicates from:
{entity_names}

Return JSON:
[["ABC Corp", "ABC Corporation", "the Company"], ...]

Only include groups with 2+ items."""

        try:
            response, _ = self.generate(system_prompt, user_prompt)
            groups = self._parse_json(response)
            if isinstance(groups, list):
                return [g for g in groups if isinstance(g, list) and len(g) >= 2]
            return []
        except Exception:
            return []

    def select_canonical(self, duplicate_group: list[str]) -> tuple[str, list[str]]:
        """Select canonical form from duplicate group. Returns (canonical, aliases)."""
        system_prompt = """You are a legal terminology expert. Select the most canonical form for a group of duplicate entity names.

Criteria: most legally precise, most complete, most commonly used, without articles unless defined term.

Return JSON: {"canonical": "...", "aliases": ["...", ...]}"""

        user_prompt = f"""Select canonical form from:
{duplicate_group}

Return JSON: {{"canonical": "Selected Form", "aliases": ["alias1", "alias2"]}}"""

        try:
            response, _ = self.generate(system_prompt, user_prompt)
            result = self._parse_json(response)
            if isinstance(result, dict):
                canonical = result.get("canonical", duplicate_group[0])
                aliases = [a for a in result.get("aliases", duplicate_group) if a != canonical]
                return canonical, aliases
        except Exception:
            pass
        return duplicate_group[0], duplicate_group[1:]

    def answer_query(
        self, query: str, context: str, contract_text: str | None = None
    ) -> tuple[str, float]:
        """Answer a question using KG context. Returns (answer, confidence)."""
        system_prompt = """You are a legal contract analysis assistant. Answer questions using the provided knowledge graph context.

Guidelines:
1. Base your answer ONLY on the provided context
2. Cite specific triples or clauses when possible
3. If context is insufficient, say so
4. Be precise with legal terminology

Format:
ANSWER: [Your answer]
CONFIDENCE: [HIGH/MEDIUM/LOW]
SOURCES: [List of relevant triples]"""

        user_prompt = f"""Question: {query}

Knowledge Graph Context:
{context}

{f'Contract Text Excerpt: {contract_text[:2000]}' if contract_text else ''}

Answer based on the provided context."""

        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}

        try:
            response, _ = self.generate(system_prompt, user_prompt)

            answer = response
            confidence = 0.7

            if "ANSWER:" in response:
                start = response.find("ANSWER:") + 7
                end = response.find("CONFIDENCE:")
                if end == -1:
                    end = len(response)
                answer = response[start:end].strip()

            if "CONFIDENCE:" in response:
                conf_text = response.split("CONFIDENCE:")[1].split("\n")[0].strip().lower()
                for key, val in confidence_map.items():
                    if key in conf_text:
                        confidence = val
                        break

            return answer, confidence
        except Exception as e:
            return f"Error answering query: {e}", 0.0


@lru_cache()
def get_llm_service() -> LLMService:
    """Get cached LLM service singleton."""
    return LLMService()
