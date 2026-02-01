"""
Query service for answering questions about contracts.

Combines knowledge graph context with LLM for accurate answers.
"""

from functools import lru_cache
from typing import Any
from uuid import UUID

import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.api import QueryResponse
from kggen_cuad.models.triple import Entity, Triple
from kggen_cuad.services.llm_service import get_llm_service, LLMService
from kggen_cuad.services.search_service import get_search_service, SearchService
from kggen_cuad.services.graph_service import get_graph_service, GraphService
from kggen_cuad.storage.redis_cache import get_redis_cache, RedisCache

logger = structlog.get_logger(__name__)


class QueryService:
    """
    Service for answering questions about contracts.

    Uses hybrid retrieval to find relevant context, then
    generates answers using the LLM.
    """

    def __init__(self):
        self.settings = get_settings()
        self._llm: LLMService | None = None
        self._search: SearchService | None = None
        self._graph: GraphService | None = None
        self._cache: RedisCache | None = None

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service()
        return self._llm

    @property
    def search(self) -> SearchService:
        if self._search is None:
            self._search = get_search_service()
        return self._search

    @property
    def graph(self) -> GraphService:
        if self._graph is None:
            self._graph = get_graph_service()
        return self._graph

    @property
    def cache(self) -> RedisCache:
        if self._cache is None:
            self._cache = get_redis_cache()
        return self._cache

    # =========================================================================
    # Main Query Interface
    # =========================================================================

    async def answer_query(
        self,
        query: str,
        contract_ids: list[UUID] | None = None,
        include_sources: bool = True,
        use_cache: bool = True,
    ) -> QueryResponse:
        """
        Answer a question about contracts.

        Args:
            query: The question to answer
            contract_ids: Optional list of contracts to search
            include_sources: Whether to include source triples
            use_cache: Whether to use cached results

        Returns:
            QueryResponse with answer, confidence, and sources
        """
        # Check cache
        if use_cache:
            cached = self.cache.get_query_result(
                query=query,
                contract_ids=[str(c) for c in contract_ids] if contract_ids else None,
            )
            if cached:
                logger.debug("query_cache_hit", query=query[:50])
                return QueryResponse(**cached)

        # Retrieve context
        context = await self._retrieve_query_context(query, contract_ids)

        # Generate answer
        answer, confidence = await self.llm.answer_query(
            query=query,
            context=context["formatted_context"],
            contract_text=context.get("contract_text"),
        )

        # Build response
        sources = []
        if include_sources:
            sources = context["triples"]

        response = QueryResponse(
            query=query,
            answer=answer,
            confidence=confidence,
            sources=sources,
            contract_ids=[str(c) for c in contract_ids] if contract_ids else [],
        )

        # Cache result
        if use_cache and confidence > 0.5:
            self.cache.set_query_result(
                query=query,
                result=response.model_dump(),
                contract_ids=[str(c) for c in contract_ids] if contract_ids else None,
            )

        logger.info(
            "query_answered",
            query=query[:50],
            confidence=confidence,
            sources=len(sources),
        )

        return response

    async def _retrieve_query_context(
        self,
        query: str,
        contract_ids: list[UUID] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant context for a query."""
        # Get formatted context from search
        formatted_context = await self.search.retrieve_context_formatted(
            query=query,
            contract_ids=contract_ids,
            max_entities=15,
            max_triples=25,
        )

        # Get raw results for sources
        raw_context = await self.search.retrieve_context(
            query=query,
            contract_ids=contract_ids,
            max_entities=15,
            max_triples=25,
        )

        return {
            "formatted_context": formatted_context,
            "entities": raw_context["entities"],
            "triples": raw_context["triples"],
        }

    # =========================================================================
    # Specialized Query Types
    # =========================================================================

    async def query_licensing(
        self,
        contract_id: UUID,
    ) -> QueryResponse:
        """Get licensing information from a contract."""
        query = "What are the licensing terms and IP rights granted in this contract?"
        return await self.answer_query(query, contract_ids=[contract_id])

    async def query_obligations(
        self,
        contract_id: UUID,
        party_name: str | None = None,
    ) -> QueryResponse:
        """Get obligations from a contract."""
        if party_name:
            query = f"What are the obligations of {party_name} in this contract?"
        else:
            query = "What are the key obligations of each party in this contract?"
        return await self.answer_query(query, contract_ids=[contract_id])

    async def query_restrictions(
        self,
        contract_id: UUID,
    ) -> QueryResponse:
        """Get restrictions from a contract."""
        query = "What restrictions or limitations apply in this contract?"
        return await self.answer_query(query, contract_ids=[contract_id])

    async def query_liability(
        self,
        contract_id: UUID,
    ) -> QueryResponse:
        """Get liability information from a contract."""
        query = "What are the liability provisions and caps in this contract?"
        return await self.answer_query(query, contract_ids=[contract_id])

    async def query_termination(
        self,
        contract_id: UUID,
    ) -> QueryResponse:
        """Get termination conditions from a contract."""
        query = "What are the termination conditions and notice requirements?"
        return await self.answer_query(query, contract_ids=[contract_id])

    async def query_governing_law(
        self,
        contract_id: UUID,
    ) -> QueryResponse:
        """Get governing law and jurisdiction information."""
        query = "What is the governing law and jurisdiction for this contract?"
        return await self.answer_query(query, contract_ids=[contract_id])

    # =========================================================================
    # Multi-Contract Analysis
    # =========================================================================

    async def compare_contracts(
        self,
        contract_ids: list[UUID],
        aspect: str = "all",
    ) -> QueryResponse:
        """
        Compare multiple contracts.

        Args:
            contract_ids: Contracts to compare
            aspect: Aspect to compare (licensing, obligations, restrictions, etc.)
        """
        aspect_queries = {
            "licensing": "Compare the licensing terms across these contracts",
            "obligations": "Compare the obligations of parties across these contracts",
            "restrictions": "Compare the restrictions across these contracts",
            "liability": "Compare the liability provisions across these contracts",
            "all": "Compare the key terms across these contracts including licensing, obligations, restrictions, and liability",
        }

        query = aspect_queries.get(aspect, aspect_queries["all"])
        return await self.answer_query(query, contract_ids=contract_ids)

    async def find_common_terms(
        self,
        contract_ids: list[UUID],
    ) -> dict[str, list[str]]:
        """Find common terms across multiple contracts."""
        common_terms = {
            "parties": [],
            "ip_assets": [],
            "obligations": [],
            "restrictions": [],
        }

        # Get entities from each contract
        entity_sets: dict[str, set] = {
            "Party": set(),
            "IPAsset": set(),
            "Obligation": set(),
            "Restriction": set(),
        }

        for contract_id in contract_ids:
            subgraph = await self.graph.get_contract_subgraph(contract_id)
            for entity in subgraph.entities:
                entity_type = entity.entity_type.value
                if entity_type in entity_sets:
                    entity_sets[entity_type].add(entity.normalized_name or entity.name)

        # Map to output
        common_terms["parties"] = list(entity_sets["Party"])
        common_terms["ip_assets"] = list(entity_sets["IPAsset"])
        common_terms["obligations"] = list(entity_sets["Obligation"])
        common_terms["restrictions"] = list(entity_sets["Restriction"])

        return common_terms

    # =========================================================================
    # Entity-Specific Queries
    # =========================================================================

    async def query_entity(
        self,
        entity_id: str,
        query: str,
    ) -> QueryResponse:
        """Answer a question about a specific entity."""
        # Get entity
        entity = await self.graph.get_entity(entity_id)
        if not entity:
            return QueryResponse(
                query=query,
                answer=f"Entity {entity_id} not found.",
                confidence=0.0,
                sources=[],
            )

        # Get entity neighborhood
        subgraph = await self.graph.get_entity_neighborhood(
            entity_id=entity_id,
            depth=2,
            max_nodes=30,
        )

        # Format context
        context_lines = [
            f"Entity: {entity.name} ({entity.entity_type.value})",
            "",
            "Related entities and relationships:",
        ]

        for triple in subgraph.triples:
            # Find subject and object names
            subject = next(
                (e for e in subgraph.entities if str(e.id) == str(triple.subject_id)),
                None,
            )
            obj = next(
                (e for e in subgraph.entities if str(e.id) == str(triple.object_id)),
                None,
            )

            if subject and obj:
                pred = triple.predicate.value.replace("_", " ").lower()
                context_lines.append(f"  - {subject.name} {pred} {obj.name}")

        context = "\n".join(context_lines)

        # Generate answer
        answer, confidence = await self.llm.answer_query(
            query=query,
            context=context,
        )

        return QueryResponse(
            query=query,
            answer=answer,
            confidence=confidence,
            sources=[],
        )

    async def get_entity_summary(
        self,
        entity_id: str,
    ) -> str:
        """Generate a summary of an entity's relationships."""
        entity = await self.graph.get_entity(entity_id)
        if not entity:
            return f"Entity {entity_id} not found."

        # Get all triples involving this entity
        triples = await self.graph.get_triples_for_entity(
            entity_id=entity_id,
            direction="both",
        )

        if not triples:
            return f"{entity.name} ({entity.entity_type.value}): No relationships found."

        # Group by predicate
        outgoing: dict[str, list[str]] = {}
        incoming: dict[str, list[str]] = {}

        for triple in triples:
            pred = triple.predicate.value.replace("_", " ").lower()
            if str(triple.subject_id) == entity_id:
                if pred not in outgoing:
                    outgoing[pred] = []
                outgoing[pred].append(str(triple.object_id))
            else:
                if pred not in incoming:
                    incoming[pred] = []
                incoming[pred].append(str(triple.subject_id))

        # Format summary
        lines = [f"## {entity.name} ({entity.entity_type.value})"]

        if outgoing:
            lines.append("\n### Relationships (outgoing):")
            for pred, targets in outgoing.items():
                lines.append(f"- {pred}: {len(targets)} target(s)")

        if incoming:
            lines.append("\n### Relationships (incoming):")
            for pred, sources in incoming.items():
                lines.append(f"- {pred}: {len(sources)} source(s)")

        return "\n".join(lines)

    # =========================================================================
    # Validation Queries
    # =========================================================================

    async def validate_triple(
        self,
        triple: Triple,
        contract_text: str | None = None,
    ) -> tuple[bool, float, str]:
        """
        Validate a triple against the source contract.

        Returns (is_valid, confidence, explanation).
        """
        # Get subject and object entities
        subject = await self.graph.get_entity(str(triple.subject_id))
        obj = await self.graph.get_entity(str(triple.object_id))

        if not subject or not obj:
            return False, 0.0, "Subject or object entity not found"

        # Create validation query
        pred = triple.predicate.value.replace("_", " ").lower()
        statement = f"{subject.name} {pred} {obj.name}"

        system_prompt = """You are a legal contract analyst. Verify if the following statement is supported by the contract text.

Respond with:
VALID: true or false
CONFIDENCE: 0.0 to 1.0
EXPLANATION: Brief explanation of your assessment"""

        user_prompt = f"""Statement to verify: {statement}

Contract text:
{contract_text[:4000] if contract_text else 'No contract text available'}

Provide your assessment."""

        response, _ = await self.llm.generate(system_prompt, user_prompt)

        # Parse response
        is_valid = "VALID: true" in response.lower()
        confidence = 0.5

        if "CONFIDENCE:" in response:
            try:
                conf_str = response.split("CONFIDENCE:")[1].split("\n")[0].strip()
                confidence = float(conf_str)
            except (ValueError, IndexError):
                pass

        explanation = ""
        if "EXPLANATION:" in response:
            explanation = response.split("EXPLANATION:")[1].strip()

        return is_valid, confidence, explanation

    # =========================================================================
    # Suggestions
    # =========================================================================

    async def suggest_queries(
        self,
        contract_id: UUID,
    ) -> list[str]:
        """Suggest relevant queries for a contract."""
        # Get basic stats
        subgraph = await self.graph.get_contract_subgraph(contract_id)

        entity_types = set(e.entity_type.value for e in subgraph.entities)
        predicate_types = set(t.predicate.value for t in subgraph.triples)

        suggestions = []

        if "Party" in entity_types:
            suggestions.append("Who are the parties to this contract?")

        if "LICENSES_TO" in predicate_types or "IPAsset" in entity_types:
            suggestions.append("What IP rights are licensed in this contract?")

        if "Obligation" in entity_types:
            suggestions.append("What are the key obligations of each party?")

        if "Restriction" in entity_types:
            suggestions.append("What restrictions apply to the parties?")

        if "LiabilityProvision" in entity_types:
            suggestions.append("What are the liability caps and limitations?")

        if "GOVERNED_BY" in predicate_types or "Jurisdiction" in entity_types:
            suggestions.append("What is the governing law for this contract?")

        if "Temporal" in entity_types:
            suggestions.append("What are the key dates and deadlines?")

        # Add general suggestions
        suggestions.extend([
            "What are the termination conditions?",
            "Are there any non-compete provisions?",
            "What warranties are provided?",
        ])

        return suggestions[:10]


@lru_cache()
def get_query_service() -> QueryService:
    """Get cached query service instance."""
    return QueryService()
