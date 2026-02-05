"""Query service — RAG pipeline for answering contract questions.

Combines hybrid search retrieval with LLM answer generation.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..config import get_settings
from ..utils.llm import get_llm_service
from ..search.service import get_search_service


@dataclass
class QueryResponse:
    """Response from a query."""
    query: str
    answer: str
    confidence: float
    sources: list[dict] = field(default_factory=list)
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "model_used": self.model_used,
        }


class QueryService:
    """RAG query service for contract analysis."""

    def __init__(self):
        self._settings = get_settings()
        self._llm = None
        self._search = None
        self._cache: dict[str, QueryResponse] = {}

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm_service()
        return self._llm

    @property
    def search(self):
        if self._search is None:
            self._search = get_search_service()
        return self._search

    def query(
        self,
        question: str,
        use_cache: bool = True,
    ) -> QueryResponse:
        """Answer a question about contracts using RAG."""
        # Check cache
        if use_cache and question in self._cache:
            return self._cache[question]

        # Retrieve context
        formatted_context = self.search.retrieve_context_formatted(
            question, max_entities=15, max_triples=25
        )
        raw_context = self.search.retrieve_context(
            question, max_entities=15, max_triples=25
        )

        # Generate answer
        answer, confidence = self.llm.answer_query(
            query=question,
            context=formatted_context,
        )

        response = QueryResponse(
            query=question,
            answer=answer,
            confidence=confidence,
            sources=raw_context.get("triples", []),
        )

        # Cache high-confidence results
        if use_cache and confidence > 0.5:
            self._cache[question] = response

        return response

    # Specialized queries

    def query_licensing(self) -> QueryResponse:
        """Get licensing information."""
        return self.query("What are the licensing terms and IP rights granted?")

    def query_obligations(self, party_name: str | None = None) -> QueryResponse:
        """Get obligations information."""
        if party_name:
            return self.query(f"What are the obligations of {party_name}?")
        return self.query("What are the key obligations of each party?")

    def query_restrictions(self) -> QueryResponse:
        """Get restrictions information."""
        return self.query("What restrictions or limitations apply?")

    def query_liability(self) -> QueryResponse:
        """Get liability information."""
        return self.query("What are the liability provisions and caps?")

    def query_termination(self) -> QueryResponse:
        """Get termination conditions."""
        return self.query("What are the termination conditions and notice requirements?")

    def query_governing_law(self) -> QueryResponse:
        """Get governing law information."""
        return self.query("What is the governing law and jurisdiction?")

    # Multi-contract

    def compare(self, aspect: str = "all") -> QueryResponse:
        """Compare contracts on a specific aspect."""
        aspects = {
            "licensing": "Compare the licensing terms across contracts",
            "obligations": "Compare the obligations across contracts",
            "restrictions": "Compare the restrictions across contracts",
            "liability": "Compare the liability provisions across contracts",
            "all": "Compare the key terms across contracts including licensing, obligations, restrictions, and liability",
        }
        question = aspects.get(aspect, aspects["all"])
        return self.query(question, use_cache=False)

    def suggest_queries(self) -> list[str]:
        """Suggest relevant queries based on indexed data."""
        return [
            "Who are the parties to this contract?",
            "What IP rights are licensed?",
            "What are the key obligations of each party?",
            "What restrictions apply to the parties?",
            "What are the liability caps and limitations?",
            "What is the governing law?",
            "What are the key dates and deadlines?",
            "What are the termination conditions?",
            "Are there any non-compete provisions?",
            "What warranties are provided?",
        ]


@lru_cache()
def get_query_service() -> QueryService:
    """Get cached query service singleton."""
    return QueryService()
