"""Web search provider interfaces and implementations."""

from typing import List, Optional, Protocol
from app.domain.search.models import WebSearchResult


class WebSearchProviderProtocol(Protocol):
    """Protocol for pluggable web search backends."""

    async def search(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """Execute web search and return raw search results."""
        ...


class FakeWebSearchProvider:
    """In-memory deterministic web search provider for tests."""

    def __init__(self, predefined_results: Optional[List[WebSearchResult]] = None) -> None:
        self.predefined_results = predefined_results or []
        self.call_history: List[str] = []

    async def search(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        self.call_history.append(query)
        return self.predefined_results[:num_results]
