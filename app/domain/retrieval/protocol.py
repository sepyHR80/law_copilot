"""Vector retriever protocol."""

from typing import List, Protocol, runtime_checkable

from app.domain.retrieval.models import (
    LexicalSearchQuery,
    RetrievalResult,
    VectorSearchQuery,
)


@runtime_checkable
class VectorRetrieverProtocol(Protocol):
    """Abstract interface for vector similarity search."""

    def search(self, query: VectorSearchQuery) -> List[RetrievalResult]:
        """Execute vector similarity retrieval.

        Args:
            query: VectorSearchQuery with embedding vector and optional filters.

        Returns:
            List of RetrievalResult objects ordered by rank.

        Raises:
            VectorDimensionError: If query vector dimension differs from expected.
            InvalidQueryError: If query is malformed.
            RetrievalError: If database or retrieval query execution fails.
        """
        ...


@runtime_checkable
class LexicalRetrieverProtocol(Protocol):
    """Abstract interface for lexical (full-text) search."""

    def search(self, query: LexicalSearchQuery) -> List[RetrievalResult]:
        """Execute full-text lexical retrieval.

        Args:
            query: LexicalSearchQuery with search text and optional filters.

        Returns:
            List of RetrievalResult objects ordered by rank.

        Raises:
            InvalidQueryError: If query is malformed.
            RetrievalError: If database or retrieval query execution fails.
        """
        ...
