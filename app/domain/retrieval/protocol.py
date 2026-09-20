"""Vector retriever protocol."""

from typing import List, Protocol, runtime_checkable

from app.domain.retrieval.models import (
    HybridSearchQuery,
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


@runtime_checkable
class HybridRetrieverProtocol(Protocol):
    """Abstract interface for hybrid (vector + lexical) retrieval."""

    async def search(self, query: HybridSearchQuery) -> List[RetrievalResult]:
        """Execute hybrid retrieval with Reciprocal Rank Fusion.

        Args:
            query: HybridSearchQuery with query text, optional vector, and parameters.

        Returns:
            List of RetrievalResult objects ordered by fused rank.

        Raises:
            InvalidQueryError: If query is malformed.
            RetrievalError: If retrieval execution fails.
        """
        ...
