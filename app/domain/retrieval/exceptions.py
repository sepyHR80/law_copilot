"""Retrieval domain exceptions."""

from typing import Optional


class RetrievalError(Exception):
    """Base exception for all retrieval-related errors."""


class VectorDimensionError(RetrievalError):
    """Raised when a query vector does not match the expected dimension."""

    def __init__(
        self,
        expected_dimension: int,
        actual_dimension: int,
        message: Optional[str] = None,
    ) -> None:
        self.expected_dimension = expected_dimension
        self.actual_dimension = actual_dimension
        msg = (
            message
            or f"Query vector dimension mismatch: expected {expected_dimension}, got {actual_dimension}."
        )
        super().__init__(msg)


class InvalidQueryError(RetrievalError):
    """Raised when a search query is invalid (e.g. empty vector, top_k <= 0)."""


class RerankingError(RetrievalError):
    """Base exception for errors during reranking."""


class RerankerModelLoadError(RerankingError):
    """Raised when a reranker model cannot be loaded or initialized."""
