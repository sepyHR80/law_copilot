"""Embedding domain exceptions."""

from typing import Optional


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when an embedding returned by the provider does not match the expected dimension."""

    def __init__(
        self,
        expected_dimension: int,
        actual_dimension: int,
        index: int,
        message: Optional[str] = None,
    ) -> None:
        self.expected_dimension = expected_dimension
        self.actual_dimension = actual_dimension
        self.index = index
        msg = (
            message
            or f"Embedding dimension mismatch at index {index}: "
            f"expected {expected_dimension}, got {actual_dimension}."
        )
        super().__init__(msg)


class EmbeddingProviderError(EmbeddingError):
    """Raised when the embedding provider fails (e.g. API error, timeout, network failure)."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when the embedding service or provider is configured with invalid parameters."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when input to the embedding service is invalid (e.g. None, empty string, whitespace)."""
