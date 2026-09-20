"""Domain embeddings package."""

from app.domain.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingValidationError,
)
from app.domain.embeddings.protocol import EmbeddingProvider

__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingValidationError",
]
