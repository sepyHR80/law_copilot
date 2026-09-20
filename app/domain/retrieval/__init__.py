"""Domain retrieval package."""

from app.domain.retrieval.exceptions import (
    InvalidQueryError,
    RetrievalError,
    VectorDimensionError,
)
from app.domain.retrieval.models import (
    RetrievalFilter,
    RetrievalResult,
    VectorSearchQuery,
)
from app.domain.retrieval.protocol import VectorRetrieverProtocol

__all__ = [
    "InvalidQueryError",
    "RetrievalError",
    "RetrievalFilter",
    "RetrievalResult",
    "VectorDimensionError",
    "VectorRetrieverProtocol",
    "VectorSearchQuery",
]
