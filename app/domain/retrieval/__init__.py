"""Domain retrieval package."""

from app.domain.retrieval.exceptions import (
    InvalidQueryError,
    RetrievalError,
    VectorDimensionError,
)
from app.domain.retrieval.models import (
    HybridSearchQuery,
    LexicalSearchQuery,
    RetrievalFilter,
    RetrievalResult,
    VectorSearchQuery,
)
from app.domain.retrieval.protocol import (
    HybridRetrieverProtocol,
    LexicalRetrieverProtocol,
    VectorRetrieverProtocol,
)

__all__ = [
    "HybridRetrieverProtocol",
    "HybridSearchQuery",
    "InvalidQueryError",
    "LexicalRetrieverProtocol",
    "LexicalSearchQuery",
    "RetrievalError",
    "RetrievalFilter",
    "RetrievalResult",
    "VectorDimensionError",
    "VectorRetrieverProtocol",
    "VectorSearchQuery",
]
