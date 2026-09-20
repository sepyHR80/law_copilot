"""Domain retrieval package."""

from app.domain.retrieval.exceptions import (
    InvalidQueryError,
    RetrievalError,
    VectorDimensionError,
)
from app.domain.retrieval.models import (
    LexicalSearchQuery,
    RetrievalFilter,
    RetrievalResult,
    VectorSearchQuery,
)
from app.domain.retrieval.protocol import (
    LexicalRetrieverProtocol,
    VectorRetrieverProtocol,
)

__all__ = [
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
