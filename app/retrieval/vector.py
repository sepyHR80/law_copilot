"""Vector search service."""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.domain.retrieval.models import RetrievalResult, VectorSearchQuery
from app.domain.retrieval.protocol import VectorRetrieverProtocol
from app.infrastructure.db.repositories.vector_search import PgVectorRetriever


class VectorSearchService:
    """Application service for vector similarity retrieval."""

    def __init__(
        self,
        retriever: Optional[VectorRetrieverProtocol] = None,
        session: Optional[Session] = None,
    ) -> None:
        if retriever is not None:
            self._retriever = retriever
        elif session is not None:
            self._retriever = PgVectorRetriever(session=session)
        else:
            raise ValueError("Either retriever or session must be provided to VectorSearchService.")

    def search(self, query: VectorSearchQuery) -> List[RetrievalResult]:
        """Execute vector search query.

        Args:
            query: VectorSearchQuery with embedding vector and optional filters.

        Returns:
            List of RetrievalResult objects ordered by rank.
        """
        return self._retriever.search(query)
