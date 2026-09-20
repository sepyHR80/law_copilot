"""Lexical search service."""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.domain.retrieval.models import LexicalSearchQuery, RetrievalResult
from app.domain.retrieval.protocol import LexicalRetrieverProtocol
from app.infrastructure.db.repositories.lexical_search import PgLexicalRetriever


class LexicalSearchService:
    """Application service for lexical (full-text) retrieval."""

    def __init__(
        self,
        retriever: Optional[LexicalRetrieverProtocol] = None,
        session: Optional[Session] = None,
        language: str = "english",
    ) -> None:
        if retriever is not None:
            self._retriever = retriever
        elif session is not None:
            self._retriever = PgLexicalRetriever(session=session, language=language)
        else:
            raise ValueError("Either retriever or session must be provided to LexicalSearchService.")

    def search(self, query: LexicalSearchQuery) -> List[RetrievalResult]:
        """Execute lexical full-text search query.

        Args:
            query: LexicalSearchQuery with query string and optional filters.

        Returns:
            List of RetrievalResult objects ordered by rank.
        """
        return self._retriever.search(query)
