"""Unit tests for Stage 11 — Lexical (FTS) Retrieval domain and service."""

from unittest.mock import MagicMock
from uuid import uuid4
import pytest
from pydantic import ValidationError

from app.domain.retrieval.exceptions import InvalidQueryError
from app.domain.retrieval.models import (
    LexicalSearchQuery,
    RetrievalFilter,
    RetrievalResult,
)
from app.domain.retrieval.protocol import LexicalRetrieverProtocol
from app.infrastructure.db.repositories.lexical_search import PgLexicalRetriever
from app.retrieval.lexical import LexicalSearchService


class TestLexicalSearchQueryModels:
    def test_valid_query_construction(self):
        """LexicalSearchQuery constructs correctly with filters and parameters."""
        doc_id = uuid4()
        filters = RetrievalFilter(
            knowledge_type="factual",
            document_type="contract",
            document_id=doc_id,
        )
        query = LexicalSearchQuery(
            query="Article 4 termination notice",
            top_k=10,
            min_score=0.1,
            filters=filters,
        )
        assert query.query == "Article 4 termination notice"
        assert query.top_k == 10
        assert query.min_score == 0.1
        assert query.filters.knowledge_type == "factual"

    def test_invalid_top_k_raises_validation_error(self):
        """top_k < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            LexicalSearchQuery(query="law clause", top_k=0)


class TestPgLexicalRetrieverValidation:
    def test_empty_query_returns_empty_list_immediately(self):
        """Empty or whitespace-only query string returns [] without hitting DB."""
        session = MagicMock()
        retriever = PgLexicalRetriever(session=session)

        assert retriever.search(LexicalSearchQuery(query="", top_k=5)) == []
        assert retriever.search(LexicalSearchQuery(query="   \t\n  ", top_k=5)) == []
        session.execute.assert_not_called()


class TestLexicalSearchService:
    def test_service_delegates_to_retriever(self):
        """LexicalSearchService accurately delegates search to the retriever."""
        mock_retriever = MagicMock(spec=LexicalRetrieverProtocol)
        sample_result = RetrievalResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            content="Article 12: Governing Law",
            score=0.75,
            rank=1,
            source={"title": "Statute of Frauds"},
        )
        mock_retriever.search.return_value = [sample_result]

        service = LexicalSearchService(retriever=mock_retriever)
        query = LexicalSearchQuery(query="governing law", top_k=5)
        results = service.search(query)

        assert len(results) == 1
        assert results[0] == sample_result
        mock_retriever.search.assert_called_once_with(query)

    def test_service_requires_retriever_or_session(self):
        """LexicalSearchService raises ValueError if neither retriever nor session provided."""
        with pytest.raises(ValueError):
            LexicalSearchService()
