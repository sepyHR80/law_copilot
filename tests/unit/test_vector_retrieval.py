"""Unit tests for Stage 10 — Vector Retrieval domain and service."""

from unittest.mock import MagicMock
from uuid import uuid4
import pytest
from pydantic import ValidationError

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
from app.infrastructure.db.repositories.vector_search import PgVectorRetriever
from app.retrieval.vector import VectorSearchService


class TestVectorSearchQueryModels:
    def test_valid_query_construction(self):
        """Query with vector and filters constructs correctly."""
        doc_id = uuid4()
        filters = RetrievalFilter(
            knowledge_type="factual",
            document_type="contract",
            source="internal",
            document_id=doc_id,
        )
        query = VectorSearchQuery(
            vector=[0.1] * 1536,
            top_k=10,
            min_score=0.7,
            filters=filters,
        )
        assert len(query.vector) == 1536
        assert query.top_k == 10
        assert query.min_score == 0.7
        assert query.filters.knowledge_type == "factual"
        assert query.filters.document_id == doc_id

    def test_invalid_top_k_raises_validation_error(self):
        """top_k < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            VectorSearchQuery(vector=[0.1] * 1536, top_k=0)


class TestVectorRetrieverValidation:
    def test_dimension_mismatch_raises_error(self):
        """Vector with wrong dimension raises VectorDimensionError."""
        session = MagicMock()
        retriever = PgVectorRetriever(session=session, expected_dimension=1536)
        query = VectorSearchQuery(vector=[0.1] * 512, top_k=5)  # 512 != 1536

        with pytest.raises(VectorDimensionError) as exc_info:
            retriever.search(query)

        err = exc_info.value
        assert err.expected_dimension == 1536
        assert err.actual_dimension == 512

    def test_empty_vector_raises_invalid_query_error(self):
        """Empty vector raises InvalidQueryError."""
        session = MagicMock()
        retriever = PgVectorRetriever(session=session, expected_dimension=1536)
        query = VectorSearchQuery(vector=[], top_k=5)

        with pytest.raises(InvalidQueryError):
            retriever.search(query)


class TestVectorSearchService:
    def test_service_delegates_to_retriever(self):
        """VectorSearchService accurately delegates search to the retriever."""
        mock_retriever = MagicMock(spec=VectorRetrieverProtocol)
        sample_result = RetrievalResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            content="Sample clause content",
            score=0.85,
            rank=1,
            source={"title": "Master Service Agreement"},
        )
        mock_retriever.search.return_value = [sample_result]

        service = VectorSearchService(retriever=mock_retriever)
        query = VectorSearchQuery(vector=[0.1] * 1536, top_k=5)
        results = service.search(query)

        assert len(results) == 1
        assert results[0] == sample_result
        mock_retriever.search.assert_called_once_with(query)

    def test_service_requires_retriever_or_session(self):
        """VectorSearchService raises ValueError if neither retriever nor session provided."""
        with pytest.raises(ValueError):
            VectorSearchService()
