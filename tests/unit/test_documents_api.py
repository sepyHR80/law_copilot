"""Unit tests for Document API endpoints including /stats and /list."""

from unittest.mock import MagicMock
from uuid import uuid4
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from app.api.routes.documents import get_db
from app.main import app


def test_documents_stats_empty():
    """Test /stats endpoint with empty database."""
    mock_db = MagicMock()
    mock_db.query.return_value.count.return_value = 0
    mock_db.query.return_value.filter.return_value.count.return_value = 0
    mock_db.query.return_value.order_by.return_value.all.return_value = []

    app.dependency_overrides[get_db] = lambda: mock_db
    client = TestClient(app)
    try:
        res = client.get("/api/v1/documents/stats")
        assert res.status_code == 200
        data = res.json()
        assert data["total_documents"] == 0
        assert data["total_chunks"] == 0
        assert data["embedded_chunks"] == 0
        assert data["embedding_coverage_pct"] == 0.0
        assert data["embedding_config"]["model"] is not None
        assert data["documents"] == []
    finally:
        app.dependency_overrides.clear()


def test_documents_stats_with_data():
    """Test /stats endpoint with documents and chunks using mock query."""
    doc_id = uuid4()
    ver_id = uuid4()

    mock_doc = MagicMock()
    mock_doc.id = doc_id
    mock_doc.title = "قانون مجازات اسلامی"
    mock_doc.document_type = "law"
    mock_doc.knowledge_type = "factual"
    mock_doc.source = "parliament"
    mock_doc.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    mock_ver = MagicMock()
    mock_ver.id = ver_id
    mock_doc.versions = [mock_ver]

    mock_db = MagicMock()

    # Query counts: total_docs=1, total_versions=1, total_chunks=10, embedded_chunks=10
    # Inside documents iteration: chunk count=10, embedded=10
    call_counts = [1, 1, 10, 10, 10, 10]

    def count_side_effect():
        if call_counts:
            return call_counts.pop(0)
        return 0

    mock_query = MagicMock()
    mock_query.count.side_effect = count_side_effect
    mock_query.filter.return_value.count.side_effect = count_side_effect
    mock_query.order_by.return_value.all.return_value = [mock_doc]

    mock_db.query.return_value = mock_query

    app.dependency_overrides[get_db] = lambda: mock_db
    client = TestClient(app)
    try:
        res = client.get("/api/v1/documents/stats")
        assert res.status_code == 200
        data = res.json()
        assert data["total_documents"] == 1
        assert data["total_versions"] == 1
        assert data["total_chunks"] == 10
        assert data["embedded_chunks"] == 10
        assert data["embedding_coverage_pct"] == 100.0
        assert len(data["documents"]) == 1
        assert data["documents"][0]["title"] == "قانون مجازات اسلامی"
        assert data["documents"][0]["coverage_pct"] == 100.0

        # Test list endpoint
        res_list = client.get("/api/v1/documents")
        assert res_list.status_code == 200
        list_data = res_list.json()
        assert len(list_data) == 1
        assert list_data[0]["title"] == "قانون مجازات اسلامی"
    finally:
        app.dependency_overrides.clear()


def test_embed_single_document_already_embedded():
    """Test /embed endpoint when all chunks already have embeddings."""
    doc_id = uuid4()
    ver_id = uuid4()
    mock_doc = MagicMock()
    mock_doc.id = doc_id
    mock_doc.title = "قانون مدنی"
    mock_ver = MagicMock()
    mock_ver.id = ver_id
    mock_doc.versions = [mock_ver]

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc
    # total_chunks = 5
    mock_db.query.return_value.filter.return_value.count.side_effect = [5, 5]
    # unembedded_chunks = []
    mock_db.query.return_value.filter.return_value.all.return_value = []

    app.dependency_overrides[get_db] = lambda: mock_db
    client = TestClient(app)
    try:
        res = client.post(f"/api/v1/documents/{doc_id}/embed")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "already_embedded"
        assert data["coverage_pct"] == 100.0
    finally:
        app.dependency_overrides.clear()


def test_embed_single_document_success():
    """Test /embed endpoint generating embeddings for unembedded chunks."""
    from unittest.mock import AsyncMock, patch

    doc_id = uuid4()
    ver_id = uuid4()
    mock_doc = MagicMock()
    mock_doc.id = doc_id
    mock_doc.title = "قانون مدنی"
    mock_ver = MagicMock()
    mock_ver.id = ver_id
    mock_doc.versions = [mock_ver]

    mock_chunk = MagicMock()
    mock_chunk.content = "ماده ۱ قانون مدنی"
    mock_chunk.embedding = None

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc
    # total_chunks = 1, new_embedded_count = 1
    mock_db.query.return_value.filter.return_value.count.side_effect = [1, 1]
    # unembedded_chunks = [mock_chunk]
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_chunk]

    with patch("app.core.config.get_settings") as mock_settings, \
         patch("app.infrastructure.embeddings.openai_provider.OpenAIEmbeddingProvider") as mock_provider_cls:

        mock_s = MagicMock()
        mock_s.embedding_api_key = "valid-key"
        mock_s.embedding_endpoint = "https://api.openai.com/v1"
        mock_s.embedding_model = "text-embedding-3-small"
        mock_s.embedding_dimension = 1536
        mock_s.embedding_batch_size = 10
        mock_settings.return_value = mock_s

        mock_instance = MagicMock()
        mock_instance.embed_texts = AsyncMock(return_value=[[0.1] * 1536])
        mock_provider_cls.return_value = mock_instance

        app.dependency_overrides[get_db] = lambda: mock_db
        client = TestClient(app)
        try:
            res = client.post(f"/api/v1/documents/{doc_id}/embed")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert data["newly_embedded"] == 1
            assert data["coverage_pct"] == 100.0
            assert mock_chunk.embedding == [0.1] * 1536
        finally:
            app.dependency_overrides.clear()


def test_embed_all_documents_success():
    """Test /embed-all endpoint generating embeddings for all unembedded chunks."""
    from unittest.mock import AsyncMock, patch

    mock_chunk = MagicMock()
    mock_chunk.content = "ماده ۱۰ قانون مدنی"
    mock_chunk.embedding = None

    mock_db = MagicMock()
    mock_db.query.return_value.count.return_value = 1
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_chunk]
    mock_db.query.return_value.filter.return_value.count.return_value = 1

    with patch("app.core.config.get_settings") as mock_settings, \
         patch("app.infrastructure.embeddings.openai_provider.OpenAIEmbeddingProvider") as mock_provider_cls:

        mock_s = MagicMock()
        mock_s.embedding_api_key = "valid-key"
        mock_s.embedding_endpoint = "https://api.openai.com/v1"
        mock_s.embedding_model = "text-embedding-3-small"
        mock_s.embedding_dimension = 1536
        mock_s.embedding_batch_size = 10
        mock_settings.return_value = mock_s

        mock_instance = MagicMock()
        mock_instance.embed_texts = AsyncMock(return_value=[[0.2] * 1536])
        mock_provider_cls.return_value = mock_instance

        app.dependency_overrides[get_db] = lambda: mock_db
        client = TestClient(app)
        try:
            res = client.post("/api/v1/documents/embed-all")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert data["newly_embedded"] == 1
            assert mock_chunk.embedding == [0.2] * 1536
        finally:
            app.dependency_overrides.clear()

