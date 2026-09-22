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
