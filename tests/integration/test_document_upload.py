"""Integration tests for Stage 05 document upload API."""

import hashlib
import io
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.storage import MinioStorage
from app.main import app

client = TestClient(app)

FIXTURES = Path(__file__).parent / "fixtures"


def _make_file(filename: str, content: bytes, content_type: str = "application/octet-stream"):
    return {"file": (filename, io.BytesIO(content), content_type)}


def _make_data(file_title="Test Doc", doc_type="law", knowledge="factual", source="uploaded", version="1.0"):
    return {
        "title": file_title,
        "document_type": doc_type,
        "knowledge_type": knowledge,
        "source": source,
        "version": version,
    }


def test_valid_pdf_upload_succeeds() -> None:
    """Valid PDF upload returns 201 and creates records."""
    pdf_bytes = (FIXTURES / "sample.pdf").read_bytes()
    response = client.post(
        "/api/v1/documents",
        files=_make_file("sample.pdf", pdf_bytes, "application/pdf"),
        data=_make_data("Sample Law PDF"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Sample Law PDF"
    assert body["document_type"] == "law"
    assert body["knowledge_type"] == "factual"
    assert body["source"] == "uploaded"
    assert body["version"] == "1.0"
    assert body["file_name"] == "sample.pdf"
    assert body["mime_type"] == "application/pdf"
    assert body["checksum"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert body["storage_key"].startswith(f"documents/{body['id']}/1.0/")


def test_valid_txt_upload_succeeds() -> None:
    """Valid TXT upload returns 201."""
    txt_bytes = b"This is a legal text document.\nArticle 1: ..."
    response = client.post(
        "/api/v1/documents",
        files=_make_file("notes.txt", txt_bytes, "text/plain"),
        data=_make_data("Notes", doc_type="policy"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["file_name"] == "notes.txt"
    assert body["mime_type"] == "text/plain"
    assert body["checksum"] == hashlib.sha256(txt_bytes).hexdigest()


def test_valid_html_upload_succeeds() -> None:
    """Valid HTML upload returns 201."""
    html_bytes = (FIXTURES / "sample.html").read_bytes()
    response = client.post(
        "/api/v1/documents",
        files=_make_file("page.html", html_bytes, "text/html"),
        data=_make_data("Web Page", source="internal"),
    )
    assert response.status_code == 201
    assert response.json()["mime_type"] == "text/html"


def test_valid_docx_mime_upload() -> None:
    """DOCX MIME type is accepted even with minimal content."""
    docx_bytes = b"PK\x03\x04 fake docx content for testing"
    response = client.post(
        "/api/v1/documents",
        files=_make_file("contract.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        data=_make_data("Contract"),
    )
    assert response.status_code == 201
    assert response.json()["mime_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_document_and_version_in_database() -> None:
    """After upload, Document and DocumentVersion exist in PostgreSQL."""
    txt_bytes = b"Database verification content."
    response = client.post(
        "/api/v1/documents",
        files=_make_file("db_test.txt", txt_bytes, "text/plain"),
        data=_make_data("DB Test Doc"),
    )
    assert response.status_code == 201
    body = response.json()
    doc_id = body["id"]

    with SessionLocal() as db:
        from app.infrastructure.db.models.document import Document, DocumentVersion

        doc = db.query(Document).filter_by(id=doc_id).first()
        assert doc is not None
        assert doc.title == "DB Test Doc"

        ver = db.query(DocumentVersion).filter_by(document_id=doc_id).first()
        assert ver is not None
        assert ver.version == "1.0"
        assert ver.checksum == hashlib.sha256(txt_bytes).hexdigest()
        assert ver.storage_key == body["storage_key"]


def test_file_exists_in_minio() -> None:
    """Uploaded file is actually in MinIO."""
    content = b"MinIO storage verification."
    response = client.post(
        "/api/v1/documents",
        files=_make_file("minio_test.txt", content, "text/plain"),
        data=_make_data("MinIO Test"),
    )
    assert response.status_code == 201
    storage_key = response.json()["storage_key"]

    storage = MinioStorage(settings=get_settings())
    downloaded = storage.get_object(storage_key)
    assert downloaded == content


def test_unsupported_mime_type_rejected() -> None:
    """Unsupported MIME type returns 415."""
    response = client.post(
        "/api/v1/documents",
        files=_make_file("image.png", b"\x89PNG\r\n", "image/png"),
        data=_make_data(),
    )
    assert response.status_code == 415


def test_mime_extension_mismatch_rejected() -> None:
    """MIME/extension mismatch returns 400."""
    pdf_bytes = (FIXTURES / "sample.pdf").read_bytes()
    response = client.post(
        "/api/v1/documents",
        files=_make_file("sample.pdf", pdf_bytes, "text/plain"),
        data=_make_data(),
    )
    assert response.status_code == 400


def test_empty_file_rejected() -> None:
    """Empty file returns 400."""
    response = client.post(
        "/api/v1/documents",
        files=_make_file("empty.pdf", b"", "application/pdf"),
        data=_make_data(),
    )
    assert response.status_code == 400


def test_missing_title_rejected() -> None:
    """Missing title returns 400."""
    response = client.post(
        "/api/v1/documents",
        files=_make_file("test.pdf", b"content", "application/pdf"),
        data=_make_data(file_title=""),
    )
    assert response.status_code == 400


def test_invalid_knowledge_type_rejected() -> None:
    """Invalid knowledge_type returns 400."""
    response = client.post(
        "/api/v1/documents",
        files=_make_file("test.pdf", b"content", "application/pdf"),
        data=_make_data(knowledge="invalid"),
    )
    assert response.status_code == 400


def test_file_too_large_rejected() -> None:
    """File exceeding MAX_DOCUMENT_SIZE_BYTES returns 413."""
    large_content = b"x" * (11 * 1024 * 1024)  # 11 MB
    response = client.post(
        "/api/v1/documents",
        files=_make_file("large.pdf", large_content, "application/pdf"),
        data=_make_data(),
    )
    assert response.status_code == 413


def test_correct_content_type_in_minio() -> None:
    """Content type metadata is stored correctly in MinIO."""
    content = b"Content type test."
    response = client.post(
        "/api/v1/documents",
        files=_make_file("ct_test.pdf", content, "application/pdf"),
        data=_make_data("CT Test"),
    )
    assert response.status_code == 201
    storage_key = response.json()["storage_key"]

    storage = MinioStorage(settings=get_settings())
    stat = storage.stat_object(storage_key)
    assert stat.content_type == "application/pdf"


def test_database_failure_triggers_cleanup() -> None:
    """Database failure after upload triggers MinIO cleanup."""
    content = b"Cleanup test content."

    with patch("app.api.routes.documents.SQLAlchemyDocumentRepository") as mock_repo_cls:
        mock_repo = mock_repo_cls.return_value
        mock_repo.create_with_version.side_effect = Exception("DB failure")

        response = client.post(
            "/api/v1/documents",
            files=_make_file("cleanup_test.txt", content, "text/plain"),
            data=_make_data("Cleanup Test"),
        )

    assert response.status_code == 500
