"""Integration tests for document parsing service."""

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.ingestion.parsers.exceptions import DocumentParsingError, UnsupportedDocumentFormatError
from app.ingestion.parsers.models import ParsedDocument
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.parsers.service import DocumentParsingService
from app.infrastructure.storage import MinioStorage

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parsing"


@pytest.fixture
def parsing_service() -> DocumentParsingService:
    storage = MinioStorage(settings=get_settings())
    storage.initialize()
    return DocumentParsingService(storage=storage, registry=ParserRegistry())


def _upload_and_get_key(filename: str, content: bytes, mime_type: str) -> tuple[str, str, str]:
    """Upload a file to MinIO and return (key, doc_id, ver_id)."""
    import uuid
    doc_id = str(uuid.uuid4())
    ver_id = str(uuid.uuid4())
    key = f"test/{doc_id}/{ver_id}/{filename}"
    storage = MinioStorage(settings=get_settings())
    storage.initialize()
    storage.put_object(key, content, content_type=mime_type)
    return key, doc_id, ver_id


class TestDocumentParsingService:
    def test_parse_pdf_from_storage(self, parsing_service):
        import uuid
        pdf_bytes = (FIXTURES / "sample.pdf").read_bytes()
        key, doc_id, ver_id = _upload_and_get_key("test.pdf", pdf_bytes, "application/pdf")

        result = parsing_service.parse_document(
            document_id=uuid.UUID(doc_id),
            document_version_id=uuid.UUID(ver_id),
            storage_key=key,
            mime_type="application/pdf",
        )

        assert isinstance(result, ParsedDocument)
        assert result.document_id == doc_id
        assert result.document_version_id == ver_id
        assert result.mime_type == "application/pdf"
        assert len(result.pages) > 0

    def test_parse_txt_from_storage(self, parsing_service):
        import uuid
        txt_bytes = (FIXTURES / "sample.txt").read_bytes()
        key, doc_id, ver_id = _upload_and_get_key("test.txt", txt_bytes, "text/plain")

        result = parsing_service.parse_document(
            document_id=uuid.UUID(doc_id),
            document_version_id=uuid.UUID(ver_id),
            storage_key=key,
            mime_type="text/plain",
        )

        assert isinstance(result, ParsedDocument)
        assert result.document_id == doc_id
        assert len(result.pages) == 1
        assert "Article 1" in result.pages[0].text

    def test_parse_html_from_storage(self, parsing_service):
        import uuid
        html_bytes = (FIXTURES / "sample.html").read_bytes()
        key, doc_id, ver_id = _upload_and_get_key("test.html", html_bytes, "text/html")

        result = parsing_service.parse_document(
            document_id=uuid.UUID(doc_id),
            document_version_id=uuid.UUID(ver_id),
            storage_key=key,
            mime_type="text/html",
        )

        assert isinstance(result, ParsedDocument)
        assert result.document_id == doc_id
        assert result.title == "Test HTML Document"

    def test_parse_docx_from_storage(self, parsing_service):
        import uuid
        docx_bytes = (FIXTURES / "sample.docx").read_bytes()
        key, doc_id, ver_id = _upload_and_get_key("test.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        result = parsing_service.parse_document(
            document_id=uuid.UUID(doc_id),
            document_version_id=uuid.UUID(ver_id),
            storage_key=key,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        assert isinstance(result, ParsedDocument)
        assert result.document_id == doc_id
        assert len(result.tables) > 0

    def test_unsupported_mime_type_raises_error(self, parsing_service):
        import uuid
        with pytest.raises(UnsupportedDocumentFormatError):
            parsing_service.parse_document(
                document_id=uuid.uuid4(),
                document_version_id=uuid.uuid4(),
                storage_key="test/key",
                mime_type="image/png",
            )

    def test_missing_object_raises_error(self, parsing_service):
        import uuid
        with pytest.raises(DocumentParsingError):
            parsing_service.parse_document(
                document_id=uuid.uuid4(),
                document_version_id=uuid.uuid4(),
                storage_key="nonexistent/missing.pdf",
                mime_type="application/pdf",
            )
