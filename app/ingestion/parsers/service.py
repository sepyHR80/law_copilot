"""Document parsing service integrating storage and parsers."""

from typing import Optional
from uuid import UUID

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.models import ParsedDocument
from app.ingestion.parsers.registry import ParserRegistry
from app.infrastructure.storage import StorageBackend


class DocumentParsingService:
    """Service for parsing documents from storage."""

    def __init__(
        self,
        storage: StorageBackend,
        registry: Optional[ParserRegistry] = None,
    ) -> None:
        self._storage = storage
        self._registry = registry or ParserRegistry()

    def parse_document(
        self,
        document_id: UUID,
        document_version_id: UUID,
        storage_key: str,
        mime_type: str,
    ) -> ParsedDocument:
        """Parse a document from storage.

        Args:
            document_id: UUID of the document.
            document_version_id: UUID of the document version.
            storage_key: MinIO storage key.
            mime_type: MIME type of the document.

        Returns:
            ParsedDocument with extracted content.

        Raises:
            DocumentParsingError: If parsing fails.
            UnsupportedDocumentFormatError: If no parser exists for the MIME type.
        """
        from app.ingestion.parsers.exceptions import DocumentParsingError

        # Get parser and parse (validates MIME type before downloading)
        parser = self._registry.get_parser(mime_type)

        # Download raw bytes
        try:
            content = self._storage.get_object(storage_key)
        except Exception as exc:
            raise DocumentParsingError(f"Failed to download document: {exc}") from exc

        # Parse
        try:
            return parser.parse(
                content=content,
                document_id=str(document_id),
                document_version_id=str(document_version_id),
            )
        except Exception as exc:
            if isinstance(exc, DocumentParsingError):
                raise
            raise DocumentParsingError(f"Failed to parse document: {exc}") from exc
