"""Abstract document parser interface."""

from abc import ABC, abstractmethod
from app.ingestion.parsers.models import ParsedDocument


class DocumentParser(ABC):
    """Abstract interface for document parsers."""

    @abstractmethod
    def parse(self, content: bytes, document_id: str, document_version_id: str) -> ParsedDocument:
        """Parse raw document bytes into a ParsedDocument.

        Args:
            content: Raw bytes of the document.
            document_id: UUID of the document.
            document_version_id: UUID of the document version.

        Returns:
            ParsedDocument with extracted content.
        """
        ...
