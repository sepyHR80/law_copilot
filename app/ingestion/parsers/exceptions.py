"""Parser-related exceptions."""

from app.domain.documents.exceptions import DocumentError


class DocumentParsingError(DocumentError):
    """Raised when document parsing fails."""


class UnsupportedDocumentFormatError(DocumentError):
    """Raised when no parser exists for the given MIME type."""


class InvalidDocumentContentError(DocumentError):
    """Raised when document content is invalid or corrupted."""
