"""Exceptions for search operations."""

from app.domain.documents.exceptions import DocumentError


class ExternalSearchError(DocumentError):
    """Raised when external search operations fail."""
