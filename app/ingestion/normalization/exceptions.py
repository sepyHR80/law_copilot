"""Normalization-related exceptions."""

from app.domain.documents.exceptions import DocumentError


class DocumentNormalizationError(DocumentError):
    """Raised when document normalization fails."""
