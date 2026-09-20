"""Exceptions for verification domain."""

from app.domain.documents.exceptions import DocumentError


class VerificationError(DocumentError):
    """Base exception for claim verification failures."""
