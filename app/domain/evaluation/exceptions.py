"""Exceptions for evaluation domain."""

from app.domain.documents.exceptions import DocumentError


class EvaluationError(DocumentError):
    """Base exception for evaluation operations."""
