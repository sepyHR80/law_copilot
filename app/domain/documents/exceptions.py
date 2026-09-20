"""Application-level exceptions for document operations."""


class DocumentError(Exception):
    """Base exception for document-related errors."""


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when the document type or MIME type is not supported."""


class DocumentTooLargeError(DocumentError):
    """Raised when the uploaded file exceeds the size limit."""


class EmptyDocumentError(DocumentError):
    """Raised when the uploaded file is empty."""


class InvalidDocumentMetadataError(DocumentError):
    """Raised when document metadata is invalid."""


class DocumentStorageError(DocumentError):
    """Raised when a storage operation fails."""
