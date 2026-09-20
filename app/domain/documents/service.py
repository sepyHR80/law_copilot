"""Document upload application service."""

import hashlib
import os
import re
from typing import Optional
from uuid import UUID, uuid4

from app.core.config import Settings, get_settings
from app.domain.documents.exceptions import (
    DocumentStorageError,
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentMetadataError,
    UnsupportedDocumentTypeError,
)
from app.domain.documents.repository import DocumentRepositoryProtocol
from app.infrastructure.storage import StorageBackend
from app.infrastructure.storage.exceptions import StorageError

# Supported MIME types and their corresponding extensions
SUPPORTED_MIME_TYPES = {
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "text/plain": [".txt"],
    "text/html": [".html"],
}

# Valid knowledge types
VALID_KNOWLEDGE_TYPES = {"factual", "stylistic"}


class DocumentService:
    """Application service for document upload operations."""

    def __init__(
        self,
        storage: StorageBackend,
        repository: DocumentRepositoryProtocol,
        settings: Optional[Settings] = None,
    ) -> None:
        self._storage = storage
        self._repository = repository
        self._settings = settings or get_settings()

    def upload_document(
        self,
        file_content: bytes,
        filename: str,
        title: str,
        document_type: str,
        knowledge_type: str,
        source: str,
        mime_type: str,
        version: str = "1.0",
    ) -> dict:
        """Upload a document: validate, store, and create database records.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: Original filename from the client.
            title: Document title.
            document_type: Type of document.
            knowledge_type: Knowledge category (factual/stylistic).
            source: Document source.
            mime_type: MIME type of the file.
            version: Version string (default "1.0").

        Returns:
            Dictionary with document metadata.

        Raises:
            UnsupportedDocumentTypeError: If MIME type is not supported.
            DocumentTooLargeError: If file exceeds size limit.
            EmptyDocumentError: If file is empty.
            InvalidDocumentMetadataError: If metadata is invalid.
            DocumentStorageError: If storage or database operation fails.
        """
        # Validate MIME type
        if mime_type not in SUPPORTED_MIME_TYPES:
            raise UnsupportedDocumentTypeError(
                f"Unsupported MIME type: {mime_type}. "
                f"Supported types: {list(SUPPORTED_MIME_TYPES.keys())}"
            )

        # Validate file extension matches MIME type
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_MIME_TYPES[mime_type]:
            raise InvalidDocumentMetadataError(
                f"File extension '{ext}' does not match MIME type '{mime_type}'"
            )

        # Validate file size
        if len(file_content) > self._settings.max_document_size_bytes:
            raise DocumentTooLargeError(
                f"File size {len(file_content)} exceeds maximum "
                f"{self._settings.max_document_size_bytes} bytes"
            )

        # Validate non-empty file
        if len(file_content) == 0:
            raise EmptyDocumentError("Uploaded file is empty")

        # Validate title
        if not title or not title.strip():
            raise InvalidDocumentMetadataError("Document title is required")

        # Validate knowledge type
        if knowledge_type not in VALID_KNOWLEDGE_TYPES:
            raise InvalidDocumentMetadataError(
                f"Invalid knowledge_type: {knowledge_type}. "
                f"Must be one of: {VALID_KNOWLEDGE_TYPES}"
            )

        # Generate document ID and storage key
        document_id = uuid4()
        safe_filename = self._sanitize_filename(filename)
        storage_key = f"documents/{document_id}/{version}/{safe_filename}"

        # Calculate checksum
        checksum = hashlib.sha256(file_content).hexdigest()

        # Upload to MinIO first
        try:
            self._storage.put_object(
                key=storage_key,
                data=file_content,
                content_type=mime_type,
            )
        except StorageError as exc:
            raise DocumentStorageError(f"Failed to store document: {exc}") from exc

        # Create database records
        try:
            document, doc_version = self._repository.create_with_version(
                document_id=document_id,
                title=title,
                document_type=document_type,
                knowledge_type=knowledge_type,
                source=source,
                version=version,
                storage_key=storage_key,
                checksum=checksum,
            )
            self._repository.commit()
        except Exception as exc:
            # Rollback and cleanup storage
            self._repository.rollback()
            try:
                self._storage.delete_object(storage_key)
            except StorageError:
                pass  # Best effort cleanup
            raise DocumentStorageError(f"Failed to create document records: {exc}") from exc

        return {
            "id": document.id,
            "title": document.title,
            "document_type": document.document_type,
            "knowledge_type": document.knowledge_type,
            "source": document.source,
            "version": doc_version.version,
            "storage_key": doc_version.storage_key,
            "checksum": doc_version.checksum,
            "file_name": safe_filename,
            "mime_type": mime_type,
            "created_at": document.created_at,
        }

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage key usage.

        Removes path components and unsafe characters.
        """
        # Get just the basename (no directory components)
        basename = os.path.basename(filename)
        # Remove any remaining path traversal patterns
        basename = basename.replace("..", "")
        # Keep only safe characters
        basename = re.sub(r"[^a-zA-Z0-9._-]", "_", basename)
        # Ensure it's not empty
        if not basename:
            basename = "document"
        return basename
