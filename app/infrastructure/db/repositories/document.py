"""SQLAlchemy implementation of the document repository."""

from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.domain.documents.repository import DocumentRepositoryProtocol
from app.infrastructure.db.models.document import Document, DocumentVersion


class SQLAlchemyDocumentRepository(DocumentRepositoryProtocol):
    """SQLAlchemy-based document repository implementation."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create_with_version(
        self,
        document_id: UUID,
        title: str,
        document_type: str,
        knowledge_type: str,
        source: str,
        version: str,
        storage_key: str,
        checksum: str,
    ) -> Tuple[Document, DocumentVersion]:
        """Create a Document and its initial DocumentVersion."""
        document = Document(
            id=document_id,
            title=title,
            document_type=document_type,
            knowledge_type=knowledge_type,
            source=source,
        )
        self._db.add(document)
        self._db.flush()

        doc_version = DocumentVersion(
            document_id=document.id,
            version=version,
            storage_key=storage_key,
            checksum=checksum,
        )
        self._db.add(doc_version)
        self._db.flush()

        return document, doc_version

    def commit(self) -> None:
        """Commit the current transaction."""
        self._db.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._db.rollback()

    def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Get a document by its ID."""
        return self._db.query(Document).filter_by(id=document_id).first()
