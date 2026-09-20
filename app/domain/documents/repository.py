"""Document repository interface (protocol).

This module defines the abstract interface for document persistence.
The domain layer depends on this protocol, not on SQLAlchemy directly.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from uuid import UUID


class DocumentRepositoryProtocol(ABC):
    """Abstract interface for document persistence operations."""

    @abstractmethod
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
    ) -> Tuple:
        """Create a Document and its initial DocumentVersion."""
        ...

    @abstractmethod
    def get_by_id(self, document_id: UUID) -> Optional[object]:
        """Get a document by its ID."""
        ...

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...
