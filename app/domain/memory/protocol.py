"""Memory repository protocol definitions."""

from typing import List, Optional, Protocol, runtime_checkable
from uuid import UUID

from app.domain.memory.models import (
    ConversationDetail,
    ConversationMessage,
    MemoryRecord,
    MemoryUpsert,
)


@runtime_checkable
class LongTermMemoryRepositoryProtocol(Protocol):
    """Protocol for persisting durable user preferences and case context."""

    def upsert_memory(self, user_id: UUID, memory: MemoryUpsert) -> MemoryRecord:
        """Upsert a memory record by (user_id, type, key)."""
        ...

    def get_memory(self, user_id: UUID, type: str, key: str) -> Optional[MemoryRecord]:
        """Retrieve a specific memory record."""
        ...

    def list_memories(self, user_id: UUID, type: Optional[str] = None) -> List[MemoryRecord]:
        """List all memories for a user, optionally filtered by type."""
        ...

    def delete_memory(self, user_id: UUID, type: str, key: str) -> bool:
        """Delete a memory record. Returns True if deleted, False if not found."""
        ...


@runtime_checkable
class ConversationRepositoryProtocol(Protocol):
    """Protocol for short-term conversation sessions and message history."""

    def create_conversation(self, user_id: UUID, title: Optional[str] = None) -> UUID:
        """Create a new conversation session."""
        ...

    def get_conversation(self, conversation_id: UUID) -> Optional[ConversationDetail]:
        """Get conversation details with messages ordered by created_at."""
        ...

    def add_message(self, conversation_id: UUID, role: str, content: str) -> ConversationMessage:
        """Append a message turn to a conversation."""
        ...

    def list_messages(self, conversation_id: UUID, limit: int = 50) -> List[ConversationMessage]:
        """List recent messages for a conversation session."""
        ...
