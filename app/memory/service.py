"""Memory application service managing short-term conversations and long-term user context."""

from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from app.domain.memory.models import (
    ConversationDetail,
    ConversationMessage,
    MemoryRecord,
    MemoryUpsert,
)
from app.domain.memory.protocol import (
    ConversationRepositoryProtocol,
    LongTermMemoryRepositoryProtocol,
)


class MemoryService:
    """Coordinates durable user preferences and conversational message history."""

    def __init__(
        self,
        memory_repo: LongTermMemoryRepositoryProtocol,
        conversation_repo: Optional[ConversationRepositoryProtocol] = None,
    ) -> None:
        self.memory_repo = memory_repo
        self.conversation_repo = conversation_repo

    # --- Long-Term Memory (Preferences & Scoped Context) ---

    def upsert_memory(self, user_id: UUID, memory: MemoryUpsert) -> MemoryRecord:
        """Upsert a user preference or context entry."""
        return self.memory_repo.upsert_memory(user_id=user_id, memory=memory)

    def get_memory(self, user_id: UUID, type: str, key: str) -> Optional[MemoryRecord]:
        """Retrieve a specific user memory."""
        return self.memory_repo.get_memory(user_id=user_id, type=type, key=key)

    def list_memories(self, user_id: UUID, type: Optional[str] = None) -> List[MemoryRecord]:
        """List memories for a user, optionally filtered by type."""
        return self.memory_repo.list_memories(user_id=user_id, type=type)

    def delete_memory(self, user_id: UUID, type: str, key: str) -> bool:
        """Delete a user memory entry."""
        return self.memory_repo.delete_memory(user_id=user_id, type=type, key=key)

    def build_scoped_context(
        self,
        user_id: UUID,
        relevant_types: Sequence[str] = ("preference", "tone", "jurisdiction", "case_context", "department"),
    ) -> str:
        """Construct scoped user preference context for prompting.

        Explicitly marked as non-authoritative style/context guidance per Spec Section 19.3.
        """
        memories = self.memory_repo.list_memories(user_id=user_id)
        filtered = [m for m in memories if m.type.lower() in relevant_types]

        if not filtered:
            return ""

        lines = ["[USER PREFERENCES & CONTEXT (NON-AUTHORITATIVE GUIDELINES)]"]
        for m in filtered:
            val_str = m.value.get("text") or m.value.get("value") or str(m.value)
            lines.append(f"- {m.type.capitalize()} ({m.key}): {val_str}")

        lines.append("(Note: User preferences guide tone and context only. They are NOT legal evidence and must not be cited as law.)")
        return "\n".join(lines)

    # --- Short-Term Memory (Conversations & Turns) ---

    def create_conversation(self, user_id: UUID, title: Optional[str] = None) -> UUID:
        """Create a new conversation session."""
        if not self.conversation_repo:
            raise NotImplementedError("Conversation repository is not configured.")
        return self.conversation_repo.create_conversation(user_id=user_id, title=title)

    def get_conversation(self, conversation_id: UUID) -> Optional[ConversationDetail]:
        """Get full conversation details with messages."""
        if not self.conversation_repo:
            raise NotImplementedError("Conversation repository is not configured.")
        return self.conversation_repo.get_conversation(conversation_id=conversation_id)

    def add_message(self, conversation_id: UUID, role: str, content: str) -> ConversationMessage:
        """Append a message turn to conversation history."""
        if not self.conversation_repo:
            raise NotImplementedError("Conversation repository is not configured.")
        return self.conversation_repo.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

    def list_messages(self, conversation_id: UUID, limit: int = 50) -> List[ConversationMessage]:
        """Retrieve recent conversation messages."""
        if not self.conversation_repo:
            raise NotImplementedError("Conversation repository is not configured.")
        return self.conversation_repo.list_messages(conversation_id=conversation_id, limit=limit)
