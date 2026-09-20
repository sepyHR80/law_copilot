"""SQLAlchemy repository implementations for Memory and Conversation models."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.domain.memory.exceptions import ConversationNotFoundError
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
from app.infrastructure.db.models.conversation import Conversation, Message
from app.infrastructure.db.models.memory import Memory


class SQLAlchemyMemoryRepository(LongTermMemoryRepositoryProtocol):
    """PostgreSQL repository for durable user preferences and case context."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_memory(self, user_id: UUID, memory: MemoryUpsert) -> MemoryRecord:
        """Upsert a memory record atomically on conflict (user_id, type, key)."""
        now = datetime.utcnow()
        new_id = uuid4()

        stmt = (
            insert(Memory)
            .values(
                id=new_id,
                user_id=user_id,
                type=memory.type,
                key=memory.key,
                value=memory.value,
                confidence=memory.confidence,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["user_id", "type", "key"],
                set_={
                    "value": memory.value,
                    "confidence": memory.confidence,
                    "updated_at": now,
                },
            )
            .returning(Memory)
        )

        result = self.session.execute(stmt)
        self.session.commit()
        orm_record = result.scalar_one()

        return MemoryRecord(
            id=orm_record.id,
            user_id=orm_record.user_id,
            type=orm_record.type,
            key=orm_record.key,
            value=orm_record.value,
            confidence=orm_record.confidence,
            created_at=orm_record.created_at,
            updated_at=orm_record.updated_at,
        )

    def get_memory(self, user_id: UUID, type: str, key: str) -> Optional[MemoryRecord]:
        """Retrieve a specific memory record by user_id, type, and key."""
        stmt = select(Memory).where(
            Memory.user_id == user_id,
            Memory.type == type,
            Memory.key == key,
        )
        record = self.session.scalars(stmt).first()
        if not record:
            return None

        return MemoryRecord(
            id=record.id,
            user_id=record.user_id,
            type=record.type,
            key=record.key,
            value=record.value,
            confidence=record.confidence,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def list_memories(self, user_id: UUID, type: Optional[str] = None) -> List[MemoryRecord]:
        """List all memories for a user, optionally filtered by type."""
        stmt = select(Memory).where(Memory.user_id == user_id)
        if type is not None:
            stmt = stmt.where(Memory.type == type)
        stmt = stmt.order_by(Memory.created_at.desc())

        records = self.session.scalars(stmt).all()
        return [
            MemoryRecord(
                id=r.id,
                user_id=r.user_id,
                type=r.type,
                key=r.key,
                value=r.value,
                confidence=r.confidence,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in records
        ]

    def delete_memory(self, user_id: UUID, type: str, key: str) -> bool:
        """Delete a memory record by user_id, type, and key."""
        stmt = select(Memory).where(
            Memory.user_id == user_id,
            Memory.type == type,
            Memory.key == key,
        )
        record = self.session.scalars(stmt).first()
        if not record:
            return False

        self.session.delete(record)
        self.session.commit()
        return True


class SQLAlchemyConversationRepository(ConversationRepositoryProtocol):
    """PostgreSQL repository for short-term conversation turns and sessions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_conversation(self, user_id: UUID, title: Optional[str] = None) -> UUID:
        """Create a new conversation session."""
        conv = Conversation(
            id=uuid4(),
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
        )
        self.session.add(conv)
        self.session.commit()
        return conv.id

    def get_conversation(self, conversation_id: UUID) -> Optional[ConversationDetail]:
        """Retrieve conversation detail with all messages."""
        conv = self.session.get(Conversation, conversation_id)
        if not conv:
            return None

        stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
        messages = self.session.scalars(stmt).all()

        return ConversationDetail(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=[
                ConversationMessage(
                    id=m.id,
                    conversation_id=m.conversation_id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at,
                )
                for m in messages
            ],
        )

    def add_message(self, conversation_id: UUID, role: str, content: str) -> ConversationMessage:
        """Append a message turn to a conversation."""
        conv = self.session.get(Conversation, conversation_id)
        if not conv:
            raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist.")

        msg = Message(
            id=uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=datetime.utcnow(),
        )
        self.session.add(msg)
        self.session.commit()

        return ConversationMessage(
            id=msg.id,
            conversation_id=msg.conversation_id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at,
        )

    def list_messages(self, conversation_id: UUID, limit: int = 50) -> List[ConversationMessage]:
        """List messages for a conversation session."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        messages = self.session.scalars(stmt).all()
        return [
            ConversationMessage(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ]
