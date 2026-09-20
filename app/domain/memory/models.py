"""Domain models for memory and conversation history."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """Represents a long-term memory entry in the database."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    type: str = Field(..., max_length=50)
    key: str = Field(..., max_length=255)
    value: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class MemoryUpsert(BaseModel):
    """Input model for writing or updating a user memory."""

    type: str = Field(..., max_length=50, min_length=1)
    key: str = Field(..., max_length=255, min_length=1)
    value: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ConversationMessage(BaseModel):
    """Represents a message turn within a conversation."""

    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    role: str = Field(..., max_length=20)
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationDetail(BaseModel):
    """Represents a conversation session and its associated messages."""

    id: UUID
    user_id: UUID
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[ConversationMessage] = Field(default_factory=list)
