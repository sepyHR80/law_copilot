"""Memory and conversation history API routes."""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.domain.memory.exceptions import ConversationNotFoundError, MemoryNotFoundError
from app.domain.memory.models import (
    ConversationDetail,
    ConversationMessage,
    MemoryRecord,
    MemoryUpsert,
)
from app.infrastructure.db.repositories.memory import (
    SQLAlchemyConversationRepository,
    SQLAlchemyMemoryRepository,
)
from app.infrastructure.db.session import SessionLocal
from app.memory.service import MemoryService

router = APIRouter(tags=["memory"])


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_memory_service(db: Session = Depends(get_db)) -> MemoryService:
    """Dependency to construct MemoryService with database session."""
    return MemoryService(
        memory_repo=SQLAlchemyMemoryRepository(db),
        conversation_repo=SQLAlchemyConversationRepository(db),
    )


class MessageCreate(BaseModel):
    role: str
    content: str


class ConversationCreate(BaseModel):
    title: Optional[str] = None


# --- Long-Term Memory Endpoints ---

@router.get("/api/v1/memories", response_model=List[MemoryRecord], status_code=status.HTTP_200_OK)
def list_user_memories(
    user_id: UUID = Query(..., description="ID of the user whose memories are listed"),
    type: Optional[str] = Query(None, description="Optional filter by memory type"),
    service: MemoryService = Depends(get_memory_service),
) -> List[MemoryRecord]:
    """List durable user preferences and context entries."""
    return service.list_memories(user_id=user_id, type=type)


@router.post("/api/v1/memories", response_model=MemoryRecord, status_code=status.HTTP_200_OK)
def upsert_user_memory(
    memory: MemoryUpsert,
    user_id: UUID = Query(..., description="ID of the user whose memory is upserted"),
    service: MemoryService = Depends(get_memory_service),
) -> MemoryRecord:
    """Upsert a user preference or context entry."""
    return service.upsert_memory(user_id=user_id, memory=memory)


@router.delete("/api/v1/memories/{type}/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_memory(
    type: str,
    key: str,
    user_id: UUID = Query(..., description="ID of the user whose memory is deleted"),
    service: MemoryService = Depends(get_memory_service),
) -> None:
    """Delete a user preference or context entry."""
    deleted = service.delete_memory(user_id=user_id, type=type, key=key)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory of type '{type}' and key '{key}' not found for user {user_id}.",
        )


# --- Short-Term Conversation Endpoints ---

@router.post("/api/v1/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    user_id: UUID = Query(..., description="Owner user ID"),
    service: MemoryService = Depends(get_memory_service),
) -> dict:
    """Create a new conversation session."""
    conv_id = service.create_conversation(user_id=user_id, title=payload.title)
    return {"conversation_id": str(conv_id)}


@router.get("/api/v1/conversations/{conversation_id}", response_model=ConversationDetail, status_code=status.HTTP_200_OK)
def get_conversation(
    conversation_id: UUID,
    service: MemoryService = Depends(get_memory_service),
) -> ConversationDetail:
    """Get conversation session details with turn history."""
    conv = service.get_conversation(conversation_id=conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found.",
        )
    return conv


@router.get("/api/v1/conversations/{conversation_id}/messages", response_model=List[ConversationMessage], status_code=status.HTTP_200_OK)
def list_conversation_messages(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    service: MemoryService = Depends(get_memory_service),
) -> List[ConversationMessage]:
    """List message turns for a conversation session."""
    return service.list_messages(conversation_id=conversation_id, limit=limit)


@router.post("/api/v1/conversations/{conversation_id}/messages", response_model=ConversationMessage, status_code=status.HTTP_201_CREATED)
def add_conversation_message(
    conversation_id: UUID,
    message: MessageCreate,
    service: MemoryService = Depends(get_memory_service),
) -> ConversationMessage:
    """Append a new message turn to conversation history."""
    try:
        return service.add_message(
            conversation_id=conversation_id,
            role=message.role,
            content=message.content,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
