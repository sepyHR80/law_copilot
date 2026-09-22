"""Chat Trace model for agent execution tracking and debugging."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base

JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
UUID_TYPE = Uuid(as_uuid=True).with_variant(UUID(as_uuid=True), "postgresql")


class ChatTrace(Base):
    """Stores full step-by-step telemetry, categorized topic, and execution path of an AI interaction."""

    __tablename__ = "chat_traces"

    id: Mapped[str] = mapped_column(UUID_TYPE, primary_key=True, default=uuid4)
    conversation_id: Mapped[Optional[str]] = mapped_column(UUID_TYPE, nullable=True, index=True)

    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    intent: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    ai_response: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(30), nullable=False, default="success", index=True)
    is_sufficient: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    execution_path: Mapped[List[Dict[str, Any]]] = mapped_column(JSON_TYPE, default=list, nullable=False)
    retrieval_data: Mapped[List[Dict[str, Any]]] = mapped_column(JSON_TYPE, default=list, nullable=False)
    citations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON_TYPE, default=list, nullable=False)
    evidence: Mapped[List[Dict[str, Any]]] = mapped_column(JSON_TYPE, default=list, nullable=False)

    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trace_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON_TYPE, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        index=True,
    )
