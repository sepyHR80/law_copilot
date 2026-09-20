"""Memory model for long-term user context."""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from sqlalchemy import String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("user_id", "type", "key", name="uq_memory_user_type_key"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, onupdate=datetime.utcnow)
