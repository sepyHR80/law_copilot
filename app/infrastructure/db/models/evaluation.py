"""Evaluation case model."""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class EvaluationCase(Base):
    __tablename__ = "evaluation_cases"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected_sources: Mapped[List[str]] = mapped_column(JSONB, default=list, nullable=False)
    expected_style: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, onupdate=datetime.utcnow)
