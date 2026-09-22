"""Pydantic models for chat tracing, execution path steps, and categories."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class StepTrace(BaseModel):
    """Represents a single step in the AI agent's execution path."""

    step: str = Field(..., description="Unique step identifier (e.g. analyze_intent, retrieve_knowledge)")
    title: str = Field(..., description="Human-readable Persian title")
    status: str = Field(default="completed", description="Status: completed, skipped, failed")
    duration_ms: int = Field(default=0, description="Duration of this step in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic data captured during this step")


class ChatTraceSummary(BaseModel):
    """Summary representation of a chat trace for list views."""

    id: UUID
    conversation_id: Optional[UUID] = None
    category: str
    intent: str
    user_query: str
    ai_response_preview: str
    status: str
    is_sufficient: bool
    model_name: Optional[str] = None
    latency_ms: Optional[int] = None
    step_count: int = 0
    created_at: datetime


class ChatTraceDetail(BaseModel):
    """Full deep trace data including entire execution path and raw payloads."""

    id: UUID
    conversation_id: Optional[UUID] = None
    category: str
    intent: str
    user_query: str
    ai_response: str
    status: str
    is_sufficient: bool
    execution_path: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_data: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    model_name: Optional[str] = None
    latency_ms: Optional[int] = None
    error_message: Optional[str] = None
    trace_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class CategoryStat(BaseModel):
    """Aggregated statistics for a chat category."""

    category: str
    count: int
    success_count: int
    insufficient_count: int
    error_count: int
    avg_latency_ms: float
    latest_at: Optional[datetime] = None


class CategoriesResponse(BaseModel):
    """Response containing category summaries and system totals."""

    total_traces: int
    categories: List[CategoryStat]
    overall_success_rate: float
    avg_system_latency_ms: float


class TraceListResponse(BaseModel):
    """Paginated list of chat traces."""

    total: int
    limit: int
    offset: int
    items: List[ChatTraceSummary]
