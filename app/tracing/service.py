"""Trace Service for recording, aggregating, and querying AI chat execution traces."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import case, desc, func, select
from sqlalchemy.orm import Session

from app.infrastructure.db.models.trace import ChatTrace
from app.infrastructure.db.session import SessionLocal
from app.tracing.categorizer import categorize_query
from app.tracing.models import (
    CategoriesResponse,
    CategoryStat,
    ChatTraceDetail,
    ChatTraceSummary,
    TraceListResponse,
)

logger = logging.getLogger(__name__)


class TraceService:
    """Service to capture and query agent execution traces and categories."""

    @staticmethod
    def ensure_table_exists(session: Session) -> None:
        """Ensure the chat_traces table exists in the database."""
        try:
            ChatTrace.__table__.create(session.get_bind(), checkfirst=True)
        except Exception as exc:
            logger.debug("ChatTrace checkfirst notice: %s", exc)

    @classmethod
    def record_trace(
        cls,
        user_query: str,
        ai_response: str,
        intent: str,
        is_sufficient: bool,
        execution_path: List[Dict[str, Any]],
        retrieval_data: Optional[List[Dict[str, Any]]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        model_name: Optional[str] = None,
        latency_ms: Optional[int] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        trace_metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> Optional[UUID]:
        """Record an interaction execution trace safely without blocking."""
        category = categorize_query(
            query=user_query,
            intent=intent,
            is_sufficient=is_sufficient,
        )

        db = session or SessionLocal()
        should_close = session is None

        try:
            cls.ensure_table_exists(db)

            # Clean/parse conversation_id if present
            conv_uuid = None
            if conversation_id:
                try:
                    conv_uuid = UUID(str(conversation_id))
                except (ValueError, TypeError):
                    conv_uuid = None

            trace_record = ChatTrace(
                id=uuid4(),
                conversation_id=conv_uuid,
                category=category,
                intent=intent,
                user_query=user_query,
                ai_response=ai_response,
                status=status,
                is_sufficient=is_sufficient,
                execution_path=execution_path or [],
                retrieval_data=retrieval_data or [],
                citations=citations or [],
                evidence=evidence or [],
                model_name=model_name,
                latency_ms=latency_ms,
                error_message=error_message,
                trace_metadata=trace_metadata or {},
                created_at=datetime.utcnow(),
            )
            db.add(trace_record)
            db.commit()
            return trace_record.id
        except Exception as exc:
            logger.warning("Failed to persist ChatTrace: %s", exc)
            try:
                db.rollback()
            except Exception:
                pass
            return None
        finally:
            if should_close:
                db.close()

    @classmethod
    def get_categories_summary(cls, session: Session) -> CategoriesResponse:
        """Aggregate all chats by category with statistics."""
        cls.ensure_table_exists(session)

        # Total count
        total = session.scalar(select(func.count(ChatTrace.id))) or 0

        # Group by category
        stmt = (
            select(
                ChatTrace.category,
                func.count(ChatTrace.id).label("count"),
                func.sum(case((ChatTrace.status == "success", 1), else_=0)).label("success_count"),
                func.sum(case((ChatTrace.is_sufficient.is_(False), 1), else_=0)).label("insufficient_count"),
                func.sum(case((ChatTrace.status.in_(["error", "rate_limited"]), 1), else_=0)).label("error_count"),
                func.avg(ChatTrace.latency_ms).label("avg_latency"),
                func.max(ChatTrace.created_at).label("latest_at"),
            )
            .group_by(ChatTrace.category)
            .order_by(desc("count"))
        )
        rows = session.execute(stmt).all()

        categories: List[CategoryStat] = []
        total_success = 0
        total_latency = 0
        latency_samples = 0

        for r in rows:
            cnt = int(r.count or 0)
            succ = int(r.success_count or 0)
            insuff = int(r.insufficient_count or 0)
            err = int(r.error_count or 0)
            avg_lat = float(r.avg_latency or 0.0)
            latest = r.latest_at

            total_success += succ
            if r.avg_latency:
                total_latency += avg_lat * cnt
                latency_samples += cnt

            categories.append(
                CategoryStat(
                    category=r.category,
                    count=cnt,
                    success_count=succ,
                    insufficient_count=insuff,
                    error_count=err,
                    avg_latency_ms=round(avg_lat, 1),
                    latest_at=latest,
                )
            )

        overall_success_rate = round((total_success / total * 100), 1) if total > 0 else 100.0
        avg_system_latency = round((total_latency / latency_samples), 1) if latency_samples > 0 else 0.0

        return CategoriesResponse(
            total_traces=total,
            categories=categories,
            overall_success_rate=overall_success_rate,
            avg_system_latency_ms=avg_system_latency,
        )

    @classmethod
    def list_traces(
        cls,
        session: Session,
        category: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> TraceListResponse:
        """Return paginated list of chat traces matching filters."""
        cls.ensure_table_exists(session)

        base_query = select(ChatTrace)

        if category and category.strip():
            base_query = base_query.where(ChatTrace.category == category.strip())

        if status and status.strip():
            base_query = base_query.where(ChatTrace.status == status.strip())

        if search and search.strip():
            term = f"%{search.strip()}%"
            base_query = base_query.where(
                ChatTrace.user_query.ilike(term) | ChatTrace.ai_response.ilike(term)
            )

        # Count total
        count_stmt = select(func.count()).select_from(base_query.subquery())
        total = session.scalar(count_stmt) or 0

        # Execute pagination
        paged_stmt = base_query.order_by(desc(ChatTrace.created_at)).limit(limit).offset(offset)
        records = session.scalars(paged_stmt).all()

        items: List[ChatTraceSummary] = []
        for r in records:
            preview = (r.ai_response or "")[:150]
            if len(r.ai_response or "") > 150:
                preview += "..."

            step_count = len(r.execution_path) if isinstance(r.execution_path, list) else 0

            items.append(
                ChatTraceSummary(
                    id=r.id,
                    conversation_id=r.conversation_id,
                    category=r.category,
                    intent=r.intent,
                    user_query=r.user_query,
                    ai_response_preview=preview,
                    status=r.status,
                    is_sufficient=r.is_sufficient,
                    model_name=r.model_name,
                    latency_ms=r.latency_ms,
                    step_count=step_count,
                    created_at=r.created_at,
                )
            )

        return TraceListResponse(
            total=total,
            limit=limit,
            offset=offset,
            items=items,
        )

    @classmethod
    def get_trace(cls, session: Session, trace_id: str) -> Optional[ChatTraceDetail]:
        """Fetch complete detailed execution trace by UUID."""
        cls.ensure_table_exists(session)
        try:
            uid = UUID(str(trace_id))
        except (ValueError, TypeError):
            return None

        record = session.get(ChatTrace, uid)
        if not record:
            return None

        return ChatTraceDetail(
            id=record.id,
            conversation_id=record.conversation_id,
            category=record.category,
            intent=record.intent,
            user_query=record.user_query,
            ai_response=record.ai_response,
            status=record.status,
            is_sufficient=record.is_sufficient,
            execution_path=record.execution_path or [],
            retrieval_data=record.retrieval_data or [],
            citations=record.citations or [],
            evidence=record.evidence or [],
            model_name=record.model_name,
            latency_ms=record.latency_ms,
            error_message=record.error_message,
            trace_metadata=record.trace_metadata or {},
            created_at=record.created_at,
        )

    @classmethod
    def delete_trace(cls, session: Session, trace_id: str) -> bool:
        """Delete a single chat trace."""
        cls.ensure_table_exists(session)
        try:
            uid = UUID(str(trace_id))
        except (ValueError, TypeError):
            return False

        record = session.get(ChatTrace, uid)
        if not record:
            return False

        session.delete(record)
        session.commit()
        return True

    @classmethod
    def clear_traces(cls, session: Session) -> int:
        """Remove all chat traces for administrative maintenance."""
        cls.ensure_table_exists(session)
        deleted = session.query(ChatTrace).delete()
        session.commit()
        return deleted
