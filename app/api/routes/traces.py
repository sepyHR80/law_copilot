"""Chat Traces & Debugging API routes."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.infrastructure.db.session import SessionLocal
from app.tracing.models import (
    CategoriesResponse,
    ChatTraceDetail,
    TraceListResponse,
)
from app.tracing.service import TraceService

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])


def get_db():
    """Dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/categories", response_model=CategoriesResponse, status_code=status.HTTP_200_OK)
def get_categories_summary(db: Session = Depends(get_db)) -> CategoriesResponse:
    """Get aggregated metrics and chat counts per topic category."""
    return TraceService.get_categories_summary(session=db)


@router.get("", response_model=TraceListResponse, status_code=status.HTTP_200_OK)
def list_traces(
    category: Optional[str] = Query(None, description="Filter by topic category"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (success, insufficient, error)"),
    search: Optional[str] = Query(None, description="Search keyword in query or response"),
    limit: int = Query(50, ge=1, le=100, description="Page size"),
    offset: int = Query(0, ge=0, description="Offset"),
    db: Session = Depends(get_db),
) -> TraceListResponse:
    """Get paginated list of chat interaction traces."""
    return TraceService.list_traces(
        session=db,
        category=category,
        status=status_filter,
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get("/{trace_id}", response_model=ChatTraceDetail, status_code=status.HTTP_200_OK)
def get_trace_detail(
    trace_id: str,
    db: Session = Depends(get_db),
) -> ChatTraceDetail:
    """Get deep execution trace with step-by-step telemetry for debugging."""
    trace = TraceService.get_trace(session=db, trace_id=trace_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with ID '{trace_id}' not found.",
        )
    return trace


@router.delete("/{trace_id}", status_code=status.HTTP_200_OK)
def delete_trace(
    trace_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a single trace record."""
    deleted = TraceService.delete_trace(session=db, trace_id=trace_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with ID '{trace_id}' not found.",
        )
    return {"status": "deleted", "id": trace_id}


@router.post("/clear", status_code=status.HTTP_200_OK)
def clear_all_traces(db: Session = Depends(get_db)) -> dict:
    """Clear all chat traces for administrative maintenance."""
    count = TraceService.clear_traces(session=db)
    return {"status": "cleared", "deleted_count": count}
