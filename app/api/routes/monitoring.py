"""Monitoring, health, readiness, and metrics endpoints for Law Copilot."""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.metrics import metrics
from app.infrastructure.db.session import SessionLocal

router = APIRouter(tags=["monitoring"])


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/healthz", status_code=status.HTTP_200_OK)
def liveness_probe() -> dict[str, str]:
    """Kubernetes liveness probe: indicates if the process is running."""
    return {"status": "alive"}


@router.get("/readyz", status_code=status.HTTP_200_OK)
def readiness_probe(db: Session = Depends(get_db)) -> dict[str, str]:
    """Kubernetes readiness probe: verifies database connectivity and core services."""
    try:
        # Check database connectivity
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    is_ready = db_status == "connected"
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return Response(
        content=f'{{"status": "{"ready" if is_ready else "not_ready"}", "database": "{db_status}"}}',
        media_type="application/json",
        status_code=status_code,
    )


@router.get("/metrics", status_code=status.HTTP_200_OK)
def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint exposing metrics in text format."""
    return Response(
        content=metrics.export_prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
