"""Reranking infrastructure implementations."""

from typing import Optional
from app.core.config import get_settings
from app.domain.retrieval.protocol import RerankerProtocol
from app.infrastructure.reranking.cross_encoder import CrossEncoderReranker
from app.infrastructure.reranking.fake import FakeReranker


def get_default_reranker(
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> RerankerProtocol:
    """Create reranker, falling back to FakeReranker if disabled or sentence-transformers is missing."""
    settings = get_settings()
    is_enabled = settings.reranker_enabled if enabled is None else enabled
    if not is_enabled:
        return FakeReranker()

    try:
        import sentence_transformers  # noqa: F401
        return CrossEncoderReranker(
            model_name=model_name or settings.reranker_model,
            batch_size=batch_size or settings.reranker_batch_size,
        )
    except (ImportError, Exception):
        return FakeReranker()


__all__ = [
    "CrossEncoderReranker",
    "FakeReranker",
    "get_default_reranker",
]

