"""Reranking service for re-scoring and ordering retrieval candidates."""

from typing import List, Optional, Sequence

from app.core.config import get_settings
from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.protocol import RerankerProtocol
from app.infrastructure.reranking.cross_encoder import CrossEncoderReranker


class RerankingService:
    """Service to rerank candidate chunks using a configured RerankerProtocol."""

    def __init__(
        self,
        reranker: Optional[RerankerProtocol] = None,
    ) -> None:
        self._reranker = reranker

    @property
    def reranker(self) -> RerankerProtocol:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank candidates using the underlying reranker model.

        Args:
            query: The search query.
            candidates: Sequence of RetrievalResult candidates.
            top_n: Optional number of top candidates to return. Defaults to settings.reranker_top_n.

        Returns:
            List of reranked RetrievalResult objects.
        """
        if not candidates:
            return []

        settings = get_settings()
        effective_top_n = top_n if top_n is not None else settings.reranker_top_n

        if not settings.reranker_enabled:
            limit = effective_top_n if effective_top_n >= 1 else len(candidates)
            return [
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    document_version_id=c.document_version_id,
                    content=c.content,
                    score=c.score,
                    rank=idx,
                    source=dict(c.source),
                )
                for idx, c in enumerate(candidates[:limit], start=1)
            ]

        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_n=effective_top_n,
        )
