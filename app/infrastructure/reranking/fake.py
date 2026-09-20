"""Deterministic Fake Reranker for offline unit tests and development."""

from typing import Callable, List, Optional, Sequence
from uuid import UUID

from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.protocol import RerankerProtocol


class FakeReranker(RerankerProtocol):
    """Deterministic in-memory reranker.

    Calculates scores based on word overlap or a custom scoring function.
    Guarantees no external model downloads during automated unit tests.
    """

    def __init__(
        self,
        custom_scorer: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.custom_scorer = custom_scorer

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank candidates based on keyword overlap or custom scorer."""
        if not candidates:
            return []

        scored_candidates: List[tuple[RetrievalResult, float]] = []
        query_words = set(query.lower().split())

        for cand in candidates:
            if self.custom_scorer:
                score = self.custom_scorer(query, cand.content)
            else:
                # Deterministic word overlap ratio
                cand_words = set(cand.content.lower().split())
                if query_words and cand_words:
                    overlap = len(query_words.intersection(cand_words))
                    score = min(1.0, overlap / len(query_words))
                else:
                    score = 0.0

            scored_candidates.append((cand, score))

        # Deterministic sort by score descending, tie-break by chunk_id string
        scored_candidates.sort(
            key=lambda entry: (entry[1], str(entry[0].chunk_id)),
            reverse=True,
        )

        reranked_results: List[RetrievalResult] = []
        limit = top_n if top_n is not None and top_n >= 1 else len(scored_candidates)

        for rank_idx, (cand, score) in enumerate(scored_candidates[:limit], start=1):
            reranked_results.append(
                RetrievalResult(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    document_version_id=cand.document_version_id,
                    content=cand.content,
                    score=round(score, 6),
                    rank=rank_idx,
                    source=dict(cand.source),
                )
            )

        return reranked_results
