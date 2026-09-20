"""Cross-Encoder Reranker implementation."""

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple
from uuid import UUID

from app.core.config import get_settings
from app.domain.retrieval.exceptions import RerankerModelLoadError, RerankingError
from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.protocol import RerankerProtocol


def _sigmoid(x: float) -> float:
    """Logistic sigmoid function mapping logits (-inf, +inf) to [0.0, 1.0]."""
    # Prevent overflow
    if x >= 40.0:
        return 1.0
    elif x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker(RerankerProtocol):
    """Cross-Encoder neural reranker using joint query-document scoring.

    Features:
    - Lazy model loading so imports/downloads don't slow down startup.
    - Configurable batch scoring to optimize memory and throughput.
    - Sigmoid normalization of logits into [0.0, 1.0].
    - Injected model_backend support for deterministic unit testing without downloads.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        model_backend: Optional[Any] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size or settings.reranker_batch_size
        self._model_backend = model_backend
        self._initialized = model_backend is not None

    def _get_backend(self) -> Any:
        """Lazy load the underlying model backend."""
        if self._initialized:
            return self._model_backend

        try:
            from sentence_transformers import CrossEncoder

            self._model_backend = CrossEncoder(self.model_name)
            self._initialized = True
            return self._model_backend
        except ImportError as exc:
            raise RerankerModelLoadError(
                f"sentence-transformers is not installed. Install it or provide a custom backend: {exc}"
            ) from exc
        except Exception as exc:
            raise RerankerModelLoadError(
                f"Failed to load CrossEncoder model '{self.model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank candidate chunks by joint cross-encoder scoring.

        Args:
            query: The search query.
            candidates: Sequence of RetrievalResult candidates.
            top_n: Optional max number of reranked candidates to return.

        Returns:
            List of RetrievalResult objects ordered by normalized score descending.
        """
        if not candidates:
            return []

        cleaned_query = query.strip()
        if not cleaned_query:
            # If query is empty, preserve original order and slice
            limit = top_n if top_n is not None and top_n >= 1 else len(candidates)
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

        backend = self._get_backend()
        pairs = [(cleaned_query, c.content) for c in candidates]

        raw_scores: List[float] = []

        try:
            # Process in batches
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]

                # Check if backend has predict or is a callable
                if hasattr(backend, "predict"):
                    batch_scores = backend.predict(batch_pairs)
                elif callable(backend):
                    batch_scores = backend(batch_pairs)
                else:
                    raise RerankingError(f"Unsupported model backend: {type(backend)}")

                # Convert ndarray / iterable to list of floats
                for score in batch_scores:
                    raw_scores.append(float(score))

        except Exception as exc:
            if isinstance(exc, (RerankerModelLoadError, RerankingError)):
                raise
            raise RerankingError(f"Inference error during cross-encoder reranking: {exc}") from exc

        # Apply sigmoid normalization and pair with candidates
        scored_candidates: List[Tuple[RetrievalResult, float]] = []
        for cand, raw_s in zip(candidates, raw_scores):
            normalized_s = _sigmoid(raw_s)
            scored_candidates.append((cand, normalized_s))

        # Deterministic sorting: highest score first, tie-break by chunk_id string
        scored_candidates.sort(
            key=lambda entry: (entry[1], str(entry[0].chunk_id)),
            reverse=True,
        )

        limit = top_n if top_n is not None and top_n >= 1 else len(scored_candidates)
        reranked_results: List[RetrievalResult] = []

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
