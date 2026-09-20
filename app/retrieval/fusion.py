"""Reciprocal Rank Fusion (RRF) algorithm.

Combines rankings from multiple heterogeneous retrieval systems (e.g. vector search,
lexical search) without requiring raw score calibration or scale normalization.
"""

from typing import Dict, List, Sequence
from uuid import UUID

from app.domain.retrieval.models import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """Combine multiple ranked lists of RetrievalResult using Reciprocal Rank Fusion.

    Formula:
        RRF_score(d) = SUM_{list i} (1 / (k + rank_i(d)))

    Conventions:
        - rank_i(d) is a 1-based integer (1, 2, 3, ...) representing the item's
          position in the given retriever's candidate list.
        - k is a smoothing constant (default: 60) that prevents top-ranked items
          from completely dominating the combined score.
        - If an item appears in multiple lists, its scores are accumulated, rewarding
          documents retrieved by multiple independent search strategies.
        - Deterministic ordering: Sorted by fused RRF score descending; ties are broken
          consistently by chunk_id string representation.

    Args:
        ranked_lists: Sequence of result lists from different retrievers (e.g. vector, lexical).
        k: Smoothing parameter (default 60, must be >= 1).

    Returns:
        Fused list of RetrievalResult objects with recalculated scores and sequential ranks (1..N).
    """
    if k < 1:
        raise ValueError("RRF smoothing constant k must be >= 1.")

    fused_scores: Dict[UUID, float] = {}
    doc_lookup: Dict[UUID, RetrievalResult] = {}

    for result_list in ranked_lists:
        for item in result_list:
            chunk_id = item.chunk_id

            # Calculate reciprocal rank contribution: 1 / (k + rank)
            # Ensure rank is at least 1
            rank = item.rank if item.rank >= 1 else 1
            reciprocal_score = 1.0 / (k + rank)

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = reciprocal_score
                doc_lookup[chunk_id] = item
            else:
                fused_scores[chunk_id] += reciprocal_score
                # Optionally enrich source metadata if current item has richer metadata
                if not doc_lookup[chunk_id].source and item.source:
                    doc_lookup[chunk_id] = item

    if not fused_scores:
        return []

    # Deterministic sorting: highest score first, tie-break by chunk_id
    sorted_items = sorted(
        fused_scores.items(),
        key=lambda entry: (entry[1], str(entry[0])),
        reverse=True,
    )

    fused_results: List[RetrievalResult] = []
    for rank_idx, (chunk_id, score) in enumerate(sorted_items, start=1):
        original = doc_lookup[chunk_id]
        fused_results.append(
            RetrievalResult(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                document_version_id=original.document_version_id,
                content=original.content,
                score=round(score, 6),
                rank=rank_idx,
                source=dict(original.source),
            )
        )

    return fused_results
