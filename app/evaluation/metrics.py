"""Standard Information Retrieval (IR) and Generation evaluation metrics.

Implements mathematically rigorous IR metrics per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 23:
- Precision@k
- Recall@k
- MRR (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Faithfulness
- Citation Completeness
"""

import math
import re
from typing import Collection, List, Sequence
from uuid import UUID


def precision_at_k(
    retrieved_ids: Sequence[UUID | str],
    relevant_ids: Collection[UUID | str],
    k: int,
) -> float:
    """Compute Precision at rank k: fraction of top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    cutoff = list(retrieved_ids)[:k]
    if not cutoff:
        return 0.0
    rel_set = {str(i) for i in relevant_ids}
    hits = sum(1 for item in cutoff if str(item) in rel_set)
    return hits / k


def recall_at_k(
    retrieved_ids: Sequence[UUID | str],
    relevant_ids: Collection[UUID | str],
    k: int,
) -> float:
    """Compute Recall at rank k: fraction of relevant items captured in top-k."""
    rel_set = {str(i) for i in relevant_ids}
    if not rel_set:
        return 1.0
    cutoff = list(retrieved_ids)[:k]
    hits = sum(1 for item in cutoff if str(item) in rel_set)
    return hits / len(rel_set)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[UUID | str],
    relevant_ids: Collection[UUID | str],
) -> float:
    """Compute Reciprocal Rank: 1 / rank of the first relevant retrieved item (0.0 if none found)."""
    rel_set = {str(i) for i in relevant_ids}
    if not rel_set:
        return 0.0
    for rank_idx, item in enumerate(retrieved_ids, start=1):
        if str(item) in rel_set:
            return 1.0 / rank_idx
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[UUID | str],
    relevant_ids: Collection[UUID | str],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k (binary relevance)."""
    if k <= 0:
        return 0.0
    rel_set = {str(i) for i in relevant_ids}
    if not rel_set:
        return 1.0

    cutoff = list(retrieved_ids)[:k]
    # DCG calculation
    dcg = 0.0
    for idx, item in enumerate(cutoff, start=1):
        gain = 1.0 if str(item) in rel_set else 0.0
        dcg += gain / math.log2(idx + 1)

    # Ideal DCG calculation
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def faithfulness_score(response_text: str, evidence_texts: Sequence[str]) -> float:
    """Compute heuristic grounding/faithfulness score based on keyword overlap with evidence."""
    clean_resp = response_text.strip()
    if not clean_resp:
        return 0.0
    if "insufficient evidence" in clean_resp.lower():
        return 1.0

    stopwords = {
        "the", "and", "or", "to", "in", "of", "for", "with", "a", "an", "is", "are",
        "was", "were", "be", "by", "on", "at", "as", "that", "this", "it", "from",
        "which", "shall", "may", "will", "can", "has", "have", "had", "not", "any",
    }
    resp_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", clean_resp.lower())) - stopwords
    if not resp_tokens:
        return 1.0

    combined_evidence = " ".join(evidence_texts).lower()
    evidence_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", combined_evidence)) - stopwords

    overlap = resp_tokens.intersection(evidence_tokens)
    return len(overlap) / len(resp_tokens)


def citation_completeness(
    cited_chunk_ids: Sequence[UUID | str],
    expected_chunk_ids: Collection[UUID | str],
) -> float:
    """Fraction of expected authoritative chunks that were cited in the response."""
    exp_set = {str(i) for i in expected_chunk_ids}
    if not exp_set:
        return 1.0
    cited_set = {str(i) for i in cited_chunk_ids}
    matched = exp_set.intersection(cited_set)
    return len(matched) / len(exp_set)
