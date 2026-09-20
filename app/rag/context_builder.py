"""Context Builder responsible for ordering evidence, preserving metadata, and bounding prompts."""

from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from app.domain.rag.exceptions import ContextBuildError
from app.domain.rag.models import Citation, EvidenceItem
from app.domain.retrieval.models import RetrievalResult


def _approx_token_count(text: str) -> int:
    """Heuristic approximation of token count (~4 characters per token)."""
    return max(1, len(text) // 4)


class ContextBuilder:
    """Constructs structured evidence contexts and verifies citation provenance."""

    def __init__(self, default_max_tokens: int = 3500) -> None:
        self.default_max_tokens = default_max_tokens

    def build_evidence_items(
        self,
        candidates: Sequence[RetrievalResult],
        max_context_tokens: Optional[int] = None,
    ) -> List[EvidenceItem]:
        """Convert retrieval results into ordered, token-bounded EvidenceItems.

        Args:
            candidates: Sequence of RetrievalResult ordered by relevance.
            max_context_tokens: Optional maximum token ceiling for evidence context.

        Returns:
            List of EvidenceItem objects with 1-based sequential evidence_id.
        """
        if not candidates:
            return []

        token_budget = max_context_tokens or self.default_max_tokens
        accumulated_tokens = 0
        evidence_items: List[EvidenceItem] = []

        for idx, cand in enumerate(candidates, start=1):
            cand_tokens = _approx_token_count(cand.content)
            if accumulated_tokens + cand_tokens > token_budget and evidence_items:
                # Stop adding once budget is reached (keep at least 1 evidence item)
                break

            accumulated_tokens += cand_tokens

            # Extract provenance metadata safely from source dictionary
            src = cand.source or {}
            page = src.get("page")
            if page is None and isinstance(src.get("metadata"), dict):
                page = src["metadata"].get("page")

            section = src.get("section")
            if section is None and isinstance(src.get("metadata"), dict):
                section = src["metadata"].get("section")

            evidence_items.append(
                EvidenceItem(
                    evidence_id=idx,
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    document_version_id=cand.document_version_id,
                    content=cand.content,
                    score=cand.score,
                    page=int(page) if page is not None and str(page).isdigit() else None,
                    section=str(section) if section else None,
                    source=dict(src),
                )
            )

        return evidence_items

    def format_context(self, evidence_items: Sequence[EvidenceItem]) -> str:
        """Format evidence items into bounded, structured context text.

        Args:
            evidence_items: Sequence of EvidenceItem objects.

        Returns:
            Formatted evidence block string.
        """
        if not evidence_items:
            return "No relevant evidence documents retrieved."

        blocks: List[str] = []
        for item in evidence_items:
            page_str = str(item.page) if item.page is not None else "N/A"
            section_str = item.section if item.section else "N/A"

            block = (
                f"[Evidence {item.evidence_id}]\n"
                f"Document ID: {item.document_id}\n"
                f"Version ID: {item.document_version_id}\n"
                f"Chunk ID: {item.chunk_id}\n"
                f"Page: {page_str}\n"
                f"Section: {section_str}\n"
                f"Relevance Score: {item.score:.4f}\n"
                f"Content:\n{item.content.strip()}"
            )
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    def resolve_citations(
        self,
        raw_citations: Sequence[Dict[str, Any]],
        evidence_items: Sequence[EvidenceItem],
    ) -> List[Citation]:
        """Resolve LLM raw citations against genuine evidence items.

        Guarantees that every returned Citation references verified metadata
        from the retrieved candidate evidence pool (no fabricated source metadata).

        Args:
            raw_citations: List of dicts produced by the LLM (e.g. [{"evidence_id": 1, ...}]).
            evidence_items: Verified EvidenceItems provided in context.

        Returns:
            List of verified Citation objects.
        """
        if not raw_citations or not evidence_items:
            return []

        # Map evidence by 1-based ID and by stringified chunk_id for flexible resolution
        evidence_by_id = {ev.evidence_id: ev for ev in evidence_items}
        evidence_by_chunk = {str(ev.chunk_id): ev for ev in evidence_items}

        verified_citations: List[Citation] = []
        seen_chunk_ids = set()

        for cit in raw_citations:
            matched_evidence: Optional[EvidenceItem] = None

            # Try resolution by evidence_id
            ev_id = cit.get("evidence_id")
            if ev_id is not None:
                try:
                    matched_evidence = evidence_by_id.get(int(ev_id))
                except (ValueError, TypeError):
                    matched_evidence = None

            # Fallback resolution by chunk_id
            if matched_evidence is None and "chunk_id" in cit:
                matched_evidence = evidence_by_chunk.get(str(cit["chunk_id"]))

            if matched_evidence is not None and matched_evidence.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(matched_evidence.chunk_id)
                verified_citations.append(
                    Citation(
                        document_id=matched_evidence.document_id,
                        document_version_id=matched_evidence.document_version_id,
                        chunk_id=matched_evidence.chunk_id,
                        page=matched_evidence.page,
                        section=matched_evidence.section,
                        snippet=cit.get("snippet"),
                    )
                )

        return verified_citations
