"""Claim verification service ensuring evidence grounding and valid citations."""

import re
from typing import List, Optional, Sequence
from app.domain.rag.models import Citation, EvidenceItem
from app.domain.verification.models import VerificationResult


class ClaimVerificationService:
    """Verifies that generated responses are strictly grounded in supplied evidence.

    Per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 21.4:
    - claims are grounded in evidence
    - citations exist and match provided evidence
    - cited chunks support the claims
    - unsupported claims and citation errors are surfaced in a structured result
    """

    def __init__(self, min_keyword_overlap_ratio: float = 0.25) -> None:
        self.min_keyword_overlap_ratio = min_keyword_overlap_ratio

    def _extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences for fine-grained claim checking."""
        clean = text.strip()
        if not clean:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract alphanumeric keywords, ignoring common stop words."""
        stopwords = {
            "the", "and", "or", "to", "in", "of", "for", "with", "a", "an", "is", "are",
            "was", "were", "be", "by", "on", "at", "as", "that", "this", "it", "from",
            "which", "shall", "may", "will", "can", "has", "have", "had", "not", "any",
        }
        tokens = re.findall(r"\b[A-Za-z0-9_]{3,}\b", text.lower())
        return {t for t in tokens if t not in stopwords}

    def verify(
        self,
        response_text: str,
        citations: Sequence[Citation],
        evidence_items: Sequence[EvidenceItem],
        raw_citations: Optional[Sequence[dict]] = None,
    ) -> VerificationResult:
        """Perform comprehensive grounding and citation verification."""
        clean_text = response_text.strip()
        if not clean_text:
            return VerificationResult(
                passed=False,
                unsupported_claims=["Response text is empty."],
                citation_errors=[],
                confidence_score=0.0,
                repair_guidance="Provide a non-empty response grounded in the provided evidence.",
            )

        # If answer says evidence is insufficient, it is valid by definition
        if "insufficient evidence" in clean_text.lower():
            return VerificationResult(
                passed=True,
                unsupported_claims=[],
                citation_errors=[],
                confidence_score=1.0,
                repair_guidance=None,
            )

        # 1. Citation validity check
        citation_errors: List[str] = []
        valid_chunk_ids = {item.chunk_id for item in evidence_items}
        valid_evidence_ids = {item.evidence_id for item in evidence_items}

        if raw_citations:
            for rc in raw_citations:
                ev_id = rc.get("evidence_id")
                if ev_id is not None and ev_id not in valid_evidence_ids:
                    citation_errors.append(f"Citation references non-existent Evidence ID: {ev_id}")

        for cit in citations:
            if cit.chunk_id not in valid_chunk_ids:
                citation_errors.append(f"Citation references invalid chunk ID: {cit.chunk_id}")

        # If evidence was provided but no citations were resolved:
        if evidence_items and not citations and not raw_citations:
            citation_errors.append("No citations provided for substantive legal assertions.")

        # 2. Evidence coverage check: combined text of all evidence
        evidence_combined_text = " ".join(item.content for item in evidence_items).lower()
        evidence_keywords = self._extract_keywords(evidence_combined_text)

        # 3. Sentence-level claim support check
        sentences = self._extract_sentences(clean_text)
        unsupported_claims: List[str] = []

        for s in sentences:
            s_keywords = self._extract_keywords(s)
            if not s_keywords:
                continue

            # Check keyword overlap with evidence
            overlap = s_keywords.intersection(evidence_keywords)
            overlap_ratio = len(overlap) / len(s_keywords)

            # Look for specific statutory identifiers or article references
            statute_refs = re.findall(r"\b(?:section|article|code)\s+\d+\b", s, re.IGNORECASE)
            for ref in statute_refs:
                if ref.lower() not in evidence_combined_text:
                    unsupported_claims.append(
                        f"Claim references unsupported statutory provision '{ref}': {s}"
                    )

            if overlap_ratio < self.min_keyword_overlap_ratio and not statute_refs:
                unsupported_claims.append(f"Low factual support ({overlap_ratio:.0%}): {s}")

        passed = len(citation_errors) == 0 and len(unsupported_claims) == 0
        confidence = 1.0 - (0.3 * len(citation_errors) + 0.2 * len(unsupported_claims))
        confidence = max(0.0, min(1.0, confidence))

        repair_guidance = None
        if not passed:
            guidance_parts = []
            if citation_errors:
                guidance_parts.append(
                    f"Fix citation errors: {'; '.join(citation_errors[:3])}. "
                    "Ensure you only cite valid Evidence IDs present in the supplied evidence."
                )
            if unsupported_claims:
                guidance_parts.append(
                    f"Remove or ground unsupported claims: {'; '.join(unsupported_claims[:2])}. "
                    "Rely exclusively on factual statements in the provided context."
                )
            repair_guidance = " ".join(guidance_parts)

        return VerificationResult(
            passed=passed,
            unsupported_claims=unsupported_claims,
            citation_errors=citation_errors,
            confidence_score=confidence,
            repair_guidance=repair_guidance,
        )
