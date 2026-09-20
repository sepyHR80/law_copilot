"""External search service with source filtering and domain validation.

Per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 21.3:
External search is only invoked as a fallback when internal evidence is insufficient.
External search must never override internal authoritative legal rules.
"""

from typing import List, Optional
from urllib.parse import urlparse

from app.domain.search.exceptions import ExternalSearchError
from app.domain.search.models import ValidatedSource, WebSearchResult
from app.infrastructure.search.provider import WebSearchProviderProtocol

TRUSTED_GOV_TLDS = {".gov", ".gov.ir", ".gov.uk", ".gouv.fr", ".gov.au"}
TRUSTED_EDU_TLDS = {".edu", ".ac.uk", ".ac.ir"}
BLOCKED_DOMAINS = {
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "blogspot.com",
    "wordpress.com",
    "pastebin.com",
    "quora.com",
}
TRUSTED_LEGAL_DOMAINS = {
    "law.cornell.edu",
    "supremecourt.gov",
    "judiciary.gov.uk",
    "eur-lex.europa.eu",
    "un.org",
    "icj-cij.org",
    "rooznamehrasmi.ir",
    "qavanin.ir",
    "dastour.ir",
}


class ExternalSearchService:
    """Service for querying external web search, filtering domains, and scoring source authority."""

    def __init__(self, provider: WebSearchProviderProtocol) -> None:
        self.provider = provider

    def _extract_domain(self, url: str) -> str:
        """Extract normalized domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    def evaluate_source(self, raw_result: WebSearchResult) -> ValidatedSource:
        """Evaluate source domain authority and filter out untrusted sources."""
        domain = raw_result.domain or self._extract_domain(raw_result.url)
        content = raw_result.snippet.strip()

        # Check blocked domains
        for blocked in BLOCKED_DOMAINS:
            if domain == blocked or domain.endswith(f".{blocked}"):
                return ValidatedSource(
                    title=raw_result.title,
                    url=raw_result.url,
                    domain=domain,
                    content=content,
                    is_trusted=False,
                    trust_score=0.1,
                    authority_category="untrusted",
                )

        # Check trusted legal domains
        if domain in TRUSTED_LEGAL_DOMAINS:
            return ValidatedSource(
                title=raw_result.title,
                url=raw_result.url,
                domain=domain,
                content=content,
                is_trusted=True,
                trust_score=0.95,
                authority_category="judiciary" if "court" in domain or "judiciary" in domain else "government",
            )

        # Check Government TLDs
        if any(domain.endswith(tld) for tld in TRUSTED_GOV_TLDS):
            return ValidatedSource(
                title=raw_result.title,
                url=raw_result.url,
                domain=domain,
                content=content,
                is_trusted=True,
                trust_score=0.9,
                authority_category="government",
            )

        # Check Academic TLDs
        if any(domain.endswith(tld) for tld in TRUSTED_EDU_TLDS):
            return ValidatedSource(
                title=raw_result.title,
                url=raw_result.url,
                domain=domain,
                content=content,
                is_trusted=True,
                trust_score=0.8,
                authority_category="academic",
            )

        # General web source: default moderate trust if content exists
        return ValidatedSource(
            title=raw_result.title,
            url=raw_result.url,
            domain=domain,
            content=content,
            is_trusted=True,
            trust_score=0.5,
            authority_category="general",
        )

    async def search_and_validate(
        self,
        query: str,
        num_results: int = 5,
        min_trust_score: float = 0.4,
    ) -> List[ValidatedSource]:
        """Execute external web search, filter untrusted domains, and sort by trust score."""
        try:
            raw_results = await self.provider.search(query=query, num_results=num_results)
        except Exception as exc:
            raise ExternalSearchError(f"External web search failed: {exc}") from exc

        validated: List[ValidatedSource] = []
        for r in raw_results:
            source = self.evaluate_source(r)
            if source.is_trusted and source.trust_score >= min_trust_score:
                validated.append(source)

        # Sort by trust_score descending
        validated.sort(key=lambda s: s.trust_score, reverse=True)
        return validated

    def format_external_context(self, sources: List[ValidatedSource]) -> str:
        """Format validated external sources into a context block with non-authoritative disclaimers."""
        if not sources:
            return "No validated external web sources found."

        blocks: List[str] = []
        for idx, src in enumerate(sources, start=1):
            block = (
                f"[EXTERNAL WEB SOURCE #{idx} - SECONDARY/NON-AUTHORITATIVE]\n"
                f"Title: {src.title}\n"
                f"Domain: {src.domain} (Category: {src.authority_category}, Trust Score: {src.trust_score:.2f})\n"
                f"URL: {src.url}\n"
                f"Content:\n{src.content}"
            )
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)
