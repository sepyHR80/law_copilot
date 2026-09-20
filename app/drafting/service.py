"""Document drafting service with strict separation of factual and stylistic knowledge."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field

from app.domain.drafting.exceptions import DraftingError, StyleLeakageError
from app.domain.drafting.models import (
    DraftDocument,
    DraftingQuery,
    DraftSection,
)
from app.domain.rag.models import Citation, EvidenceItem
from app.domain.retrieval.exceptions import InvalidQueryError
from app.domain.retrieval.models import HybridSearchQuery, RetrievalFilter, RetrievalResult
from app.domain.retrieval.protocol import HybridRetrieverProtocol, RerankerProtocol
from app.infrastructure.reranking.cross_encoder import CrossEncoderReranker
from app.llm.service import LLMService
from app.rag.context_builder import ContextBuilder

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

DEFAULT_DRAFTING_SYSTEM_PROMPT = """You are a professional legal drafting engine for Law Copilot.
Your task is to draft high-quality, structured legal documents based on the provided factual evidence and stylistic references.

STRICT GROUNDING & KNOWLEDGE SEPARATION RULES:
1. FACTUAL EVIDENCE ([FACTUAL EVIDENCE #ID]):
   - Authoritative legal rules, contractual terms, statutes, or case facts.
   - Strictly ground all substantive legal claims in Factual Evidence.
   - Cite corresponding Factual Evidence ID in citations.

2. STYLE REFERENCES ([STYLE REFERENCE #ID]):
   - For structure, tone, formal layout, paragraph flow, and terminology ONLY.
   - NEVER cite a Style Reference as legal authority or evidence ID!
   - Citations are reserved EXCLUSIVELY for Factual Evidence.
   - Do NOT import names, party details, or specific historical facts from style references.

3. NO FABRICATION:
   - Do not fabricate legal authority or factual allegations not supported by the factual evidence.

Provide your output strictly in valid JSON matching this schema:
{
  "title": "Document Title",
  "document_type": "contract|notice|complaint|letter|other",
  "sections": [
    {
      "heading": "Section Heading",
      "content": "Section text...",
      "citations": [
        {
          "evidence_id": 1,
          "snippet": "Quoted text from factual evidence"
        }
      ]
    }
  ]
}"""

DEFAULT_DRAFTING_USER_TEMPLATE = """DRAFTING REQUEST:
Topic: {topic}
Document Type: {document_type}
Recipient / Parties: {recipient}
Jurisdiction: {jurisdiction}
Additional Instructions: {additional_instructions}

AUTHORITATIVE FACTUAL EVIDENCE:
{factual_evidence_block}

STYLISTIC REFERENCES (DO NOT CITE AS AUTHORITY):
{style_references_block}

Please generate the complete legal draft in valid JSON following the required schema."""


class _DraftSectionLLMSchema(BaseModel):
    heading: str
    content: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)


class _DraftDocumentLLMSchema(BaseModel):
    title: str
    document_type: str = "legal_document"
    sections: List[_DraftSectionLLMSchema] = Field(default_factory=list)


class DocumentDraftingService:
    """Coordinates legal document drafting using dual-channel retrieval:

    - Channel 1 (Factual): Legal rules, contracts, statutes (binding authority).
    - Channel 2 (Stylistic): Past templates, tone, layout (non-authoritative).
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetrieverProtocol,
        reranker: Optional[RerankerProtocol] = None,
        llm_service: Optional[LLMService] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompts_dir: Optional[Path] = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_service = llm_service or LLMService()
        self.context_builder = context_builder or ContextBuilder()
        self.prompts_dir = prompts_dir or PROMPTS_DIR

    def _load_prompt(self, filename: str, default: str) -> str:
        """Load prompt template from disk with fallback."""
        file_path = self.prompts_dir / filename
        if file_path.is_file():
            try:
                return file_path.read_text(encoding="utf-8").strip()
            except Exception:
                return default
        return default

    def _format_style_context(self, style_items: Sequence[EvidenceItem]) -> str:
        """Format style reference items into a dedicated non-authoritative block."""
        if not style_items:
            return "No specific stylistic templates provided. Follow standard professional legal drafting standards."

        blocks: List[str] = []
        for item in style_items:
            block = (
                f"[Style Reference {item.evidence_id}]\n"
                f"Document ID: {item.document_id}\n"
                f"Chunk ID: {item.chunk_id}\n"
                f"Tone/Format Guidance:\n{item.content.strip()}"
            )
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    async def draft(self, query: DraftingQuery) -> DraftDocument:
        """Execute dual-channel retrieval and grounded document generation."""
        cleaned_topic = query.topic.strip()
        if not cleaned_topic:
            raise InvalidQueryError("Drafting topic must not be empty.")

        # -------------------------------------------------------------
        # 1. Channel 1: Factual Retrieval (knowledge_type="factual")
        # -------------------------------------------------------------
        factual_filter = RetrievalFilter(knowledge_type="factual")
        factual_search_query = HybridSearchQuery(
            text_query=cleaned_topic,
            top_k=query.factual_top_k,
            candidate_k=max(20, query.factual_top_k * 3),
            filters=factual_filter,
        )
        factual_candidates = await self.hybrid_retriever.search(factual_search_query)

        if factual_candidates:
            reranked_factual = self.reranker.rerank(
                query=cleaned_topic,
                candidates=factual_candidates,
                top_n=query.factual_top_k,
            )
        else:
            reranked_factual = []

        factual_evidence = self.context_builder.build_evidence_items(reranked_factual)
        factual_context = self.context_builder.format_context(factual_evidence)

        # -------------------------------------------------------------
        # 2. Channel 2: Stylistic Retrieval (knowledge_type="stylistic")
        # -------------------------------------------------------------
        style_evidence: List[EvidenceItem] = []
        if query.style_top_k > 0:
            style_filter = RetrievalFilter(knowledge_type="stylistic")
            style_query_text = f"{query.document_type or ''} {cleaned_topic}".strip()
            style_search_query = HybridSearchQuery(
                text_query=style_query_text,
                top_k=query.style_top_k,
                candidate_k=max(10, query.style_top_k * 3),
                filters=style_filter,
            )
            style_candidates = await self.hybrid_retriever.search(style_search_query)

            if style_candidates:
                reranked_style = self.reranker.rerank(
                    query=style_query_text,
                    candidates=style_candidates,
                    top_n=query.style_top_k,
                )
            else:
                reranked_style = []

            style_evidence = self.context_builder.build_evidence_items(reranked_style)

        style_context = self._format_style_context(style_evidence)

        # -------------------------------------------------------------
        # 3. Prompt Assembly
        # -------------------------------------------------------------
        system_prompt = self._load_prompt("drafting_system.txt", DEFAULT_DRAFTING_SYSTEM_PROMPT)
        user_template = self._load_prompt("drafting_user.txt", DEFAULT_DRAFTING_USER_TEMPLATE)

        user_prompt = user_template.format(
            topic=cleaned_topic,
            document_type=query.document_type or "legal document",
            recipient=query.recipient or "Not specified",
            jurisdiction=query.jurisdiction or "Not specified",
            additional_instructions=query.additional_instructions or "None",
            factual_evidence_block=factual_context,
            style_references_block=style_context,
        )

        # -------------------------------------------------------------
        # 4. LLM Generation
        # -------------------------------------------------------------
        model_name = "unknown"
        llm_usage = None

        try:
            parsed, llm_resp = await self.llm_service.structured_complete(
                prompt=user_prompt,
                schema=_DraftDocumentLLMSchema,
                system_prompt=system_prompt,
            )
            raw_title = parsed.title
            raw_doc_type = parsed.document_type
            raw_sections = parsed.sections
            model_name = llm_resp.model
            llm_usage = llm_resp.usage
        except Exception:
            # Fallback to general completion and JSON parsing
            llm_resp = await self.llm_service.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            model_name = llm_resp.model
            llm_usage = llm_resp.usage
            content = llm_resp.content.strip()

            try:
                data = json.loads(content)
                raw_title = data.get("title", f"Draft {query.document_type or 'Document'}")
                raw_doc_type = data.get("document_type", query.document_type or "legal_document")
                raw_sections = [
                    _DraftSectionLLMSchema(
                        heading=s.get("heading", "Section"),
                        content=s.get("content", ""),
                        citations=s.get("citations", []),
                    )
                    for s in data.get("sections", [])
                ]
            except Exception:
                raw_title = f"Draft {query.document_type or 'Document'}"
                raw_doc_type = query.document_type or "legal_document"
                raw_sections = [
                    _DraftSectionLLMSchema(
                        heading="Main Content",
                        content=content,
                        citations=[],
                    )
                ]

        # -------------------------------------------------------------
        # 5. Strict Verification: Resolve Citations against Factual KB Only
        #    Style documents must NEVER be cited as legal authority.
        # -------------------------------------------------------------
        style_chunk_ids = {str(item.chunk_id) for item in style_evidence}

        processed_sections: List[DraftSection] = []
        all_document_citations: List[Citation] = []
        seen_citations_keys = set()

        for s in raw_sections:
            # Check for illegal style citations in raw citations
            for cit in s.citations:
                chunk_id = str(cit.get("chunk_id", ""))
                if chunk_id and chunk_id in style_chunk_ids:
                    raise StyleLeakageError(
                        f"Style reference chunk {chunk_id} was cited as legal authority. "
                        "Style references cannot be cited as binding evidence."
                    )

            # Resolve citations strictly against factual evidence
            verified_citations = self.context_builder.resolve_citations(
                raw_citations=s.citations,
                evidence_items=factual_evidence,
            )

            processed_sections.append(
                DraftSection(
                    heading=s.heading,
                    content=s.content,
                    citations=verified_citations,
                )
            )

            for vc in verified_citations:
                key = (vc.document_id, vc.chunk_id)
                if key not in seen_citations_keys:
                    seen_citations_keys.add(key)
                    all_document_citations.append(vc)

        # -------------------------------------------------------------
        # 6. Compose Full Document Text
        # -------------------------------------------------------------
        full_text_lines = [f"# {raw_title}\n"]
        for section in processed_sections:
            full_text_lines.append(f"## {section.heading}\n{section.content}\n")
        full_text = "\n".join(full_text_lines).strip()

        return DraftDocument(
            title=raw_title,
            document_type=raw_doc_type,
            sections=processed_sections,
            full_text=full_text,
            citations=all_document_citations,
            factual_evidence=factual_evidence,
            style_references=style_evidence,
            model=model_name,
            usage=llm_usage,
        )
