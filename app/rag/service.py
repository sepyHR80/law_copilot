"""End-to-end grounded RAG Question Answering service."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.rag.models import Citation, EvidenceItem, RAGQuery, RAGResponse
from app.domain.retrieval.exceptions import InvalidQueryError
from app.domain.retrieval.models import HybridSearchQuery
from app.domain.retrieval.protocol import HybridRetrieverProtocol, RerankerProtocol
from app.infrastructure.reranking.cross_encoder import CrossEncoderReranker
from app.llm.service import LLMService
from app.rag.context_builder import ContextBuilder

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

DEFAULT_SYSTEM_PROMPT = """You are a professional, rigorous legal research assistant.
Answer the legal question strictly and exclusively based on the supplied evidence documents.
If the supplied evidence does not contain enough information to answer the question accurately,
state that there is insufficient evidence and set is_sufficient to false.
Provide your response in JSON format:
{
  "answer": "...",
  "is_sufficient": true,
  "citations": [{"evidence_id": 1, "snippet": "..."}]
}"""

DEFAULT_USER_TEMPLATE = """LEGAL QUESTION:
{question}

SUPPLIED EVIDENCE:
{evidence_block}

Please provide a grounded answer based strictly on the supplied evidence."""


class _LLMRAGSchema(BaseModel):
    """Internal schema for structured RAG completion."""

    answer: str
    is_sufficient: bool = True
    citations: List[Dict[str, Any]] = Field(default_factory=list)


class RAGService:
    """Orchestrates end-to-end grounded legal QA:

    query -> hybrid retrieval -> reranking -> context builder -> LLM -> answer + citations.
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
        """Load prompt template from file, with fallback."""
        file_path = self.prompts_dir / filename
        if file_path.is_file():
            try:
                return file_path.read_text(encoding="utf-8").strip()
            except Exception:
                return default
        return default

    async def answer(self, query: RAGQuery) -> RAGResponse:
        """Execute grounded legal QA pipeline for a question."""
        cleaned_question = query.question.strip()
        if not cleaned_question:
            raise InvalidQueryError("Question must not be empty.")

        # 1. Hybrid retrieval across vector + lexical FTS
        search_query = HybridSearchQuery(
            text_query=cleaned_question,
            top_k=query.top_k,
            candidate_k=query.candidate_k,
            filters=query.filters,
        )
        candidates = await self.hybrid_retriever.search(search_query)

        # 2. Reranking candidates
        if candidates:
            reranked_candidates = self.reranker.rerank(
                query=cleaned_question,
                candidates=candidates,
                top_n=query.top_k,
            )
        else:
            reranked_candidates = []

        # 3. Context building
        evidence_items = self.context_builder.build_evidence_items(
            candidates=reranked_candidates,
            max_context_tokens=query.max_context_tokens,
        )

        # Handle zero evidence early: deterministic insufficient evidence
        if not evidence_items:
            return RAGResponse(
                question=cleaned_question,
                answer=(
                    "Based on the provided documents, there is insufficient evidence to answer this question."
                ),
                citations=[],
                evidence=[],
                is_sufficient=False,
                model="none",
                usage=None,
            )

        evidence_text = self.context_builder.format_context(evidence_items)

        # 4. Prompt loading & formatting
        system_prompt = self._load_prompt("rag_system.txt", DEFAULT_SYSTEM_PROMPT)
        user_template = self._load_prompt("rag_user.txt", DEFAULT_USER_TEMPLATE)
        user_prompt = user_template.format(
            question=cleaned_question,
            evidence_block=evidence_text,
        )

        # 5. LLM generation
        try:
            parsed_result, llm_response = await self.llm_service.structured_complete(
                prompt=user_prompt,
                schema=_LLMRAGSchema,
                system_prompt=system_prompt,
            )
            answer_text = parsed_result.answer
            is_sufficient = parsed_result.is_sufficient
            raw_citations = parsed_result.citations
        except Exception:
            # Fallback to standard completion
            llm_response = await self.llm_service.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            content = llm_response.content.strip()
            try:
                data = json.loads(content)
                answer_text = data.get("answer", content)
                is_sufficient = bool(data.get("is_sufficient", True))
                raw_citations = data.get("citations", [])
            except Exception:
                answer_text = content
                is_sufficient = "insufficient evidence" not in content.lower()
                raw_citations = []

        # 6. Strict citation verification (prevents fabricated metadata)
        citations = self.context_builder.resolve_citations(
            raw_citations=raw_citations,
            evidence_items=evidence_items,
        )

        return RAGResponse(
            question=cleaned_question,
            answer=answer_text,
            citations=citations,
            evidence=evidence_items,
            is_sufficient=is_sufficient,
            model=llm_response.model,
            usage=llm_response.usage,
        )
