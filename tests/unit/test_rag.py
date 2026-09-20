"""Unit tests for Stage 15 — Basic RAG grounded QA pipeline."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.domain.rag.models import Citation, EvidenceItem, RAGQuery, RAGResponse
from app.domain.retrieval.exceptions import InvalidQueryError
from app.domain.retrieval.models import HybridSearchQuery, RetrievalResult
from app.infrastructure.llm.fake import FakeLLMProvider
from app.infrastructure.reranking.fake import FakeReranker
from app.llm.service import LLMService
from app.main import app
from app.rag.context_builder import ContextBuilder
from app.rag.service import RAGService


def _create_retrieval_result(
    content: str,
    score: float = 0.8,
    page: int | None = 1,
    section: str | None = "Section 2.1",
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        document_version_id=uuid4(),
        content=content,
        score=score,
        rank=1,
        source={
            "type": "hybrid",
            "page": page,
            "section": section,
            "metadata": {"page": page, "section": section},
        },
    )


class TestContextBuilder:
    """Test ContextBuilder logic for metadata preservation, token bounding, and citation resolution."""

    def test_build_evidence_items_preserves_metadata(self) -> None:
        builder = ContextBuilder()
        c1 = _create_retrieval_result("Clause A content", page=3, section="Termination")
        c2 = _create_retrieval_result("Clause B content", page=5, section="Liability")

        items = builder.build_evidence_items([c1, c2])

        assert len(items) == 2
        assert items[0].evidence_id == 1
        assert items[0].chunk_id == c1.chunk_id
        assert items[0].document_id == c1.document_id
        assert items[0].page == 3
        assert items[0].section == "Termination"

        assert items[1].evidence_id == 2
        assert items[1].chunk_id == c2.chunk_id
        assert items[1].page == 5
        assert items[1].section == "Liability"

    def test_build_evidence_items_bounds_tokens(self) -> None:
        builder = ContextBuilder()
        # Each item is ~100 characters -> ~25 tokens
        c1 = _create_retrieval_result("Item 1 " * 20)
        c2 = _create_retrieval_result("Item 2 " * 20)
        c3 = _create_retrieval_result("Item 3 " * 20)

        # Budget allows at most 1 item
        items = builder.build_evidence_items([c1, c2, c3], max_context_tokens=30)
        assert len(items) == 1
        assert items[0].chunk_id == c1.chunk_id

    def test_build_evidence_items_empty_candidates(self) -> None:
        builder = ContextBuilder()
        assert builder.build_evidence_items([]) == []

    def test_format_context_structure(self) -> None:
        builder = ContextBuilder()
        c = _create_retrieval_result("This agreement shall be governed by German law.", page=2, section="Choice of Law")
        items = builder.build_evidence_items([c])
        context_str = builder.format_context(items)

        assert "[Evidence 1]" in context_str
        assert str(c.document_id) in context_str
        assert str(c.chunk_id) in context_str
        assert "Page: 2" in context_str
        assert "Section: Choice of Law" in context_str
        assert "German law" in context_str

    def test_format_context_empty(self) -> None:
        builder = ContextBuilder()
        assert "No relevant evidence" in builder.format_context([])

    def test_resolve_citations_by_evidence_id(self) -> None:
        builder = ContextBuilder()
        c = _create_retrieval_result("Indemnity clause", page=7, section="Indemnification")
        items = builder.build_evidence_items([c])

        raw_citations = [{"evidence_id": 1, "snippet": "Indemnity clause"}]
        citations = builder.resolve_citations(raw_citations, items)

        assert len(citations) == 1
        assert citations[0].chunk_id == c.chunk_id
        assert citations[0].document_id == c.document_id
        assert citations[0].document_version_id == c.document_version_id
        assert citations[0].page == 7
        assert citations[0].section == "Indemnification"
        assert citations[0].snippet == "Indemnity clause"

    def test_resolve_citations_by_chunk_id(self) -> None:
        builder = ContextBuilder()
        c = _create_retrieval_result("Confidentiality text", page=1, section="NDA")
        items = builder.build_evidence_items([c])

        raw_citations = [{"chunk_id": str(c.chunk_id), "snippet": "Confidentiality text"}]
        citations = builder.resolve_citations(raw_citations, items)

        assert len(citations) == 1
        assert citations[0].chunk_id == c.chunk_id

    def test_resolve_citations_rejects_hallucinated_ids(self) -> None:
        builder = ContextBuilder()
        c = _create_retrieval_result("Valid text")
        items = builder.build_evidence_items([c])

        hallucinated_citations = [
            {"evidence_id": 99, "snippet": "Non-existent evidence 99"},
            {"chunk_id": str(uuid4()), "snippet": "Fabricated chunk UUID"},
        ]
        citations = builder.resolve_citations(hallucinated_citations, items)
        # Both must be rejected since they do not match retrieved evidence
        assert citations == []

    def test_resolve_citations_deduplicates(self) -> None:
        builder = ContextBuilder()
        c = _create_retrieval_result("Single clause")
        items = builder.build_evidence_items([c])

        dup_citations = [
            {"evidence_id": 1, "snippet": "Snippet 1"},
            {"evidence_id": 1, "snippet": "Snippet 2"},
        ]
        citations = builder.resolve_citations(dup_citations, items)
        assert len(citations) == 1


class TestRAGService:
    """Test full RAG pipeline with mock/fake components."""

    @pytest.mark.asyncio
    async def test_end_to_end_grounded_answer(self) -> None:
        cand1 = _create_retrieval_result("The notice period for termination is 30 days.", page=4, section="Termination")
        cand2 = _create_retrieval_result("Governing jurisdiction is Berlin, Germany.", page=8, section="Jurisdiction")

        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand1, cand2]

        fake_reranker = FakeReranker()

        llm_answer = json.dumps({
            "answer": "Termination requires a 30-day notice period under Berlin jurisdiction.",
            "is_sufficient": True,
            "citations": [
                {"evidence_id": 1, "snippet": "notice period for termination is 30 days"},
                {"evidence_id": 2, "snippet": "jurisdiction is Berlin"},
            ],
        })
        fake_llm = FakeLLMProvider(responses=[llm_answer])
        llm_service = LLMService(provider=fake_llm)

        rag_service = RAGService(
            hybrid_retriever=mock_retriever,
            reranker=fake_reranker,
            llm_service=llm_service,
        )

        query = RAGQuery(question="What is the termination notice period?", top_k=2)
        response = await rag_service.answer(query)

        assert response.is_sufficient is True
        assert "30-day notice" in response.answer
        assert len(response.evidence) == 2
        assert len(response.citations) == 2

        # Verify citation identity matches actual chunks
        cit_chunk_ids = {c.chunk_id for c in response.citations}
        assert cand1.chunk_id in cit_chunk_ids
        assert cand2.chunk_id in cit_chunk_ids

    @pytest.mark.asyncio
    async def test_insufficient_evidence_when_zero_chunks_retrieved(self) -> None:
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = []

        fake_llm = FakeLLMProvider()
        llm_service = LLMService(provider=fake_llm)

        rag_service = RAGService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=llm_service,
        )

        query = RAGQuery(question="What are the patent claim terms?")
        response = await rag_service.answer(query)

        assert response.is_sufficient is False
        assert "insufficient evidence" in response.answer.lower()
        assert response.citations == []
        assert response.evidence == []
        # No LLM call should be wasted
        assert len(fake_llm.call_history) == 0

    @pytest.mark.asyncio
    async def test_insufficient_evidence_reported_by_llm(self) -> None:
        cand = _create_retrieval_result("General introduction to the company.")
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand]

        llm_answer = json.dumps({
            "answer": "The provided documents do not mention liability caps.",
            "is_sufficient": False,
            "citations": [],
        })
        fake_llm = FakeLLMProvider(responses=[llm_answer])
        llm_service = LLMService(provider=fake_llm)

        rag_service = RAGService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=llm_service,
        )

        query = RAGQuery(question="What is the liability cap?")
        response = await rag_service.answer(query)

        assert response.is_sufficient is False
        assert response.citations == []

    @pytest.mark.asyncio
    async def test_empty_question_raises_error(self) -> None:
        rag_service = RAGService(
            hybrid_retriever=AsyncMock(),
            reranker=FakeReranker(),
        )
        with pytest.raises(InvalidQueryError, match="not be empty"):
            await rag_service.answer(RAGQuery(question="   "))

    def test_prompt_files_exist_and_contain_grounding_rules(self) -> None:
        prompts_dir = Path(__file__).resolve().parent.parent.parent / "prompts"
        sys_prompt_file = prompts_dir / "rag_system.txt"
        user_prompt_file = prompts_dir / "rag_user.txt"

        assert sys_prompt_file.is_file(), "rag_system.txt must exist in prompts/"
        assert user_prompt_file.is_file(), "rag_user.txt must exist in prompts/"

        sys_content = sys_prompt_file.read_text()
        assert "evidence" in sys_content.lower()
        assert "insufficient" in sys_content.lower()
        assert "citations" in sys_content.lower()


class TestRAGApi:
    """Test the FastAPI RAG query endpoint."""

    def test_api_rag_query_endpoint(self) -> None:
        from app.api.routes.rag import get_rag_service

        cand = _create_retrieval_result("Arbitration takes place in Munich.", page=10, section="Arbitration")

        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand]

        llm_answer = json.dumps({
            "answer": "Arbitration venue is Munich.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 1, "snippet": "Arbitration takes place in Munich"}],
        })
        fake_llm = FakeLLMProvider(responses=[llm_answer])
        llm_service = LLMService(provider=fake_llm)

        fake_service = RAGService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=llm_service,
        )

        app.dependency_overrides[get_rag_service] = lambda: fake_service

        client = TestClient(app)
        try:
            res = client.post("/api/v1/rag/query", json={"question": "Where is arbitration held?"})
            assert res.status_code == 200
            data = res.json()
            assert data["is_sufficient"] is True
            assert "Munich" in data["answer"]
            assert len(data["citations"]) == 1
            assert str(cand.chunk_id) == data["citations"][0]["chunk_id"]
        finally:
            app.dependency_overrides.clear()
