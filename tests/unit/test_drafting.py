"""Unit tests for Stage 18 — Style Retrieval + Document Generation."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.agent.nodes import create_analyze_intent_node
from app.domain.drafting.exceptions import DraftingError, StyleLeakageError
from app.domain.drafting.models import (
    DraftDocument,
    DraftingQuery,
    DraftSection,
)
from app.domain.rag.models import Citation, EvidenceItem
from app.domain.retrieval.exceptions import InvalidQueryError
from app.domain.retrieval.models import HybridSearchQuery, RetrievalResult
from app.drafting.service import DocumentDraftingService
from app.infrastructure.llm.fake import FakeLLMProvider
from app.infrastructure.reranking.fake import FakeReranker
from app.llm.service import LLMService
from app.main import app
from app.rag.context_builder import ContextBuilder


def _create_result(
    content: str,
    score: float = 0.85,
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    knowledge_type: str = "factual",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        document_version_id=uuid4(),
        content=content,
        score=score,
        rank=1,
        source={
            "knowledge_type": knowledge_type,
            "section": "General Terms",
            "page": 1,
        },
    )


class TestDraftingDomainModels:
    """Test drafting domain models."""

    def test_draft_section_creation(self) -> None:
        sec = DraftSection(
            heading="Recitals",
            content="Whereas the parties desire to enter into this agreement...",
            citations=[],
        )
        assert sec.heading == "Recitals"
        assert len(sec.citations) == 0

    def test_draft_query_defaults(self) -> None:
        q = DraftingQuery(topic="Non-Disclosure Agreement")
        assert q.topic == "Non-Disclosure Agreement"
        assert q.factual_top_k == 5
        assert q.style_top_k == 3
        assert q.document_type == "agreement"

    def test_draft_query_empty_topic_validation(self) -> None:
        with pytest.raises(Exception):
            DraftingQuery(topic="")


class TestDocumentDraftingService:
    """Test DocumentDraftingService dual-channel retrieval and style-fact separation."""

    @pytest.mark.asyncio
    async def test_dual_channel_retrieval_queries(self) -> None:
        """Verify that factual and stylistic retrieval channels use distinct filters."""
        mock_retriever = AsyncMock()
        mock_retriever.search.side_effect = [
            [_create_result("Factual statute clause", knowledge_type="factual")],
            [_create_result("Style layout template", knowledge_type="stylistic")],
        ]

        service = DocumentDraftingService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=FakeLLMProvider(responses=[json.dumps({
                "title": "Service Agreement",
                "document_type": "contract",
                "sections": [
                    {
                        "heading": "1. Scope",
                        "content": "Consultant shall provide services according to Factual Clause.",
                        "citations": [{"evidence_id": 1}],
                    }
                ],
            })])),
            context_builder=ContextBuilder(),
        )

        query = DraftingQuery(topic="Consulting Agreement", document_type="contract")
        result = await service.draft(query)

        assert mock_retriever.search.call_count == 2
        calls = mock_retriever.search.call_args_list

        # Channel 1: Factual
        call_1_query: HybridSearchQuery = calls[0].args[0]
        assert call_1_query.filters is not None
        assert call_1_query.filters.knowledge_type == "factual"
        assert call_1_query.text_query == "Consulting Agreement"

        # Channel 2: Stylistic
        call_2_query: HybridSearchQuery = calls[1].args[0]
        assert call_2_query.filters is not None
        assert call_2_query.filters.knowledge_type == "stylistic"
        assert "contract" in call_2_query.text_query

        assert len(result.factual_evidence) == 1
        assert len(result.style_references) == 1
        assert len(result.sections) == 1
        assert len(result.citations) == 1

    @pytest.mark.asyncio
    async def test_style_leakage_raises_error(self) -> None:
        """Verify that citing a stylistic chunk as legal authority raises StyleLeakageError."""
        mock_retriever = AsyncMock()
        style_chunk_id = uuid4()

        factual_res = _create_result("Factual statutory rule")
        style_res = _create_result("Style template structure", chunk_id=style_chunk_id)

        mock_retriever.search.side_effect = [[factual_res], [style_res]]

        # LLM attempts to cite the style chunk as an authority
        leakage_response = json.dumps({
            "title": "Complaint Draft",
            "document_type": "complaint",
            "sections": [
                {
                    "heading": "Claims",
                    "content": "Defendant breached duty as set forth in style template.",
                    "citations": [{"chunk_id": str(style_chunk_id)}],
                }
            ],
        })

        service = DocumentDraftingService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=FakeLLMProvider(responses=[leakage_response])),
            context_builder=ContextBuilder(),
        )

        query = DraftingQuery(topic="Breach of Contract Complaint")
        with pytest.raises(StyleLeakageError, match="Style reference chunk"):
            await service.draft(query)

    @pytest.mark.asyncio
    async def test_style_references_never_become_factual_citations(self) -> None:
        """Verify that style references are never treated as factual citations."""
        mock_retriever = AsyncMock()
        factual_res = _create_result("Civil Code Section 1234: Interest is capped at 10%.")
        style_res = _create_result("FORM: 'NOW COMES Plaintiff by and through undersigned counsel...'")

        mock_retriever.search.side_effect = [[factual_res], [style_res]]

        llm_response = json.dumps({
            "title": "Demand Letter",
            "document_type": "letter",
            "sections": [
                {
                    "heading": "Notice of Default",
                    "content": "Interest rate is 10% under Civil Code Section 1234.",
                    "citations": [{"evidence_id": 1}],
                }
            ],
        })

        service = DocumentDraftingService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=FakeLLMProvider(responses=[llm_response])),
            context_builder=ContextBuilder(),
        )

        draft = await service.draft(DraftingQuery(topic="Demand Letter for Unpaid Invoices"))

        assert len(draft.citations) == 1
        assert draft.citations[0].chunk_id == factual_res.chunk_id
        # Confirm the style reference chunk is NOT in the citations
        assert draft.citations[0].chunk_id != style_res.chunk_id

    @pytest.mark.asyncio
    async def test_drafting_with_no_style_references(self) -> None:
        """Verify drafting functions properly when no style references exist."""
        mock_retriever = AsyncMock()
        factual_res = _create_result("Governing Law: Laws of California.")
        # Empty style search results
        mock_retriever.search.side_effect = [[factual_res], []]

        llm_response = json.dumps({
            "title": "Confidentiality Agreement",
            "document_type": "agreement",
            "sections": [
                {
                    "heading": "Governing Law",
                    "content": "This agreement shall be governed by California law.",
                    "citations": [{"evidence_id": 1}],
                }
            ],
        })

        service = DocumentDraftingService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=FakeLLMProvider(responses=[llm_response])),
            context_builder=ContextBuilder(),
        )

        draft = await service.draft(DraftingQuery(topic="Confidentiality Agreement", style_top_k=0))
        assert len(draft.style_references) == 0
        assert len(draft.factual_evidence) == 1
        assert "Governing Law" in draft.full_text

    @pytest.mark.asyncio
    async def test_drafting_empty_topic_raises_error(self) -> None:
        service = DocumentDraftingService(
            hybrid_retriever=AsyncMock(),
            reranker=FakeReranker(),
            llm_service=LLMService(provider=FakeLLMProvider(responses=["OK"])),
        )
        with pytest.raises(InvalidQueryError):
            await service.draft(DraftingQuery(topic="   "))

    @pytest.mark.asyncio
    async def test_unstructured_llm_fallback(self) -> None:
        """Verify service handles unstructured non-JSON LLM response gracefully."""
        mock_retriever = AsyncMock()
        mock_retriever.search.side_effect = [[_create_result("Statutory facts")], []]

        raw_text = "This is a drafted contract. All rights are reserved."
        service = DocumentDraftingService(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=FakeLLMProvider(responses=[raw_text, raw_text])),
            context_builder=ContextBuilder(),
        )

        draft = await service.draft(DraftingQuery(topic="Sample Agreement"))
        assert len(draft.sections) >= 1
        assert "drafted contract" in draft.full_text


class TestAgentDraftingIntent:
    """Test intent classification routing for document drafting."""

    @pytest.mark.asyncio
    async def test_analyze_intent_classifies_drafting(self) -> None:
        analyze_intent = create_analyze_intent_node()

        for q in [
            "Draft a non-disclosure agreement",
            "Please write a contract for software development",
            "Draft a complaint against landlord",
            "Prepare a notice of termination",
            "Document generation for employment offer",
        ]:
            state = {"query": q, "trace_metadata": {}}
            res = await analyze_intent(state)
            assert res["intent"] == "document_generation"


class TestDraftingAPI:
    """Test FastAPI endpoint POST /api/v1/drafting/draft."""

    def test_draft_endpoint_success(self) -> None:
        mock_service = AsyncMock()
        mock_service.draft.return_value = DraftDocument(
            title="Non-Disclosure Agreement",
            document_type="contract",
            sections=[
                DraftSection(
                    heading="Confidentiality",
                    content="The Receiving Party shall maintain confidentiality.",
                    citations=[],
                )
            ],
            full_text="# Non-Disclosure Agreement\n\n## Confidentiality\nThe Receiving Party shall maintain confidentiality.",
            citations=[],
            factual_evidence=[],
            style_references=[],
            model="fake-llm",
        )

        from app.api.routes.drafting import get_drafting_service

        app.dependency_overrides[get_drafting_service] = lambda: mock_service
        try:
            client = TestClient(app)
            response = client.post(
                "/api/v1/drafting/draft",
                json={"topic": "Mutual Non-Disclosure Agreement", "document_type": "contract"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Non-Disclosure Agreement"
            assert len(data["sections"]) == 1
            assert "Confidentiality" in data["full_text"]
        finally:
            app.dependency_overrides.pop(get_drafting_service, None)

    def test_draft_endpoint_style_leakage_422(self) -> None:
        mock_service = AsyncMock()
        mock_service.draft.side_effect = StyleLeakageError("Style reference chunk cited as authority.")

        from app.api.routes.drafting import get_drafting_service

        app.dependency_overrides[get_drafting_service] = lambda: mock_service
        try:
            client = TestClient(app)
            response = client.post(
                "/api/v1/drafting/draft",
                json={"topic": "Bad Draft Request"},
            )
            assert response.status_code == 422
            assert "Style leakage detected" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_drafting_service, None)
