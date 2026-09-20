"""Unit tests for Stage 16 — LangGraph Agent orchestration."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.agent.edges import route_evidence, route_intent, route_verification
from app.agent.graph import LegalAgent, build_legal_agent_graph
from app.agent.models import AgentRequest, AgentResponse
from app.agent.nodes import (
    create_analyze_intent_node,
    create_assess_evidence_node,
    create_handle_general_node,
    create_handle_insufficient_node,
    create_prepare_query_node,
    create_verify_answer_node,
)
from app.agent.state import AgentState
from app.domain.rag.models import EvidenceItem
from app.domain.retrieval.models import RetrievalResult
from app.infrastructure.llm.fake import FakeLLMProvider
from app.infrastructure.reranking.fake import FakeReranker
from app.llm.service import LLMService
from app.main import app
from app.rag.context_builder import ContextBuilder


def _create_candidate(
    content: str,
    page: int = 1,
    section: str = "Intro",
    chunk_id: UUID | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        content=content,
        score=0.9,
        rank=1,
        source={"type": "hybrid", "page": page, "section": section},
    )


class TestAgentNodesAndEdges:
    """Test individual node functions and routing edges in isolation."""

    @pytest.mark.asyncio
    async def test_prepare_query_node_normalizes_whitespace(self) -> None:
        node = create_prepare_query_node()
        state: AgentState = {"query": "  what   is  the   statute   of   limitations?  "}
        res = await node(state)
        assert res["normalized_query"] == "what is the statute of limitations?"

    @pytest.mark.asyncio
    async def test_assess_evidence_node_empty_vs_non_empty(self) -> None:
        node = create_assess_evidence_node()

        empty_res = await node({"retrieval_results": []})
        assert empty_res["is_sufficient"] is False

        non_empty_res = await node({"retrieval_results": [_create_candidate("doc text")]})
        assert non_empty_res["is_sufficient"] is True

    @pytest.mark.asyncio
    async def test_verify_answer_insufficient_bypasses_citation_check(self) -> None:
        node = create_verify_answer_node(ContextBuilder())
        state: AgentState = {
            "draft": "Insufficient evidence found.",
            "is_sufficient": False,
            "raw_citations": [],
            "selected_evidence": [],
        }
        res = await node(state)
        assert res["verification_passed"] is True
        assert res["citations"] == []

    def test_routing_edges(self) -> None:
        # route_intent
        assert route_intent({"intent": "general"}) == "handle_general"
        assert route_intent({"intent": "legal_qa"}) == "prepare_query"

        # route_evidence
        assert route_evidence({"is_sufficient": False, "retrieval_results": []}) == "handle_insufficient"
        assert route_evidence({"is_sufficient": True, "retrieval_results": [_create_candidate("text")]}) == "build_context"

        # route_verification
        assert route_verification({"verification_passed": True}) == "__end__"
        assert route_verification({"verification_passed": False, "retry_count": 0, "max_retries": 2}) == "repair_draft"
        assert route_verification({"verification_passed": False, "retry_count": 2, "max_retries": 2}) == "__end__"


class TestLegalAgentGraph:
    """Test the LangGraph agent state machine transitions and boundedness."""

    @pytest.mark.asyncio
    async def test_general_conversational_routing(self) -> None:
        mock_retriever = AsyncMock()
        fake_llm = FakeLLMProvider()

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
        )
        agent = LegalAgent(compiled_graph)

        res = await agent.run(AgentRequest(query="Hello there!"))

        assert res.intent == "general"
        assert "Law Copilot" in res.response
        assert res.citations == []
        assert res.evidence == []
        assert res.is_sufficient is True
        # Verify no retrieval was called for general greetings
        mock_retriever.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_legal_qa_with_sufficient_evidence(self) -> None:
        cand = _create_candidate("Confidentiality obligations survive for 5 years after termination.", page=6, section="Confidentiality")
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand]

        valid_answer = json.dumps({
            "answer": "Confidentiality obligations last for 5 years post-termination.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 1, "snippet": "survive for 5 years"}],
        })
        fake_llm = FakeLLMProvider(responses=[valid_answer])

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
        )
        agent = LegalAgent(compiled_graph)

        res = await agent.run(AgentRequest(query="How long does confidentiality survive?"))

        assert res.intent == "legal_qa"
        assert "5 years" in res.response
        assert len(res.citations) == 1
        assert res.citations[0].chunk_id == cand.chunk_id
        assert len(res.evidence) == 1
        assert res.is_sufficient is True
        assert res.retry_count == 0

    @pytest.mark.asyncio
    async def test_insufficient_evidence_when_zero_results(self) -> None:
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = []

        fake_llm = FakeLLMProvider()

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
        )
        agent = LegalAgent(compiled_graph)

        res = await agent.run(AgentRequest(query="What are the non-compete terms?"))

        assert res.intent == "legal_qa"
        assert res.is_sufficient is False
        assert "insufficient evidence" in res.response.lower()
        assert res.citations == []
        assert res.evidence == []
        # No LLM calls wasted
        assert len(fake_llm.call_history) == 0

    @pytest.mark.asyncio
    async def test_verification_repair_cycle(self) -> None:
        cand = _create_candidate("Severability clause ensures validity of remainder.", page=9, section="Severability")
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand]

        # First draft has a hallucinated evidence ID (99) -> triggers repair
        first_bad_draft = json.dumps({
            "answer": "Severability protects remainder.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 99, "snippet": "non-existent"}],
        })
        # Repaired draft fixes the citation ID to 1
        second_good_draft = json.dumps({
            "answer": "Severability clause ensures remaining provisions remain valid.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 1, "snippet": "validity of remainder"}],
        })

        fake_llm = FakeLLMProvider(responses=[first_bad_draft, second_good_draft])

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
        )
        agent = LegalAgent(compiled_graph)

        res = await agent.run(AgentRequest(query="Explain the severability clause", max_retries=2))

        assert res.intent == "legal_qa"
        assert res.retry_count == 1
        assert len(res.citations) == 1
        assert res.citations[0].chunk_id == cand.chunk_id
        assert len(fake_llm.call_history) == 2

    @pytest.mark.asyncio
    async def test_bounded_retries_prevent_infinite_loop(self) -> None:
        cand = _create_candidate("Governing law is Delaware.")
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand]

        # Drafts continuously cite hallucinated IDs
        bad_draft = json.dumps({
            "answer": "Delaware law applies.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 999, "snippet": "hallucinated"}],
        })

        fake_llm = FakeLLMProvider(responses=[bad_draft, bad_draft, bad_draft, bad_draft])

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
        )
        agent = LegalAgent(compiled_graph)

        # Max retries set to 2
        res = await agent.run(AgentRequest(query="What is the governing law?", max_retries=2))

        # Must terminate deterministically after max retries
        assert res.retry_count == 2
        assert "Delaware" in res.response
        # Hallucinated citations are stripped
        assert res.citations == []


class TestAgentApi:
    """Test the agent chat HTTP endpoint."""

    def test_agent_chat_api_endpoint(self) -> None:
        from app.api.routes.agent import get_legal_agent

        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = []

        fake_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=FakeLLMProvider()),
            context_builder=ContextBuilder(),
        )
        fake_agent = LegalAgent(fake_graph)

        app.dependency_overrides[get_legal_agent] = lambda: fake_agent

        client = TestClient(app)
        try:
            res = client.post("/api/v1/agent/chat", json={"query": "Hello"})
            assert res.status_code == 200
            data = res.json()
            assert data["intent"] == "general"
            assert "Law Copilot" in data["response"]
        finally:
            app.dependency_overrides.clear()
