"""LangGraph legal agent assembly and orchestrator."""

from pathlib import Path
from typing import Any, Optional
from langgraph.graph import END, START, StateGraph

from app.agent.edges import route_evidence, route_intent, route_verification
from app.agent.models import AgentRequest, AgentResponse
from app.agent.nodes import (
    create_analyze_intent_node,
    create_assess_evidence_node,
    create_build_context_node,
    create_generate_draft_node,
    create_handle_general_node,
    create_handle_insufficient_node,
    create_load_memory_node,
    create_prepare_query_node,
    create_repair_draft_node,
    create_retrieve_knowledge_node,
    create_verify_answer_node,
)
from app.agent.state import AgentState
from app.domain.retrieval.protocol import HybridRetrieverProtocol, RerankerProtocol
from app.infrastructure.reranking.cross_encoder import CrossEncoderReranker
from app.llm.service import LLMService
from app.rag.context_builder import ContextBuilder



def build_legal_agent_graph(
    hybrid_retriever: HybridRetrieverProtocol,
    reranker: Optional[RerankerProtocol] = None,
    llm_service: Optional[LLMService] = None,
    context_builder: Optional[ContextBuilder] = None,
    memory_service: Optional[Any] = None,
    prompts_dir: Optional[Path] = None,
    checkpointer: Optional[Any] = None,
):
    """Construct and compile the constrained LangGraph state machine.

    Coordinates modular services without raw SQL/storage/provider logic.
    Guarantees bounded retry execution and prevents infinite loops.
    """
    reranker = reranker or CrossEncoderReranker()
    llm_service = llm_service or LLMService()
    context_builder = context_builder or ContextBuilder()

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("load_memory", create_load_memory_node(memory_service))
    workflow.add_node("analyze_intent", create_analyze_intent_node())
    workflow.add_node("handle_general", create_handle_general_node())
    workflow.add_node("prepare_query", create_prepare_query_node())
    workflow.add_node(
        "retrieve_knowledge",
        create_retrieve_knowledge_node(hybrid_retriever=hybrid_retriever, reranker=reranker),
    )
    workflow.add_node("assess_evidence", create_assess_evidence_node())
    workflow.add_node("handle_insufficient", create_handle_insufficient_node())
    workflow.add_node("build_context", create_build_context_node(context_builder))
    workflow.add_node(
        "generate_draft",
        create_generate_draft_node(llm_service=llm_service, prompts_dir=prompts_dir or Path("prompts")),
    )
    workflow.add_node("verify_answer", create_verify_answer_node(context_builder))
    workflow.add_node("repair_draft", create_repair_draft_node(llm_service))

    # Add transitions
    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "analyze_intent")

    workflow.add_conditional_edges(
        "analyze_intent",
        route_intent,
        {
            "handle_general": "handle_general",
            "prepare_query": "prepare_query",
        },
    )

    workflow.add_edge("handle_general", END)
    workflow.add_edge("prepare_query", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "assess_evidence")

    workflow.add_conditional_edges(
        "assess_evidence",
        route_evidence,
        {
            "handle_insufficient": "handle_insufficient",
            "build_context": "build_context",
        },
    )

    workflow.add_edge("handle_insufficient", END)
    workflow.add_edge("build_context", "generate_draft")
    workflow.add_edge("generate_draft", "verify_answer")

    workflow.add_conditional_edges(
        "verify_answer",
        route_verification,
        {
            END: END,
            "repair_draft": "repair_draft",
        },
    )

    workflow.add_edge("repair_draft", "verify_answer")

    return workflow.compile(checkpointer=checkpointer)



class LegalAgent:
    """High-level client interface for invoking the LangGraph legal agent."""

    def __init__(self, compiled_graph) -> None:
        self.graph = compiled_graph

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Run the agent state machine on an AgentRequest."""
        initial_state: AgentState = {
            "query": request.query,
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "filters": request.filters,
            "retry_count": 0,
            "max_retries": request.max_retries,
            "errors": [],
            "trace_metadata": {},
        }

        config = {}
        if request.conversation_id:
            config["configurable"] = {"thread_id": str(request.conversation_id)}

        final_state = await self.graph.ainvoke(initial_state, config=config if config else None)

        return AgentResponse(
            query=request.query,
            response=final_state.get("final_response") or final_state.get("draft") or "",
            intent=final_state.get("intent", "legal_qa"),
            citations=final_state.get("citations", []),
            evidence=final_state.get("selected_evidence", []),
            is_sufficient=final_state.get("is_sufficient", True),
            retry_count=final_state.get("retry_count", 0),
            trace_metadata=final_state.get("trace_metadata", {}),
        )
