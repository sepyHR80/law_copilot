import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.graph import LegalAgent, build_legal_agent_graph
from app.agent.models import AgentRequest, AgentResponse
from app.core.config import get_settings
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.reranking import get_default_reranker
from app.llm.service import LLMService
from app.rag.context_builder import ContextBuilder
from app.retrieval.hybrid import HybridSearchService
from app.tracing.service import TraceService

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_legal_agent(db: Session = Depends(get_db)) -> LegalAgent:
    """Constructs LegalAgent with injected services."""
    settings = get_settings()

    embedding_provider = OpenAIEmbeddingProvider(
        endpoint=settings.embedding_endpoint,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        expected_dimension=settings.embedding_dimension,
        timeout=float(settings.embedding_timeout),
        batch_size=settings.embedding_batch_size,
    )

    hybrid_retriever = HybridSearchService(
        session=db,
        embedding_provider=embedding_provider,
    )

    reranker = get_default_reranker()

    llm_service = LLMService()
    context_builder = ContextBuilder()

    compiled_graph = build_legal_agent_graph(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_service=llm_service,
        context_builder=context_builder,
    )

    return LegalAgent(compiled_graph)


@router.post("/chat", response_model=AgentResponse, status_code=status.HTTP_200_OK)
async def chat_with_agent(
    request: AgentRequest,
    agent: LegalAgent = Depends(get_legal_agent),
) -> AgentResponse:
    """Invoke the LangGraph legal agent state machine."""
    t0 = time.perf_counter()
    try:
        response = await agent.run(request)
        total_latency_ms = int((time.perf_counter() - t0) * 1000)

        # Extract evidence preview
        retrieval_data = []
        for ev in response.evidence:
            retrieval_data.append({
                "chunk_id": str(ev.chunk_id),
                "document_id": str(ev.document_id),
                "content_preview": ev.content[:300] + ("..." if len(ev.content) > 300 else ""),
                "score": ev.score,
                "page": ev.page,
                "section": ev.section,
                "title": ev.source.get("title") if isinstance(ev.source, dict) else None,
            })

        citations_data = [
            c.model_dump(mode="json") if hasattr(c, "model_dump") else c
            for c in response.citations
        ]

        settings = get_settings()
        trace_id = TraceService.record_trace(
            user_query=request.query,
            ai_response=response.response,
            intent=response.intent,
            is_sufficient=response.is_sufficient,
            execution_path=response.execution_path,
            retrieval_data=retrieval_data,
            citations=citations_data,
            evidence=[e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in response.evidence],
            conversation_id=request.conversation_id,
            model_name=settings.llm_model,
            latency_ms=total_latency_ms,
            status="success" if response.is_sufficient else "insufficient",
            trace_metadata=response.trace_metadata,
        )

        if trace_id:
            response.trace_id = str(trace_id)
        return response
    except Exception as exc:
        total_latency_ms = int((time.perf_counter() - t0) * 1000)
        err_str = str(exc).lower()
        is_rate_limit = "rate limit" in err_str or "quota exceeded" in err_str or "429" in err_str
        err_msg = (
            "سقف درخواست‌های روزانه هوش مصنوعی موقتاً تکمیل شده است. "
            "سیستم به صورت خودکار مدل‌های جایگزین را امتحان می‌کند؛ لطفاً چند لحظه دیگر مجدداً تلاش فرمایید."
            if is_rate_limit else
            f"خطا در پردازش هوش مصنوعی: {exc}"
        )

        trace_id = TraceService.record_trace(
            user_query=request.query,
            ai_response=err_msg,
            intent="legal_qa",
            is_sufficient=False,
            execution_path=[{
                "step": "error",
                "title": "خطا در پردازش",
                "status": "failed",
                "duration_ms": total_latency_ms,
                "details": {"error": str(exc)},
                "timestamp": datetime.utcnow().isoformat(),
            }],
            conversation_id=request.conversation_id,
            latency_ms=total_latency_ms,
            status="rate_limited" if is_rate_limit else "error",
            error_message=str(exc),
        )

        if is_rate_limit:
            return AgentResponse(
                query=request.query,
                response=err_msg,
                intent="legal_qa",
                citations=[],
                evidence=[],
                is_sufficient=False,
                trace_id=str(trace_id) if trace_id else None,
                trace_metadata={"rate_limit_handled": True, "trace_id": str(trace_id) if trace_id else None},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        ) from exc
