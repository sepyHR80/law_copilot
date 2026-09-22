"""Unit tests for Chat Tracing and Categorization System.

Ensures strict compliance with:
- Zero regex, zero hardcoded keyword heuristics.
- Grounding in Knowledge Base search (document titles & sources).
- LLM inference classification.
"""

from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.db.base import Base
from app.infrastructure.db.models.trace import ChatTrace
from app.main import app
from app.tracing.categorizer import categorize_query, categorize_with_llm
from app.tracing.service import TraceService
from app.api.routes.traces import get_db
from app.infrastructure.llm.fake import FakeLLMProvider
from app.llm.service import LLMService


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing TraceService."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[ChatTrace.__table__])
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine, tables=[ChatTrace.__table__])


class TestLegalCategorizer:
    """Test query categorization strictly grounded in Knowledge Base search and LLM."""

    def test_kb_search_grounded_categorization(self):
        """Categories derived directly from Knowledge Base search results."""
        # 1. Criminal law grounded in KB statute title
        assert categorize_query(
            "مجازات سرقت حدی چیست؟",
            retrieval_data=[{"title": "قانون مجازات اسلامی"}],
        ) == "حقوق کیفری و مجازات"

        # 2. Civil law grounded in KB civil code title
        assert categorize_query(
            "شرایط فسخ قرارداد بیع چیست؟",
            retrieval_data=[{"title": "قانون مدنی"}],
        ) == "حقوق مدنی و قراردادها"

        # 3. Procedural law grounded in KB procedural code title
        assert categorize_query(
            "مهلت تجدیدنظرخواهی در دادگاه حقوقی",
            retrieval_data=[{"title": "قانون آیین دادرسی دادگاه‌های عمومی و انقلاب"}],
        ) == "آیین دادرسی و امور قضایی"

        # 4. Commercial law grounded in KB commercial code
        assert categorize_query(
            "نحوه وصول چک صیادی و سفته",
            retrieval_data=[{"title": "قانون تجارت"}],
        ) == "حقوق تجارت و کار"

        # 5. Labor law grounded in KB labor code evidence
        assert categorize_query(
            "محاسبه سنوات و اخراج کارگر",
            evidence=[{"source": {"title": "قانون کار و تامین اجتماعی"}}],
        ) == "حقوق تجارت و کار"

        # 6. Drafting grounded in KB contract evidence
        assert categorize_query(
            "استعلام سند قرارداد",
            citations=[{"source": {"title": "قرارداد پیمانکاری عمومی"}}],
        ) == "تنظیم و تدوین اسناد حقوقی"

        # 7. Custom uploaded document in Knowledge Base
        assert categorize_query(
            "دستورالعمل نظارت بر بورس",
            retrieval_data=[{"title": "دستورالعمل نظارت بر معاملات بورس کالا"}],
        ) == "پایگاه دانش: دستورالعمل نظارت بر معاملات بورس کالا"

    def test_llm_driven_categorization(self):
        """Categories derived directly from LLM classification."""
        # Explicit LLM category parameter
        assert categorize_query(
            "تحلیل موضوع",
            llm_category="حقوق کیفری و مجازات",
        ) == "حقوق کیفری و مجازات"

        # LLM category in trace_metadata
        assert categorize_query(
            "تحلیل موضوع",
            trace_metadata={"category": "حقوق مدنی و قراردادها"},
        ) == "حقوق مدنی و قراردادها"

    def test_intent_categorization(self):
        """Conversational and drafting intents without regex."""
        assert categorize_query("سلام، وقت بخیر", intent="general") == "گفتگوی عمومی و راهنمایی"
        assert categorize_query("لطفاً یک نمونه قرارداد بنویس", intent="document_generation") == "تنظیم و تدوین اسناد حقوقی"

    def test_insufficient_evidence_categorization(self):
        """Insufficient evidence search returns standardized category."""
        assert categorize_query("سوالی که در دیتابیس نیست", is_sufficient=False) == "شواهد ناکافی در پایگاه دانش"

    def test_strict_zero_regex_verification(self):
        """Verify that app.tracing.categorizer contains absolutely no regex imports or calls."""
        import inspect
        from app.tracing import categorizer

        source = inspect.getsource(categorizer)
        assert "import re" not in source, "Found 'import re' in categorizer.py!"
        assert "from re " not in source, "Found 'from re' in categorizer.py!"
        assert "re.findall" not in source, "Found 're.findall' in categorizer.py!"
        assert "re.search" not in source, "Found 're.search' in categorizer.py!"
        assert "re.match" not in source, "Found 're.match' in categorizer.py!"

    @pytest.mark.asyncio
    async def test_async_llm_categorizer(self):
        """Test categorize_with_llm with FakeLLMProvider."""
        fake_llm = FakeLLMProvider()
        llm_service = LLMService(provider=fake_llm)
        res = await categorize_with_llm("سلام، چطور کمکم می‌کنی؟", llm_service)
        assert res["intent"] == "general"
        assert res["category"] == "گفتگوی عمومی و راهنمایی"

    @pytest.mark.asyncio
    async def test_llm_categorizer_with_markdown_code_fences(self):
        """Test categorize_with_llm when LLM outputs json inside markdown code blocks."""
        from app.tracing.categorizer import extract_json_from_response

        # Test extraction directly
        raw = "```json\n{\n  \"intent\": \"general\",\n  \"category\": \"greeting\"\n}\n```"
        extracted = extract_json_from_response(raw)
        assert extracted == {"intent": "general", "category": "greeting"}

        # Test with custom handler in FakeLLMProvider
        from app.domain.llm.models import LLMResponse, LLMUsage
        fenced_provider = FakeLLMProvider(
            custom_handler=lambda req: LLMResponse(
                content="```json\n{\n  \"intent\": \"general\",\n  \"category\": \"گفتگوی عمومی و راهنمایی\"\n}\n```",
                model="gemini",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )
        )
        service = LLMService(provider=fenced_provider)
        res = await categorize_with_llm("سلام و درود", service)
        assert res["intent"] == "general"
        assert res["category"] == "گفتگوی عمومی و راهنمایی"


class TestTraceService:
    """Test TraceService persistence, aggregation, and querying."""

    def test_record_and_get_trace(self, in_memory_db):
        trace_id = TraceService.record_trace(
            user_query="حکم سرقت چیست؟",
            ai_response="سرقت بر دو نوع حدی و تعزیری است.",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[
                {"step": "received", "title": "دریافت پرسش", "duration_ms": 2, "status": "completed"},
                {"step": "retrieve_knowledge", "title": "بازیابی ترکیبی", "duration_ms": 45, "status": "completed"},
            ],
            retrieval_data=[{"title": "قانون مجازات اسلامی"}],
            model_name="gemini-3.5-flash-lite",
            latency_ms=120,
            session=in_memory_db,
        )

        assert trace_id is not None

        # Fetch detail
        detail = TraceService.get_trace(session=in_memory_db, trace_id=str(trace_id))
        assert detail is not None
        assert detail.user_query == "حکم سرقت چیست؟"
        assert detail.category == "حقوق کیفری و مجازات"
        assert detail.model_name == "gemini-3.5-flash-lite"
        assert len(detail.execution_path) == 2

    def test_categories_summary(self, in_memory_db):
        # Insert diverse traces with knowledge base search grounding
        TraceService.record_trace(
            user_query="شرایط سرقت",
            ai_response="پاسخ",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[],
            retrieval_data=[{"title": "قانون مجازات اسلامی"}],
            latency_ms=100,
            session=in_memory_db,
        )
        TraceService.record_trace(
            user_query="فسخ قرارداد",
            ai_response="پاسخ",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[],
            retrieval_data=[{"title": "قانون مدنی"}],
            latency_ms=150,
            session=in_memory_db,
        )
        TraceService.record_trace(
            user_query="سلام",
            ai_response="درود",
            intent="general",
            is_sufficient=True,
            execution_path=[],
            latency_ms=50,
            session=in_memory_db,
        )

        summary = TraceService.get_categories_summary(session=in_memory_db)
        assert summary.total_traces == 3
        assert len(summary.categories) == 3
        assert summary.overall_success_rate == 100.0

    def test_list_traces_with_filter(self, in_memory_db):
        TraceService.record_trace(
            user_query="سرقت مسلحانه",
            ai_response="پاسخ کیفری",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[],
            retrieval_data=[{"title": "قانون مجازات اسلامی"}],
            session=in_memory_db,
        )
        TraceService.record_trace(
            user_query="اجاره مسکونی",
            ai_response="پاسخ مدنی",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[],
            retrieval_data=[{"title": "قانون مدنی"}],
            session=in_memory_db,
        )

        # Filter by criminal category
        criminal_res = TraceService.list_traces(
            session=in_memory_db,
            category="حقوق کیفری و مجازات",
        )
        assert criminal_res.total == 1
        assert "سرقت" in criminal_res.items[0].user_query

        # Filter by search term
        search_res = TraceService.list_traces(
            session=in_memory_db,
            search="مسکونی",
        )
        assert search_res.total == 1
        assert "اجاره" in search_res.items[0].user_query


class TestTracesApi:
    """Test REST API endpoints for traces."""

    def test_traces_api_endpoints(self, in_memory_db):
        # Override get_db dependency
        app.dependency_overrides[get_db] = lambda: in_memory_db

        try:
            # Seed a trace grounded in knowledge base civil code
            TraceService.record_trace(
                user_query="ماده ۱۹۰ قانون مدنی",
                ai_response="شرایط اساسی صحت معامله",
                intent="legal_qa",
                is_sufficient=True,
                execution_path=[{"step": "received", "title": "دریافت", "duration_ms": 1, "status": "completed"}],
                retrieval_data=[{"title": "قانون مدنی"}],
                session=in_memory_db,
            )

            client = TestClient(app)

            # Test categories endpoint
            cat_resp = client.get("/api/v1/traces/categories")
            assert cat_resp.status_code == 200
            cat_data = cat_resp.json()
            assert cat_data["total_traces"] == 1
            assert cat_data["categories"][0]["category"] == "حقوق مدنی و قراردادها"

            # Test list traces endpoint
            list_resp = client.get("/api/v1/traces")
            assert list_resp.status_code == 200
            list_data = list_resp.json()
            assert list_data["total"] == 1
            trace_id = list_data["items"][0]["id"]

            # Test single trace detail endpoint
            detail_resp = client.get(f"/api/v1/traces/{trace_id}")
            assert detail_resp.status_code == 200
            detail_data = detail_resp.json()
            assert detail_data["user_query"] == "ماده ۱۹۰ قانون مدنی"
            assert len(detail_data["execution_path"]) == 1

            # Test delete
            del_resp = client.delete(f"/api/v1/traces/{trace_id}")
            assert del_resp.status_code == 200
        finally:
            app.dependency_overrides.clear()
