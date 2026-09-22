"""Unit tests for Chat Tracing and Categorization System."""

from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.db.base import Base
from app.infrastructure.db.models.trace import ChatTrace
from app.main import app
from app.tracing.categorizer import categorize_query
from app.tracing.service import TraceService
from app.api.routes.traces import get_db


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
    """Test query categorization engine across Iranian legal branches."""

    def test_criminal_categorization(self):
        assert categorize_query("مجازات سرقت حدی چیست؟") == "حقوق کیفری و مجازات"
        assert categorize_query("حکم قتل عمد و قصاص") == "حقوق کیفری و مجازات"
        assert categorize_query("ارکان جرم کلاهبرداری") == "حقوق کیفری و مجازات"

    def test_civil_categorization(self):
        assert categorize_query("شرایط فسخ قرارداد بیع چیست؟") == "حقوق مدنی و قراردادها"
        assert categorize_query("نحوه مطالبه مهریه و نفقه زوجه") == "حقوق مدنی و قراردادها"
        assert categorize_query("قواعد ارث و ماترک متوفی") == "حقوق مدنی و قراردادها"

    def test_procedural_categorization(self):
        assert categorize_query("مهلت تجدیدنظرخواهی در دادگاه حقوقی") == "آیین دادرسی و امور قضایی"
        assert categorize_query("صلاحیت شورای حل اختلاف") == "آیین دادرسی و امور قضایی"

    def test_commercial_labor_categorization(self):
        assert categorize_query("نحوه وصول چک صیادی و سفته") == "حقوق تجارت و کار"
        assert categorize_query("محاسبه سنوات و اخراج کارگر در قانون کار") == "حقوق تجارت و کار"

    def test_drafting_categorization(self):
        assert categorize_query("لطفاً یک نمونه قرارداد اجاره بنویس") == "تنظیم و تدوین اسناد حقوقی"

    def test_general_conversational_categorization(self):
        assert categorize_query("سلام، وقت بخیر", intent="general") == "گفتگوی عمومی و راهنمایی"

    def test_insufficient_evidence_categorization(self):
        assert categorize_query("سوالی که در دیتابیس نیست", is_sufficient=False) == "شواهد ناکافی در پایگاه دانش"


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
        # Insert diverse traces
        TraceService.record_trace(
            user_query="شرایط سرقت",
            ai_response="پاسخ",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[],
            latency_ms=100,
            session=in_memory_db,
        )
        TraceService.record_trace(
            user_query="فسخ قرارداد",
            ai_response="پاسخ",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[],
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
            session=in_memory_db,
        )
        TraceService.record_trace(
            user_query="اجاره مسکونی",
            ai_response="پاسخ مدنی",
            intent="legal_qa",
            is_sufficient=True,
            execution_path=[],
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
            # Seed a trace
            TraceService.record_trace(
                user_query="ماده ۱۹۰ قانون مدنی",
                ai_response="شرایط اساسی صحت معامله",
                intent="legal_qa",
                is_sufficient=True,
                execution_path=[{"step": "received", "title": "دریافت", "duration_ms": 1, "status": "completed"}],
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
