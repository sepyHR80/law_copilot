"""Unit tests for the pipeline processing SSE endpoint and frontend static files."""

import io
import json
import pytest
from fastapi.testclient import TestClient
import pymupdf

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def parse_sse_events(raw_text: str):
    """Helper to parse raw SSE text into a list of (event_type, json_data) tuples."""
    events = []
    blocks = raw_text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        event_type = "message"
        data_str = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                data_str += line.replace("data:", "").strip()
        if data_str:
            events.append((event_type, json.loads(data_str)))
    return events


class TestFrontendStaticRoutes:
    """Verify that Persian frontend static files are properly served by FastAPI."""

    def test_root_index_html(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "سامانه هوشمند حقوقی Law Copilot" in response.text
        assert 'dir="rtl"' in response.text

    def test_upload_page_html(self, client: TestClient):
        response = client.get("/upload.html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "بارگذاری و پردازش اسناد در پایگاه دانش" in response.text
        assert "Chunking" in response.text
        assert 'dir="rtl"' in response.text

    def test_chat_page_html(self, client: TestClient):
        response = client.get("/chat.html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "گفتگوی حقوقی با دستیار هوشمند" in response.text
        assert 'dir="rtl"' in response.text

    def test_static_css_and_js(self, client: TestClient):
        css_resp = client.get("/css/styles.css")
        assert css_resp.status_code == 200
        assert "Vazirmatn" in css_resp.text
        assert "direction: rtl" in css_resp.text

        upload_js_resp = client.get("/js/upload.js")
        assert upload_js_resp.status_code == 200
        assert "handlePipelineEvent" in upload_js_resp.text

        chat_js_resp = client.get("/js/chat.js")
        assert chat_js_resp.status_code == 200
        assert "renderAssistantResponse" in chat_js_resp.text


class TestPipelineProcessingSSE:
    """Verify document processing pipeline with SSE streaming progress."""

    def test_process_txt_document_pipeline(self, client: TestClient):
        sample_txt = (
            "فصل اول - مقررات عمومی\n\n"
            "ماده ۱ - قوانین باید در سراسر قلمرو جمهوری اسلامی ایران رعایت گردند.\n\n"
            "ماده ۲ - اموال غیرمنقول شامل زمین و بنا و هر آنچه به آنها متصل است می‌باشد."
        ).encode("utf-8")

        response = client.post(
            "/api/v1/pipeline/process",
            files={"file": ("civil_code.txt", sample_txt, "text/plain")},
            data={
                "title": "قانون مدنی",
                "document_type": "law",
                "knowledge_type": "factual",
                "source": "مجلس شورای اسلامی",
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = parse_sse_events(response.text)
        assert len(events) >= 5

        stages = [data.get("stage") for _, data in events]
        assert "upload" in stages
        assert "parsing" in stages
        assert "normalization" in stages
        assert "chunking" in stages
        assert "complete" in stages

        # Check parsing event details
        parsing_ev = next(d for _, d in events if d.get("stage") == "parsing" and d.get("status") == "completed")
        assert parsing_ev["text_extracted"] is True
        assert parsing_ev["total_chars"] > 0
        assert "ماده ۱" in parsing_ev["sample_text"]

        # Check chunking event details
        chunking_ev = next(d for _, d in events if d.get("stage") == "chunking" and d.get("status") == "completed")
        assert chunking_ev["total_chunks"] >= 1
        assert len(chunking_ev["sample_chunks"]) >= 1

        # Check complete event details
        complete_ev = next(d for _, d in events if d.get("stage") == "complete")
        assert complete_ev["status"] == "success"
        assert complete_ev["progress"] == 100

    def test_process_pdf_document_pipeline(self, client: TestClient):
        # Create an in-memory PDF using pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text(
            (50, 72),
            "ماده ۱۰ - قراردادهای خصوصی نسبت به کسانی که آن را منعقد نموده‌اند، در صورتی که مخالف صریح قانون نباشد، نافذ است.",
        )
        pdf_bytes = doc.tobytes()
        doc.close()

        response = client.post(
            "/api/v1/pipeline/process",
            files={"file": ("test_contract_law.pdf", pdf_bytes, "application/pdf")},
            data={
                "title": "ماده ۱۰ قانون مدنی",
                "document_type": "law",
                "knowledge_type": "factual",
                "source": "قانون مدنی",
            },
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        assert len(events) >= 4

        # Validate that PDF parsing reported text extraction status
        parsing_ev = next(d for _, d in events if d.get("stage") == "parsing" and d.get("status") == "completed")
        assert parsing_ev["is_pdf"] is True
        assert parsing_ev["text_extracted"] is True
        assert parsing_ev["pages_count"] >= 1

        # Validate chunking produces chunks
        chunking_ev = next(d for _, d in events if d.get("stage") == "chunking" and d.get("status") == "completed")
        assert chunking_ev["total_chunks"] >= 1

    def test_process_empty_file_error(self, client: TestClient):
        response = client.post(
            "/api/v1/pipeline/process",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"title": "Empty File"},
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        error_ev = next((d for _, d in events if d.get("status") == "error"), None)
        assert error_ev is not None
        assert "خالی" in error_ev["message"] or "empty" in error_ev["message"].lower()

    def test_process_unsupported_file_type(self, client: TestClient):
        response = client.post(
            "/api/v1/pipeline/process",
            files={"file": ("malicious.exe", b"binary content", "application/x-msdownload")},
            data={"title": "Binary File"},
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        error_ev = next((d for _, d in events if d.get("status") == "error"), None)
        assert error_ev is not None
        assert "پشتیبانی نمی‌شود" in error_ev["message"]

    def test_process_txt_document_pipeline_with_embedding(self, client: TestClient):
        """Verify pipeline emits embedding stage started, in_progress, and completed events."""
        from unittest.mock import AsyncMock, patch

        sample_txt = (
            "ماده ۱ - قوانین باید در سراسر قلمرو جمهوری اسلامی ایران رعایت گردند.\n\n"
            "ماده ۲ - اموال غیرمنقول شامل زمین و بنا می‌باشد."
        ).encode("utf-8")

        with patch("app.core.config.get_settings") as mock_settings, \
             patch("app.infrastructure.embeddings.openai_provider.OpenAIEmbeddingProvider") as mock_provider_cls:

            mock_s = mock_settings.return_value
            mock_s.embedding_api_key = "valid-active-key"
            mock_s.embedding_endpoint = "https://api.openai.com/v1"
            mock_s.embedding_model = "text-embedding-3-small"
            mock_s.embedding_dimension = 1536
            mock_s.embedding_batch_size = 5

            mock_instance = mock_provider_cls.return_value
            mock_instance.embed_texts = AsyncMock(side_effect=lambda texts: [[0.05] * 1536 for _ in texts])

            response = client.post(
                "/api/v1/pipeline/process",
                files={"file": ("sample_statute.txt", sample_txt, "text/plain")},
                data={
                    "title": "قانون آزمایشی",
                    "document_type": "law",
                    "knowledge_type": "factual",
                    "source": "مجلس",
                },
            )

            assert response.status_code == 200
            events = parse_sse_events(response.text)
            embedding_events = [d for _, d in events if d.get("stage") == "embedding"]
            assert len(embedding_events) >= 2

            started_ev = next((d for d in embedding_events if d.get("status") == "started"), None)
            assert started_ev is not None
            assert started_ev["total_to_embed"] >= 1
            assert started_ev["embedded_so_far"] == 0

            completed_ev = next((d for d in embedding_events if d.get("status") == "completed"), None)
            assert completed_ev is not None
            assert completed_ev["percentage"] == 100.0
            assert completed_ev["embedded_so_far"] == completed_ev["total_to_embed"]

