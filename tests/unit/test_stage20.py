"""Unit tests for Stage 20 — Evaluation, Observability, and Productionization."""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.core.metrics import MetricsRegistry, metrics
from app.core.telemetry import TelemetrySpan, scrub_secrets, trace_span
from app.domain.evaluation.models import BenchmarkSuiteResult, RetrievalTestCase
from app.domain.retrieval.models import RetrievalResult
from app.evaluation.metrics import (
    citation_completeness,
    faithfulness_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.runner import BenchmarkEvaluationRunner
from app.main import app


class TestEvaluationMetrics:
    """Test IR and generation evaluation metrics calculations."""

    def test_precision_at_k(self) -> None:
        u1, u2, u3, u4 = uuid4(), uuid4(), uuid4(), uuid4()
        retrieved = [u1, u2, u3, u4]
        relevant = {u1, u3}

        # top-1: 1 hit in 1 = 1.0
        assert precision_at_k(retrieved, relevant, k=1) == 1.0
        # top-2: 1 hit in 2 = 0.5
        assert precision_at_k(retrieved, relevant, k=2) == 0.5
        # top-4: 2 hits in 4 = 0.5
        assert precision_at_k(retrieved, relevant, k=4) == 0.5
        # empty
        assert precision_at_k([], relevant, k=5) == 0.0

    def test_recall_at_k(self) -> None:
        u1, u2, u3, u4 = uuid4(), uuid4(), uuid4(), uuid4()
        retrieved = [u1, u2, u3, u4]
        relevant = {u1, u3}

        # top-1 captures 1 of 2 = 0.5
        assert recall_at_k(retrieved, relevant, k=1) == 0.5
        # top-3 captures 2 of 2 = 1.0
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_mean_reciprocal_rank(self) -> None:
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        # Relevant item at position 2 -> MRR = 1/2 = 0.5
        assert mean_reciprocal_rank([u1, u2, u3], relevant_ids={u2}) == 0.5
        # Relevant item at position 1 -> MRR = 1.0
        assert mean_reciprocal_rank([u1, u2, u3], relevant_ids={u1}) == 1.0
        # No relevant item -> 0.0
        assert mean_reciprocal_rank([u1, u2], relevant_ids={u3}) == 0.0

    def test_ndcg_at_k(self) -> None:
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        # Perfect ranking: relevant item is #1
        assert ndcg_at_k([u1, u2, u3], relevant_ids={u1}, k=3) == 1.0
        # Suboptimal ranking: relevant item is #2
        score = ndcg_at_k([u2, u1, u3], relevant_ids={u1}, k=3)
        assert 0.0 < score < 1.0

    def test_faithfulness_score(self) -> None:
        evidence = ["Contractual freedom is guaranteed under Civil Code Article 10."]
        # High overlap
        resp_grounded = "Civil Code Article 10 guarantees contractual freedom."
        assert faithfulness_score(resp_grounded, evidence) >= 0.7

        # Zero evidence / refusal is faithful by definition
        assert faithfulness_score("There is insufficient evidence to answer.", evidence) == 1.0

    def test_citation_completeness(self) -> None:
        c1, c2, c3 = uuid4(), uuid4(), uuid4()
        assert citation_completeness(cited_chunk_ids=[c1, c2], expected_chunk_ids=[c1, c2, c3]) == pytest.approx(2 / 3)


class TestBenchmarkEvaluationRunner:
    """Test automated evaluation runner."""

    @pytest.mark.asyncio
    async def test_runner_executes_suite(self) -> None:
        mock_retriever = AsyncMock()
        # Return empty candidates
        mock_retriever.search.return_value = []

        runner = BenchmarkEvaluationRunner(retriever=mock_retriever)
        suite_res: BenchmarkSuiteResult = await runner.run_suite()

        assert suite_res.total_cases > 0
        assert suite_res.passed is True
        assert "retrieval_cases" in suite_res.details


class TestTelemetryAndSecretScrubbing:
    """Test telemetry tracing and secret scrubbing."""

    def test_scrub_secrets_redacts_keys_and_values(self) -> None:
        payload = {
            "api_key": "sk-1234567890123456789012345",
            "password": "supersecretpassword",
            "normal_field": "public text",
            "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "nested": {
                "secret_key": "nestedsecret",
                "safe_field": 42,
            },
        }
        scrubbed = scrub_secrets(payload)

        assert scrubbed["api_key"] == "[REDACTED]"
        assert scrubbed["password"] == "[REDACTED]"
        assert scrubbed["auth_header"] == "[REDACTED]"
        assert scrubbed["normal_field"] == "public text"
        assert scrubbed["nested"]["secret_key"] == "[REDACTED]"
        assert scrubbed["nested"]["safe_field"] == 42

    def test_trace_span_context_manager(self) -> None:
        with trace_span("test_operation", attributes={"password": "123", "param": "val"}) as span:
            span.set_attribute("user_query", "search contracts")
            assert span.name == "test_operation"
            assert span.attributes["password"] == "[REDACTED]"
            assert span.attributes["user_query"] == "search contracts"

        assert span.status == "OK"
        assert span.duration_seconds >= 0.0


class TestMetricsRegistry:
    """Test Prometheus metrics registry."""

    def test_metrics_recording_and_export(self) -> None:
        reg = MetricsRegistry()
        reg.record_request("/api/v1/rag/query", 200, 0.12)
        reg.record_tokens("fake-llm", prompt_tokens=15, completion_tokens=30)
        reg.record_verification_failure()
        reg.record_web_fallback()

        output = reg.export_prometheus_text()

        assert "law_copilot_requests_total" in output
        assert '/api/v1/rag/query' in output
        assert "law_copilot_llm_tokens_total" in output
        assert "fake-llm" in output
        assert "law_copilot_verification_failures_total 1" in output
        assert "law_copilot_web_fallback_total 1" in output


class TestMonitoringEndpoints:
    """Test /healthz, /readyz, and /metrics API endpoints."""

    def test_liveness_probe_healthz(self) -> None:
        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_readiness_probe_readyz(self) -> None:
        client = TestClient(app)
        response = client.get("/readyz")
        # In test environment, DB connects or returns structured JSON
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "database" in data

    def test_prometheus_metrics_endpoint(self) -> None:
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "law_copilot_requests_total" in response.text
