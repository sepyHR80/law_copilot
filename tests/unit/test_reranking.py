"""Unit tests for Stage 13 — Reranking."""

from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4
import pytest

from app.core.config import Settings
from app.domain.retrieval.exceptions import RerankerModelLoadError, RerankingError
from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.protocol import RerankerProtocol
from app.infrastructure.reranking.cross_encoder import CrossEncoderReranker, _sigmoid
from app.infrastructure.reranking.fake import FakeReranker
from app.retrieval.reranking import RerankingService


def _create_candidate(content: str, score: float = 0.5, chunk_id: UUID | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        content=content,
        score=score,
        rank=1,
        source={"type": "hybrid"},
    )


class TestRerankerProtocolConformity:
    """Verify implementations conform to RerankerProtocol."""

    def test_fake_reranker_implements_protocol(self) -> None:
        reranker = FakeReranker()
        assert isinstance(reranker, RerankerProtocol)

    def test_cross_encoder_reranker_implements_protocol(self) -> None:
        reranker = CrossEncoderReranker(model_backend=lambda pairs: [0.0] * len(pairs))
        assert isinstance(reranker, RerankerProtocol)


class TestFakeReranker:
    """Test deterministic FakeReranker logic."""

    def test_empty_candidates_returns_empty(self) -> None:
        reranker = FakeReranker()
        assert reranker.rerank("contract clause", []) == []

    def test_keyword_overlap_scoring(self) -> None:
        reranker = FakeReranker()
        c1 = _create_candidate("unrelated text regarding sports and weather")
        c2 = _create_candidate("the contract termination clause shall apply")
        c3 = _create_candidate("contract agreement between both parties")

        results = reranker.rerank("contract clause termination", [c1, c2, c3])

        assert len(results) == 3
        # c2 has 3 matching words (contract, termination, clause), c3 has 1 (contract), c1 has 0
        assert results[0].chunk_id == c2.chunk_id
        assert results[0].rank == 1
        assert results[0].score == 1.0  # 3 / 3
        assert results[1].chunk_id == c3.chunk_id
        assert results[1].rank == 2
        assert results[2].chunk_id == c1.chunk_id
        assert results[2].rank == 3
        assert results[2].score == 0.0

    def test_top_n_filtering(self) -> None:
        reranker = FakeReranker()
        c1 = _create_candidate("contract part one")
        c2 = _create_candidate("contract part two")
        c3 = _create_candidate("contract part three")

        results = reranker.rerank("contract", [c1, c2, c3], top_n=2)
        assert len(results) == 2
        assert [r.rank for r in results] == [1, 2]

    def test_custom_scorer(self) -> None:
        def my_scorer(q: str, c: str) -> float:
            return float(len(c))

        reranker = FakeReranker(custom_scorer=my_scorer)
        c_short = _create_candidate("short")
        c_long = _create_candidate("a very long candidate text")

        results = reranker.rerank("any query", [c_short, c_long])
        assert results[0].chunk_id == c_long.chunk_id
        assert results[1].chunk_id == c_short.chunk_id


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker with custom backend (no downloads)."""

    def test_sigmoid_normalization(self) -> None:
        # Standard logits
        assert pytest.approx(_sigmoid(0.0), rel=1e-5) == 0.5
        assert _sigmoid(10.0) > 0.999
        assert _sigmoid(-10.0) < 0.001
        # Boundary clipping
        assert _sigmoid(50.0) == 1.0
        assert _sigmoid(-50.0) == 0.0

    def test_empty_candidates(self) -> None:
        reranker = CrossEncoderReranker(model_backend=lambda p: [])
        assert reranker.rerank("test query", []) == []

    def test_empty_query_preserves_order(self) -> None:
        reranker = CrossEncoderReranker(model_backend=lambda p: [])
        c1 = _create_candidate("cand 1", score=0.9)
        c2 = _create_candidate("cand 2", score=0.8)

        results = reranker.rerank("   ", [c1, c2], top_n=1)
        assert len(results) == 1
        assert results[0].chunk_id == c1.chunk_id
        assert results[0].rank == 1

    def test_reranking_with_injected_backend(self) -> None:
        c1 = _create_candidate("chunk A: low relevance")
        c2 = _create_candidate("chunk B: high relevance")
        c3 = _create_candidate("chunk C: medium relevance")

        # Mock model backend returning logits: c1 -> -2.0, c2 -> 4.0, c3 -> 1.0
        def mock_predict(pairs: list[tuple[str, str]]) -> list[float]:
            scores = []
            for _, doc in pairs:
                if "chunk B" in doc:
                    scores.append(4.0)
                elif "chunk C" in doc:
                    scores.append(1.0)
                else:
                    scores.append(-2.0)
            return scores

        reranker = CrossEncoderReranker(
            model_name="mock-model",
            batch_size=16,
            model_backend=mock_predict,
        )

        results = reranker.rerank("query", [c1, c2, c3])

        assert len(results) == 3
        # B should be rank 1, C rank 2, A rank 3
        assert results[0].chunk_id == c2.chunk_id
        assert results[0].rank == 1
        assert 0.9 < results[0].score < 1.0  # sigmoid(4.0) ~ 0.982

        assert results[1].chunk_id == c3.chunk_id
        assert results[1].rank == 2
        assert 0.7 < results[1].score < 0.8  # sigmoid(1.0) ~ 0.731

        assert results[2].chunk_id == c1.chunk_id
        assert results[2].rank == 3
        assert 0.1 < results[2].score < 0.2  # sigmoid(-2.0) ~ 0.119

    def test_batch_scoring_invocations(self) -> None:
        candidates = [_create_candidate(f"chunk {i}") for i in range(10)]
        call_batch_sizes: list[int] = []

        mock_backend = MagicMock()
        mock_backend.predict.side_effect = lambda batch: [0.0] * len(batch)

        def track_predict(batch: list[tuple[str, str]]) -> list[float]:
            call_batch_sizes.append(len(batch))
            return [0.0] * len(batch)

        reranker = CrossEncoderReranker(
            batch_size=4,
            model_backend=track_predict,
        )

        reranker.rerank("query", candidates)
        # 10 candidates with batch size 4 -> 4, 4, 2
        assert call_batch_sizes == [4, 4, 2]

    def test_deterministic_tie_breaking(self) -> None:
        id1 = UUID("00000000-0000-0000-0000-000000000001")
        id2 = UUID("00000000-0000-0000-0000-000000000002")

        c1 = _create_candidate("text 1", chunk_id=id1)
        c2 = _create_candidate("text 2", chunk_id=id2)

        # Both get identical logit = 0.0
        reranker = CrossEncoderReranker(
            model_backend=lambda pairs: [0.0, 0.0],
        )

        results = reranker.rerank("query", [c1, c2])
        # In tie-breaking: reverse sort by (score, str(chunk_id))
        # "00000000-0000-0000-0000-000000000002" > "00000000-0000-0000-0000-000000000001"
        assert results[0].chunk_id == id2
        assert results[1].chunk_id == id1

    def test_model_load_failure_raises_reranker_model_load_error(self) -> None:
        reranker = CrossEncoderReranker(model_name="non-existent-model")

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RerankerModelLoadError):
                reranker.rerank("query", [_create_candidate("text")])

    def test_backend_inference_error_raises_reranking_error(self) -> None:
        def failing_backend(pairs: list) -> list:
            raise RuntimeError("GPU out of memory")

        reranker = CrossEncoderReranker(model_backend=failing_backend)
        with pytest.raises(RerankingError, match="Inference error"):
            reranker.rerank("query", [_create_candidate("text")])


class TestRerankingService:
    """Test the RerankingService application layer."""

    def test_service_delegates_to_reranker(self) -> None:
        fake_reranker = FakeReranker()
        service = RerankingService(reranker=fake_reranker)

        c1 = _create_candidate("irrelevant doc")
        c2 = _create_candidate("indemnification clause agreement")

        results = service.rerank("indemnification clause", [c1, c2], top_n=1)
        assert len(results) == 1
        assert results[0].chunk_id == c2.chunk_id
        assert results[0].rank == 1

    def test_service_disabled_preserves_candidate_order(self) -> None:
        service = RerankingService()
        c1 = _create_candidate("doc 1", score=0.9)
        c2 = _create_candidate("doc 2", score=0.8)
        c3 = _create_candidate("doc 3", score=0.7)

        with patch("app.retrieval.reranking.get_settings") as mock_settings:
            mock_settings.return_value = Settings(reranker_enabled=False, reranker_top_n=2)
            results = service.rerank("query", [c1, c2, c3])

            assert len(results) == 2
            assert results[0].chunk_id == c1.chunk_id
            assert results[0].rank == 1
            assert results[1].chunk_id == c2.chunk_id
            assert results[1].rank == 2

    def test_service_empty_candidates_returns_empty(self) -> None:
        service = RerankingService(reranker=FakeReranker())
        assert service.rerank("query", []) == []
