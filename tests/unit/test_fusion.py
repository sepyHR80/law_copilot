"""Unit tests for Stage 12 — Reciprocal Rank Fusion (RRF) algorithm."""

from uuid import UUID, uuid4
import pytest

from app.domain.retrieval.models import RetrievalResult
from app.retrieval.fusion import reciprocal_rank_fusion


def _make_result(chunk_id: UUID, content: str, rank: int, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=uuid4(),
        document_version_id=uuid4(),
        content=content,
        score=score,
        rank=rank,
        source={"title": f"Doc for {content}"},
    )


class TestReciprocalRankFusion:
    def test_rank_fusion_correctness(self):
        """Verify mathematical calculation: RRF(d) = 1/(k + rank_vec) + 1/(k + rank_lex)."""
        id1 = uuid4()
        # chunk 1 is rank 1 in vector (1 / (60 + 1) = 1/61) and rank 2 in lexical (1 / (60 + 2) = 1/62)
        vec_item = _make_result(id1, "Clause A", rank=1)
        lex_item = _make_result(id1, "Clause A", rank=2)

        fused = reciprocal_rank_fusion([[vec_item], [lex_item]], k=60)

        assert len(fused) == 1
        expected_score = round((1.0 / 61.0) + (1.0 / 62.0), 6)
        assert fused[0].score == expected_score
        assert fused[0].rank == 1
        assert fused[0].chunk_id == id1

    def test_vector_only_result(self):
        """When only vector search returns results, RRF correctly scores and ranks them."""
        id1, id2 = uuid4(), uuid4()
        vec_results = [
            _make_result(id1, "Vector Top 1", rank=1),
            _make_result(id2, "Vector Top 2", rank=2),
        ]
        lex_results = []

        fused = reciprocal_rank_fusion([vec_results, lex_results], k=60)

        assert len(fused) == 2
        assert fused[0].chunk_id == id1
        assert fused[0].score == round(1.0 / 61.0, 6)
        assert fused[0].rank == 1
        assert fused[1].chunk_id == id2
        assert fused[1].score == round(1.0 / 62.0, 6)
        assert fused[1].rank == 2

    def test_lexical_only_result(self):
        """When only lexical search returns results, RRF correctly scores and ranks them."""
        id1 = uuid4()
        vec_results = []
        lex_results = [_make_result(id1, "Lexical Only", rank=1)]

        fused = reciprocal_rank_fusion([vec_results, lex_results], k=60)

        assert len(fused) == 1
        assert fused[0].chunk_id == id1
        assert fused[0].score == round(1.0 / 61.0, 6)
        assert fused[0].rank == 1

    def test_overlapping_result(self):
        """A chunk appearing in both lists gets boosted over a chunk appearing in only one."""
        id_both = uuid4()
        id_vec_only = uuid4()

        # id_both is rank 2 in vector, rank 2 in lexical
        # id_vec_only is rank 1 in vector, but absent in lexical
        vec_list = [
            _make_result(id_vec_only, "Vector 1", rank=1),
            _make_result(id_both, "In Both", rank=2),
        ]
        lex_list = [
            _make_result(id_both, "In Both", rank=2),
        ]

        fused = reciprocal_rank_fusion([vec_list, lex_list], k=60)

        assert len(fused) == 2
        # id_both score: 1/62 + 1/62 = 0.032258
        # id_vec_only score: 1/61 = 0.016393
        # id_both must rank 1!
        assert fused[0].chunk_id == id_both
        assert fused[0].rank == 1
        assert fused[1].chunk_id == id_vec_only
        assert fused[1].rank == 2

    def test_disjoint_result_sets(self):
        """Two disjoint result sets are correctly merged and interleaved."""
        id_vec = uuid4()
        id_lex = uuid4()

        vec_list = [_make_result(id_vec, "Vec Doc", rank=1)]
        lex_list = [_make_result(id_lex, "Lex Doc", rank=2)]

        fused = reciprocal_rank_fusion([vec_list, lex_list], k=60)

        assert len(fused) == 2
        # Rank 1 in vector has score 1/61 > Rank 2 in lexical with score 1/62
        assert fused[0].chunk_id == id_vec
        assert fused[0].rank == 1
        assert fused[1].chunk_id == id_lex
        assert fused[1].rank == 2

    def test_duplicate_handling(self):
        """Duplicate results inside the same list or across lists do not corrupt results."""
        id1 = uuid4()
        # Same chunk twice in list 1 (edge case) and once in list 2
        vec_list = [_make_result(id1, "Item", rank=1), _make_result(id1, "Item", rank=1)]
        lex_list = [_make_result(id1, "Item", rank=1)]

        fused = reciprocal_rank_fusion([vec_list, lex_list], k=60)

        # Output must be deduplicated
        assert len(fused) == 1
        assert fused[0].chunk_id == id1

    def test_deterministic_ordering(self):
        """Tied items are ordered deterministically by chunk_id string."""
        # Two distinct chunks with identical rank in disjoint single lists
        id_a = UUID("00000000-0000-0000-0000-000000000001")
        id_b = UUID("00000000-0000-0000-0000-000000000002")

        vec_list = [_make_result(id_b, "B", rank=1)]
        lex_list = [_make_result(id_a, "A", rank=1)]

        # Call fusion multiple times and verify exact same order
        for _ in range(5):
            fused = reciprocal_rank_fusion([vec_list, lex_list], k=60)
            assert len(fused) == 2
            # Both have score 1/61, tie broken by str(chunk_id)
            assert fused[0].score == fused[1].score
            assert fused[0].rank == 1
            assert fused[1].rank == 2

    def test_empty_input_lists(self):
        """Empty input lists return empty result list."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_invalid_k_raises_error(self):
        """k < 1 raises ValueError."""
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], k=0)
