"""Domain models for benchmark evaluation and regression testing."""

from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class RetrievalTestCase(BaseModel):
    """Evaluation test case for retrieval accuracy."""

    case_id: str
    query: str
    expected_chunk_ids: List[UUID]
    knowledge_type: Optional[str] = "factual"
    k_values: List[int] = Field(default_factory=lambda: [1, 3, 5])


class GenerationTestCase(BaseModel):
    """Evaluation test case for generation grounding and faithfulness."""

    case_id: str
    question: str
    expected_key_facts: List[str]
    forbidden_claims: List[str] = Field(default_factory=list)
    expect_sufficient: bool = True


class StyleTestCase(BaseModel):
    """Evaluation test case for style and document generation."""

    case_id: str
    topic: str
    document_type: str
    expected_sections: List[str]
    forbid_style_citations: bool = True


class RetrievalEvaluationScore(BaseModel):
    """Computed metrics for retrieval evaluation."""

    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]


class BenchmarkSuiteResult(BaseModel):
    """Aggregated evaluation results across the benchmark suite."""

    total_cases: int
    retrieval_cases_evaluated: int
    generation_cases_evaluated: int
    mean_mrr: float
    mean_ndcg_at_5: float
    mean_faithfulness: float
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
