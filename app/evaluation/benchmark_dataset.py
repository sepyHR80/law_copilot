"""Pre-packaged benchmark dataset for Law Copilot evaluation and regression testing."""

from uuid import UUID
from app.domain.evaluation.models import (
    GenerationTestCase,
    RetrievalTestCase,
    StyleTestCase,
)

# Standard deterministic chunk identifiers for evaluation cases
CHUNK_ID_CIVIL_10 = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID_CIVIL_11 = UUID("10000000-0000-0000-0000-000000000002")
CHUNK_ID_LABOR_24 = UUID("20000000-0000-0000-0000-000000000001")
CHUNK_ID_COMMERCIAL_5 = UUID("30000000-0000-0000-0000-000000000001")

BENCHMARK_RETRIEVAL_CASES = [
    RetrievalTestCase(
        case_id="ret_case_01",
        query="freedom of contract validity under civil law",
        expected_chunk_ids=[CHUNK_ID_CIVIL_10],
        knowledge_type="factual",
        k_values=[1, 3, 5],
    ),
    RetrievalTestCase(
        case_id="ret_case_02",
        query="severance pay calculation on employment termination",
        expected_chunk_ids=[CHUNK_ID_LABOR_24],
        knowledge_type="factual",
        k_values=[1, 3, 5],
    ),
    RetrievalTestCase(
        case_id="ret_case_03",
        query="statutory limitations on commercial promissory notes",
        expected_chunk_ids=[CHUNK_ID_COMMERCIAL_5],
        knowledge_type="factual",
        k_values=[1, 3, 5],
    ),
]

BENCHMARK_GENERATION_CASES = [
    GenerationTestCase(
        case_id="gen_case_01",
        question="Are private contracts enforceable if they differ from statutory norms?",
        expected_key_facts=["valid", "unless contrary to mandatory law", "freedom of contract"],
        forbidden_claims=["automatic nullity", "treble damages"],
        expect_sufficient=True,
    ),
    GenerationTestCase(
        case_id="gen_case_02",
        question="What is the severance compensation required under the Labor Code?",
        expected_key_facts=["one month salary per year of service"],
        forbidden_claims=["no compensation", "five years severance penalty"],
        expect_sufficient=True,
    ),
    GenerationTestCase(
        case_id="gen_case_03",
        question="What is the regulation governing interplanetary commercial transport?",
        expected_key_facts=["insufficient evidence"],
        forbidden_claims=["Space Travel Act of 2029"],
        expect_sufficient=False,
    ),
]

BENCHMARK_STYLE_CASES = [
    StyleTestCase(
        case_id="style_case_01",
        topic="Mutual Non-Disclosure Agreement",
        document_type="agreement",
        expected_sections=["Confidential Information", "Obligations", "Term and Termination"],
        forbid_style_citations=True,
    ),
    StyleTestCase(
        case_id="style_case_02",
        topic="Notice of Breach of Contract",
        document_type="notice",
        expected_sections=["Default Description", "Cure Period", "Reservation of Rights"],
        forbid_style_citations=True,
    ),
]
