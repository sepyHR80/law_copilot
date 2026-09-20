# ⚖️ Law Copilot — Enterprise Grounded Legal AI Platform

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-FF6F00.svg)](https://github.com/langchain-ai/langgraph)
[![Tests](https://img.shields.io/badge/Tests-281%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

**Law Copilot** is a production-grade, enterprise legal research and document generation assistant designed specifically for rigorous legal compliance, zero-hallucination question answering, and structured document drafting.

Built strictly around the principle of **Grounded Authority**, Law Copilot separates binding legal truth from stylistic formatting, enforces citation provenance, and provides end-to-end observability, security, and benchmark evaluation.

---

## 📑 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [Core Invariants & Design Principles](#-core-invariants--design-principles)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start & Local Setup](#-quick-start--local-setup)
- [Running with Docker Compose](#-running-with-docker-compose)
- [API Reference & Usage Examples](#-api-reference--usage-examples)
- [Evaluation & Benchmark System](#-evaluation--benchmark-system)
- [Observability, Tracing & Metrics](#-observability-tracing--metrics)
- [Security Baseline](#-security-baseline)
- [Production & Kubernetes Deployment](#-production--kubernetes-deployment)
- [Testing & Verification](#-testing--verification)

---

## 🏛️ Architectural Overview

```mermaid
flowchart TD
    User([User Request / API Client]) --> Gateway[FastAPI API Gateway]
    
    subgraph Ingestion ["1. Document Ingestion Pipeline"]
        Upload[Upload API & Deduplication] --> Parser[Parsers: PDF, DOCX, HTML, TXT]
        Parser --> Normalizer[Canonical Normalizer]
        Normalizer --> Chunking[Structure-Aware Chunking]
        Chunking --> Embeddings[OpenAI / TEI Embedding Provider]
        Embeddings --> StorageDB[(PostgreSQL 16 + pgvector + GIN FTS)]
        Upload --> MinIO[(MinIO / S3 Object Storage)]
    end

    subgraph AgentEngine ["2. LangGraph Agent State Machine"]
        Gateway --> MemoryLoad[Load Scoped Memory & Context]
        MemoryLoad --> IntentNode{Classify Intent}
        IntentNode -->|General Greeting| GeneralResp[Conversational Response]
        IntentNode -->|Document Drafting| DraftingService[Dual-Channel Drafting]
        IntentNode -->|Legal Research| RetrievalPrep[Prepare Query]
        
        RetrievalPrep --> HybridSearch[Hybrid Retrieval: Vector + FTS]
        HybridSearch --> Reranker[Cross-Encoder Reranker]
        Reranker --> SufficiencyCheck{Evidence Sufficient?}
        
        SufficiencyCheck -->|Yes| BuildContext[Token-Bounded Context Builder]
        SufficiencyCheck -->|No| ExternalFallback[External Web Search Fallback]
        ExternalFallback --> DomainFilter[Domain Authority & Trust Filter]
        DomainFilter --> BuildContext
        
        BuildContext --> LLMGen[Provider-Agnostic LLM Engine]
        LLMGen --> VerifierNode{Claim & Citation Verifier}
        VerifierNode -->|Passed| FinalResp([Grounded Response + Citations])
        VerifierNode -->|Failed & Retries < Max| RepairDraft[Bounded Repair Node]
        RepairDraft --> VerifierNode
        VerifierNode -->|Max Retries Exceeded| FinalResp
    end

    subgraph DraftingEngine ["3. Knowledge-Separated Document Generation"]
        DraftingService --> FactRetrieval[Channel 1: Factual Retrieval]
        DraftingService --> StyleRetrieval[Channel 2: Stylistic Retrieval]
        FactRetrieval --> DraftingLLM[Drafting Generator]
        StyleRetrieval --> DraftingLLM
        DraftingLLM --> LeakageCheck{Style Leakage Check}
        LeakageCheck -->|Zero Style as Authority| VerifiedDraft([Structured Draft Document])
    end

    subgraph Observability ["4. Observability & Telemetry"]
        Gateway -.-> OTel[OpenTelemetry Distributed Tracing]
        Gateway -.-> PromMetrics[Prometheus Metrics Scraper]
        OTel -.-> Jaeger[Jaeger UI]
        PromMetrics -.-> Grafana[Grafana Dashboards]
    end
```

---

## 🛡️ Core Invariants & Design Principles

1. **No Hallucinated Citations**: Every citation returned by the system is verified against authentic, retrieved document chunks (`chunk_id`, `document_id`, `version_id`, and `page_number`). If an assertion cannot be grounded, it is rejected.
2. **Strict Knowledge Separation (Anti-Leakage Invariant)**:
   - **Factual Knowledge Base (`knowledge_type="factual"`)**: Laws, statutes, binding regulations, and contracts. Sole authority for legal facts and citations.
   - **Stylistic Knowledge Base (`knowledge_type="stylistic"`)**: Templates, past complaints, and formatting guides. Used *only* for tone, layout, and terminology.
   - **Hard Rule**: Stylistic documents must **never** be cited as legal authority.
3. **Internal Precedence**: Internal authoritative sources always take precedence. External web search is executed *only* as a fallback when internal evidence is insufficient.
4. **Bounded Agent Execution**: The LangGraph state machine enforces finite retries (`max_retries <= 5`) to prevent infinite repair loops and resource exhaustion.
5. **Zero-Leakage Telemetry**: Sensitive tokens, authorization headers, API keys, and passwords are recursively scrubbed from all traces, logs, and metrics.

---

## ⚡ Key Features

- **Multi-Format Ingestion**: High-throughput parsing for PDF (with PyMuPDF), Word documents (DOCX), HTML, and plain text with SHA-256 deduplication.
- **Structure-Aware Chunking**: Preserves legal hierarchy (Articles, Sections, Clauses, Recitals, and Tables) without breaking sentence coherence.
- **Hybrid Retrieval + Reciprocal Rank Fusion (RRF)**:
  - Dense cosine similarity search via `pgvector` HNSW / IVFFlat indexes.
  - Sparse lexical keyword search via PostgreSQL GIN indexes and English/multi-lingual text search dictionaries.
  - Reciprocal Rank Fusion (`RRF score = 1 / (60 + rank)`) combining vector and keyword ranks.
- **Cross-Encoder Reranking**: Fine-grained relevance re-scoring using transformer cross-encoders.
- **LangGraph Legal Agent**: Autonomous multi-turn state machine coordinating intent classification, memory loading, dynamic retrieval, and bounded self-correction.
- **Durable Memory & Scoped Context**: Multi-user session management with atomic PostgreSQL `ON CONFLICT` memory upserts and short-term message turn histories.
- **Model Context Protocol (MCP)**: Integrated `KnowledgeMCPServer` exposing the standardized `search_knowledge` tool for interoperable tool calling.
- **Automated Claim Verification**: Sentence-level parsing comparing statements against evidence text to detect fabricated legal numbers, dates, or unauthorized statutes.
- **Complete Evaluation Suite**: Benchmark datasets with exact computations for Precision@k, Recall@k, MRR, NDCG@k, Faithfulness, and Citation Completeness.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **API & Runtime** | FastAPI, Uvicorn, Python 3.12, Pydantic v2 |
| **Agent Orchestration** | LangGraph, LangChain Core |
| **Database & Vector Search** | PostgreSQL 16, pgvector, SQLAlchemy 2.0, Alembic, psycopg3 |
| **Object Storage** | MinIO (S3-compatible API), AWS S3 |
| **Embeddings & LLM** | OpenAI API, LiteLLM gateway, Cross-Encoder models |
| **Observability** | OpenTelemetry SDK, Prometheus, Jaeger, Grafana |
| **Testing & Quality** | Pytest, pytest-asyncio, AnyIO |

---

## 📁 Project Directory Structure

```text
law-copilot/
├── app/
│   ├── main.py                          # FastAPI application initialization & root probes
│   ├── core/
│   │   ├── config.py                    # Pydantic Settings & environment validation
│   │   ├── metrics.py                   # Prometheus metrics registry and collectors
│   │   └── telemetry.py                 # OpenTelemetry tracing & secret scrubbing
│   ├── domain/                          # Pure business logic & interfaces (No DB/SDK deps)
│   │   ├── documents/                   # Document domain models & exceptions
│   │   ├── drafting/                    # Legal drafting models & StyleLeakageError
│   │   ├── embeddings/                  # Embedding contracts & vector representations
│   │   ├── evaluation/                  # IR & generation evaluation models
│   │   ├── llm/                         # LLM request/response abstractions
│   │   ├── mcp/                         # Model Context Protocol schemas
│   │   ├── memory/                      # Long-term preferences & conversation turns
│   │   ├── rag/                         # Citation, EvidenceItem, and QA schemas
│   │   ├── retrieval/                   # Retrieval filters, query models & protocols
│   │   ├── search/                      # Validated external search source models
│   │   └── verification/                # Claim verification results & error schemas
│   ├── infrastructure/                  # Concrete adapters and I/O implementations
│   │   ├── db/                          # SQLAlchemy models, session, repositories
│   │   ├── embeddings/                  # OpenAI & local embedding providers
│   │   ├── llm/                         # LiteLLM & Fake provider implementations
│   │   ├── mcp/                         # Knowledge MCP Server & Client
│   │   ├── reranking/                   # Cross-encoder & fake rerankers
│   │   ├── search/                      # Web search provider integrations
│   │   └── storage/                     # MinIO object storage client
│   ├── ingestion/                       # Parsers, normalizers & chunking engines
│   │   ├── parsers/                     # PDF, DOCX, HTML, TXT extractors
│   │   ├── normalization/               # Canonical document transformations
│   │   └── chunking/                    # Structure-aware chunking strategies
│   ├── retrieval/                       # Vector, Lexical FTS & Hybrid RRF services
│   ├── rag/                             # Grounded QA service & ContextBuilder
│   ├── agent/                           # LangGraph nodes, edges, state & graph
│   ├── memory/                          # Scoped memory service & session manager
│   ├── drafting/                        # Document drafting service (dual-channel)
│   ├── search/                          # External web search & trust filtering
│   ├── verification/                    # Claim grounding & citation verifier
│   ├── evaluation/                      # Benchmark dataset & IR metric calculators
│   └── api/                             # Versioned REST controllers
│       ├── router.py                    # Consolidated API router
│       └── routes/                      # Documents, RAG, Agent, Drafting, Memory, Monitoring
├── docs/                                # Production architecture & ops runbooks
│   ├── kubernetes_architecture.md       # K8s Deployment, HPA & Ingress manifests
│   ├── operational_runbook.md           # Backup, disaster recovery & secret rotation
│   └── security_baseline.md             # Security audit & defense-in-depth checklist
├── observability/
│   └── prometheus.yml                   # Prometheus scraper configuration
├── prompts/                             # Version-controlled system & user prompt templates
├── tests/                               # Comprehensive unit & integration test suites
│   ├── unit/                            # 281 passing unit tests (100% offline & mocked)
│   └── integration/                     # Database & object storage integration tests
├── alembic/                             # Alembic migrations & version history
├── docker-compose.yml                   # Multi-container enterprise local runtime
├── Dockerfile                           # Production container definition
├── pyproject.toml                       # Dependencies and package configuration
└── README.md                            # Comprehensive documentation
```

---

## 🚀 Quick Start & Local Setup

### Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- Docker & Docker Compose (optional for database & services)

### 1. Clone the Repository
```bash
git clone https://github.com/sepyHR80/law_copilot.git
cd law_copilot
```

### 2. Install Dependencies
Using `uv`:
```bash
uv sync
```

### 3. Environment Configuration
Copy `.env.example` to `.env` or set environment variables:
```bash
cp .env.example .env
```
Key configuration parameters in `.env`:
```ini
APP_ENV=development
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/law_copilot
STORAGE_BACKEND=minio
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_DOCUMENTS=law-documents

# Embedding & LLM
OPENAI_API_KEY=your-api-key
EMBEDDING_MODEL=text-embedding-3-small
LLM_DEFAULT_MODEL=gpt-4o-mini
```

### 4. Run Database Migrations
```bash
uv run alembic upgrade head
```

### 5. Start the Development Server
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API is now live at: `http://localhost:8000`  
Interactive OpenAPI Documentation (Swagger UI): `http://localhost:8000/docs`

---

## 🐳 Running with Docker Compose

To launch the full enterprise stack (FastAPI, PostgreSQL with pgvector, MinIO, Prometheus, Jaeger, and Grafana):

```bash
docker-compose up -d --build
```

### Exposed Service Endpoints
| Service | URL | Default Credentials |
|---|---|---|
| **FastAPI Backend** | `http://localhost:8000` | None |
| **API Docs (Swagger UI)** | `http://localhost:8000/docs` | None |
| **MinIO Console** | `http://localhost:9001` | `minioadmin` / `minioadmin` |
| **Prometheus** | `http://localhost:9090` | None |
| **Jaeger Tracing UI** | `http://localhost:16686` | None |
| **Grafana Dashboards** | `http://localhost:3000` | `admin` / `admin` |

---

## 📡 API Reference & Usage Examples

### 1. Document Upload
Upload contracts, statutes, or past complaints in PDF, DOCX, TXT, or HTML:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample_contract.pdf" \
  -F "document_type=contract" \
  -F "knowledge_type=factual" \
  -F "title=Master Services Agreement"
```

### 2. Grounded Legal QA (RAG)
Query uploaded legal documents with verified citation provenance:
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the termination conditions specified in Section 4?",
    "top_k": 5
  }'
```
**Sample Response:**
```json
{
  "question": "What are the termination conditions specified in Section 4?",
  "answer": "Either party may terminate the agreement upon 30 days prior written notice in the event of a material breach.",
  "citations": [
    {
      "document_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "chunk_id": "4a2b9f81-5c3e-4b2a-8c9e-1f2e3d4c5b6a",
      "page": 4,
      "section": "Section 4.1 Termination for Cause",
      "snippet": "Either party may terminate this Agreement upon thirty (30) days written notice..."
    }
  ],
  "is_sufficient": true,
  "model": "gpt-4o-mini"
}
```

### 3. Legal Agent Query
Execute multi-turn conversational reasoning with dynamic intent routing and bounded self-correction:
```bash
curl -X POST "http://localhost:8000/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can the consultant demand additional compensation for out-of-scope work?",
    "max_retries": 2,
    "enable_external_search": false
  }'
```

### 4. Document Drafting (Dual-Channel)
Draft a formal agreement or notice separating substantive law from stylistic templates:
```bash
curl -X POST "http://localhost:8000/api/v1/drafting/draft" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Non-Disclosure Agreement for Technology Transfer",
    "document_type": "contract",
    "recipient": "Acme Innovations Corp.",
    "jurisdiction": "California",
    "factual_top_k": 5,
    "style_top_k": 3,
    "additional_instructions": "Include a strict 3-year confidentiality term and mutual indemnification."
  }'
```

### 5. Health, Readiness & Metrics
```bash
# Liveness probe
curl http://localhost:8000/healthz

# Readiness probe (verifies database & downstream connections)
curl http://localhost:8000/readyz

# Prometheus metrics scrape endpoint
curl http://localhost:8000/metrics
```

---

## 📈 Evaluation & Benchmark System

Law Copilot includes a built-in Information Retrieval (IR) and Generation evaluation engine ([`app/evaluation/`](file:///Users/speyhr/Downloads/law/law-copilot/app/evaluation/)).

### Metrics Supported
- **Precision@k**: $\frac{|\text{Relevant Chunks} \cap \text{Top-}k \text{ Retrieved}|}{k}$
- **Recall@k**: $\frac{|\text{Relevant Chunks} \cap \text{Top-}k \text{ Retrieved}|}{|\text{Total Relevant Chunks}|}$
- **Mean Reciprocal Rank (MRR)**: $\frac{1}{\text{rank}_1}$ of the first relevant document.
- **NDCG@k**: Normalized Discounted Cumulative Gain with logarithmic position penalties.
- **Faithfulness**: Grounding ratio checking that claims made in the answer are strictly supported by evidence keywords.
- **Citation Completeness**: Verification that all required statutory chunks are cited.

### Running the Evaluation Suite
```bash
uv run python -c "
import asyncio
from app.evaluation.runner import BenchmarkEvaluationRunner
runner = BenchmarkEvaluationRunner()
print(asyncio.run(runner.run_suite()))
"
```

---

## 🔍 Observability, Tracing & Metrics

### Distributed Tracing with OpenTelemetry & Jaeger
Law Copilot generates structured spans covering every execution step:
```text
HTTP /api/v1/agent/query
  ├── load_memory
  ├── analyze_intent
  ├── retrieve_knowledge (Hybrid: Vector + GIN FTS)
  ├── cross_encoder_rerank
  ├── build_context
  ├── llm_complete
  └── verify_answer (Claim & Citation Validation)
```
View live traces in Jaeger UI at `http://localhost:16686`.

### Prometheus Metrics
Exposed at `GET /metrics`:
- `law_copilot_requests_total{endpoint, status}`: Request throughput by route and status code.
- `law_copilot_request_duration_seconds`: Histogram of operational latencies.
- `law_copilot_llm_tokens_total{model, type}`: Real-time LLM prompt and completion token accounting.
- `law_copilot_verification_failures_total`: Counter for hallucination and citation mismatches caught by the verifier.
- `law_copilot_web_fallback_total`: Count of external search queries triggered.

---

## 🔒 Security Baseline

As audited in [`docs/security_baseline.md`](docs/security_baseline.md):
- **100% Parameterized Queries**: SQLAlchemy core expressions prevent SQL injection.
- **Safe Object Storage Keys**: Deterministic UUID/SHA-256 hashed keys eliminate directory traversal (`../`) vulnerabilities.
- **Strict File Type Validation**: Magic byte inspection and MIME-type validation restrict uploads to authorized document types.
- **Automatic Secret Scrubbing**: Recursively redacts tokens and passwords from logs and spans.
- **Bounded Resource Consumption**: Strict maximum upload sizes (50MB ceiling) and execution timeouts.

---

## ☸️ Production & Kubernetes Deployment

For enterprise container orchestration, production manifests are provided in [`docs/kubernetes_architecture.md`](docs/kubernetes_architecture.md):
- **Deployments & Services**: 3-replica API deployment with zero-downtime rolling update strategy.
- **Horizontal Pod Autoscaler (HPA)**: Automatic scaling based on CPU utilization and request concurrency.
- **Health Probes**: Configured `/healthz` (liveness) and `/readyz` (readiness) probes.
- **Operational Runbook**: Disaster recovery, PostgreSQL + pgvector backups, and secret rotation guidelines in [`docs/operational_runbook.md`](docs/operational_runbook.md).

---

## ✅ Testing & Verification

The test suite contains **281 passing tests** with 0 failures, covering all unit, domain, infrastructure, and agent behaviors:

```bash
# Run all unit and integration tests
uv run pytest -v

# Run specific stage tests
uv run pytest tests/unit/test_drafting.py -v   # Stage 18
uv run pytest tests/unit/test_stage19.py -v    # Stage 19
uv run pytest tests/unit/test_stage20.py -v    # Stage 20

# Verify Alembic database migrations
uv run alembic current
```

---

## 📄 License
Proprietary & Confidential. Developed for enterprise legal compliance.
