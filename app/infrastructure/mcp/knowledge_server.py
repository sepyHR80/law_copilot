"""Knowledge MCP Server implementation.

Provides the search_knowledge tool boundary per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 21.1.
"""

from typing import Any, Dict, List, Optional
from app.domain.mcp.exceptions import MCPToolExecutionError
from app.domain.mcp.models import (
    MCPEvidenceItem,
    MCPSearchKnowledgeRequest,
    MCPSearchKnowledgeResponse,
)
from app.domain.retrieval.models import HybridSearchQuery, RetrievalFilter
from app.domain.retrieval.protocol import HybridRetrieverProtocol, RerankerProtocol


class KnowledgeMCPServer:
    """Primary Knowledge MCP Server exposing the search_knowledge capability."""

    def __init__(
        self,
        hybrid_retriever: HybridRetrieverProtocol,
        reranker: Optional[RerankerProtocol] = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return standardized tool declarations exposed by this MCP server."""
        return [
            {
                "name": "search_knowledge",
                "description": (
                    "Search authoritative internal legal knowledge base. "
                    "Returns structured evidence items with verified provenance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text."},
                        "knowledge_type": {
                            "type": "string",
                            "enum": ["factual", "stylistic"],
                            "default": "factual",
                            "description": "Whether to retrieve factual authority or stylistic references.",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 8,
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Number of evidence items to return.",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional metadata filters for narrowing retrieval candidates.",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    async def search_knowledge(
        self,
        request: MCPSearchKnowledgeRequest,
    ) -> MCPSearchKnowledgeResponse:
        """Execute the search_knowledge tool and return structured evidence."""
        try:
            # Build filters
            effective_filters = request.filters or RetrievalFilter()
            effective_filters.knowledge_type = request.knowledge_type

            search_query = HybridSearchQuery(
                text_query=request.query,
                top_k=request.top_k,
                candidate_k=max(20, request.top_k * 3),
                filters=effective_filters,
            )

            candidates = await self.hybrid_retriever.search(search_query)

            if candidates and self.reranker:
                candidates = self.reranker.rerank(
                    query=request.query,
                    candidates=candidates,
                    top_n=request.top_k,
                )

            evidence_items: List[MCPEvidenceItem] = []
            for cand in candidates:
                src = cand.source or {}
                page = src.get("page")
                section = src.get("section")
                evidence_items.append(
                    MCPEvidenceItem(
                        chunk_id=cand.chunk_id,
                        document_id=cand.document_id,
                        document_version_id=cand.document_version_id,
                        content=cand.content,
                        score=cand.score,
                        page=int(page) if page is not None and str(page).isdigit() else None,
                        section=str(section) if section else None,
                        knowledge_type=request.knowledge_type,
                        source=dict(src),
                    )
                )

            return MCPSearchKnowledgeResponse(
                evidence=evidence_items,
                total_count=len(evidence_items),
                knowledge_type=request.knowledge_type or "factual",
            )
        except Exception as exc:
            raise MCPToolExecutionError(f"Failed to execute search_knowledge: {exc}") from exc

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool call by name."""
        if tool_name == "search_knowledge":
            req = MCPSearchKnowledgeRequest.model_validate(arguments)
            response = await self.search_knowledge(req)
            return response.model_dump(mode="json")
        raise MCPToolExecutionError(f"Unknown MCP tool: '{tool_name}'")
