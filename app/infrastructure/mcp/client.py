"""Client for invoking MCP servers and tools."""

from typing import Any, Dict
from app.domain.mcp.exceptions import MCPToolExecutionError
from app.infrastructure.mcp.knowledge_server import KnowledgeMCPServer


class KnowledgeMCPClient:
    """In-process client providing a standardized tool boundary to KnowledgeMCPServer."""

    def __init__(self, server: KnowledgeMCPServer) -> None:
        self.server = server

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool on the MCP server."""
        return await self.server.execute_tool(tool_name=tool_name, arguments=arguments)
