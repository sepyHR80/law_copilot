"""Exceptions for MCP tools."""

from app.domain.documents.exceptions import DocumentError


class MCPError(DocumentError):
    """Base exception for MCP operations."""


class MCPToolExecutionError(MCPError):
    """Raised when an MCP tool execution fails."""
