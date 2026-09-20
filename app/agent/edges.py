"""Conditional routing edges for LangGraph legal agent."""

from typing import Literal
from langgraph.graph import END

from app.agent.state import AgentState


def route_intent(state: AgentState) -> str:
    """Route based on user intent."""
    if state.get("intent") == "general":
        return "handle_general"
    return "prepare_query"


def route_evidence(state: AgentState) -> str:
    """Route based on retrieved evidence sufficiency."""
    if not state.get("is_sufficient", True) or not state.get("retrieval_results"):
        return "handle_insufficient"
    return "build_context"


def route_verification(state: AgentState) -> str:
    """Route based on verification pass/fail with bounded retry count.

    Guarantees no infinite loops by terminating when retry_count >= max_retries.
    """
    if state.get("verification_passed", True):
        return END

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        return "repair_draft"

    # Max retries exceeded: force termination to prevent infinite loops
    return END
