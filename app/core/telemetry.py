"""OpenTelemetry tracing and secret scrubbing for Law Copilot.

Per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 24 and 25:
- Clean distributed tracing across API -> Agent -> Retrieval -> Reranker -> LLM -> Verification.
- Zero secret leakage: automatically scrubs API keys, authorization headers, passwords, and tokens.
"""

from contextlib import contextmanager
import logging
import re
import time
from typing import Any, Dict, Iterator, Optional
import uuid

logger = logging.getLogger("law_copilot.telemetry")

# Patterns for identifying and masking sensitive data
SENSITIVE_KEY_PATTERNS = {
    "key",
    "secret",
    "password",
    "token",
    "auth",
    "authorization",
    "credential",
    "api_key",
    "access_key",
    "secret_key",
    "private_key",
}

BEARER_PATTERN = re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{10,}", re.IGNORECASE)
API_KEY_PATTERN = re.compile(r"(?:sk-[a-zA-Z0-9]{20,}|key-[a-zA-Z0-9]{20,})", re.IGNORECASE)


def scrub_secrets(obj: Any) -> Any:
    """Recursively scrub sensitive keys and token values from dictionaries, lists, and strings."""
    if isinstance(obj, dict):
        clean_dict = {}
        for k, v in obj.items():
            k_lower = str(k).lower()
            if any(s in k_lower for s in SENSITIVE_KEY_PATTERNS):
                clean_dict[k] = "[REDACTED]"
            else:
                clean_dict[k] = scrub_secrets(v)
        return clean_dict

    if isinstance(obj, list):
        return [scrub_secrets(item) for item in obj]

    if isinstance(obj, str):
        masked = BEARER_PATTERN.sub("Bearer [REDACTED]", obj)
        masked = API_KEY_PATTERN.sub("[REDACTED_API_KEY]", masked)
        return masked

    return obj


class TelemetrySpan:
    """Represents an active OpenTelemetry-compatible span."""

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id or uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_span_id = parent_span_id
        self.attributes = scrub_secrets(attributes or {})
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0
        self.status = "OK"
        self.error: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a sanitized attribute on the span."""
        scrubbed = scrub_secrets({key: value})
        self.attributes.update(scrubbed)

    def finish(self, status: str = "OK", error: Optional[str] = None) -> None:
        """Complete the span execution."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.status = status
        self.error = error


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Iterator[TelemetrySpan]:
    """Context manager for tracing an operational span with automatic sanitization."""
    span = TelemetrySpan(name=name, attributes=attributes, trace_id=trace_id)
    try:
        yield span
        span.finish(status="OK")
    except Exception as exc:
        span.finish(status="ERROR", error=str(exc))
        raise
