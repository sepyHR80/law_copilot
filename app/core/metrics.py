"""Prometheus metrics registry and collector for Law Copilot.

Tracks essential operational engineering metrics per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 24.2:
- request count
- request latency
- token usage
- verification failure rate
- web fallback rate
"""

from collections import defaultdict
import threading
from typing import Dict, Tuple


class MetricsRegistry:
    """Thread-safe Prometheus metrics registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Counters
        self.requests_total: Dict[Tuple[str, int], int] = defaultdict(int)
        self.llm_tokens_total: Dict[Tuple[str, str], int] = defaultdict(int)  # (model, token_type) -> count
        self.verification_failures_total: int = 0
        self.web_fallback_total: int = 0

        # Latency buckets (in seconds): 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf
        self.latency_buckets = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        self.request_durations: Dict[str, list[float]] = defaultdict(list)

    def record_request(self, endpoint: str, status_code: int, duration_seconds: float) -> None:
        """Record an incoming API request completion."""
        with self._lock:
            self.requests_total[(endpoint, status_code)] += 1
            self.request_durations[endpoint].append(duration_seconds)

    def record_tokens(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM token usage."""
        with self._lock:
            self.llm_tokens_total[(model, "prompt")] += prompt_tokens
            self.llm_tokens_total[(model, "completion")] += completion_tokens

    def record_verification_failure(self) -> None:
        """Record a failed claim verification."""
        with self._lock:
            self.verification_failures_total += 1

    def record_web_fallback(self) -> None:
        """Record an external web search fallback trigger."""
        with self._lock:
            self.web_fallback_total += 1

    def export_prometheus_text(self) -> str:
        """Export metrics formatted according to Prometheus exposition format."""
        lines = []

        # 1. requests_total
        lines.append("# HELP law_copilot_requests_total Total number of HTTP requests processed.")
        lines.append("# TYPE law_copilot_requests_total counter")
        with self._lock:
            for (endpoint, code), count in self.requests_total.items():
                lines.append(f'law_copilot_requests_total{{endpoint="{endpoint}",status="{code}"}} {count}')

            if not self.requests_total:
                lines.append('law_copilot_requests_total{endpoint="total",status="200"} 0')

        # 2. llm_tokens_total
        lines.append("# HELP law_copilot_llm_tokens_total Total tokens consumed by LLM operations.")
        lines.append("# TYPE law_copilot_llm_tokens_total counter")
        with self._lock:
            for (model, t_type), count in self.llm_tokens_total.items():
                lines.append(f'law_copilot_llm_tokens_total{{model="{model}",type="{t_type}"}} {count}')

            if not self.llm_tokens_total:
                lines.append('law_copilot_llm_tokens_total{model="none",type="prompt"} 0')

        # 3. verification_failures_total
        lines.append("# HELP law_copilot_verification_failures_total Total claim verification failures.")
        lines.append("# TYPE law_copilot_verification_failures_total counter")
        lines.append(f"law_copilot_verification_failures_total {self.verification_failures_total}")

        # 4. web_fallback_total
        lines.append("# HELP law_copilot_web_fallback_total Total external web search fallback queries.")
        lines.append("# TYPE law_copilot_web_fallback_total counter")
        lines.append(f"law_copilot_web_fallback_total {self.web_fallback_total}")

        lines.append("")
        return "\n".join(lines)


# Global singleton registry
metrics = MetricsRegistry()
