"""Fake LLM Provider for deterministic offline testing."""

import json
from typing import Any, Callable, List, Optional
from pydantic import BaseModel

from app.domain.llm.exceptions import LLMResponseValidationError
from app.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from app.domain.llm.protocol import LLMProvider


class FakeLLMProvider(LLMProvider):
    """Deterministic in-memory LLM provider for tests.

    Never makes external network calls or requires API keys.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        custom_handler: Optional[Callable[[LLMRequest], LLMResponse]] = None,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._custom_handler = custom_handler
        self.call_history: List[LLMRequest] = []
        self._closed = False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a deterministic response and record request in history."""
        # Automatic handling for intent/category classification requests during agent tests
        is_intent_query = any(
            "تحلیل قصد" in (m.content or "") or "دسته‌بندی موضوعی" in (m.content or "")
            for m in request.messages
        )
        if is_intent_query:
            user_msg = request.messages[-1].content.lower() if request.messages else ""
            if any(w in user_msg for w in ["hello", "سلام", "درود", "hi", "hey", "چه کار", "چطور", "کمک"]):
                intent_val = "general"
                cat_val = "گفتگوی عمومی و راهنمایی"
            elif any(w in user_msg for w in ["draft", "تنظیم", "قرارداد", "دادخواست"]):
                intent_val = "document_generation"
                cat_val = "تنظیم و تدوین اسناد حقوقی"
            else:
                intent_val = "legal_qa"
                cat_val = "حقوق مدنی و قراردادها"
            fake_json = json.dumps({"intent": intent_val, "category": cat_val})
            return LLMResponse(
                content=fake_json,
                model="fake-model",
                usage=LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                finish_reason="stop",
            )

        self.call_history.append(request)

        if self._custom_handler:
            return self._custom_handler(request)

        content = self._responses.pop(0) if self._responses else "Default fake LLM response."
        parsed_result: Optional[Any] = None

        if request.response_format is not None:
            if isinstance(request.response_format, type) and issubclass(request.response_format, BaseModel):
                try:
                    parsed_json = json.loads(content)
                    parsed_result = request.response_format.model_validate(parsed_json)
                except Exception as exc:
                    raise LLMResponseValidationError(
                        f"Failed to validate response against schema {request.response_format.__name__}: {exc}"
                    ) from exc

        return LLMResponse(
            content=content,
            parsed=parsed_result,
            model=request.model or "fake-model",
            usage=LLMUsage(
                prompt_tokens=10,
                completion_tokens=len(content.split()),
                total_tokens=10 + len(content.split()),
            ),
            finish_reason="stop",
        )

    async def close(self) -> None:
        """Close fake provider."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed
