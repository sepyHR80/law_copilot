"""Unit tests for Stage 14 — LLM Service + LiteLLM integration."""

import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock
import httpx
import openai
import pytest
from pydantic import BaseModel, Field

from app.domain.llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from app.domain.llm.models import ChatMessage, LLMRequest, LLMResponse, LLMUsage
from app.domain.llm.protocol import LLMProvider
from app.infrastructure.llm.fake import FakeLLMProvider
from app.infrastructure.llm.openai_provider import OpenAILLMProvider
from app.llm.service import LLMService


class SampleLegalSummary(BaseModel):
    summary: str
    jurisdiction: str
    key_points: List[str] = Field(default_factory=list)


class TestLLMProtocolConformity:
    """Verify implementations conform to LLMProvider protocol."""

    def test_fake_llm_implements_protocol(self) -> None:
        provider = FakeLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_openai_provider_implements_protocol(self) -> None:
        provider = OpenAILLMProvider(base_url="http://mock-url/v1", api_key="test-key")
        assert isinstance(provider, LLMProvider)


class TestFakeLLMProvider:
    """Test FakeLLMProvider in-memory behavior."""

    @pytest.mark.asyncio
    async def test_basic_canned_generation(self) -> None:
        canned = ["First response", "Second response"]
        provider = FakeLLMProvider(responses=canned)

        req1 = LLMRequest(messages=[ChatMessage(role="user", content="Hello")])
        res1 = await provider.generate(req1)
        assert res1.content == "First response"
        assert len(provider.call_history) == 1

        req2 = LLMRequest(messages=[ChatMessage(role="user", content="World")])
        res2 = await provider.generate(req2)
        assert res2.content == "Second response"
        assert len(provider.call_history) == 2

    @pytest.mark.asyncio
    async def test_structured_response_parsing(self) -> None:
        valid_json = json.dumps({
            "summary": "Termination clause review",
            "jurisdiction": "DE",
            "key_points": ["30 days notice required", "Written notice only"],
        })
        provider = FakeLLMProvider(responses=[valid_json])

        req = LLMRequest(
            messages=[ChatMessage(role="user", content="Extract summary")],
            response_format=SampleLegalSummary,
        )
        res = await provider.generate(req)

        assert isinstance(res.parsed, SampleLegalSummary)
        assert res.parsed.summary == "Termination clause review"
        assert res.parsed.jurisdiction == "DE"
        assert len(res.parsed.key_points) == 2

    @pytest.mark.asyncio
    async def test_structured_response_validation_failure(self) -> None:
        invalid_json = "not valid json"
        provider = FakeLLMProvider(responses=[invalid_json])

        req = LLMRequest(
            messages=[ChatMessage(role="user", content="Extract summary")],
            response_format=SampleLegalSummary,
        )
        with pytest.raises(LLMResponseValidationError):
            await provider.generate(req)

    @pytest.mark.asyncio
    async def test_close_lifecycle(self) -> None:
        provider = FakeLLMProvider()
        assert not provider.is_closed
        await provider.close()
        assert provider.is_closed


class TestOpenAILLMProvider:
    """Test OpenAILLMProvider using injected mock client (fully offline)."""

    def test_configuration_validation(self) -> None:
        with pytest.raises(LLMConfigurationError, match="base_url must not be empty"):
            OpenAILLMProvider(base_url="  ")

        with pytest.raises(LLMConfigurationError, match="timeout must be greater than 0"):
            OpenAILLMProvider(base_url="http://mock/v1", timeout=0.0)

        with pytest.raises(LLMConfigurationError, match="default_model must not be empty"):
            OpenAILLMProvider(base_url="http://mock/v1", default_model="  ")

    @pytest.mark.asyncio
    async def test_empty_messages_raises_error(self) -> None:
        mock_client = AsyncMock()
        provider = OpenAILLMProvider(base_url="http://mock/v1", client=mock_client)
        req = LLMRequest(messages=[])

        with pytest.raises(LLMConfigurationError, match="at least one message"):
            await provider.generate(req)

    @pytest.mark.asyncio
    async def test_standard_completion(self) -> None:
        mock_client = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Analysis of legal contract."
        mock_choice.finish_reason = "stop"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "gpt-4o-mini"
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 25
        mock_completion.usage.total_tokens = 40

        mock_client.chat.completions.create.return_value = mock_completion

        provider = OpenAILLMProvider(
            base_url="http://mock/v1",
            default_model="gpt-4o-mini",
            client=mock_client,
        )

        req = LLMRequest(
            messages=[
                ChatMessage(role="system", content="You are a legal assistant."),
                ChatMessage(role="user", content="Analyze clause."),
            ],
            temperature=0.2,
            max_tokens=200,
        )
        res = await provider.generate(req)

        assert res.content == "Analysis of legal contract."
        assert res.model == "gpt-4o-mini"
        assert res.finish_reason == "stop"
        assert res.usage is not None
        assert res.usage.total_tokens == 40

        # Verify call arguments
        mock_client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": "Analyze clause."},
            ],
            temperature=0.2,
            timeout=60.0,
            max_tokens=200,
        )

    @pytest.mark.asyncio
    async def test_structured_output_with_beta_parse(self) -> None:
        mock_client = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"summary": "Test", "jurisdiction": "FR", "key_points": []}'
        expected_obj = SampleLegalSummary(summary="Test", jurisdiction="FR", key_points=[])
        mock_choice.message.parsed = expected_obj
        mock_choice.finish_reason = "stop"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "gpt-4o-mini"
        mock_completion.usage.prompt_tokens = 20
        mock_completion.usage.completion_tokens = 10
        mock_completion.usage.total_tokens = 30

        mock_client.beta.chat.completions.parse.return_value = mock_completion

        provider = OpenAILLMProvider(
            base_url="http://mock/v1",
            client=mock_client,
        )

        req = LLMRequest(
            messages=[ChatMessage(role="user", content="Extract summary")],
            response_format=SampleLegalSummary,
        )
        res = await provider.generate(req)

        assert res.parsed == expected_obj
        mock_client.beta.chat.completions.parse.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_timeout_error_mapping(self) -> None:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=httpx.Request("POST", "http://mock/v1")
        )

        provider = OpenAILLMProvider(base_url="http://mock/v1", client=mock_client)
        req = LLMRequest(messages=[ChatMessage(role="user", content="Prompt")])

        with pytest.raises(LLMTimeoutError, match="timed out"):
            await provider.generate(req)

    @pytest.mark.asyncio
    async def test_rate_limit_error_mapping(self) -> None:
        mock_client = AsyncMock()
        mock_response = httpx.Response(429, request=httpx.Request("POST", "http://mock/v1"))
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": "Rate limit exceeded"},
        )

        provider = OpenAILLMProvider(base_url="http://mock/v1", client=mock_client)
        req = LLMRequest(messages=[ChatMessage(role="user", content="Prompt")])

        with pytest.raises(LLMRateLimitError, match="rate limit"):
            await provider.generate(req)

    @pytest.mark.asyncio
    async def test_connection_error_mapping(self) -> None:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=httpx.Request("POST", "http://mock/v1")
        )

        provider = OpenAILLMProvider(base_url="http://mock/v1", client=mock_client)
        req = LLMRequest(messages=[ChatMessage(role="user", content="Prompt")])

        with pytest.raises(LLMProviderError, match="Failed to connect"):
            await provider.generate(req)

    @pytest.mark.asyncio
    async def test_api_status_error_mapping(self) -> None:
        mock_client = AsyncMock()
        mock_response = httpx.Response(500, request=httpx.Request("POST", "http://mock/v1"))
        mock_client.chat.completions.create.side_effect = openai.APIStatusError(
            message="Internal Server Error",
            response=mock_response,
            body={"error": "Internal Server Error"},
        )

        provider = OpenAILLMProvider(base_url="http://mock/v1", client=mock_client)
        req = LLMRequest(messages=[ChatMessage(role="user", content="Prompt")])

        with pytest.raises(LLMProviderError, match="status 500"):
            await provider.generate(req)


class TestLLMService:
    """Test LLMService application layer."""

    @pytest.mark.asyncio
    async def test_service_complete_helper(self) -> None:
        fake_provider = FakeLLMProvider(responses=["Legal analysis completed."])
        service = LLMService(provider=fake_provider)

        res = await service.complete(
            prompt="Analyze clause",
            system_prompt="You are a legal expert.",
            model="custom-model",
            temperature=0.1,
        )

        assert res.content == "Legal analysis completed."
        assert len(fake_provider.call_history) == 1
        recorded_req = fake_provider.call_history[0]
        assert recorded_req.model == "custom-model"
        assert recorded_req.temperature == 0.1
        assert len(recorded_req.messages) == 2
        assert recorded_req.messages[0].role == "system"
        assert recorded_req.messages[0].content == "You are a legal expert."
        assert recorded_req.messages[1].role == "user"
        assert recorded_req.messages[1].content == "Analyze clause"

    @pytest.mark.asyncio
    async def test_service_structured_complete_helper(self) -> None:
        json_data = json.dumps({
            "summary": "Non-disclosure summary",
            "jurisdiction": "US",
            "key_points": ["Point 1"],
        })
        fake_provider = FakeLLMProvider(responses=[json_data])
        service = LLMService(provider=fake_provider)

        parsed_data, res = await service.structured_complete(
            prompt="Extract summary",
            schema=SampleLegalSummary,
        )

        assert isinstance(parsed_data, SampleLegalSummary)
        assert parsed_data.summary == "Non-disclosure summary"
        assert parsed_data.jurisdiction == "US"
        assert res.content == json_data

    @pytest.mark.asyncio
    async def test_service_close_delegation(self) -> None:
        fake_provider = FakeLLMProvider()
        service = LLMService(provider=fake_provider)

        assert not fake_provider.is_closed
        await service.close()
        assert fake_provider.is_closed
