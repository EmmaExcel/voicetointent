from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from schemas.financial import TransactionIntent


MOCK_INTENT = TransactionIntent(
    intent="send_money",
    amount=5000,
    currency="NGN",
    recipient={
        "id": "ben_001",
        "full_name": "Jane Doe",
        "account_number": "0123456789",
        "bank_name": "Guaranty Trust Bank (GTB)",
    },
)

MOCK_INTENT_JSON = MOCK_INTENT.model_dump_json()


class TestOllamaProvider:

    def test_extract_calls_chat_and_validates(self):
        from core.llm.ollama_provider import OllamaProvider

        mock_client = MagicMock()
        mock_client.chat.return_value.message.content = MOCK_INTENT_JSON

        provider = OllamaProvider.__new__(OllamaProvider)
        provider.model = "qwen:4b"
        provider.client = mock_client

        result = provider.extract("Send 5k to mum", TransactionIntent)

        assert result.intent == "send_money"
        assert result.amount == 5000
        mock_client.chat.assert_called_once()


class TestOpenAIProvider:

    def test_extract_strict_strategy(self):
        from core.llm.openai_provider import OpenAIProvider, _STRATEGY_STRICT

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = MOCK_INTENT_JSON
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.model = "gpt-4o-mini"
        provider.client = mock_client
        provider.disable_thinking = False
        provider._strategy = _STRATEGY_STRICT

        result = provider.extract("Send 5k to mum", TransactionIntent)

        assert result.amount == 5000

    def test_clean_json_strips_markdown(self):
        from core.llm.openai_provider import OpenAIProvider

        raw = "```json\n{\"key\": \"value\"}\n```"
        assert OpenAIProvider._clean_json(raw) == '{"key": "value"}'

    def test_clean_json_strips_think_tags(self):
        from core.llm.openai_provider import OpenAIProvider

        raw = "<think>reasoning here</think>{\"key\": \"value\"}"
        assert OpenAIProvider._clean_json(raw) == '{"key": "value"}'


class TestGeminiProvider:

    def test_extract_validates_response(self):
        from core.llm.gemini_provider import GeminiProvider

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = MOCK_INTENT_JSON

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.model_name = "gemini-1.5-flash"
        provider.client = mock_client

        result = provider.extract("Send 5k to mum", TransactionIntent)

        assert result.intent == "send_money"
        assert result.recipient.full_name == "Jane Doe"


class TestAnthropicProvider:

    def test_extract_uses_tool_use_block(self):
        from core.llm.anthropic_provider import AnthropicProvider

        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.name = "extract_intent"
        mock_block.input = MOCK_INTENT.model_dump()

        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [mock_block]

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-3-5-haiku-latest"
        provider.client = mock_client

        result = provider.extract("Send 5k to mum", TransactionIntent)

        assert result.amount == 5000

    def test_extract_raises_when_no_tool_block(self):
        from core.llm.anthropic_provider import AnthropicProvider

        mock_block = MagicMock()
        mock_block.type = "text"

        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [mock_block]

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-3-5-haiku-latest"
        provider.client = mock_client

        with pytest.raises(RuntimeError, match="tool_use"):
            provider.extract("Send 5k to mum", TransactionIntent)
