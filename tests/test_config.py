from __future__ import annotations

import pytest


def test_ollama_provider_created(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen:4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    import config
    config.get_llm_provider.cache_clear()

    from unittest.mock import patch
    with patch("core.llm.ollama_provider.Client"):
        provider = config.get_llm_provider()
        from core.llm.ollama_provider import OllamaProvider
        assert isinstance(provider, OllamaProvider)

    config.get_llm_provider.cache_clear()


def test_openai_provider_created(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    import config
    config.get_llm_provider.cache_clear()

    from unittest.mock import patch
    with patch("core.llm.openai_provider.OpenAI"):
        provider = config.get_llm_provider()
        from core.llm.openai_provider import OpenAIProvider
        assert isinstance(provider, OpenAIProvider)

    config.get_llm_provider.cache_clear()


def test_openai_raises_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import config
    config.get_llm_provider.cache_clear()

    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        config.get_llm_provider()

    config.get_llm_provider.cache_clear()


def test_gemini_raises_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import config
    config.get_llm_provider.cache_clear()

    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        config.get_llm_provider()

    config.get_llm_provider.cache_clear()


def test_anthropic_raises_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import config
    config.get_llm_provider.cache_clear()

    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        config.get_llm_provider()

    config.get_llm_provider.cache_clear()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")

    import config
    config.get_llm_provider.cache_clear()

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        config.get_llm_provider()

    config.get_llm_provider.cache_clear()
