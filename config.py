from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class WhisperConfig:
    model: str = os.getenv("WHISPER_MODEL", "tiny.en")
    device: str = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")


class MicConfig:
    silence_threshold_seconds: float = float(
        os.getenv("MIC_SILENCE_THRESHOLD_SECONDS", "3.0")
    )
    silence_rms_threshold: float = float(
        os.getenv("MIC_SILENCE_RMS_THRESHOLD", "0.005")
    )
    sample_rate: int = int(os.getenv("MIC_SAMPLE_RATE", "16000"))


@lru_cache(maxsize=1)
def get_llm_provider():
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider_name == "ollama":
        from core.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "qwen:4b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider_name == "lmstudio":
        from core.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=os.getenv("LMSTUDIO_MODEL", "qwen/qwen3.5-9b"),
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        )

    if provider_name == "openai":
        from core.llm.openai_provider import OpenAIProvider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")
        return OpenAIProvider(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
        )

    if provider_name == "gemini":
        from core.llm.gemini_provider import GeminiProvider
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set in .env")
        return GeminiProvider(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            api_key=api_key,
        )

    if provider_name == "anthropic":
        from core.llm.anthropic_provider import AnthropicProvider
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set in .env")
        return AnthropicProvider(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider_name}'. "
        "Choose from: ollama, lmstudio, openai, gemini, anthropic"
    )
