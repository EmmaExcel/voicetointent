from __future__ import annotations

from typing import TypeVar

from ollama import Client

from pydantic import BaseModel

from core.llm.base import LLMProvider, build_system_prompt

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(LLMProvider):

    def __init__(self, model: str = "qwen:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = Client(host=base_url)

    def extract(self, text: str, schema: type[T], system_prompt: str = "") -> T:
        system = build_system_prompt(schema, override=system_prompt)

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            format=schema.model_json_schema(),
            options={"temperature": 0.0},
        )

        return schema.model_validate_json(response.message.content)
