from __future__ import annotations

from typing import TypeVar

from google import genai
from google.genai import types

from pydantic import BaseModel

from core.llm.base import LLMProvider, build_system_prompt

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(LLMProvider):

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = ""):
        self.model_name = model
        self.client = genai.Client(api_key=api_key)

    def extract(self, text: str, schema: type[T], system_prompt: str = "") -> T:
        system = build_system_prompt(schema, override=system_prompt)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return schema.model_validate_json(raw)
