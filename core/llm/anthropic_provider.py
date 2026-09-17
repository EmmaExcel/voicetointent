from __future__ import annotations

import json
from typing import TypeVar

import anthropic

from pydantic import BaseModel

from core.llm.base import LLMProvider, build_system_prompt

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(LLMProvider):

    def __init__(self, model: str = "claude-3-5-haiku-latest", api_key: str = ""):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract(self, text: str, schema: type[T], system_prompt: str = "") -> T:
        system = build_system_prompt(schema, override=system_prompt)
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        tool_definition = {
            "name": "extract_intent",
            "description": "Extract structured intent from the user's spoken command.",
            "input_schema": schema.model_json_schema(),
        }

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=(
                f"{system}\n\n"
                f"The output must conform to this JSON schema:\n{schema_json}"
            ),
            tools=[tool_definition],
            tool_choice={"type": "tool", "name": "extract_intent"},
            messages=[{"role": "user", "content": text}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_intent":
                return schema.model_validate(block.input)

        raise RuntimeError("Anthropic did not return a tool_use block.")
