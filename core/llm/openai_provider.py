from __future__ import annotations

import json
import re
from typing import TypeVar

from openai import BadRequestError, OpenAI

from pydantic import BaseModel

from core.llm.base import LLMProvider, build_system_prompt

T = TypeVar("T", bound=BaseModel)

_STRATEGY_STRICT = "strict"
_STRATEGY_NO_STRICT = "no_strict"
_STRATEGY_PROMPT = "prompt"


class OpenAIProvider(LLMProvider):

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str | None = None,
        disable_thinking: bool = True,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.disable_thinking = disable_thinking
        self._strategy: str | None = None

    def extract(self, text: str, schema: type[T], system_prompt: str = "") -> T:
        system = build_system_prompt(schema, override=system_prompt)
        user = f"/no_think\n{text}" if self.disable_thinking else text

        if self._strategy == _STRATEGY_STRICT:
            raw = self._try_structured(user, system, schema, strict=True)
            return schema.model_validate_json(raw)

        if self._strategy == _STRATEGY_NO_STRICT:
            raw = self._try_structured(user, system, schema, strict=False)
            return schema.model_validate_json(raw)

        if self._strategy == _STRATEGY_PROMPT:
            raw = self._prompt_extract(user, system, schema)
            return schema.model_validate_json(raw)

        raw = self._try_structured(user, system, schema, strict=True)
        if raw:
            self._strategy = _STRATEGY_STRICT
            return schema.model_validate_json(raw)

        raw = self._try_structured(user, system, schema, strict=False)
        if raw:
            self._strategy = _STRATEGY_NO_STRICT
            return schema.model_validate_json(raw)

        raw = self._prompt_extract(user, system, schema)
        if not raw:
            raise RuntimeError(
                "All extraction strategies failed. "
                "Check the model is loaded and accessible."
            )

        self._strategy = _STRATEGY_PROMPT
        return schema.model_validate_json(raw)

    def _try_structured(self, user: str, system: str, schema: type[T], strict: bool) -> str:
        try:
            schema_obj: dict = {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            }
            if strict:
                schema_obj["strict"] = True

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_schema", "json_schema": schema_obj},
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()

        except (BadRequestError, Exception) as exc:
            print(f"[openai] structured({'strict' if strict else 'no-strict'}) failed: {exc}")
            return ""

    def _prompt_extract(self, user: str, system: str, schema: type[T]) -> str:
        field_lines = []
        example: dict = {}

        for name, field_info in schema.model_fields.items():
            desc = field_info.description or ""
            field_lines.append(f"  - {name}: {desc}")
            annotation = str(field_info.annotation)
            if "int" in annotation:
                example[name] = 0
            elif "float" in annotation:
                example[name] = 0.0
            else:
                example[name] = name.upper()

        fields_str = "\n".join(field_lines)
        example_str = json.dumps(example, indent=2)

        rich_system = (
            f"{system}\n\n"
            "Extract the following fields and return ONLY a JSON object. "
            "No explanation, no markdown:\n\n"
            f"Fields:\n{fields_str}\n\n"
            f"Example output format (replace placeholders with real values):\n"
            f"{example_str}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": rich_system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            return self._clean_json(raw)

        except BadRequestError as exc:
            print(f"[openai] prompt-only failed: {exc}")
            return ""

    @staticmethod
    def _clean_json(raw: str) -> str:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return match.group(0).strip() if match else raw.strip()
