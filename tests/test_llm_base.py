from __future__ import annotations

from core.llm.base import build_system_prompt, _GENERIC_SYSTEM_PROMPT
from pydantic import BaseModel


class SchemaWithPrompt(BaseModel):
    value: str

    @classmethod
    def system_prompt(cls) -> str:
        return "custom prompt for this schema"


class SchemaWithoutPrompt(BaseModel):
    value: str


def test_uses_schema_classmethod_when_present():
    result = build_system_prompt(SchemaWithPrompt)
    assert result == "custom prompt for this schema"


def test_falls_back_to_generic_when_no_classmethod():
    result = build_system_prompt(SchemaWithoutPrompt)
    assert result == _GENERIC_SYSTEM_PROMPT


def test_override_takes_priority_over_classmethod():
    result = build_system_prompt(SchemaWithPrompt, override="override prompt")
    assert result == "override prompt"


def test_override_takes_priority_over_generic():
    result = build_system_prompt(SchemaWithoutPrompt, override="override prompt")
    assert result == "override prompt"


def test_empty_override_falls_through_to_classmethod():
    result = build_system_prompt(SchemaWithPrompt, override="")
    assert result == "custom prompt for this schema"
