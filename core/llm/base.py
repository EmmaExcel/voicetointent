from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_GENERIC_SYSTEM_PROMPT = (
    "You are a precise data extraction assistant processing voice-transcribed text. "
    "The input may contain speech recognition errors. "
    "Extract the requested fields and return only a valid JSON object."
)


def build_system_prompt(schema_cls: type[BaseModel], override: str = "") -> str:
    if override:
        return override

    if hasattr(schema_cls, "system_prompt") and callable(schema_cls.system_prompt):
        return schema_cls.system_prompt()

    return _GENERIC_SYSTEM_PROMPT


class LLMProvider(ABC):

    @abstractmethod
    def extract(self, text: str, schema: type[T], system_prompt: str = "") -> T:
        ...
