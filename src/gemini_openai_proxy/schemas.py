"""Pydantic models mirroring the subset of the OpenAI API we implement.

Content fields are typed as ``Any`` on purpose: clients send slightly different
shapes (plain string, list of parts, null) and we normalize in `content.py` so
that unexpected shapes degrade gracefully instead of 500-ing.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def unix_ts() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Any = ""
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_none(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("content") is None:
            data = {**data, "content": ""}
        return data


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "default"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    reasoning_effort: str | None = None  # "high"/"medium" -> extended thinking

    @property
    def wants_extended_thinking(self) -> bool:
        return (self.reasoning_effort or "").lower() in {"high", "medium", "max", "enabled", "enable"}

    @property
    def include_usage(self) -> bool:
        return bool((self.stream_options or {}).get("include_usage", False))


def make_usage(prompt_chars: int, completion_chars: int) -> dict[str, int]:
    prompt = max(1, prompt_chars // 4)
    completion = max(1, completion_chars // 4)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=unix_ts)
    owned_by: str = "google"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class ImagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str = Field(min_length=1)
    model: str = "default"
    n: int = Field(default=1, ge=1, le=8)
    response_format: Literal["b64_json"] = "b64_json"
    size: str | None = None  # accepted for compatibility, ignored upstream


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: str | None = None


class ImagesResponse(BaseModel):
    created: int = Field(default_factory=unix_ts)
    data: list[ImageData]
