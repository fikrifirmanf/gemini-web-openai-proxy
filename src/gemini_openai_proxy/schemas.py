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


class ResponsesRequest(BaseModel):
    """OpenAI **Responses API** (`POST /v1/responses`) — required by Codex ≥0.145.

    Only the subset Codex actually sends is modelled; unknown fields are allowed
    and ignored (tools currently have no Gemini Web counterpart).
    """

    model_config = ConfigDict(extra="allow")

    model: str = "default"
    input: str | list[Any] = ""
    instructions: str | None = None
    stream: bool = False
    reasoning: dict[str, Any] | None = None

    @property
    def effort(self) -> str | None:
        effort = (self.reasoning or {}).get("effort")
        return str(effort) if effort else None

    def to_chat_request(self, extra_system: str = "") -> ChatCompletionRequest:
        """Translate Responses-API items into chat-completions messages.

        `extra_system` carries the emulated tool-calling protocol when the
        request declares tools (see tools_emulation.py).
        """
        system = (self.instructions or "").strip()
        if extra_system.strip():
            system = f"{system}\n\n{extra_system}".strip() if system else extra_system.strip()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if isinstance(self.input, str):
            messages.append({"role": "user", "content": self.input})
        else:
            for item in self.input:
                if not isinstance(item, dict):
                    messages.append({"role": "user", "content": str(item)})
                    continue
                kind = item.get("type", "message")
                if kind == "message":
                    role = str(item.get("role", "user"))
                    messages.append({"role": role, "content": _flatten_content(item.get("content"))})
                elif kind == "function_call":
                    name = str(item.get("name") or "tool")
                    messages.append({
                        "role": "assistant",
                        "content": f"[Tool request] {name} {item.get('arguments') or ''}",
                    })
                elif kind == "function_call_output":
                    name = str(item.get("call_id") or "tool_call")
                    messages.append({
                        "role": "tool",
                        "content": f"[tool result for {name}]\n{_flatten_content(item.get('output'))}",
                    })
                # reasoning / item_reference / mcp_* -> no Gemini Web equivalent.

        messages = [m for m in messages if str(m.get("content") or "").strip() or m["role"] == "system"]
        if not messages:
            messages.append({"role": "user", "content": "(no content)"})
        return ChatCompletionRequest(
            model=self.model or "default",
            messages=messages,
            stream=self.stream,
            reasoning_effort=self.effort,
        )


def _flatten_content(content: Any) -> str:
    """Responses-API content is a string or a parts list; keep only the text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("output") or "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("output") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def new_response_id() -> str:
    return "resp_" + uuid.uuid4().hex


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
