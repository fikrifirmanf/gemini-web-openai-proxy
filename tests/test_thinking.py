"""reasoning_effort -> Gemini Web extended_thinking wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from gemini_openai_proxy.config import Settings
from gemini_openai_proxy.schemas import ChatCompletionRequest, ChatMessage, ResponsesRequest
from gemini_openai_proxy.upstream import GeminiService


def make_service(**overrides) -> GeminiService:
    settings = Settings(
        send_interval=overrides.pop("send_interval", 0.0),
        quota_cooldown=overrides.pop("quota_cooldown", 60.0),
        data_dir=overrides.pop("data_dir", None) or Path("."),
    )
    assert not overrides, f"unused overrides: {list(overrides)}"
    return GeminiService(settings)


class _RecordingChat:
    def __init__(self, sink: dict[str, Any]):
        self._sink = sink

    async def send_message(self, prompt, files=None, **kwargs):
        self._sink.update(kwargs)
        from tests.conftest import FakeOutput
        return FakeOutput(text="ok")

    async def send_message_stream(self, prompt, files=None, **kwargs):
        self._sink.update(kwargs)
        from tests.conftest import FakeOutput
        yield FakeOutput(text="ok", text_delta="ok")


class _NoFilesCollector:
    files: list = []
    notes: list = []

    async def collect(self, messages, *, only_last: bool = False) -> None:
        return None


def _service_with_capture(sink: dict[str, Any]) -> tuple[GeminiService, _RecordingChat]:
    svc = make_service()
    chat = _RecordingChat(sink)
    svc.client = type("C", (), {"start_chat": staticmethod(lambda model=None: chat)})()
    svc._ensure_ready = lambda: asyncio.sleep(0)  # type: ignore[method-assign]
    svc._sessions = {}  # type: ignore[attr-defined]
    svc._media_collector = lambda: _NoFilesCollector()  # type: ignore[method-assign]
    svc.resolve_model = lambda name: (None, name)  # type: ignore[method-assign]
    return svc, chat


def test_chat_non_stream_forwards_extended_thinking():
    sink: dict[str, Any] = {}
    svc, _ = _service_with_capture(sink)
    req = ChatCompletionRequest.model_validate({
        "model": "gemini-flash", "messages": [
            {"role": "user", "content": "think hard"}], "reasoning_effort": "high",
    })

    async def go():
        return await svc.chat_non_stream(req)

    output, _m, _n = asyncio.run(go())
    assert sink.get("extended_thinking") is True
    assert str(output.text) == "ok"


def test_chat_stream_without_effort_is_standard():
    sink: dict[str, Any] = {}
    svc, _svc_chat = _service_with_capture(sink)
    req = ChatCompletionRequest.model_validate({
        "model": "gemini-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    async def go():
        return [chunk async for chunk in svc.chat_stream(req)]

    asyncio.run(go())
    assert sink.get("extended_thinking") is False


def test_responses_reasoning_maps_through():
    req = ResponsesRequest.model_validate({
        "model": "gemini-pro",
        "input": "hi",
        "reasoning": {"effort": "medium"},
    })
    chat = req.to_chat_request()
    assert chat.reasoning_effort == "medium"
    assert chat.wants_extended_thinking is True

    plain = ResponsesRequest.model_validate({"model": "gemini-pro", "input": "hi"}).to_chat_request()
    assert plain.wants_extended_thinking is False


def test_thinking_changes_session_cache_key():
    svc: GeminiService = make_service()
    msg = ChatMessage(role="user", content="hi")
    key_plain = svc._hash_for([msg], None, "m")
    key_think = svc._hash_for([msg], None, "m", True)
    assert key_plain != key_think
