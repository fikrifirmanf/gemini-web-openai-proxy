"""Tests for the emulated tool-calling path (Codex support on /v1/responses)."""

from __future__ import annotations

import json

import pytest

from gemini_openai_proxy.config import Settings
from gemini_openai_proxy.schemas import ResponsesRequest
from gemini_openai_proxy.tools_emulation import build_tool_instructions, parse_tool_calls


SHELL_TOOL = {
    "type": "function",
    "name": "shell",
    "description": "Runs a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "array"}}},
}

CALL_JSON = '{"tool_calls": [{"name": "shell", "arguments": {"command": ["ls", "-la"]}}]}'


# ----------------------------------------------------------------- parser


def test_parse_exact_tool_call_object():
    calls = parse_tool_calls(CALL_JSON)
    assert calls == [{"name": "shell", "arguments": '{"command": ["ls", "-la"]}'}]


def test_parse_tolerates_fences_and_chatter_once():
    fenced = f'Sure, running it now:\n```json\n{CALL_JSON}\n```\n(Tools will run.)'
    calls = parse_tool_calls(fenced)
    assert calls and calls[0]["name"] == "shell"


def test_parse_plain_text_returns_none():
    assert parse_tool_calls("The file list is:\n- a\n- b") is None
    assert parse_tool_calls("") is None
    assert parse_tool_calls("not json at all {") is None


def test_parse_arguments_as_string_and_alias_names():
    as_string = '{"tool_calls": [{"name": "shell", "arguments": "{}"}]}'
    assert parse_tool_calls(as_string) == [{"name": "shell", "arguments": "{}"}]

    alias = '{"function_calls": [{"name": "apply_patch"}]}'
    assert parse_tool_calls(alias) == [{"name": "apply_patch", "arguments": "{}"}]


# ------------------------------------------------------------ instructions


def test_build_tool_instructions_lists_functions():
    text = build_tool_instructions([SHELL_TOOL])
    assert '"name": "shell"' in text
    assert "Tool-calling protocol" in text
    assert "ENTIRE reply must be exactly one JSON object" in text.replace("\n", " ")


def test_build_tool_instructions_skips_non_function_types():
    text = build_tool_instructions([SHELL_TOOL, {"type": "web_search"}])
    assert '"name": "shell"' in text
    assert '"web_search"' not in text.replace('"shell"', "")  # no new occurrences


# ------------------------------------------------------------------ schema


def test_to_chat_request_with_tool_protocol_and_call_items():
    req = ResponsesRequest.model_validate({
        "instructions": "You are Codex.",
        "input": [
            {"type": "message", "role": "user", "content": "make a file"},
            {"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
        ],
    })
    chat = req.to_chat_request(extra_system="protocol-section")
    roles = [m.role for m in chat.messages]
    assert roles[0] == "system"
    assert "protocol-section" in chat.messages[0].content
    assert any("[Tool request] shell" in m.content for m in chat.messages)
    assert any("[tool result for c1]" in m.content for m in chat.messages)


# ------------------------------------------------------------------- route


@pytest.fixture()
def tool_client(app, monkeypatch):
    """App whose stub service replies with a scripted tool-call JSON."""
    from fastapi.testclient import TestClient

    svc = app.state.gemini

    async def fake_non_stream(request):
        from tests.conftest import FakeOutput
        return FakeOutput(text=CALL_JSON, text_delta=CALL_JSON), request.model, []

    async def fake_stream(request):
        from tests.conftest import FakeOutput
        yield FakeOutput(text='{"tool_ca', text_delta='{"tool_ca')
        yield FakeOutput(text=CALL_JSON, text_delta='lls": [{"name": "shell", "arguments": {"command": ["ls"]}}]}')

    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]
    svc.chat_stream = fake_stream  # type: ignore[method-assign]
    with TestClient(app) as test_client:
        yield test_client


def test_non_stream_tool_call_becomes_function_call_item(tool_client):
    resp = tool_client.post("/v1/responses", json={
        "model": "gemini-pro",
        "input": "list files",
        "tools": [SHELL_TOOL],
        "stream": False,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["output"]) == 1
    item = body["output"][0]
    assert item["type"] == "function_call"
    assert item["name"] == "shell"
    assert json.loads(item["arguments"]) == {"command": ["ls", "-la"]}
    assert item["call_id"].startswith("call_")


def test_stream_tool_call_emits_function_call_events(tool_client):
    events = [
        line.split(": ", 1)[1]
        for line in tool_client.post("/v1/responses", json={
            "model": "gemini-flash",
            "input": "list files",
            "tools": [SHELL_TOOL],
            "stream": True,
        }).text.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "response.created"
    assert "response.function_call_arguments.delta" in events
    assert events[-1] == "response.completed"
    assert "response.output_text.delta" not in events  # buffering hides the JSON


def test_stream_plain_reply_stays_incremental(app):
    """No tools -> true incremental text deltas (the pre-emulation behaviour)."""
    from fastapi.testclient import TestClient

    deltas = []
    with TestClient(app) as client:
        with client.stream("POST", "/v1/responses", json={
            "model": "gemini-flash", "input": "hi", "stream": True
        }) as resp:
            for line in resp.iter_lines():
                if line.startswith("event: response.output_text.delta"):
                    deltas.append(line)
    assert len(deltas) >= 2  # stub yields two pieces, streamed live
