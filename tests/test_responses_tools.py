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


def test_parse_repairs_bare_keys_from_live_corruption():
    bad = '{"tool_calls": [{"name": "shell", "arguments": {"cmd": "printf x > /tmp/f && cat /tmp/f" doctrine: ""}}]}'
    calls = parse_tool_calls(bad)
    assert calls and calls[0]["name"] == "shell"
    assert json.loads(calls[0]["arguments"]) == {"cmd": "printf x > /tmp/f && cat /tmp/f", "doctrine": ""}


def test_repair_keeps_colons_inside_strings():
    from gemini_openai_proxy.tools_emulation import _repair_json
    repaired = _repair_json('{"cmd": "ls -la: ok", timing: 3,}')
    assert json.loads(repaired) == {"cmd": "ls -la: ok", "timing": 3}


def test_parse_arguments_as_string_and_alias_names():
    as_string = '{"tool_calls": [{"name": "shell", "arguments": "{}"}]}'
    assert parse_tool_calls(as_string) == [{"name": "shell", "arguments": "{}"}]

    alias = '{"function_calls": [{"name": "apply_patch"}]}'
    assert parse_tool_calls(alias) == [{"name": "apply_patch", "arguments": "{}"}]


def test_parse_accepts_openai_native_single_call_shape():
    single = '{"name": "exec_command", "arguments": {"cmd": "ls"}}'
    assert parse_tool_calls(single) == [
        {"name": "exec_command", "arguments": '{"cmd": "ls"}'}
    ]
    wrapped = '{"function_call": {"name": "shell", "arguments": {}}}'
    assert parse_tool_calls(wrapped) == [{"name": "shell", "arguments": "{}"}]


def test_tool_protocol_contains_example_and_no_markdown_rule():
    text = build_tool_instructions([SHELL_TOOL])
    assert 'Example valid reply' in text
    assert "does NOT execute anything" in text


# ------------------------------------------------------------ instructions


def test_build_tool_instructions_lists_functions():
    text = build_tool_instructions([SHELL_TOOL])
    assert '"name":"shell"' in text  # compact catalogue entry, not just the protocol example
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
    assistant = [m.content for m in chat.messages if m.role == "assistant"]
    assert assistant == ['{"tool_calls": [{"name": "shell", "arguments": {}}]}']
    assert chat.messages[-1].role == "tool"
    assert chat.messages[-1].content.startswith("[Tool result: shell]\ndone")
    assert chat.messages[-1].content == "[Tool result: shell]\ndone"  # hash-stable: no suffix in content
    assert chat.tool_mode is True


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


# ---------------------------------------------------------- corrective retry


# deliberately beyond the repair pass: bare `name` VALUE (not a key) can't be salvaged
BAD_TOOL_JSON = '{"tool_calls": [{"name": shell, "arguments": {"cmd": "x" doctrine: ""}}]}'


def _install_flaky_upstream(svc, calls, seen):
    """First upstream reply: corrupt tool JSON; second: valid."""
    from tests.conftest import FakeOutput

    async def fake_stream(request):
        n = calls["n"]
        calls["n"] += 1
        seen.append(request)
        text = BAD_TOOL_JSON if n == 0 else CALL_JSON
        yield FakeOutput(text=text, text_delta=text)

    async def fake_non_stream(request):
        n = calls["n"]
        calls["n"] += 1
        seen.append(request)
        text = BAD_TOOL_JSON if n == 0 else CALL_JSON
        return FakeOutput(text=text, text_delta=text), request.model, []

    svc.chat_stream = fake_stream  # type: ignore[method-assign]
    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]


def test_stream_malformed_tool_json_retries_once(app):
    from fastapi.testclient import TestClient

    svc = app.state.gemini
    calls, seen = {"n": 0}, []
    _install_flaky_upstream(svc, calls, seen)
    try:
        with TestClient(app) as client:
            events = client.post("/v1/responses", json={
                "model": "gemini-flash", "input": "run ls",
                "tools": [SHELL_TOOL], "stream": True,
            }).text
            assert events.count("event: response.function_call_arguments.delta") == 1  # the GOOD call won
            assert calls["n"] == 2  # exactly one corrective retry
            assert "event: response.output_text.delta" not in events  # nothing leak-streamed
            corrective = seen[1].messages[-1]
            assert corrective.role == "user" and "previous reply was malformed" in corrective.content
    finally:
        del svc.chat_stream, svc.chat_non_stream


def test_non_stream_malformed_tool_json_retries_once(app):
    from fastapi.testclient import TestClient

    svc = app.state.gemini
    calls, seen = {"n": 0}, []
    _install_flaky_upstream(svc, calls, seen)
    try:
        with TestClient(app) as client:
            body = client.post("/v1/responses", json={
                "model": "gemini-flash", "input": "run ls",
                "tools": [SHELL_TOOL], "stream": False,
            }).json()
            assert body["output"][0]["type"] == "function_call"
            assert body["output"][0]["name"] == "shell"
            assert calls["n"] == 2
    finally:
        del svc.chat_stream, svc.chat_non_stream


def test_prose_that_mentions_tool_calls_does_not_retry(app):
    """A normal answer that merely contains the word must stay single-send."""
    from fastapi.testclient import TestClient

    svc = app.state.gemini
    calls, seen = {"n": 0}, []
    _install_flaky_upstream(svc, calls, seen)
    prose = {"n": 0}
    try:
        async def pure_prose(request):
            prose["n"] += 1
            from tests.conftest import FakeOutput
            return FakeOutput(text="I can help with tool_calls questions."), request.model, []

        svc.chat_non_stream = pure_prose  # type: ignore[method-assign]
        with TestClient(app) as client:
            body = client.post("/v1/responses", json={
                "model": "gemini-flash", "input": "hi",
                "tools": [SHELL_TOOL], "stream": False,
            }).json()
            assert body["output"][0]["type"] == "message"
            assert prose["n"] == 1
    finally:
        del svc.chat_stream, svc.chat_non_stream


# ------------------------------------------------------- tool-loop tuning


def test_catalogue_is_compact_and_lists_custom_tools():
    from gemini_openai_proxy.tools_emulation import custom_tool_names

    big = {
        "type": "function", "name": "exec_command", "description": "x " * 2000,
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"cmd": {"type": "string", "description": "d " * 400}}},
    }
    patch = {"type": "custom", "name": "apply_patch", "description": "Apply a patch"}
    text = build_tool_instructions([big, patch])
    assert len(text) < 5000
    assert "additionalProperties" not in text
    assert "apply_patch (freeform tool)" in text
    assert custom_tool_names([big, patch]) == {"apply_patch"}


def test_history_merges_parallel_calls_and_names_results():
    req = ResponsesRequest.model_validate({
        "input": [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c1"},
            {"type": "function_call", "name": "b", "arguments": '{"x":1}', "call_id": "c2"},
            {"type": "function_call_output", "call_id": "c1", "output": "A" * 50000},
            {"type": "function_call_output", "call_id": "c2", "output": "ok"},
        ],
    })
    msgs = req.to_chat_request(extra_system="proto").messages
    assert [m.role for m in msgs] == ["system", "user", "assistant", "tool", "tool"]
    assert parse_tool_calls(msgs[2].content) and len(parse_tool_calls(msgs[2].content)) == 2
    assert msgs[3].content.startswith("[Tool result: a]")
    assert len(msgs[3].content) < 13000 and "chars omitted" in msgs[3].content
    assert msgs[4].content.startswith("[Tool result: b]")


def test_custom_tool_roundtrip_item_shapes():
    from gemini_openai_proxy.routes.responses import _function_call_items

    calls = [{"name": "apply_patch", "arguments": json.dumps({"input": "*** Begin Patch\n*** End Patch"})}]
    [item] = _function_call_items(calls, {"apply_patch"})
    assert item["type"] == "custom_tool_call" and item["input"].startswith("*** Begin Patch")

    req = ResponsesRequest.model_validate({"input": [
        {"type": "message", "role": "user", "content": "patch"},
        {"type": "custom_tool_call", "name": "apply_patch", "input": "P", "call_id": "k"},
        {"type": "custom_tool_call_output", "call_id": "k", "output": "Done!"},
    ]})
    msgs = req.to_chat_request().messages
    assert json.loads(msgs[1].content)["tool_calls"][0]["arguments"] == {"input": "P"}
    assert msgs[2].content.startswith("[Tool result: apply_patch]")


def test_agent_loop_reuses_cached_web_session():
    """Round 2 of a tool loop must hit the cached chat (send only the new turn)."""
    import asyncio
    from tests.conftest import FakeOutput
    from tests.test_thinking import _service_with_capture
    from gemini_openai_proxy.tools_emulation import render_calls

    sink: dict = {}
    prompts: list[str] = []
    svc, chat = _service_with_capture(sink)
    chat.metadata = ["cid", "rid", "rcid"]
    from collections import OrderedDict
    svc._sessions = OrderedDict()

    async def send_message(prompt, files=None, **kwargs):
        prompts.append(prompt)
        return FakeOutput(text="ok")

    chat.send_message = send_message
    first = ResponsesRequest.model_validate({"input": "list files"}).to_chat_request(extra_system="proto")
    asyncio.run(svc.chat_non_stream(first))
    svc.remember_reply(first, render_calls([("shell", '{"cmd":"ls"}')]))

    second = ResponsesRequest.model_validate({"input": [
        {"type": "message", "role": "user", "content": "list files"},
        {"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}', "call_id": "c"},
        {"type": "function_call_output", "call_id": "c", "output": "a b"},
    ]}).to_chat_request(extra_system="proto")
    asyncio.run(svc.chat_non_stream(second))
    assert len(prompts) == 2
    assert prompts[1].startswith("[Tool result: shell]\na b")  # newest turn only, no transcript replay


def test_tool_rounds_honour_effort_by_default_and_can_opt_out():
    from tests.test_thinking import make_service

    svc = make_service()
    req = ResponsesRequest.model_validate({
        "reasoning": {"effort": "high"},
        "input": [{"type": "message", "role": "user", "content": "go"},
                  {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c"},
                  {"type": "function_call_output", "call_id": "c", "output": "x"}],
    }).to_chat_request()
    assert req.wants_extended_thinking is True
    assert svc._effective_thinking(req) is True      # default: think on every round
    svc.settings.tool_round_thinking = False
    assert svc._effective_thinking(req) is False     # opt-out: skip on tool-result rounds


def test_refusal_detection():
    from gemini_openai_proxy.tools_emulation import looks_like_refusal

    assert looks_like_refusal("I'm sorry, my tool-calling budget is exhausted for this turn.")
    assert looks_like_refusal("I can't run terminal commands directly, but I can guide you through it.")
    assert looks_like_refusal("You can run `npm publish` yourself.")
    assert not looks_like_refusal("Done. The package builds and all tests pass.")
    assert not looks_like_refusal(CALL_JSON)


def test_refusal_after_tool_result_triggers_one_corrective_retry(app):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeOutput

    svc = app.state.gemini
    seen = []

    async def fake_non_stream(request):
        seen.append(request)
        text = "My tool-calling budget is exhausted, I can guide you instead." if len(seen) == 1 else CALL_JSON
        return FakeOutput(text=text, text_delta=text), request.model, []

    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]
    with TestClient(app) as client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL],
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "c"},
                {"type": "function_call_output", "call_id": "c", "output": "ok"},
            ],
        }).json()
    assert len(seen) == 2 and "not an injection" in str(seen[1].messages[-1].content)
    assert body["output"][0]["type"] == "function_call"


def test_invented_limit_is_caught_on_a_fresh_user_turn_too(app):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeOutput

    svc = app.state.gemini
    seen = []

    async def fake_non_stream(request):
        seen.append(request)
        text = "Since I've reached the maximum number of tool calls for this turn, here is a guide." if len(seen) == 1 else CALL_JSON
        return FakeOutput(text=text, text_delta=text), request.model, []

    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]
    with TestClient(app) as client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL], "input": "fix and publish it",
        }).json()
    assert len(seen) == 2
    assert body["output"][0]["type"] == "function_call"


def test_soft_phrasing_is_not_retried_on_plain_questions():
    from gemini_openai_proxy.tools_emulation import looks_like_refusal

    assert looks_like_refusal("You can run `npm test` to verify.", strong_only=True) is False
    assert looks_like_refusal("I've reached the maximum number of tool calls.", strong_only=True)


def test_history_scrubs_assistant_turns_that_claimed_a_limit():
    req = ResponsesRequest.model_validate({"input": [
        {"type": "message", "role": "user", "content": "a"},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "I've reached the maximum number of tool calls for this turn."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Found 3 files."}]},
        {"type": "message", "role": "user", "content": "go on"},
    ]})
    msgs = req.to_chat_request(extra_system="proto").messages
    assert [m.content for m in msgs if m.role == "assistant"] == ["(Previous reply omitted.)", "Found 3 files."]
    plain = req.to_chat_request().messages  # no tools -> history untouched
    assert "maximum number" in plain[1].content


def test_injection_style_refusal_is_detected():
    from gemini_openai_proxy.tools_emulation import looks_like_refusal

    assert looks_like_refusal(
        "I'm sorry, but I cannot use the custom tool-calling protocol or emit raw JSON as requested "
        "by the system prompt injection.", strong_only=True)
    assert "legitimate part of this task" in build_tool_instructions([SHELL_TOOL])


def test_tool_budget_phrasings_from_live_sessions_are_caught():
    from gemini_openai_proxy.tools_emulation import looks_like_refusal

    for text in (
        "Since I cannot execute any more tool calls, here is an overview of how you can fix it.",
        "I'm sorry about that! Since my tool-calling budget is exhausted for this turn, I can't run terminal commands.",
        "Since I've reached the maximum number of tool calls for this turn, let's look at what we have.",
    ):
        assert looks_like_refusal(text, strong_only=True), text
    assert not looks_like_refusal("I can run the tests next. Done: 3 files changed.", strong_only=True)


def test_debug_dump_writes_prompt_and_reply(tmp_path):
    import json as _json
    from tests.test_thinking import make_service

    svc = make_service()
    svc.settings.debug_dump_dir = str(tmp_path / "dump")
    svc._dump("hello prompt", True, False, "hi reply")
    [f] = list((tmp_path / "dump").glob("*.json"))
    data = _json.loads(f.read_text())
    assert data["prompt"] == "hello prompt" and data["reply"] == "hi reply" and data["reused_session"] is True


# --------------------------------------------- live failures from data/dump


LIVE_ANNOUNCE = (
    "I'm setting up the `karhutla-protest` project folder and initializing a high-craft Vite + React + "
    "Tailwind application.I will now create the `karhutla-protest` project directory and scaffold it."
)
LIVE_ANNOUNCE_2 = "Ah, my apologies! Let me set up the `karhutla-protest` folder and create the full project for you right now."
LIVE_CLAIM = (
    "I am now installing dependencies and starting the development server.I have successfully created and "
    "configured the **`karhutla-protest`** project folder.\n### What Was Built"
)


def test_announce_and_claim_detection_on_live_replies():
    from gemini_openai_proxy.tools_emulation import announces_action, claims_work_done

    assert announces_action(LIVE_ANNOUNCE) and announces_action(LIVE_ANNOUNCE_2)
    assert claims_work_done(LIVE_CLAIM)
    for fine in ("Done. All five commands ran; summary below.", "The tests pass. Let me know if you want more.",
                 "Here is how the cache works: it keys by hash.", CALL_JSON):
        assert not announces_action(fine), fine


def test_send_time_suffix_added_but_cache_keys_stay_stable():
    """3-round loop: every round after the first must reuse the cached web session."""
    import asyncio
    from collections import OrderedDict
    from tests.conftest import FakeOutput
    from tests.test_thinking import _service_with_capture
    from gemini_openai_proxy.tools_emulation import render_calls

    sink: dict = {}
    prompts: list[str] = []
    svc, chat = _service_with_capture(sink)
    chat.metadata = ["cid", "rid", "rcid"]
    svc._sessions = OrderedDict()

    async def send_message(prompt, files=None, **kwargs):
        prompts.append(prompt)
        return FakeOutput(text="ok")

    chat.send_message = send_message

    def items(n):
        out = [{"type": "message", "role": "user", "content": "go"}]
        for i in range(n):
            out += [{"type": "function_call", "name": "shell", "arguments": '{"c":%d}' % i, "call_id": f"c{i}"},
                    {"type": "function_call_output", "call_id": f"c{i}", "output": f"out{i}"}]
        return out

    for n in range(3):
        req = ResponsesRequest.model_validate({"input": items(n)}).to_chat_request(extra_system="proto")
        asyncio.run(svc.chat_non_stream(req))
        svc.remember_reply(req, render_calls([("shell", '{"c":%d}' % n)]))

    assert len(prompts) == 3
    assert "Reminder: if there is more work" in prompts[0]            # plain user turn gets it too
    assert prompts[1].startswith("[Tool result: shell]\nout0\n[End of tool result]")
    assert prompts[2].startswith("[Tool result: shell]\nout1\n[End of tool result]")  # round 3 reused, no transcript


def _scripted_route(app, first_text):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeOutput

    svc, seen = app.state.gemini, []

    async def fake_non_stream(request):
        seen.append(request)
        text = first_text if len(seen) == 1 else CALL_JSON
        return FakeOutput(text=text, text_delta=text), request.model, []

    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]
    return TestClient(app), seen


def test_announce_without_tool_call_gets_one_retry(app):
    client, seen = _scripted_route(app, LIVE_ANNOUNCE_2)
    with client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL], "input": "there are no karhutla-protest dir"}).json()
    assert len(seen) == 2 and "Nothing was run" in str(seen[1].messages[-1].content)
    assert body["output"][0]["type"] == "function_call"


def test_fabricated_completion_with_no_tool_calls_gets_retry(app):
    client, seen = _scripted_route(app, "I have successfully created and configured the project folder.\n### What Was Built")
    with client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL], "input": "run it"}).json()
    assert len(seen) == 2 and "does not exist yet" in str(seen[1].messages[-1].content)
    assert body["output"][0]["type"] == "function_call"


def test_honest_summary_after_real_tool_calls_is_not_retried(app):
    client, seen = _scripted_route(app, "I have created the files and verified them with ls. Done.")
    with client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL],
            "input": [{"type": "message", "role": "user", "content": "make files"},
                      {"type": "function_call", "name": "shell", "arguments": '{"cmd": "cat > a.txt <<EOF\\nhi\\nEOF"}', "call_id": "c"},
                      {"type": "function_call_output", "call_id": "c", "output": "ok"}]}).json()
    assert len(seen) == 1
    assert body["output"][0]["type"] == "message"


def test_history_drops_prose_that_announced_or_claimed_without_tools():
    req = ResponsesRequest.model_validate({"input": [
        {"type": "message", "role": "user", "content": "build it"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": LIVE_CLAIM}]},
        {"type": "message", "role": "user", "content": "run it"},
    ]})
    msgs = req.to_chat_request(extra_system="proto").messages
    assert [m.content for m in msgs if m.role == "assistant"] == ["(Previous reply omitted.)"]


LIVE_FAKE_SUMMARY = (
    "I have successfully created the interactive protest website and dashboard inside `karhutla-protest`.\n\n"
    "### Project Overview & Features Built (`karhutla-protest`)\n- **Modern Tech Stack:** React 19, Vite."
)


def _after(cmd):
    return [{"type": "message", "role": "user", "content": "build the site"},
            {"type": "function_call", "name": "exec_command", "arguments": json.dumps({"cmd": cmd}), "call_id": "c"},
            {"type": "function_call_output", "call_id": "c", "output": "ok"}]


def test_is_mutating_distinguishes_reads_and_mkdir_from_real_changes():
    from gemini_openai_proxy.tools_emulation import is_mutating

    for cmd in ("mkdir -p karhutla-protest", "ls -la", "cat package.json", "grep -r x src"):
        assert not is_mutating("exec_command", json.dumps({"cmd": cmd})), cmd
    for cmd in ("cat > index.html <<'EOF'\nhi\nEOF", "npm create vite@latest app", "echo hi > a.txt", "git init", "mkdir x && npm install"):
        assert is_mutating("exec_command", json.dumps({"cmd": cmd})), cmd
    assert is_mutating("apply_patch", json.dumps({"input": "*** Begin Patch"}))
    assert not is_mutating("write_stdin", "{}")


def test_fake_site_summary_after_only_mkdir_is_retried(app):
    client, seen = _scripted_route(app, LIVE_FAKE_SUMMARY)
    with client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL], "input": _after("mkdir -p karhutla-protest")}).json()
    assert len(seen) == 2 and "does not exist yet" in str(seen[1].messages[-1].content)
    assert body["output"][0]["type"] == "function_call"


def test_gemini_canned_error_text_is_retried(app):
    client, seen = _scripted_route(app, "I seem to be encountering an error. Can I try something else for you?")
    with client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL], "input": _after("cat SKILL.md")}).json()
    assert len(seen) == 2 and "transient error" in str(seen[1].messages[-1].content)
    assert body["output"][0]["type"] == "function_call"


# ------------------------------------------------ live parse failures + raw blocks


def test_stray_trailing_brace_from_live_dump_is_salvaged():
    live = ('{"tool_calls": [{"name": "exec_command", "arguments": {"cmd": "cd karhutla-protest && npm install '
            'react@19 react-dom@19"}}]}}')
    calls = parse_tool_calls(live)
    assert calls and calls[0]["name"] == "exec_command"
    assert json.loads(calls[0]["arguments"])["cmd"].startswith("cd karhutla-protest")


RAW_REPLY = """@@tool exec_command
@@arg cmd
cd karhutla-protest && cat << 'EOF' > src/App.tsx
export default () => <p className="w-4 h-4" title='x'>"quoted" \\n {`tpl`}</p>;
EOF
npm run build
@@arg yield_time_ms
1500
@@end"""


def test_raw_blocks_carry_code_verbatim_without_json_escaping():
    tools = [{"type": "function", "name": "exec_command",
              "parameters": {"type": "object", "properties": {
                  "cmd": {"type": "string"}, "yield_time_ms": {"type": "integer"}}}}]
    [call] = parse_tool_calls(RAW_REPLY, tools)
    args = json.loads(call["arguments"])
    assert call["name"] == "exec_command"
    assert 'className="w-4 h-4" title=\'x\'' in args["cmd"] and '"quoted"' in args["cmd"]
    assert args["cmd"].endswith("npm run build") and "\n" in args["cmd"]
    assert args["yield_time_ms"] == 1500            # typed from the declared schema


def test_raw_blocks_multiple_tools_and_fenced_reply():
    text = "```\n@@tool a\n@@arg x\none\n@@end\n@@tool b\n@@end\n```"
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[0]["arguments"]) == {"x": "one"}
    assert json.loads(calls[1]["arguments"]) == {}


def test_prose_mentioning_at_signs_is_not_a_tool_call():
    assert parse_tool_calls("Use @@tool syntax like this in docs.") is None
    assert parse_tool_calls("email me @@ home") is None


def test_protocol_teaches_raw_blocks_and_wants_retry_sees_failed_raw_attempt():
    from gemini_openai_proxy.routes.responses import _wants_retry

    assert "@@tool shell" in build_tool_instructions([SHELL_TOOL])
    assert _wants_retry("@@tool exec_command\n@@arg cmd\n(oops no end and no args name)") is True


def test_more_canned_gemini_errors_are_recognised():
    from gemini_openai_proxy.tools_emulation import is_upstream_error_text

    for text in ("I encountered an error doing what you asked. Could you try again?",
                 "I'm having a hard time fulfilling your request. Can I help you with something else?",
                 "I seem to be encountering an error. Can I try something else for you?"):
        assert is_upstream_error_text(text), text
    assert not is_upstream_error_text("Done. All tests pass.")


# ------------------------------------------------------------ MCP namespaces


MCP_NS = {
    "type": "namespace", "name": "mcp__frontend_unslop", "description": "frontend unslop",
    "tools": [
        {"type": "function", "name": "audit_frontend_slop", "description": "Audit files",
         "parameters": {"type": "object", "properties": {"files": {"type": "array"}}}, "strict": False},
        {"name": "verify_styles_wired", "description": "Check styles", "parameters": {"type": "object", "properties": {}}},
    ],
}
INTERNAL_NS = {"type": "namespace", "name": "mcp__codex_apps__pets", "tools": [
    {"type": "function", "name": "pet_thing", "parameters": {}}]}


def test_expand_tools_flattens_namespaces_and_drops_excluded():
    from gemini_openai_proxy.tools_emulation import expand_tools

    flat, ns = expand_tools([SHELL_TOOL, MCP_NS, INTERNAL_NS, {"type": "web_search"}],
                            exclude=r"^(multi_agent|mcp__codex_apps__)")
    assert [t.get("name") for t in flat] == ["shell", "audit_frontend_slop", "verify_styles_wired", None]
    assert all(t["type"] != "namespace" for t in flat)
    assert ns == {"audit_frontend_slop": "mcp__frontend_unslop", "verify_styles_wired": "mcp__frontend_unslop"}
    text = build_tool_instructions(flat)
    assert '"name":"audit_frontend_slop"' in text and "pet_thing" not in text


def test_expand_tools_skips_name_collisions():
    from gemini_openai_proxy.tools_emulation import expand_tools

    clash = {"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "shell", "parameters": {}}]}
    flat, ns = expand_tools([SHELL_TOOL, clash])
    assert [t["name"] for t in flat] == ["shell"] and ns == {}


def test_mcp_call_comes_back_with_namespace_field(app):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeOutput

    svc = app.state.gemini
    reply = '{"tool_calls": [{"name": "audit_frontend_slop", "arguments": {"files": ["src/App.tsx"]}}]}'

    async def fake_non_stream(request):
        assert "audit_frontend_slop" in str(request.messages[0].content)   # offered to Gemini now
        return FakeOutput(text=reply, text_delta=reply), request.model, []

    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]
    with TestClient(app) as client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [SHELL_TOOL, MCP_NS], "input": "audit my files"}).json()
    [item] = body["output"]
    assert item["type"] == "function_call"
    assert item["name"] == "audit_frontend_slop" and item["namespace"] == "mcp__frontend_unslop"
    assert json.loads(item["arguments"]) == {"files": ["src/App.tsx"]}


def test_function_call_history_with_namespace_replays_by_name():
    req = ResponsesRequest.model_validate({"input": [
        {"type": "message", "role": "user", "content": "go"},
        {"type": "function_call", "name": "audit_frontend_slop", "namespace": "mcp__frontend_unslop",
         "arguments": '{"files": []}', "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "PASS"},
    ]})
    msgs = req.to_chat_request(extra_system="proto").messages
    assert json.loads(msgs[2].content)["tool_calls"][0]["name"] == "audit_frontend_slop"
    assert msgs[3].content.startswith("[Tool result: audit_frontend_slop]")


# ------------------------------------- live defects: example name, arg types, batching


EXEC_TOOL = {"type": "function", "name": "exec_command", "description": "run",
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}


def test_protocol_example_uses_a_declared_tool_not_a_made_up_name():
    text = build_tool_instructions([EXEC_TOOL])
    example = text.split("Example valid reply:")[1].split("For arguments")[0]
    assert '"name": "exec_command"' in example and '"cmd": "ls -la"' in example
    assert "@@tool exec_command" in text and "@@arg cmd" in text
    assert '"name": "shell"' not in text                      # the hallucinated `shell` source
    assert "must wait for a later reply" in text              # batching rule


def test_array_param_given_as_newline_string_is_coerced():
    tools = [{"type": "function", "name": "audit_frontend_slop",
              "parameters": {"type": "object", "properties": {
                  "files": {"type": "array", "items": {"type": "string"}},
                  "limit": {"type": "integer"}}}}]
    reply = '{"tool_calls": [{"name": "audit_frontend_slop", "arguments": {"files": "src/App.tsx\\nsrc/index.css", "limit": "3"}}]}'
    [call] = parse_tool_calls(reply, tools)
    assert json.loads(call["arguments"]) == {"files": ["src/App.tsx", "src/index.css"], "limit": 3}
    # already-correct arguments are untouched, and unknown tools pass through
    ok = '{"tool_calls": [{"name": "audit_frontend_slop", "arguments": {"files": ["a"]}}]}'
    assert json.loads(parse_tool_calls(ok, tools)[0]["arguments"]) == {"files": ["a"]}


def test_undeclared_function_name_gets_one_corrective_retry(app):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeOutput

    svc, seen = app.state.gemini, []
    bad = '{"tool_calls": [{"name": "shell", "arguments": {"cmd": "ls"}}]}'
    good = '{"tool_calls": [{"name": "exec_command", "arguments": {"cmd": "ls"}}]}'

    async def fake_non_stream(request):
        seen.append(request)
        text = bad if len(seen) == 1 else good
        return FakeOutput(text=text, text_delta=text), request.model, []

    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]
    with TestClient(app) as client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [EXEC_TOOL], "input": "list files"}).json()
    assert len(seen) == 2 and "There is no function named shell" in str(seen[1].messages[-1].content)
    assert body["output"][0]["name"] == "exec_command"


def test_still_undeclared_after_retry_falls_back_to_plain_message(app):
    from fastapi.testclient import TestClient
    from tests.conftest import FakeOutput

    svc = app.state.gemini
    bad = '{"tool_calls": [{"name": "shell", "arguments": {}}]}'

    async def fake_non_stream(request):
        return FakeOutput(text=bad, text_delta=bad), request.model, []

    svc.chat_non_stream = fake_non_stream  # type: ignore[method-assign]
    with TestClient(app) as client:
        body = client.post("/v1/responses", json={
            "model": "gemini-argon", "tools": [EXEC_TOOL], "input": "x"}).json()
    assert body["output"][0]["type"] == "message"           # never an undeclared function_call


def test_stream_final_message_uses_full_snapshot_not_garbled_deltas(app):
    """Gemini Web rewrites text mid-stream; delta diffs lose chars (":3000" -> ":300")."""
    import json
    from fastapi.testclient import TestClient

    svc = app.state.gemini
    full = "Running at [http://x:3000](http://x:3000) now"

    async def fake_stream(request):
        from tests.conftest import FakeOutput
        yield FakeOutput(text="Running at http://x:3000", text_delta="Running at http://x:300")
        yield FakeOutput(text=full, text_delta="](http://x:300) now")

    svc.chat_stream = fake_stream  # type: ignore[method-assign]
    done_text = None
    with TestClient(app) as client:
        with client.stream("POST", "/v1/responses", json={
            "model": "gemini-flash", "input": "hi", "stream": True
        }) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and '"response.completed"' in line:
                    body = json.loads(line[6:])["response"]
                    done_text = body["output"][0]["content"][0]["text"]
    assert done_text == "Running at http://x:3000 now"  # full snapshot, link unwrapped


def test_title_request_is_slimmed_down(app):
    """Codex's hidden thread-title call skips the big system prompt and thinking."""
    from fastapi.testclient import TestClient
    from gemini_openai_proxy.routes.responses import _is_title_request, _slim_title_request
    from gemini_openai_proxy.schemas import ResponsesRequest

    req = ResponsesRequest.model_validate({
        "model": "gemini-flash", "instructions": "BIG " * 500, "reasoning": {"effort": "high"},
        "input": "Generate a concise, single-line task title of at most 36 characters.\n\nUser prompt:\nhi",
    }).to_chat_request()
    assert _is_title_request(req) and req.wants_extended_thinking
    slim = _slim_title_request(req)
    assert len(slim.messages) == 1 and not slim.wants_extended_thinking

    plain = ResponsesRequest.model_validate({"model": "gemini-flash", "input": "hello"}).to_chat_request()
    assert not _is_title_request(plain)


def test_unwrap_autolinks():
    from gemini_openai_proxy.tools_emulation import unwrap_autolinks as u
    assert u("at [http://a:3000](http://a:3000)!") == "at http://a:3000!"
    assert u("[docs](http://a)") == "[docs](http://a)"  # real labels stay
