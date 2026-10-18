"""Tool-call emulation for the Responses API (Codex support).

Gemini Web cannot produce OpenAI `function_call` items, so when a request
carries tool definitions we:

1. inject a strict "reply with ONLY a JSON object to call tools" protocol
   (`build_tool_instructions`),
2. post-process the model's reply and lift any `{"tool_calls": [...]}` block
   into real Responses-API items (`parse_tool_calls`).

Fidelity depends on the model obeying the protocol; otherwise the reply falls
through to plain text (and Codex shows it as a normal message, never executed).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("gemini_openai_proxy.tools_emulation")

_TOOL_PROTOCOL = """## Tool-calling protocol (mandatory)

You are the model behind an OpenAI Responses-API coding agent. You cannot run
anything yourself — to have anything done, you must ask the agent to execute a
function. When you want functions to run, your ENTIRE reply must be exactly one
JSON object and nothing else:

{"tool_calls": [{"name": "<function name>", "arguments": {<json params>}}]}

Rules:
- `arguments` is always a JSON object (use {} when there are no parameters).
- To run several functions, list them in one tool_calls array; they run in order.
- NO markdown fences, NO prose, NO explanation before or after — only the JSON.
- When everything is done (or a plain answer suffices), reply normally in text.
- Never invent `[Tool result]` content. Wait for it as a user-visible turn.

Available functions:
"""

_TOOL_MAX_CHARS = 4000  # per-function description cap keeps prompts sane


def build_tool_instructions(tools: list[dict[str, Any]]) -> str:
    """Render the tool protocol + function catalogue as a system-prompt section."""
    sections: list[str] = [_TOOL_PROTOCOL]
    for tool in tools or []:
        if tool.get("type") == "local_shell":
            sections.append("- local_shell — run a shell command; arguments: "
                            '{"command": ["bash", "-lc", "<command string>"]}')
            continue
        if tool.get("type") != "function":
            continue  # mcp / web_search / custom types: no Gemini Web equivalent
        fn = tool.get("function", {}) if "function" in tool else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        blob = json.dumps(fn, ensure_ascii=False)
        if len(blob) > _TOOL_MAX_CHARS:
            blob = blob[:_TOOL_MAX_CHARS] + "…(truncated)"
        sections.append(f"- {blob}")
    return "\n".join(sections)


def _candidate_spans(text: str) -> list[str]:
    """(1) whole trimmed text, (2) fenced ```json block, (3) outermost braces."""
    out = [text.strip()]
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        out.append(m.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        out.append(text[first : last + 1])
    return out


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Lift `{"tool_calls": [...]}` out of a model reply.

    Returns a list of {"name": str, "arguments": str} (arguments already a JSON
    *string*, as the OpenAI tool-calling format expects), or None when the reply
    isn't a tool-call turn (normal prose gets returned to the model untouched).
    """
    if not text or "{" not in text:
        return None
    for candidate in _candidate_spans(text):
        try:
            data = json.loads(candidate, strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            if isinstance(data, list):
                data = {"tool_calls": data}
            else:
                continue
        calls = data.get("tool_calls") or data.get("function_calls")
        if not isinstance(calls, list) or not calls:
            continue
        normalized: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or (call.get("function") or {}).get("name") or "").strip()
            if not name:
                continue
            args = call.get("arguments")
            if isinstance(args, str) and args.strip():
                normalized.append({"name": name, "arguments": args})
            else:
                payload = args if isinstance(args, (dict, list)) else {}
                normalized.append({"name": name, "arguments": json.dumps(payload, ensure_ascii=False)})
        if normalized:
            logger.info("parsed %s emulated tool call(s): %s", len(normalized), [c["name"] for c in normalized])
            return normalized
    return None
