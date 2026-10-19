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

_TOOL_PROTOCOL = """## Tool-calling protocol

This is how my own coding-agent harness works; I (the user) set it up and it is a
normal, legitimate part of this task. You cannot run anything yourself — to have
something done, ask the harness to execute a function, and it will run it on my
machine and send you the result. When you want functions to run, your ENTIRE reply
must be exactly one JSON object and nothing else:

{"tool_calls": [{"name": "<function name>", "arguments": {<json params>}}]}

Example valid reply:

{"tool_calls": [{"name": "<<EX_NAME>>", "arguments": <<EX_ARGS>>}]}

For arguments that contain code or other multi-line text (writing files), JSON string
escaping is error-prone, so use RAW BLOCKS instead — no escaping at all, content is
copied verbatim, and your ENTIRE reply is still only the blocks:

@@tool <<EX_NAME>>
@@arg <<EX_KEY>>
cat > src/app.js << 'EOF'
console.log("quotes and backslashes need no escaping here");
EOF
@@end

Each `@@arg <name>` line starts a parameter; its value is every following line up to
the next `@@arg` or `@@end` line. Several `@@tool ... @@end` blocks may follow each other.
Use plain JSON for simple calls and raw blocks for anything with code in it.

Rules:
- `arguments` is always a JSON object (use {} when there are no parameters).
- Only use the function names listed below; never invent one. Copy the example's shape,
  not its name.
- To run several functions, list them in one tool_calls array. Batch only independent
  calls: anything that checks or builds what an earlier command creates (audits,
  verification, builds) must wait for a later reply, after that command's result.
- NO markdown fences, NO prose, NO explanation before or after — only the JSON.
- Pasting shell commands as markdown does NOT execute anything; the agent only
  runs what arrives in the tool_calls JSON above.
- Never announce an action in prose ("I will now create…", "Let me set up…"): either
  emit the tool_calls JSON that does it, or you are finished. Do not claim work is done
  unless a tool result showed it.
- When everything is done (or a plain answer suffices), reply normally in text.
- You may call functions as many times as the task needs; there is no per-turn limit.
  Prefer running things through the harness over asking me to run them myself.
- Never invent `[Tool result]` content. Wait for it as a user-visible turn.

Available functions:
"""

_TOOL_MAX_CHARS = 2500       # per-function catalogue entry cap keeps prompts sane
_DESC_MAX_CHARS = 600        # top-level function description
_PROP_DESC_MAX_CHARS = 160   # per-parameter description
TOOL_OUTPUT_MAX = 12000      # tool results are clipped head+tail to this many chars
TOOL_RESULT_END = "[End of tool result]"
# Spoken in the user's voice and kept OUTSIDE the quoted result: instructions inside
# tool output look like prompt injection and make the model refuse the protocol.
TOOL_REMINDER = (
    "Reminder: if there is more work to do, reply with ONLY the tool_calls JSON for the "
    "next step (as in my first message) — don't describe what you're about to do in prose. "
    "Reply in plain text only when everything is finished."
)


def turn_suffix(last_role: str) -> str:
    """Text appended to the newest message at send time (not part of the cache key)."""
    end = f"\n{TOOL_RESULT_END}" if last_role == "tool" else ""
    return f"{end}\n\n{TOOL_REMINDER}"

def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def clip_output(text: str, limit: int = TOOL_OUTPUT_MAX) -> str:
    """Keep head and tail of a long tool result (errors usually sit at the end)."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n…[{omitted} chars omitted]…\n{text[-half:]}"


def _compact_schema(node: Any) -> Any:
    """Shrink a JSON schema for the prompt: drop boilerplate, trim descriptions."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("additionalProperties", "$schema", "strict"):
                continue
            if key == "description" and isinstance(value, str):
                out[key] = _clip(" ".join(value.split()), _PROP_DESC_MAX_CHARS)
            else:
                out[key] = _compact_schema(value)
        return out
    if isinstance(node, list):
        return [_compact_schema(v) for v in node]
    return node


def render_calls(calls: list[tuple[str, str]]) -> str:
    """Canonical JSON for (name, arguments-json-string) pairs.

    Used for the assistant turn in replayed history *and* for the session-cache
    alias, so both sides hash to the same text.
    """
    items: list[dict[str, Any]] = []
    for name, args in calls:
        try:
            parsed: Any = json.loads(args) if args and args.strip() else {}
        except ValueError:
            parsed = args
        items.append({"name": name, "arguments": parsed})
    return json.dumps({"tool_calls": items}, ensure_ascii=False)


def custom_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    """Freeform (`type: custom`, e.g. Codex apply_patch) tools take a raw string input."""
    return {
        str(t.get("name")) for t in tools or []
        if t.get("type") == "custom" and t.get("name")
    }


# Invented limits: unambiguous, safe to act on in any round.
_STRONG_REFUSAL = re.compile(
    r"prompt injection|injection attempt|(?:cannot|can't|won't|will not) (?:use|follow) the (?:custom )?tool[- ]?calling protocol|"
    r"tool[- ]?call(?:ing|s)?\s+(?:budget|limit|quota)|"
    r"(?:maximum|max(?:imum)? number|limit) of tool[- ]?calls?|"
    r"(?:reached|hit|exceeded|used up|run out of)\s+(?:the |my )?(?:\w+\s+){0,2}(?:tool[- ]?calls?|tool budget)|"
    r"budget (?:is |has been )?exhausted|"
    r"(?:can(?:'|no)t|unable to|not able to|no longer able to)\s+(?:\w+\s+){0,3}"
    r"(?:run|execute|make|use|call|invoke|perform)\s+(?:\w+\s+){0,3}"
    r"(?:commands?|terminal|shell|tools?|tool[- ]?calls?|functions?)",
    re.I,
)
# Softer "do it yourself" phrasing: only meaningful mid-task, right after a tool result.
_SOFT_REFUSAL = re.compile(
    r"guide you (?:through|on)|run (?:this|these|the following) (?:command|yourself)s?\b|"
    r"you (?:can|could|should|will need to) (?:run|execute|type) ",
    re.I,
)


def looks_like_refusal(text: str, strong_only: bool = False) -> bool:
    """Prose where the model disclaims tool use (chat-persona drift / invented limits)."""
    if not text or parse_tool_calls(text) is not None:
        return False
    return bool(_STRONG_REFUSAL.search(text)) or (not strong_only and bool(_SOFT_REFUSAL.search(text)))


_VERBS = (
    r"(?:creat|set(?:ting)?\s+up|build|writ|install|run|start|initializ|generat|scaffold|implement|"
    r"mak|add|updat|fix|open|check|list|read|search|look|inspect|configur|edit|modif|delet|remov)"
)
# "I will now create…", "Let me set up…", "I'm creating the directory…"
_ANNOUNCE = re.compile(
    rf"\bI(?:'ll|\s+will|\s+am|'m|\s+shall)\s+(?:now\s+|just\s+|first\s+|going\s+to\s+)*{_VERBS}\w*|"
    rf"\blet\s+me\s+(?:\w+\s+){{0,2}}{_VERBS}\w*|\bI(?:'m|\s+am)\s+now\b",
    re.I,
)
# "I have successfully created…", "What was built"
_CLAIM = re.compile(
    r"\bI(?:'ve|\s+have)\s+(?:\w+\s+)?(?:creat|built|install|written|wrote|configur|set\s+up|implement|generat|scaffold)\w*|"
    r"\bsuccessfully\s+(?:creat|built|install|configur|implement|generat)\w*|\bwhat\s+(?:was|i)\s+built\b|"
    r"\bfeatures?\s+built\b|\bproject\s+overview\b",
    re.I,
)
# Gemini Web's own canned failure, returned as if it were an answer.
_ERROR_TEXT = re.compile(
    r"i seem to be encountering an error|can i try something else|something went wrong|"
    r"i'?m having trouble (?:with )?(?:that|this)|an error occurred while|"
    r"i encountered an error|having a hard time fulfilling|could you try again|"
    r"can i help you with something else",
    re.I,
)
# Shell commands that create/change something (plain mkdir/ls/cat/grep do not count).
_MUTATING_CMD = re.compile(
    r"(?:^|[\s;&|(])(?:tee|touch|cp|mv|rm|git\s+(?:init|add|commit|clone)|sed\s+-i|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:create|init|i|install|add|remove|uninstall)|npx\s+(?:-y\s+)?create|"
    r"pip3?\s+install|go\s+(?:mod|get)|cargo\s+(?:add|init|new)|"
    r"(?:python3?|node)\b[^\n]*(?:write|open\([^)]*['\"][wa]|>))|"
    r">>?\s*(?!/dev/null)[^\s&|]|<<",
)


def is_upstream_error_text(text: str) -> bool:
    return bool(text) and len(text) < 400 and bool(_ERROR_TEXT.search(text))


def is_mutating(name: str, arguments: str) -> bool:
    """Did this tool call (try to) change the workspace? Used to catch fake "done" claims."""
    lowered = name.lower()
    if "patch" in lowered or lowered in {"write_file", "create_file", "edit_file"}:
        return True
    if lowered == "write_stdin":
        return False
    try:
        args = json.loads(arguments) if arguments and arguments.strip() else {}
    except ValueError:
        return bool(_MUTATING_CMD.search(arguments))
    cmd = args.get("cmd") or args.get("command") if isinstance(args, dict) else None
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return bool(cmd) and bool(_MUTATING_CMD.search(str(cmd)))


def announces_action(text: str) -> bool:
    """Prose that promises an action instead of emitting the tool call for it."""
    return bool(text) and parse_tool_calls(text) is None and bool(_ANNOUNCE.search(text[:600]))


def claims_work_done(text: str) -> bool:
    return bool(text) and parse_tool_calls(text) is None and bool(_CLAIM.search(text[:1500]))


def scrub_history(content: str, tools_used_this_turn: bool) -> str:
    """Replace a past assistant turn that would teach the model to narrate instead of act.

    Invented limits and "I will now…" announcements are never legitimate; "I have
    created…" is only legitimate when a tool actually ran earlier in that turn.
    """
    if (
        looks_like_refusal(content, strong_only=True)
        or announces_action(content)
        or (not tools_used_this_turn and claims_work_done(content))
    ):
        return "(Previous reply omitted.)"
    return content


def scrub_refusal_history(content: str) -> str:  # kept for callers/tests of the old name
    return scrub_history(content, tools_used_this_turn=True)


def expand_tools(
    tools: list[dict[str, Any]] | None, exclude: str = ""
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Flatten Responses-API `namespace` tools (how Codex ships MCP tools).

    Returns (flat tool list, {function name: namespace}). The namespace is echoed
    back on the emitted `function_call` item so Codex can route it to the right
    MCP server. Namespaces matching `exclude` (regex) are dropped, and a nested
    name already taken (by a top-level tool or an earlier namespace) is skipped.
    """
    flat: list[dict[str, Any]] = []
    namespace_of: dict[str, str] = {}
    taken = {str(t.get("name")) for t in tools or [] if t.get("type") != "namespace" and t.get("name")}
    pattern = re.compile(exclude) if exclude else None
    dropped: list[str] = []
    for tool in tools or []:
        if tool.get("type") != "namespace":
            flat.append(tool)
            continue
        ns = str(tool.get("name") or "")
        if pattern and pattern.search(ns):
            dropped.append(ns)
            continue
        for sub in tool.get("tools") or []:
            if not isinstance(sub, dict) or not sub.get("name") or sub.get("type", "function") != "function":
                continue
            name = str(sub["name"])
            if name in taken:
                logger.warning("tool name %r in namespace %s collides with another tool; skipped", name, ns)
                continue
            taken.add(name)
            namespace_of[name] = ns
            flat.append({**sub, "type": "function"})
    if dropped:
        logger.info("tool emulation: excluded namespaces (GWOP_TOOL_EXCLUDE): %s", ", ".join(dropped))
    return flat, namespace_of


def _example_call(tools: list[dict[str, Any]]) -> tuple[str, str, str]:
    """(name, arguments-json, arg-key) for the protocol example, from a REAL declared tool.

    Models copy the example verbatim; a made-up name there gets called for real.
    """
    functions = [t.get("function", t) for t in tools or [] if t.get("type") == "function"]
    for fn in functions:
        props = (fn.get("parameters") or {}).get("properties") or {}
        for key in ("cmd", "command"):
            kind = (props.get(key) or {}).get("type")
            if kind == "string":
                return str(fn["name"]), json.dumps({key: "ls -la"}), key
            if kind == "array":
                return str(fn["name"]), json.dumps({key: ["bash", "-lc", "ls -la"]}), key
    if functions and functions[0].get("name"):
        return str(functions[0]["name"]), "{}", "input"
    return "tool_name", "{}", "input"


def build_tool_instructions(tools: list[dict[str, Any]]) -> str:
    """Render the tool protocol + function catalogue as a system-prompt section."""
    ex_name, ex_args, ex_key = _example_call(tools)
    protocol = (
        _TOOL_PROTOCOL.replace("<<EX_NAME>>", ex_name)
        .replace("<<EX_ARGS>>", ex_args)
        .replace("<<EX_KEY>>", ex_key)
    )
    sections: list[str] = [protocol]
    skipped: list[str] = []
    for tool in tools or []:
        if tool.get("type") == "local_shell":
            sections.append("- local_shell — run a shell command; arguments: "
                            '{"command": ["bash", "-lc", "<command string>"]}')
            continue
        if tool.get("type") == "custom":
            name = str(tool.get("name") or "").strip()
            if name:
                desc = _clip(str(tool.get("description") or ""), 1500)
                sections.append(
                    f'- {name} (freeform tool) — arguments: {{"input": "<the complete raw text '
                    f'for this tool, exactly as it must be passed>"}}. {desc}'
                )
            continue
        if tool.get("type") != "function":
            skipped.append(f'{tool.get("type")}:{tool.get("name") or tool.get("server_label") or "?"}')
            continue  # mcp / web_search types: no Gemini Web equivalent
        fn = tool.get("function", {}) if "function" in tool else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        entry = {
            "name": name,
            "description": _clip(" ".join(str(fn.get("description") or "").split()), _DESC_MAX_CHARS),
            "parameters": _compact_schema(fn.get("parameters") or {}),
        }
        blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        if len(blob) > _TOOL_MAX_CHARS:
            blob = blob[:_TOOL_MAX_CHARS] + "…(truncated)"
        sections.append(f"- {blob}")
    if skipped:
        logger.info("tool emulation: %d tool(s) not offered to Gemini (unsupported type): %s",
                    len(skipped), ", ".join(skipped[:12]))
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


_BARE_KEY = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\s*):")


def _repair_json(candidate: str) -> str:
    """Common Gemini Web corruptions: bare keys (`doctrine: ""`), a missing
    comma before them, and stray trailing commas inside objects.
    Second-chance parse only, after strict JSON failed. String-aware:
    `key:` patterns inside string values stay untouched."""
    out: list[str] = []
    i, in_str, escaped, prev_sig = 0, False, False, ""
    while i < len(candidate):
        ch = candidate[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
                prev_sig = '"'
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        m = _BARE_KEY.match(candidate, i)
        if m:
            if prev_sig not in ("", "{", "[", ","):
                out.append(",")  # value ends right before the bare key: ',' was lost
            out.append(f'"{m.group(1)}"{m.group(2)}:')
            prev_sig = ":"
            i = m.end()
            continue
        out.append(ch)
        if not ch.isspace():
            prev_sig = ch
        i += 1
    return re.sub(r",\s*([}\]])", r"\1", "".join(out))


_RAW_TOOL = re.compile(r"^@@tool[ \t]+([\w.\-:]+)[ \t]*$")
_RAW_ARG = re.compile(r"^@@arg[ \t]+([\w.\-]+)[ \t]*$")
_RAW_END = re.compile(r"^@@end[ \t]*$")


def _schema_types(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, str]]:
    types: dict[str, dict[str, str]] = {}
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        props = (fn.get("parameters") or {}).get("properties") or {}
        if fn.get("name") and isinstance(props, dict):
            types[str(fn["name"])] = {
                k: str(v.get("type")) for k, v in props.items() if isinstance(v, dict) and v.get("type")
            }
    return types


def _coerce(value: str, kind: str | None) -> Any:
    if kind in ("integer", "number"):
        try:
            return int(value) if kind == "integer" else float(value)
        except ValueError:
            return value
    if kind == "boolean":
        return value.strip().lower() in {"true", "1", "yes"}
    if kind in ("array", "object"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _parse_raw_blocks(text: str, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """`@@tool name` / `@@arg key` / raw lines / `@@end` — no JSON escaping needed."""
    if "@@tool" not in text:
        return None
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]  # the whole reply wrapped in one fence
    types = _schema_types(tools)
    calls: list[dict[str, Any]] = []
    name: str | None = None
    args: dict[str, Any] = {}
    key: str | None = None
    buf: list[str] = []

    def close_arg() -> None:
        nonlocal key, buf
        if name is not None and key is not None:
            args[key] = _coerce("\n".join(buf).strip("\n"), types.get(name, {}).get(key))
        key, buf = None, []

    def close_tool() -> None:
        nonlocal name, args
        close_arg()
        if name is not None:
            calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False)})
        name, args = None, {}

    for line in lines:
        if (m := _RAW_TOOL.match(line)):
            close_tool()
            name = m.group(1)
        elif name is not None and (m := _RAW_ARG.match(line)):
            close_arg()
            key = m.group(1)
        elif name is not None and _RAW_END.match(line):
            close_tool()
        elif key is not None:
            buf.append(line)
    close_tool()
    return calls or None


def _to_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass
    return [part.strip() for part in re.split(r"[\n,]", value) if part.strip()]


def _coerce_call_args(calls: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Fix scalar-vs-array / number-as-string mistakes against the declared schema.

    Seen live: `"files": "a.tsx\nb.css"` for an array parameter made an audit tool
    check nothing and report PASS.
    """
    types = _schema_types(tools)
    for call in calls:
        props = types.get(call["name"])
        if not props:
            continue
        try:
            args = json.loads(call["arguments"])
        except ValueError:
            continue
        if not isinstance(args, dict):
            continue
        changed = False
        for key, value in list(args.items()):
            kind = props.get(key)
            if kind == "array" and isinstance(value, str):
                args[key], changed = _to_list(value), True
            elif kind in ("integer", "number", "boolean") and isinstance(value, str):
                args[key], changed = _coerce(value, kind), True
        if changed:
            call["arguments"] = json.dumps(args, ensure_ascii=False)
    return calls


def parse_tool_calls(text: str, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]] | None:
    """Lift tool calls out of a model reply (JSON object or raw `@@tool` blocks).

    Returns a list of {"name": str, "arguments": str} (arguments already a JSON
    *string*, as the OpenAI tool-calling format expects), or None when the reply
    isn't a tool-call turn (normal prose gets returned to the model untouched).
    `tools` (the request's declarations) is only used to type raw-block values.
    """
    if not text:
        return None
    raw_calls = _parse_raw_blocks(text, tools)
    if raw_calls:
        logger.info("parsed %s raw-block tool call(s): %s", len(raw_calls), [c["name"] for c in raw_calls])
        return _coerce_call_args(raw_calls, tools)
    if "{" not in text:
        return None
    decoder = json.JSONDecoder(strict=False)
    for candidate in _candidate_spans(text):
        for attempt in (candidate, _repair_json(candidate)):
            try:
                try:
                    data = json.loads(attempt, strict=False)
                except json.JSONDecodeError:
                    # a valid object followed by junk (stray "}" etc.): keep the object
                    start = attempt.find("{")
                    if start == -1:
                        raise
                    data, end = decoder.raw_decode(attempt, start)
                    logger.info("tool JSON salvaged: ignored %d trailing char(s)", len(attempt) - end)
                if attempt is not candidate:
                    logger.info("tool JSON salvaged after bare-key/comma repair")
            except (json.JSONDecodeError, ValueError):
                continue
            calls = _normalize_calls(data)
            if calls:
                logger.info("parsed %s emulated tool call(s): %s", len(calls), [c["name"] for c in calls])
                return _coerce_call_args(calls, tools)
    return None


def _normalize_calls(data: Any) -> list[dict[str, Any]] | None:
    if isinstance(data, list):
        data = {"tool_calls": data}
    if not isinstance(data, dict):
        return None
    calls = data.get("tool_calls") or data.get("function_calls")
    if calls is None:
        # OpenAI-native single-call shape the web model sometimes prefers:
        # {"name": "...", "arguments": {...}} (or wrapped in "function_call")
        single = data["function_call"] if isinstance(data.get("function_call"), dict) else data
        if isinstance(single, dict) and str(single.get("name") or "").strip():
            calls = [single]
    if not isinstance(calls, list) or not calls:
        return None
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
    return normalized or None


_AUTOLINK = re.compile(r"\[(https?://[^\s\]]+)\]\(\1/?\)")


def unwrap_autolinks(text: str) -> str:
    """`[http://x](http://x)` -> `http://x`: Gemini Web wraps bare URLs, which a terminal shows verbatim."""
    return _AUTOLINK.sub(r"\1", text) if text and "](" in text else text
