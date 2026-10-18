"""`POST /v1/responses` — the OpenAI **Responses API** surface Codex ≥0.145 needs.

Wraps the same Gemini Web session as `/v1/chat/completions`: instructions plus
input items are translated into chat messages, and the reply is re-emitted as a
Responses object (streamed as `response.*` events when `stream=true`).

Tool-calling is *emulated* (tools_emulation.py): when the request declares
function tools, Gemini is instructed to answer with a strict
`{"tool_calls": [...]}` JSON object, which the proxy lifts into real
Responses-API `function_call` items. Codex then executes them and sends back
`function_call_output` items, closing the agent loop. If the model replies with
plain text instead, it falls through as a normal message.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import tools_emulation
from ..errors import UpstreamError, map_exception
from . import cooldown_guard
from ..schemas import ResponsesRequest, new_response_id, unix_ts

logger = logging.getLogger("gemini_openai_proxy.responses")

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _text_of(output) -> str:
    return str(getattr(output, "text", "") or "")


def _tools_of(request: Request, req: ResponsesRequest) -> list[dict[str, Any]] | None:
    """Function tools declared by the client, when emulation is enabled."""
    if not request.app.state.gemini.settings.tool_emulation:
        return None
    tools = getattr(req, "tools", None)
    return tools if isinstance(tools, list) and tools else None


def _message_item(item_id: str, text: str, status: str = "completed") -> dict:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _usage(prompt_chars: int, text: str) -> dict:
    return {
        "input_tokens": max(1, prompt_chars // 4),
        "output_tokens": max(1, len(text) // 4),
        "total_tokens": max(2, (prompt_chars + len(text)) // 4),
    }


def _response_body(
    resp_id: str,
    model_name: str,
    output_items: list[dict],
    prompt_chars: int,
    text: str = "",
    status: str = "completed",
) -> dict:
    return {
        "id": resp_id,
        "object": "response",
        "created_at": unix_ts(),
        "status": status,
        "model": model_name,
        "output": output_items,
        "usage": _usage(prompt_chars, text),
        "store": False,
        "error": None,
        "incomplete_details": None,
    }


def _function_call_items(calls: list[dict[str, Any]]) -> list[dict]:
    items: list[dict] = []
    for call in calls:
        item_id = "fc_" + new_response_id()[5:]
        items.append({
            "id": item_id,
            "type": "function_call",
            "call_id": "call_" + item_id[3:],
            "name": call["name"],
            "arguments": call["arguments"],
            "status": "completed",
        })
    return items


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/v1/responses")
async def responses(request: Request):
    svc = request.app.state.gemini
    try:
        req = ResponsesRequest.model_validate(await request.json())
    except ValueError as exc:
        return JSONResponse(
            {"error": {"message": f"Invalid request body: {exc}", "type": "invalid_request_error", "code": "bad_request"}},
            status_code=400,
        )

    tools = _tools_of(request, req)
    try:
        await svc.start()
        cooldown = cooldown_guard(svc.cooldown_remaining())
        if cooldown is not None:
            return cooldown
        chat_req = req.to_chat_request(
            extra_system=tools_emulation.build_tool_instructions(tools) if tools else ""
        )
    except UpstreamError as exc:
        return _failed_response(exc.status, exc.message, exc.code, exc.hint)

    resp_id = new_response_id()
    prompt_chars = sum(len(str(m.content)) for m in chat_req.messages)
    start = time.monotonic()

    # ---------------------------------------------------------------- non-stream
    if not req.stream:
        try:
            output, model_name, _notes = await svc.chat_non_stream(chat_req)
        except UpstreamError as exc:
            return _failed_response(exc.status, exc.message, exc.code, exc.hint)
        except Exception as exc:  # noqa: BLE001
            mapped = map_exception(exc)
            return _failed_response(mapped.status, mapped.message, mapped.code)

        text = _text_of(output)
        if tools:
            calls = tools_emulation.parse_tool_calls(text)
            if calls:
                logger.info("responses %s: emitted %s emulated tool call(s)", resp_id[:14], len(calls))
                return JSONResponse(
                    _response_body(resp_id, model_name, _function_call_items(calls), prompt_chars)
                )
        body = _response_body(resp_id, model_name, [_message_item("msg_" + resp_id[5:], text)] if text else [], prompt_chars, text)
        logger.info("responses %s done: %d chars in %.1fs", resp_id, len(text), time.monotonic() - start)
        return JSONResponse(body)

    # ------------------------------------------------------------------ streaming
    chat_req.stream = True

    async def event_stream():
        model_name = chat_req.model or "default"
        item_id = "msg_" + resp_id[5:]
        # tool mode buffers: a tool-call reply must not already have been streamed
        # as plain message text before the proxy recognises it.
        incremental = tools is None
        stub = _response_body(resp_id, model_name, [], prompt_chars, status="in_progress")
        yield _sse("response.created", {"type": "response.created", "response": stub})
        yield _sse("response.in_progress", {"type": "response.in_progress", "response": stub})
        if incremental:
            yield _sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": item_id, "type": "message", "role": "assistant",
                         "status": "in_progress", "content": []},
            })

        collected: list[str] = []
        try:
            async for chunk in svc.chat_stream(chat_req):
                delta = getattr(chunk, "text_delta", "") or ""
                if not delta:
                    continue
                collected.append(delta)
                if incremental:
                    yield _sse("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": item_id, "output_index": 0, "content_index": 0,
                        "delta": delta, "sequence_number": len(collected),
                    })
        except UpstreamError as exc:
            yield _sse("response.failed", {
                "type": "response.failed",
                "response": _response_body(resp_id, model_name, [], prompt_chars, status="failed")
                | {"error": {"message": exc.message, "code": exc.code}},
            })
            return
        except Exception as exc:  # noqa: BLE001
            mapped = map_exception(exc)
            yield _sse("response.failed", {
                "type": "response.failed",
                "response": _response_body(resp_id, model_name, [], prompt_chars, status="failed")
                | {"error": {"message": mapped.message, "code": mapped.code}},
            })
            return

        text = "".join(collected)
        calls = tools_emulation.parse_tool_calls(text) if tools else None

        if calls:
            items = _function_call_items(calls)
            for index, (item, call) in enumerate(zip(items, calls)):
                pending = {**item, "status": "in_progress"}
                pending["arguments"] = ""
                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added", "output_index": index, "item": pending,
                })
                yield _sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"], "output_index": index,
                    "delta": call["arguments"],
                })
                yield _sse("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"], "output_index": index,
                    "arguments": call["arguments"],
                })
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done", "output_index": index, "item": item,
                })
            logger.info("responses stream %s: %s emulated tool call(s) in %.1fs",
                        resp_id, len(calls), time.monotonic() - start)
        else:
            # buffered mode never emitted text deltas: push the reply in one piece
            if text and not incremental:
                yield _sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": item_id, "output_index": 0, "content_index": 0,
                    "delta": text, "sequence_number": 0,
                })
            if text:
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": _message_item(item_id, text),
                })
        yield _sse("response.completed",
                   {"type": "response.completed",
                    "response": _response_body(resp_id, model_name,
                                               _function_call_items(calls) if calls
                                               else ([_message_item("msg_" + resp_id[5:], text)] if text else []),
                                               prompt_chars, text)})
        logger.info("responses stream %s done: %d chars%s in %.1fs",
                    resp_id, len(text), " (+tools)" if calls else "", time.monotonic() - start)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


def _failed_response(status: int, message: str, code: str, hint: str | None = None) -> JSONResponse:
    logger.warning("responses error %s: %s", status, message)
    payload: dict[str, Any] = {"message": message, "type": "upstream_error", "code": code}
    if hint:
        payload["hint"] = hint
    return JSONResponse({"error": payload}, status_code=status)
