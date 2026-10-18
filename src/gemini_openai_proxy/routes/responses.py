"""`POST /v1/responses` — the OpenAI **Responses API** surface Codex ≥0.145 needs.

Wraps the same Gemini Web session as `/v1/chat/completions`: instructions plus
input items are translated into chat messages, and the reply is re-emitted as a
Responses object (streamed as `response.*` events when `stream=true`).

Codex's agent actions (running shell commands, applying patches) require the
model to emit `function_call` items — Gemini Web cannot produce those, so tool
definitions are accepted but never used: turns work as chat, not as tools.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


def _message_item(item_id: str, text: str, status: str = "completed") -> dict:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response_body(resp_id: str, model_name: str, text: str, prompt_chars: int, status: str = "completed") -> dict:
    return {
        "id": resp_id,
        "object": "response",
        "created_at": unix_ts(),
        "status": status,
        "model": model_name,
        "output": [_message_item("msg_" + resp_id[5:], text)] if text else [],
        "usage": {
            "input_tokens": max(1, prompt_chars // 4),
            "output_tokens": max(1, len(text) // 4),
            "total_tokens": max(2, (prompt_chars + len(text)) // 4),
        },
        "store": False,
        "error": None,
        "incomplete_details": None,
    }


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

    try:
        await svc.start()
        cooldown = cooldown_guard(svc.cooldown_remaining())
        if cooldown is not None:
            return cooldown
        chat_req = req.to_chat_request()
    except UpstreamError as exc:
        return JSONResponse(
            {"error": {"message": exc.message, "type": "upstream_error", "code": exc.code}},
            status_code=exc.status,
        )

    resp_id = new_response_id()
    prompt_chars = sum(len(str(m.content)) for m in chat_req.messages)
    start = time.monotonic()

    if not req.stream:
        try:
            output, model_name, _notes = await svc.chat_non_stream(chat_req)
        except UpstreamError as exc:
            return _failed_response(exc)
        except Exception as exc:  # noqa: BLE001
            return _failed_response(map_exception(exc))
        text = _text_of(output)
        body = _response_body(resp_id, model_name, text, prompt_chars)
        logger.info("responses %s done: %d chars in %.1fs", resp_id, len(text), time.monotonic() - start)
        return JSONResponse(body)

    chat_req.stream = True

    async def event_stream():
        model_name = chat_req.model or "default"
        created = _response_body(resp_id, model_name, "", prompt_chars, status="in_progress")
        yield _sse("response.created", {"type": "response.created", "response": created})
        yield _sse("response.in_progress", {"type": "response.in_progress", "response": created})

        collected: list[str] = []
        item_id = "msg_" + resp_id[5:]
        try:
            output = None
            async for chunk in svc.chat_stream(chat_req):
                output = chunk
                delta = getattr(chunk, "text_delta", "") or ""
                if delta:
                    collected.append(delta)
                    yield _sse("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                        "sequence_number": len(collected),
                    })
            text = "".join(collected) or _text_of(output if output is not None else None)
        except UpstreamError as exc:
            yield _sse("response.failed", {
                "type": "response.failed",
                "response": _response_body(resp_id, model_name, "", prompt_chars,
                                           status="failed")
                | {"error": {"message": exc.message, "code": exc.code}},
            })
            return
        except Exception as exc:  # noqa: BLE001
            mapped = map_exception(exc)
            yield _sse("response.failed", {
                "type": "response.failed",
                "response": _response_body(resp_id, model_name, "", prompt_chars,
                                           status="failed")
                | {"error": {"message": mapped.message, "code": mapped.code}},
            })
            return

        text = "".join(collected)
        final = _response_body(resp_id, model_name, text, prompt_chars)
        if text:
            yield _sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
            })
            yield _sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": _message_item(item_id, text),
            })
        yield _sse("response.completed", {"type": "response.completed", "response": final})
        logger.info("responses stream %s done: %d chars in %.1fs", resp_id, len(text), time.monotonic() - start)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


def _failed_response(exc: UpstreamError) -> JSONResponse:
    logger.warning("responses error %s: %s", exc.status, exc.message)
    return JSONResponse(
        {
            "error": {
                "message": exc.message,
                "type": "upstream_error",
                "code": exc.code,
                **({"hint": exc.hint} if exc.hint else {}),
            }
        },
        status_code=exc.status,
    )
