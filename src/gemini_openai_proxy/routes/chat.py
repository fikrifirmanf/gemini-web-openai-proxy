"""`POST /v1/chat/completions` — non-streaming JSON + `stream=true` SSE."""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..errors import UpstreamError, map_exception
from . import cooldown_guard
from ..schemas import ChatCompletionRequest, make_usage, new_completion_id, unix_ts

logger = logging.getLogger("gemini_openai_proxy.chat")

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _delta_chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": unix_ts(),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _image_markdown(images: list) -> str:
    """Markdown for *generated* images only; web-scraped URLs are noise for clients."""
    lines: list[str] = []
    for image in images or []:
        if type(image).__name__ == "GeneratedImage":
            title = (image.alt or image.title or "generated image").strip() or "generated image"
            title = title.split("\n")[0][:120]
            lines.append(f"![{title}]({image.url})")
    return "\n".join(lines)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    svc = request.app.state.gemini
    try:
        req = ChatCompletionRequest.model_validate(await request.json())
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
        _system, _history, _text, _last, _mobj, model_name = svc._split(req)
    except UpstreamError as exc:
        return _error_response(exc)

    if req.stream:
        return StreamingResponse(
            _stream(svc, req, model_name),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    return await _non_stream(svc, req)


async def _non_stream(svc, req: ChatCompletionRequest):
    cid = new_completion_id()
    prompt_chars = sum(len(str(m.content)) for m in req.messages)
    try:
        output, model_name, notes = await svc.chat_non_stream(req)
    except UpstreamError as exc:
        return _error_response(exc)

    content = output.text or ""
    if svc.settings.images_in_chat:
        mark = _image_markdown(output.images or [])
        if mark:
            content = f"{content}\n\n{mark}" if content.strip() else mark
    if notes:
        logger.info("attachment notes: %s", "; ".join(notes))

    message: dict = {"role": "assistant", "content": content}
    if svc.settings.include_thoughts and getattr(output, "thoughts", None):
        message["reasoning_content"] = output.thoughts

    return JSONResponse(
        {
            "id": cid,
            "object": "chat.completion",
            "created": unix_ts(),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": make_usage(prompt_chars, len(content)),
        }
    )


def _stream(svc, req: ChatCompletionRequest, model_name: str):
    """Plain function: calling it returns the SSE async generator directly."""
    cid = new_completion_id()
    start = time.monotonic()
    completion_chars = 0
    trap = None

    async def event_stream():
        nonlocal completion_chars, trap
        yield f"data: {json.dumps(_delta_chunk(cid, model_name, {'role': 'assistant'}))}\n\n"
        try:
            output = None
            async for chunk in svc.chat_stream(req):
                output = chunk
                if svc.settings.include_thoughts:
                    thoughts = getattr(chunk, "thoughts_delta", "") or ""
                    if thoughts:
                        yield f"data: {json.dumps(_delta_chunk(cid, model_name, {'reasoning_content': thoughts}))}\n\n"
                delta = getattr(chunk, "text_delta", "") or ""
                if delta:
                    completion_chars += len(delta)
                    yield f"data: {json.dumps(_delta_chunk(cid, model_name, {'content': delta}))}\n\n"
            if output is not None and svc.settings.images_in_chat:
                mark = _image_markdown(getattr(output, "images", None) or [])
                if mark:
                    yield f"data: {json.dumps(_delta_chunk(cid, model_name, {'content': '\n\n' + mark}))}\n\n"
        except UpstreamError as exc:
            trap = exc
            yield f"data: {json.dumps({'error': {'message': exc.message, 'type': 'upstream_error', 'code': exc.code, 'hint': exc.hint}})}\n\n"
        except Exception as exc:  # noqa: BLE001
            mapped = map_exception(exc)
            trap = mapped
            yield f"data: {json.dumps({'error': {'message': mapped.message, 'type': 'upstream_error', 'code': mapped.code}})}\n\n"

        finish = _delta_chunk(cid, model_name, {}, finish="stop")
        if req.include_usage:
            usage = make_usage(sum(len(str(m.content)) for m in req.messages), completion_chars)
            finish["usage"] = usage
            finish["choices"] = []
        yield f"data: {json.dumps(finish)}\n\n"
        yield "data: [DONE]\n\n"
        if trap is not None:
            logger.warning("stream %s ended early after %.1fs: %s", cid[:16], time.monotonic() - start, trap.message)
        else:
            logger.info("stream %s done: %d chars in %.1fs", cid[:16], completion_chars, time.monotonic() - start)

    return event_stream()


def _error_response(exc: UpstreamError) -> JSONResponse:
    logger.warning("upstream error %s: %s", exc.status, exc.message)
    body: dict = {"message": exc.message, "type": "upstream_error", "code": exc.code}
    if exc.hint:
        body["hint"] = exc.hint
    return JSONResponse({"error": body}, status_code=exc.status)
