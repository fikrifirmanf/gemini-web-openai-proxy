"""One-line summary of each route module (app assembly lives in `main.py`)."""

import asyncio
import contextlib
from typing import Any, AsyncIterator

from fastapi.responses import JSONResponse

SSE_KEEPALIVE_FRAME = ": keepalive\n\n"


async def with_keepalive(source: AsyncIterator[Any], interval: float) -> AsyncIterator[Any]:
    """Re-yield `source`, emitting `None` after every `interval`s of silence.

    Buffered replies (tool-call emulation) and slow thinking models can sit quiet
    for minutes; clients and reverse proxies drop idle SSE connections and retry,
    which re-sends the whole prompt upstream. A `None` tells the caller to write a
    keepalive frame. `interval <= 0` disables it.
    """
    it = source.__aiter__()
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval if interval > 0 else None)
            if not done:
                yield None
                continue
            task, pending = pending, None
            try:
                item = task.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pending
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


def cooldown_guard(cooldown_seconds: float) -> JSONResponse | None:
    """Park new model traffic while an upstream 429 cooldown is active.

    Returning requests-discipline as a 429 we own keeps authenticated clients from
    hammering Google past a dry quota — the pattern that escalates blocks.
    """
    if cooldown_seconds <= 0:
        return None
    remaining = int(cooldown_seconds) + 1
    return JSONResponse(
        {
            "error": {
                "message": (
                    f"Upstream rate-limited recently; new requests are cooldown-capped "
                    f"for another ~{remaining}s. Clients should use exponential backoff."
                ),
                "type": "rate_limit_error",
                "code": "rate_limited",
            }
        },
        status_code=429,
        headers={"Retry-After": str(remaining)},
    )


from . import chat, images, meta, responses  # noqa: E402,F401

__all__ = ["chat", "images", "meta", "responses", "cooldown_guard", "with_keepalive", "SSE_KEEPALIVE_FRAME"]
