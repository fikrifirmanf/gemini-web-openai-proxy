"""One-line summary of each route module (app assembly lives in `main.py`)."""

from fastapi.responses import JSONResponse


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


from . import chat, images, meta  # noqa: E402,F401

__all__ = ["chat", "images", "meta", "cooldown_guard"]
