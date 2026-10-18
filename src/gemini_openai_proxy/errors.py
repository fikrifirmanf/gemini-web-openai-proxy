"""Map gemini_webapi / httpx exceptions onto HTTP responses."""

from __future__ import annotations

import httpx
from gemini_webapi import (
    APIError,
    AuthError,
    GeminiError,
    ImageGenerationError,
    ModelInvalidError,
    TemporarilyBlockedError,
    TimeoutError as UpstreamTimeout,
)
from gemini_webapi.exceptions import UsageLimitExceededError

_STATUS_TEXT = {
    400: "bad_request",
    401: "upstream_auth_error",
    403: "refused",
    404: "not_found",
    429: "upstream_rate_limited",
    502: "upstream_error",
    503: "upstream_unavailable",
    504: "upstream_timeout",
}


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.hint = hint

    @property
    def code(self) -> str:
        return _STATUS_TEXT.get(self.status, "upstream_error")


_HINTS = {
    401: "Re-copy __Secure-1PSID / __Secure-1PSIDTS from gemini.google.com "
         "(F12 > Network > any request > Request Cookies) and restart the proxy.",
    429: "Upstream quota is exhausted or the IP is temporarily blocked. "
         "Try again later or lower GWOP_MAX_CONCURRENCY.",
}


def map_exception(exc: BaseException) -> UpstreamError:
    if isinstance(exc, AuthError):
        return UpstreamError(401, "Gemini Web session is invalid or expired.", hint=_HINTS[401])
    if isinstance(exc, (UsageLimitExceededError, TemporarilyBlockedError)):
        return UpstreamError(429, f"Upstream rate limited: {exc}")
    if isinstance(exc, UpstreamTimeout):
        return UpstreamError(504, f"Upstream request timed out: {exc}")
    if isinstance(exc, ModelInvalidError):
        return UpstreamError(400, f"Upstream rejected the model selection: {exc}")
    if isinstance(exc, (GeminiError, ImageGenerationError, APIError)):
        return UpstreamError(502, f"Upstream rejected the request: {exc}")
    if isinstance(exc, httpx.HTTPError):
        return UpstreamError(502, f"Could not reach upstream: {exc.__class__.__name__}: {exc}")
    if isinstance(exc, ValueError):
        return UpstreamError(400, str(exc))
    return UpstreamError(502, f"Unexpected upstream failure: {exc.__class__.__name__}: {exc}")
