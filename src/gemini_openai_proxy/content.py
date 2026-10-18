"""Turn OpenAI-style messages into (prompt, files) inputs for gemini_webapi.

gemini_webapi accepts `files: list[str | Path | bytes]` per message — we
materialize every image/file reference (data URL, http URL, local path) into a
temp file and collect them, plus build the flattened prompts:

* single-turn  -> plain user text (+ optional system preamble)
* multi-turn, cache miss -> transcript with role tags + a continuation rule
* multi-turn, cache hit  -> only the newest user message (session knows the rest)
"""

from __future__ import annotations

import base64
import binascii
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

DATA_URL_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.S)

_EXT_BY_FMT = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
    "gif": "gif",
    "heic": "heic",
    "pdf": "pdf",
    "csv": "txt",
    "json": "txt",
    "plain": "txt",
    "markdown": "txt",
    "octet-stream": "bin",
}

MAGIC = {
    b"\x89PNG": "png",
    b"\xff\xd8\xff": "jpg",
    b"GIF8": "gif",
    b"%PDF": "pdf",
}


class MediaError(ValueError):
    """An image/file part could not be decoded or downloaded."""


def _ext_for_media_type(media_type: str) -> str:
    fmt = (media_type or "").split("/", 1)[-1].lower()
    return _EXT_BY_FMT.get(fmt.split("+")[0], "bin")


def _sniff(raw: bytes) -> str:
    for magic, ext in MAGIC.items():
        if raw.startswith(magic):
            return ext
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return "bin"


def write_temp(raw: bytes, suffix: str, holder: list[Path]) -> Path:
    fd = tempfile.NamedTemporaryFile(prefix="gwop_", suffix=f".{suffix}", delete=False)
    path = Path(fd.name)
    fd.close()
    path.write_bytes(raw)
    holder.append(path)
    return path


def decode_data_url(url: str, max_bytes: int) -> bytes:
    match = DATA_URL_RE.match(url)
    if not match:
        raise MediaError("Malformed data URL in attachment part.")
    try:
        raw = base64.b64decode(match.group(3), validate=False)
    except (binascii.Error, ValueError) as exc:
        raise MediaError(f"Invalid base64 in data URL: {exc}") from None
    if len(raw) > max_bytes:
        raise MediaError(f"Attachment exceeds the {max_bytes} byte limit.")
    return raw


async def download_url(url: str, http: httpx.AsyncClient, max_bytes: int) -> bytes:
    try:
        resp = await http.get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gemini.google.com/"},
            follow_redirects=True,
            timeout=60,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise MediaError(
            f"Could not download attachment {url[:96]}: {exc.__class__.__name__}"
        ) from None
    if len(resp.content) > max_bytes:
        raise MediaError(f"Attachment exceeds the {max_bytes} byte limit.")
    if not resp.content:
        raise MediaError(f"Attachment {url[:96]} downloaded as an empty body.")
    return resp.content


def text_of(content: Any) -> str:  # type: ignore[name-defined]
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _media_urls_of(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    urls: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if str(part.get("type", "")).lower() not in ("image_url", "file_url", "input_image", "image"):
            continue
        inner = part.get("image_url") or part.get("file_url") or part.get("url")
        url = inner.get("url") if isinstance(inner, dict) else (inner if isinstance(inner, str) else None)
        if isinstance(url, str):
            urls.append(url)
    return urls


def media_urls_of(content: Any) -> list[str]:
    """Public alias — callers outside content.py shouldn't touch privates."""
    return _media_urls_of(content)


def build_turn_text(role: str, content: Any) -> tuple[str, list[str]]:
    text = text_of(content)
    return f"[{role.capitalize()}]: {text}", _media_urls_of(content)


def _remote_url_ext(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lstrip(":.").lower()
    return suffix if suffix in _EXT_BY_FMT.values() else ""


class MediaCollector:
    """Materializes attachment references into temp files (needs a running loop)."""

    def __init__(self, *, max_bytes: int, http: httpx.AsyncClient):
        self.max_bytes = max_bytes
        self.http = http
        self.files: list[Path] = []
        self.notes: list[str] = []

    async def collect(self, messages: list[Any], *, only_last: bool = False) -> None:
        targets = messages[-1:] if only_last else messages
        for msg in targets:
            if getattr(msg, "role", "") != "user":
                continue
            for url in _media_urls_of(msg.content):
                try:
                    self.files.append(await self._materialize(url))
                except MediaError as exc:
                    self.notes.append(str(exc))
                except Exception as exc:  # noqa: BLE001 - never let one bad part kill the call
                    self.notes.append(f"{url[:64]}: {exc.__class__.__name__}")

    async def _materialize(self, url: str) -> Path:
        if url.startswith("data:"):
            header = url.split(",", 1)[0]
            fmt = header[5:].split(";", 1)[0]
            raw = decode_data_url(url, self.max_bytes)
            return write_temp(raw, _ext_for_media_type(fmt) if fmt else _sniff(raw), self.files)
        if url.startswith(("http://", "https://")):
            raw = await download_url(url, self.http, self.max_bytes)
            return write_temp(raw, _remote_url_ext(url) or _sniff(raw), self.files)
        local = Path(url).expanduser()
        if local.is_file():
            return local
        raise MediaError(f"Unsupported attachment source: {url[:64]}")


def make_fallback_prompt(
    history: list[tuple[str, str]], final_user: str, system: str | None, has_files: bool
) -> str:
    """Transcript prompt for the no-cached-session path."""
    lines: list[str] = []
    if system:
        lines.append("System instruction (apply throughout):\n" + system)
    for role, text in history:
        if text or role == "User":
            lines.append(f"[{role}]: {text}")
    marker = " (a file/image is attached to this message)" if has_files else ""
    lines.append(
        "You are the [Assistant] in this conversation. Respond only to the final "
        "[User] message below and continue the conversation naturally. Do not "
        "repeat earlier replies."
    )
    lines.append(f"[User]{marker}: {final_user}")
    return "\n\n".join(lines)


def flatten_single_prompt(user_text: str, system: str | None) -> str:
    if not system:
        return user_text
    return f"System instruction (apply throughout):\n{system}\n\n---\n\n{user_text}"
