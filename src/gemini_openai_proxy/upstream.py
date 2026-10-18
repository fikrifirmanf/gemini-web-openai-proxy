"""GeminiService — owns the upstream GeminiClient and adapts it to the OpenAI API.

Responsibilities:
* lifecycle: resolve cookies -> GeminiClient -> init() (auto-refresh on)
* conversation cache: OpenAI requests are stateless; when a request's message
  history was already sent upstream, we reuse the live ChatSession and only
  forward the newest user turn
* generation: single-turn, multi-turn, streaming, image generation
* resilience: auth failures flip the client to `ready=False`; the next request
  transparently re-inits (existing session cache is dropped with it)
"""

from __future__ import annotations

import asyncio
import base64 as b64lib
import contextlib
import hashlib
import json
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from gemini_webapi import GeminiClient
from gemini_webapi.types import AvailableModel, Image

from .config import Settings
from .content import MediaCollector, build_turn_text, flatten_single_prompt, make_fallback_prompt, media_urls_of
from .cookies import GeminiCookies, resolve_cookies
from .errors import UpstreamError, map_exception

_ROLE_LABEL = {"user": "User", "assistant": "Assistant", "tool": "Assistant"}
_SYSTEM_ROLES = {"system", "developer"}


def _cache_key(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(slots=True)
class SessionEntry:
    chat: Any
    model_name: str
    created: float = field(default_factory=time.time)


class GeminiService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: GeminiClient | None = None
        self._init_lock = asyncio.Lock()
        self._gate = asyncio.Semaphore(max(1, settings.max_concurrency))
        self._sessions: OrderedDict[str, SessionEntry] = OrderedDict()
        self.ready = False
        self.last_init_error = ""
        self.cookie_source = ""
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ state

    @property
    def started(self) -> bool:
        return self.client is not None

    def _models(self) -> list[AvailableModel]:
        return list(self.client.list_models() or []) if self.client else []

    def banner(self) -> dict[str, Any]:
        account = getattr(self.client, "account_status", None)
        out = {
            "service": "gemini-openai-proxy",
            "ready": self.ready,
            "cookie_source": self.cookie_source,
            "cookie_path": str(self.settings.data_dir / "gemini_cookies"),
            "upstream_proxy": self.settings.upstream_proxy or None,
            "temporary_chats": self.settings.temporary_chats,
            "models": [
                {"id": m.model_name, "display": m.display_name, "aliases": m.aliases}
                for m in self._models()
            ],
        }
        if account is not None:
            out["account_status"] = str(getattr(account, "name", int(account)))
        return out

    # ------------------------------------------------------------ lifecycle

    def _resolve_cookies(self) -> GeminiCookies:
        try:
            cookies = resolve_cookies(
                cookie_1psid=self.settings.cookie_1psid,
                cookie_1psidts=self.settings.cookie_1psidts,
                cookie_raw=self.settings.cookie_raw,
                cookie_file=self.settings.cookie_file,
            )
        except Exception as exc:
            raise UpstreamError(
                503, f"Could not read the configured cookie source: {exc!r}",
                hint="Check GWOP_COOKIE_FILE / GWOP_GEMINI_COOKIES_RAW values.",
            ) from exc
        if not cookies:
            raise UpstreamError(
                503,
                "No Gemini cookies found.",
                hint=(
                    "Set one of: GWOP_GEMINI_1PSID (+ GWOP_GEMINI_1PSIDTS) | "
                    "GWOP_GEMINI_COOKIES_RAW='__Secure-1PSID=...; __Secure-1PSIDTS=...' | "
                    "GWOP_COOKIE_FILE=/path/cookies.txt. "
                    "Copy from gemini.google.com: F12 > Network > any request > Request Cookies."
                ),
            )
        self.cookie_source = (
            "file:" + self.settings.cookie_file
            if self.settings.cookie_file
            else "raw-header"
            if self.settings.cookie_raw
            else "env"
        )
        return cookies

    async def start(self) -> None:
        async with self._init_lock:
            if self.client is None or not self.ready:
                await self._start_inner()

    async def stop(self) -> None:
        if self.client:
            with contextlib.suppress(Exception):
                await self.client.close()
            self.client = None
        if self._http:
            await self._http.aclose()
            self._http = None
        self.ready = False

    async def reinit(self) -> dict[str, Any]:
        """Admin endpoint: tear down and rebuild the upstream client."""
        async with self._init_lock:
            if self.client:
                with contextlib.suppress(Exception):
                    await self.client.close()
                self.client = None
            self.ready = False
            await self._start_inner()
        return self.banner()

    async def _start_inner(self) -> None:
        cookies = self._resolve_cookies()
        proxy = self.settings.upstream_proxy or None
        if self._http:
            await self._http.aclose()
        self._http = httpx.AsyncClient(proxy=proxy, timeout=90)

        client = GeminiClient(cookies.secure_1psid, cookies.secure_1psidts, proxy=proxy)
        if cookies.extra:
            with contextlib.suppress(Exception):
                merged = dict(client.cookies)
                merged.update(cookies.extra)
                client.cookies = merged
        try:
            await client.init(
                timeout=self.settings.init_timeout,
                auto_refresh=True,
                verbose=self.settings.log_level == "DEBUG",
            )
        except Exception as exc:  # noqa: BLE001
            mapped = map_exception(exc)
            self.last_init_error = mapped.message
            self.ready = False
            raise UpstreamError(
                mapped.status, f"Gemini Web init failed: {mapped.message}", hint=mapped.hint
            ) from exc

        self.client = client
        self._sessions.clear()
        self.ready = True
        self.last_init_error = ""

    async def _ensure_ready(self) -> GeminiClient:
        if self.client and self.ready:
            return self.client
        await self.start()
        assert self.client is not None
        return self.client

    # --------------------------------------------------------------- helpers

    def resolve_model(self, requested: str) -> tuple[AvailableModel | None, str]:
        """(resolved_or_None, display_name). None = use account default."""
        name = (requested or "").strip()
        if not name or name in {"default", "gemini-web", "gemini-web-default", "auto"}:
            return None, "default"
        assert self.client is not None
        try:
            resolved = self.client.resolve_model(name)
        except ValueError:
            if self.settings.strict_model:
                known = [m.model_name for m in self._models()] or ["(registry empty)"]
                raise UpstreamError(
                    400,
                    f"Unknown model {name!r}. Advertised models: {', '.join(known)}.",
                ) from None
            return None, name  # lenient: account default, but echo the requested name
        return resolved, resolved.model_name

    def openai_models(self) -> list[dict[str, Any]]:
        now = int(time.time())
        out = [
            {"id": m.model_name, "object": "model", "created": now, "owned_by": "google"}
            for m in self._models()
        ]
        if not out:
            out.append({"id": "default", "object": "model", "created": now, "owned_by": "google"})
        return out

    def _media_collector(self) -> MediaCollector:
        assert self._http is not None
        return MediaCollector(max_bytes=self.settings.max_image_bytes, http=self._http)

    @staticmethod
    def _cleanup_files(files: list[Path]) -> None:
        for path in files:
            if path.name.startswith("gwop_"):
                with contextlib.suppress(OSError):
                    path.unlink()

    # ------------------------------------------------------------- chat APIs

    def _split(
        self, request: Any
    ) -> tuple[str | None, list[tuple[str, str]], str, Any, Any, str]:
        """(system, history_pairs, final_user_text, final_msg, model_obj, display_name)."""
        model, display_name = self.resolve_model(request.model)
        last = request.messages[-1]
        if str(last.role) != "user":
            raise UpstreamError(400, "The last message must have role 'user'.")

        system_parts: list[str] = []
        history: list[tuple[str, str]] = []
        for msg in request.messages[:-1]:
            label = _ROLE_LABEL.get(str(msg.role), "System")
            text, _urls = build_turn_text(label, msg.content)
            if str(msg.role) in _SYSTEM_ROLES:
                system_parts.append(text)
            elif text.strip():
                history.append((label, text))
        system = "\n\n".join(p for p in system_parts if p.strip()) or None

        final_text = build_turn_text("User", last.content)[0].removeprefix("[User]: ")
        if not final_text.strip() and not media_urls_of(last.content):
            raise UpstreamError(400, "The final user message is empty.")
        return system, history, final_text, last, model, display_name

    @staticmethod
    def _hash_for(messages: list[Any], system: str | None, model_name: str) -> str:
        payload = [
            {"role": str(m.role), "text": build_turn_text(str(m.role), m.content)[0]}
            for m in messages
        ]
        if system:
            payload.append({"role": "system", "text": system})
        payload.append({"model": model_name})
        return _cache_key(payload)

    def _remember(self, key: str, chat: Any, display_name: str) -> None:
        metadata = getattr(chat, "metadata", None)
        if not metadata or not metadata[0]:
            return
        self._sessions[key] = SessionEntry(chat=chat, model_name=display_name)
        self._sessions.move_to_end(key)
        while len(self._sessions) > self.settings.session_cache_size:
            self._sessions.popitem(last=False)

    async def chat_non_stream(self, request: Any) -> tuple[Any, str, list[str]]:
        """Returns (final ModelOutput, advertised model name, media notes)."""
        await self._ensure_ready()
        system, history, final_text, _last, model_obj, display_name = self._split(request)
        messages = request.messages
        full_key = self._hash_for(messages, system, display_name)
        prefix_key = self._hash_for(messages[:-1], system, display_name)
        entry = self._sessions.get(prefix_key)

        if entry is not None:
            collector = self._media_collector()
            await collector.collect(messages, only_last=True)
            files, notes = collector.files, collector.notes
            try:
                output = await self._send(entry.chat, final_text, files)
            finally:
                self._cleanup_files(files)
            self._remember(full_key, entry.chat, display_name)
            return output, display_name, notes

        collector = self._media_collector()
        await collector.collect(messages)
        files, notes = collector.files, collector.notes
        prompt = (
            make_fallback_prompt(history, final_text, system, has_files=bool(files))
            if history
            else flatten_single_prompt(final_text, system)
        )
        chat = self.client.start_chat(model=model_obj)
        try:
            output = await self._send(chat, prompt, files)
        finally:
            self._cleanup_files(files)
        self._remember(full_key, chat, display_name)
        return output, display_name, notes

    async def chat_stream(self, request: Any) -> AsyncGenerator[Any, None]:
        """Yields incremental ModelOutput chunks; the final one carries images."""
        await self._ensure_ready()
        system, history, final_text, _last, model_obj, display_name = self._split(request)
        messages = request.messages
        full_key = self._hash_for(messages, system, display_name)
        prefix_key = self._hash_for(messages[:-1], system, display_name)
        entry = self._sessions.get(prefix_key)

        if entry is not None:
            collector = self._media_collector()
            await collector.collect(messages, only_last=True)
            files = collector.files
            chat = entry.chat
            prompt = final_text
        else:
            collector = self._media_collector()
            await collector.collect(messages)
            files = collector.files
            chat = self.client.start_chat(model=model_obj)
            prompt = (
                make_fallback_prompt(history, final_text, system, has_files=bool(files))
                if history
                else flatten_single_prompt(final_text, system)
            )

        try:
            async for chunk in self._send_raw(chat, prompt, files):
                yield chunk
        finally:
            self._cleanup_files(files)
            self._remember(full_key, chat, display_name)

    # ---------------------------------------------------------- send helpers

    async def _send(self, chat: Any, prompt: str, files: list[Path]) -> Any:
        async with self._gate:
            try:
                return await chat.send_message(
                    prompt,
                    files=files or None,
                    temporary=self.settings.temporary_chats,
                )
            except Exception as exc:  # noqa: BLE001
                mapped = map_exception(exc)
                if mapped.status == 401:
                    self.ready = False  # next request will re-init with fresh cookies
                raise mapped from exc

    def _send_raw(self, chat: Any, prompt: str, files: list[Path]) -> AsyncGenerator[Any, None]:
        async def _gen():
            async with self._gate:
                try:
                    async for chunk in chat.send_message_stream(
                        prompt,
                        files=files or None,
                        temporary=self.settings.temporary_chats,
                    ):
                        yield chunk
                except Exception as exc:  # noqa: BLE001
                    mapped = map_exception(exc)
                    if mapped.status == 401:
                        self.ready = False  # next request will re-init
                    raise mapped from exc

        return _gen()

    # -------------------------------------------------------- images / misc

    async def fetch_image_b64(self, image: Image) -> str:
        """Download one upstream image; direct fetch first, image.save() as fallback."""
        assert self._http is not None
        try:
            resp = await self._http.get(
                image.url,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gemini.google.com/"},
                follow_redirects=True,
                timeout=90,
            )
            resp.raise_for_status()
            if resp.content:
                return b64lib.b64encode(resp.content).decode()
        except httpx.HTTPError:
            pass
        with tempfile.TemporaryDirectory(prefix="gwop_img_") as tmp:
            try:
                await image.save(path=tmp, filename="img.png")
                data = (Path(tmp) / "img.png").read_bytes()
            except Exception as exc:  # noqa: BLE001
                raise map_exception(exc) from exc
        return b64lib.b64encode(data).decode()

    def status(self) -> dict[str, Any]:
        info: dict[str, Any] = self.banner()
        if self.client:
            with contextlib.suppress(Exception):
                info["usage"] = self.client.usage_info
            with contextlib.suppress(Exception):
                info["abuse"] = self.client.abuse_status
        return info
