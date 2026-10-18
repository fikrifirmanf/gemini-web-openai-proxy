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
import logging
import os
import random
import tempfile
import time

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX (Windows): lock unsupported
    fcntl = None  # type: ignore[assignment]
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from curl_cffi import CurlFollow, CurlHttpVersion
from curl_cffi.requests import AsyncSession as CurlAsyncSession
from gemini_webapi import AuthError, GeminiClient
from gemini_webapi.constants import BROWSER_TYPE
from gemini_webapi.types import AvailableModel, Image
from gemini_webapi.utils.rotate_1psidts import rotate_1psidts

from .config import Settings
from .content import MediaCollector, build_turn_text, flatten_single_prompt, make_fallback_prompt, media_urls_of, text_of
from .cookies import GeminiCookies, resolve_cookies
from .errors import UpstreamError, map_exception

logger = logging.getLogger("gemini_openai_proxy.upstream")

_ROLE_LABEL = {"user": "User", "assistant": "Assistant", "tool": "User"}
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
        self._recovery_task: asyncio.Task | None = None
        self._recovery_kick: asyncio.Task | None = None
        self._recovery_attempts = 0
        self._recovery_last = 0.0
        self._owns_rotation_session = False
        self._last_send_at = 0.0
        self._cooldown_until = 0.0
        self._instance_lock_fd: Any = None
        self._pace_lock = asyncio.Lock()

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
        out["recovery"] = self.recovery_info()
        out["quota_cooldown_seconds"] = round(self.cooldown_remaining())
        return out

    # ------------------------------------------------------------ recovery

    def recovery_info(self) -> dict[str, Any]:
        max_attempts = max(1, self.settings.recovery_max_attempts)
        return {
            "enabled": bool(self.settings.auto_recovery and self.settings.has_any_cookie),
            "attempts": self._recovery_attempts,
            "dormant": self._recovery_attempts >= max_attempts,
        }

    def _account_status_name(self) -> str | None:
        account = getattr(self.client, "account_status", None) if self.client else None
        if account is None:
            return None
        return str(getattr(account, "name", account))

    def _account_ok(self) -> bool:
        return self._account_status_name() == "AVAILABLE"

    def start_recovery(self) -> None:
        """Spawn the background self-healing loop (idempotent, no-op when disabled)."""
        if not (self.settings.auto_recovery and self.settings.has_any_cookie):
            return
        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    def stop_recovery(self) -> None:
        if self._recovery_task:
            self._recovery_task.cancel()
            self._recovery_task = None
        if self._recovery_kick:
            self._recovery_kick.cancel()
            self._recovery_kick = None

    def kick_recovery(self) -> None:
        """One recovery attempt soon; called when a request hits an auth error.

        Debounced into a single scheduled round while one is already pending.
        """
        if not (self.settings.auto_recovery and self.settings.has_any_cookie):
            return

        async def _kick() -> None:
            try:
                await asyncio.sleep(3.0)  # let concurrent 401s pile up, then heal once
                await self.maybe_recover()
            except Exception as exc:  # noqa: BLE001
                logger.debug("recovery kick aborted: %r", exc)
            finally:
                self._recovery_kick = None

        if self._recovery_kick is not None and not self._recovery_kick.done():
            return
        try:
            self._recovery_kick = asyncio.get_running_loop().create_task(_kick())
        except RuntimeError:  # no loop: e.g. called from a sync test context
            pass

    async def maybe_recover(self) -> bool:
        """One bounded attempt to authenticate an unauthenticated session.

        Returns True if the upstream is healthy (or became healthy now). Rounds are
        debounced (>=60s apart); after `recovery_max_attempts` consecutive failures
        the circuit opens until a manual `/admin/reinit` or a process restart.
        """
        now = time.monotonic()
        if now - self._recovery_last < 60.0 and self._recovery_attempts > 0:
            return self.ready and self._account_ok()
        self._recovery_last = now

        if self.ready and self._account_ok():
            self._recovery_attempts = 0
            return True

        max_attempts = max(1, self.settings.recovery_max_attempts)
        if self._recovery_attempts >= max_attempts:
            return False  # dormant: stays quiet until reinit() re-arms it

        self._recovery_attempts += 1
        try:
            ok = await self._recovery_round()
        except UpstreamError as exc:
            logger.warning("auto-recovery round %s failed: %s", self._recovery_attempts, exc.message)
            if exc.hint:
                logger.warning("auto-recovery hint: %s", exc.hint)
            ok = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto-recovery round %s failed: %r", self._recovery_attempts, exc)
            ok = False

        if ok:
            logger.info("auto-recovery: session restored after %s attempt(s)", self._recovery_attempts)
            self._recovery_attempts = 0
        elif self._recovery_attempts >= max_attempts:
            logger.critical(
                "auto-recovery dormant after %s failed rounds. Rotate yourself: re-copy "
                "fresh __Secure-1PSID/__Secure-1PSIDTS into .env, then POST /admin/reinit "
                "(/reinit in scripts/chat.py).",
                self._recovery_attempts,
            )
        return bool(ok and self.ready and self._account_ok())

    async def _recovery_round(self) -> bool:
        """Rotate __Secure-1PSIDTS from the long-lived 1PSID, then rebuild + verify."""
        cookies = self._resolve_cookies()  # also validates a cookie source exists
        self._drop_cookie_cache(cookies)   # force a real network rotation below

        curl_session = self._rotation_session(cookies)
        try:
            new_1psidts = await rotate_1psidts(
                curl_session, verbose=self.settings.log_level == "DEBUG"
            )
        except AuthError:
            raise UpstreamError(
                401,
                "Google refused to rotate cookies for the stored session (401). Rotation is "
                "only granted to sessions Google still trusts — a copied pair that Google "
                "has already rotated away needs fresh values.",
                hint=(
                    "Re-copy fresh __Secure-1PSID + __Secure-1PSIDTS into .env, then "
                    "POST /admin/reinit. To avoid this state, let the proxy's own "
                    "auto-refresh (every ~10 min while healthy) do the rotating and keep "
                    "the cookies.txt source stable."
                ),
            ) from None
        finally:
            if self._owns_rotation_session:
                with contextlib.suppress(Exception):
                    await curl_session.close()

        if not new_1psidts:
            raise UpstreamError(
                502,
                "Cookie rotation succeeded but minted no __Secure-1PSIDTS.",
                hint="Google may have invalidated the session; re-copy fresh cookies and reinit.",
            )

        async with self._init_lock:
            if self.client:
                with contextlib.suppress(Exception):
                    await self.client.close()
                self.client = None
            self.ready = False
            await self._start_inner()
        return self.ready and self._account_ok()

    def _rotation_session(self, cookies: GeminiCookies):
        """Reuse the live TLS session when possible, else build a browser-fingerprinted one."""
        try:
            live = self.client._live_client if self.client else None  # noqa: SLF001
        except Exception:  # noqa: BLE001 - property raises when the client is down
            live = None
        if live is not None:
            self._owns_rotation_session = False
            return live

        session = CurlAsyncSession(  # same fingerprint the lib uses for init
            impersonate=BROWSER_TYPE,
            proxy=self.settings.upstream_proxy or None,
            allow_redirects=CurlFollow.SAFE,
            http_version=CurlHttpVersion.NONE,
            verify=True,
        )
        jar = {"__Secure-1PSID": cookies.secure_1psid, "__Secure-1PSIDTS": cookies.secure_1psidts or ""}
        if cookies.extra:
            jar.update(cookies.extra)
        for name, value in jar.items():
            if value:
                session.cookies.set(name, value, domain=".google.com", path="/", secure=True)
        self._owns_rotation_session = True
        return session

    @staticmethod
    def _drop_cookie_cache(cookies: GeminiCookies) -> None:
        """Remove the cached cookie entry so rotation can't skip via the <60s rule.

        Mirrors gemini_webapi's cache layout: .cached_cookies_<1PSID>.json.
        """
        cache_dir = Path(os.environ.get("GEMINI_COOKIE_PATH") or (Path("data").resolve() / "gemini_cookies"))
        with contextlib.suppress(OSError):
            (cache_dir / f".cached_cookies_{cookies.secure_1psid}.json").unlink()

    async def _recovery_loop(self) -> None:
        interval = max(30.0, float(self.settings.recovery_interval))
        logger.info("auto-recovery loop started (checks every ~%ss)", interval)
        try:
            while True:
                await asyncio.sleep(interval + random.uniform(0.0, interval * 0.25))
                await self.maybe_recover()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("auto-recovery loop crashed: %r — restart the proxy to re-arm", exc)

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
        """Admin endpoint: tear down and rebuild the upstream client.

        Re-reads .env/env first, so users can refresh cookies without restarting.
        """
        self.settings = Settings.load()  # picks up edited cookie values
        self._recovery_attempts = 0  # manual action re-arms the circuit breaker
        self._recovery_last = 0.0
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
        last_role = str(last.role)
        # A tool result is the model's next *input* (Codex agent loop), so it
        # closes the turn exactly like a user message would.
        if last_role not in ("user", "tool"):
            raise UpstreamError(400, "The last message must have role 'user' (or 'tool' result).")

        system_parts: list[str] = []
        history: list[tuple[str, str]] = []
        for msg in request.messages[:-1]:
            label = _ROLE_LABEL.get(str(msg.role), "System")
            text, _urls = text_of(msg.content), []
            if str(msg.role) in _SYSTEM_ROLES:
                system_parts.append(text)
            elif text.strip():
                # raw text: make_fallback_prompt adds the "[Label]:" prefix
                history.append((label, text))
        system = "\n\n".join(p for p in system_parts if p.strip()) or None

        final_text = build_turn_text("User", last.content)[0].removeprefix("[User]: ")
        if not final_text.strip() and not media_urls_of(last.content):
            raise UpstreamError(400, "The final user message is empty.")
        return system, history, final_text, last, model, display_name

    @staticmethod
    def _hash_for(messages: list[Any], system: str | None, model_name: str, thinking: bool = False) -> str:
        payload = [
            {"role": str(m.role), "text": build_turn_text(str(m.role), m.content)[0]}
            for m in messages
        ]
        if system:
            payload.append({"role": "system", "text": system})
        if thinking:
            payload.append({"thinking": True})
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
        thinking = request.wants_extended_thinking
        system, history, final_text, _last, model_obj, display_name = self._split(request)
        messages = request.messages
        full_key = self._hash_for(messages, system, display_name, thinking)
        prefix_key = self._hash_for(messages[:-1], system, display_name, thinking)
        entry = self._sessions.get(prefix_key)

        if entry is not None:
            collector = self._media_collector()
            await collector.collect(messages, only_last=True)
            files, notes = collector.files, collector.notes
            try:
                output = await self._send(entry.chat, final_text, files, thinking)
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
            output = await self._send(chat, prompt, files, thinking)
        finally:
            self._cleanup_files(files)
        self._remember(full_key, chat, display_name)
        return output, display_name, notes

    async def chat_stream(self, request: Any) -> AsyncGenerator[Any, None]:
        """Yields incremental ModelOutput chunks; the final one carries images."""
        await self._ensure_ready()
        thinking = request.wants_extended_thinking
        system, history, final_text, _last, model_obj, display_name = self._split(request)
        messages = request.messages
        full_key = self._hash_for(messages, system, display_name, thinking)
        prefix_key = self._hash_for(messages[:-1], system, display_name, thinking)
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
            async for chunk in self._send_raw(chat, prompt, files, thinking):
                yield chunk
        finally:
            self._cleanup_files(files)
            self._remember(full_key, chat, display_name)

    # ---------------------------------------------------------- send helpers

    async def _respect_send_interval(self) -> None:
        """Jittered minimum gap between upstream sends so bursts look like typing.

        The pace lock serializes the sleep+stamp so concurrent gate holders can't
        fire two sends closer than ~send_interval apart.
        """
        interval = max(0.0, float(self.settings.send_interval))
        async with self._pace_lock:
            if interval <= 0:
                self._last_send_at = time.monotonic()
                return
            target = self._last_send_at + interval * random.uniform(1.0, 1.5)
            wait = target - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_send_at = time.monotonic()

    # ------------------------------------------------------- instance guard

    def acquire_instance_lock(self) -> None:
        """Refuse a second proxy process on the same cookie set.

        Two instances would let Google's cookie auto-refresh and ours race each
        other (each rotation step can invalidate the other's TS), which is how
        sessions die mysteriously mid-day. Non-blocking fcntl lock on the data dir.
        """
        if not getattr(self.settings, "single_instance", True) or fcntl is None:
            return
        path = self.settings.data_dir / "gemini_session.lock"
        try:
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UpstreamError(503, f"Could not create data dir: {exc}") from exc
        fd = open(path, "a+")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fd.close()
            raise UpstreamError(
                409,
                "Another gemini-web-openai-proxy instance is already running with this "
                "data_dir/cookie set.",
                hint=(
                    "Stop the other process (or set GWOP_SINGLE_INSTANCE=0 to accept the "
                    "risk of cookie-rotation races between the two instances)."
                ),
            ) from exc
        with contextlib.suppress(OSError):
            fd.seek(0)
            fd.truncate()
            fd.write(f"{os.getpid()}\n")
            fd.flush()
        self._instance_lock_fd = fd  # keep open for the lifetime of the process

    def release_instance_lock(self) -> None:
        fd = self._instance_lock_fd
        if fd is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                fd.close()
            self._instance_lock_fd = None

    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

    def _mark_rate_cool(self) -> None:
        """After an upstream 429, stop generating new requests to Google for a while.

        Retrying immediately against a dry quota / rate-limited IP is exactly the
        traffic pattern that escalates a temporary block into an abuse flag.
        """
        if self.settings.quota_cooldown > 0:
            self._cooldown_until = time.monotonic() + self.settings.quota_cooldown
            logger.warning(
                "upstream 429: parking new model traffic for ~%.0fs (GWOP_QUOTA_COOLDOWN)",
                self.settings.quota_cooldown,
            )

    @staticmethod
    def _on_send_mapped(mapped: UpstreamError, self: "GeminiService") -> None:
        """Shared side effects of send failures (401 → rebuild soon, 429 → cooldown)."""
        if mapped.status == 401:
            self.ready = False  # next request will re-init with fresh cookies
            self.kick_recovery()  # self-heal in the background immediately
        elif mapped.status == 429:
            self._mark_rate_cool()

    async def send_upstream(self, chat: Any, prompt: str, files: list[Path]) -> Any:
        """Disciplined one-shot send used by the images endpoint (gated + paced)."""
        async with self._gate:
            await self._respect_send_interval()
            try:
                return await chat.send_message(
                    prompt,
                    files=files or None,
                    temporary=self.settings.temporary_chats,
                )
            except Exception as exc:  # noqa: BLE001
                mapped = map_exception(exc)
                self._on_send_mapped(mapped, self)
                raise mapped from exc

    async def _send(self, chat: Any, prompt: str, files: list[Path], thinking: bool = False) -> Any:
        async with self._gate:
            await self._respect_send_interval()
            try:
                return await chat.send_message(
                    prompt,
                    files=files or None,
                    temporary=self.settings.temporary_chats,
                    extended_thinking=thinking,
                )
            except Exception as exc:  # noqa: BLE001
                mapped = map_exception(exc)
                self._on_send_mapped(mapped, self)
                raise mapped from exc

    def _send_raw(self, chat: Any, prompt: str, files: list[Path], thinking: bool = False) -> AsyncGenerator[Any, None]:
        async def _gen():
            async with self._gate:
                await self._respect_send_interval()
                try:
                    async for chunk in chat.send_message_stream(
                        prompt,
                        files=files or None,
                        temporary=self.settings.temporary_chats,
                        extended_thinking=thinking,
                    ):
                        yield chunk
                except Exception as exc:  # noqa: BLE001
                    mapped = map_exception(exc)
                    self._on_send_mapped(mapped, self)
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
