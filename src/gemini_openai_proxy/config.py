"""Settings loaded from environment variables (prefix `GWOP_`).

A project-root `.env` file is auto-loaded into the environment first; real
environment variables always win over file values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "GWOP_"


def _load_dotenv(env_file: Path = Path(".env")) -> None:
    """Minimal dotenv: lines of KEY=VALUE (quotes optional, # comments ok)."""
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("export "):
            line = line.removeprefix("export ")
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"").strip()
        if key and key not in os.environ:  # file never overrides a real env var
            os.environ[key] = value


def _get(key: str, default: str = "") -> str:
    return os.environ.get(ENV_PREFIX + key, os.environ.get(key, default)).strip()


def _get_bool(key: str, default: bool) -> bool:
    raw = _get(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{ENV_PREFIX}{key} must be an integer, got {raw!r}") from None


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{ENV_PREFIX}{key} must be a number, got {raw!r}") from None


@dataclass(slots=True)
class Settings:
    # server
    host: str = "127.0.0.1"
    port: int = 8011
    api_key: str = ""
    log_level: str = "INFO"

    # upstream (gemini web)
    cookie_1psid: str = ""
    cookie_1psidts: str = ""
    cookie_raw: str = ""          # "k=v; k2=v2" cookie header form
    cookie_file: str = ""         # Netscape cookies.txt
    upstream_proxy: str = ""      # e.g. http://127.0.0.1:7897

    # behaviour
    data_dir: Path = field(default_factory=lambda: Path("data").resolve())
    request_timeout: float = 300.0
    init_timeout: float = 120.0
    max_concurrency: int = 3
    session_cache_size: int = 256
    temporary_chats: bool = True  # don't save conversations into Gemini history
    strict_model: bool = False    # unknown model -> 400 instead of silently using default
    images_in_chat: bool = True   # append generated-image markdown to chat replies
    include_thoughts: bool = True # surface model thoughts as reasoning_content
    max_image_bytes: int = 20 * 1024 * 1024

    # self-healing: when Google reports the session UNAUTHENTICATED, rotate
    # __Secure-1PSIDTS from the long-lived 1PSID and rebuild the client
    auto_recovery: bool = True
    recovery_interval: float = 60.0   # seconds between recovery checks
    recovery_max_attempts: int = 5    # failed rounds in a row -> go dormant

    # abuse-safety: keep upstream traffic cadence human-like and stop retry
    # storms (the fastest way to get an account flagged)
    send_interval: float = 1.5        # min seconds between upstream sends (jittered)
    quota_cooldown: float = 300.0     # park new model traffic for Ns after an upstream 429
    single_instance: bool = True      # flock data_dir; refuse a 2nd proxy on same cookies

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv()
        data_dir = Path(_get("DATA_DIR", "data")).expanduser().resolve()
        s = cls(
            host=_get("HOST", "127.0.0.1"),
            port=_get_int("PORT", 8011),
            api_key=_get("API_KEY") or os.environ.get("API_KEY", ""),
            log_level=_get("LOG_LEVEL", "INFO").upper(),
            cookie_1psid=_get("GEMINI_1PSID"),
            cookie_1psidts=_get("GEMINI_1PSIDTS"),
            cookie_raw=_get("GEMINI_COOKIES_RAW"),
            cookie_file=_get("COOKIE_FILE"),
            upstream_proxy=_get("HTTP_PROXY"),
            data_dir=data_dir,
            request_timeout=_get_float("REQUEST_TIMEOUT", 300.0),
            init_timeout=_get_float("INIT_TIMEOUT", 120.0),
            max_concurrency=_get_int("MAX_CONCURRENCY", 3),
            session_cache_size=_get_int("SESSION_CACHE_SIZE", 256),
            temporary_chats=_get_bool("TEMPORARY_CHATS", True),
            strict_model=_get_bool("STRICT_MODEL", False),
            images_in_chat=_get_bool("IMAGES_IN_CHAT", True),
            include_thoughts=_get_bool("INCLUDE_THOUGHTS", True),
            auto_recovery=_get_bool("AUTO_RECOVERY", True),
            recovery_interval=_get_float("RECOVERY_INTERVAL", 60.0),
            recovery_max_attempts=_get_int("RECOVERY_MAX_ATTEMPTS", 5),
            send_interval=_get_float("SEND_INTERVAL", 1.5),
            quota_cooldown=_get_float("QUOTA_COOLDOWN", 300.0),
            single_instance=_get_bool("SINGLE_INSTANCE", True),
        )
        # gemini_webapi persists auto-refreshed __Secure-1PSIDTS cookies here.
        # Must be set before the gemini_webapi module is imported.
        if not os.environ.get("GEMINI_COOKIE_PATH"):
            os.environ["GEMINI_COOKIE_PATH"] = str(data_dir / "gemini_cookies")
        return s

    @property
    def has_any_cookie(self) -> bool:
        return bool(self.cookie_1psid or self.cookie_raw or self.cookie_file)
