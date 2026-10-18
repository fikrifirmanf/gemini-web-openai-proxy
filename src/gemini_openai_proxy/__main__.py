"""CLI entrypoint: `python -m gemini_openai_proxy` (or `gemini-openai-proxy`)."""

from __future__ import annotations

import argparse
import sys

from .config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gemini-openai-proxy",
        description="OpenAI-compatible API proxy on top of a Gemini Web session.",
    )
    parser.add_argument("--host", help="listen address (default GWOP_HOST, 127.0.0.1)")
    parser.add_argument("--port", type=int, help="listen port (default GWOP_PORT, 8011)")
    parser.add_argument("--log-level", help="DEBUG|INFO|WARNING|ERROR (default GWOP_LOG_LEVEL)")
    args = parser.parse_args(argv)

    settings = Settings.load()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.log_level:
        settings.log_level = args.log_level.upper()

    if not settings.has_any_cookie and not _cookie_cache_warm(settings):
        print(_no_cookie_help(), file=sys.stderr)

    from .main import create_app  # after settings/env are final

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print("uvicorn is required: uv pip install 'uvicorn[standard]'", file=sys.stderr)
        return 2

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    return 0


def _cookie_cache_warm(settings: Settings) -> bool:
    """Previously auto-refreshed cookies make cold env-less starts workable."""
    cache = settings.data_dir / "gemini_cookies"
    if not cache.is_dir():
        return False
    return any(p.name.startswith(".cached_cookies_") for p in cache.glob("*.json"))


def _no_cookie_help() -> str:
    return (
        "\nNo Gemini cookies configured yet. This run will serve 401/503 until you add them.\n\n"
        "  1. Log in at https://gemini.google.com\n"
        "  2. F12 > Network tab > refresh > click any request\n"
        "  3. Copy cookie values '__Secure-1PSID' and '__Secure-1PSIDTS'\n"
        "  4. Restart with:\n"
        "     GWOP_GEMINI_1PSID='<1PSID>' GWOP_GEMINI_1PSIDTS='<1PSIDTS>' gemini-openai-proxy\n\n"
        "Tip: __Secure-1PSIDTS refreshes automatically while the proxy runs, so the pair\n"
        "only needs to be re-copied when the account logs out or Google rotates them.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
