"""FastAPI app factory: routers, auth middleware, upstream lifecycle."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import UpstreamError
from .routes import chat, images, meta
from .upstream import GeminiService

logger = logging.getLogger("gemini_openai_proxy")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc: GeminiService = app.state.gemini
        logger.info(
            "starting upstream: cookie_source=%s data_dir=%s proxy=%s",
            svc.cookie_source or "resolving...",
            settings.data_dir,
            settings.upstream_proxy or "-",
        )
        try:
            svc.acquire_instance_lock()  # abort startup on a 2nd proxy with the same cookies
        except UpstreamError as exc:
            logger.critical("startup aborted: %s — %s", exc.message, exc.hint or "")
            raise
        try:
            await svc.start()
            banner = svc.banner()
            logger.info("upstream ready. models: %s", [m["id"] for m in banner["models"]])
            account = banner.get("account_status")
            if account and account != "AVAILABLE":
                logger.warning(
                    "Google reports account status %s — anonymous Gemini only. "
                    "Auto-recovery will keep trying cookie rotation; if it stays "
                    "unauthenticated, re-copy fresh __Secure-1PSID/__Secure-1PSIDTS cookies.",
                    account,
                )
        except UpstreamError as exc:
            svc.last_init_error = exc.message
            logger.critical("upstream init failed (%s): %s — %s", exc.status, exc.message, exc.hint or "")
            logger.critical("requests will keep retrying init; fix cookies/API key usage and reinit via POST /admin/reinit")
        svc.start_recovery()  # background self-healing; also covers init failed at boot
        yield
        svc.stop_recovery()
        svc.release_instance_lock()
        await svc.stop()

    app = FastAPI(
        title="Gemini Web OpenAI Proxy",
        version="0.1.0",
        description="OpenAI-compatible API backed by a Google Gemini Web session.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.gemini = GeminiService(settings)

    app.include_router(meta.router)
    app.include_router(chat.router)
    app.include_router(images.router)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # optional bearer auth; anything that is not model traffic stays open
        if settings.api_key and request.method != "OPTIONS":
            if request.url.path not in {"/health", "/readyz"} and not request.url.path.startswith("/docs"):
                auth = request.headers.get("authorization", "")
                if auth != f"Bearer {settings.api_key}":
                    return JSONResponse(
                        {"error": {"message": "Missing or invalid bearer token", "type": "auth_error", "code": "unauthorized"}},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
        started = time.monotonic()
        try:
            response = await call_next(request)
        except UpstreamError as exc:
            response = JSONResponse(
                {"error": {"message": exc.message, "type": "upstream_error", "code": exc.code}},
                status_code=exc.status,
            )
        except Exception as exc:  # noqa: BLE001 - final safety net
            if request.url.path != "/health":
                logger.exception("unhandled error on %s", request.url.path)
            response = JSONResponse(
                {"error": {"message": f"{exc.__class__.__name__}: {exc}", "type": "internal_error", "code": "internal"}},
                status_code=500,
            )
        elapsed = (time.monotonic() - started) * 1000
        if request.url.path != "/health":
            logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed)
        return response

    return app


app = create_app()
