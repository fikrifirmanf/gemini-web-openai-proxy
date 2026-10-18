"""Meta endpoints: health, readiness, model list, admin controls."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..errors import UpstreamError

logger = logging.getLogger("gemini_openai_proxy.meta")

router = APIRouter()


def service(request: Request):
    return request.app.state.gemini


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "gemini-openai-proxy"}


@router.get("/readyz")
async def readyz(request: Request):
    svc = service(request)
    body: dict[str, Any] = {"ready": svc.ready}
    if not svc.ready:
        body["error"] = svc.last_init_error or "upstream client not initialized"
    return JSONResponse(body, status_code=200 if svc.ready else 503)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    svc = service(request)
    await svc.start()
    return {"object": "list", "data": svc.openai_models()}


@router.post("/admin/reinit")
async def admin_reinit(request: Request) -> dict[str, Any]:
    svc = service(request)
    try:
        return await svc.reinit()
    except UpstreamError as exc:
        return {"reinitialized": False, "error": exc.message, "hint": exc.hint}


@router.get("/admin/status")
async def admin_status(request: Request) -> dict[str, Any]:
    svc = service(request)
    return svc.status()
