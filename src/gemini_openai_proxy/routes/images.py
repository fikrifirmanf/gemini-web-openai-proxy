"""`POST /v1/images/generations` — returns `b64_json` images from Gemini Web.

OpenAI's schema has no field for reference images; pass them through as an
extra `image` key (the OpenAI SDK's `extra_body={"image": [url, ...]}`).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..content import MediaCollector
from ..errors import UpstreamError, map_exception
from . import cooldown_guard
from ..schemas import ImagesRequest, ImagesResponse, ImageData

logger = logging.getLogger("gemini_openai_proxy.images")

router = APIRouter()


@router.post("/v1/images/generations")
async def images_generations(request: Request):
    svc = request.app.state.gemini
    try:
        req = ImagesRequest.model_validate(await request.json())
    except ValueError as exc:
        return JSONResponse(
            {"error": {"message": f"Invalid request body: {exc}", "type": "invalid_request_error", "code": "bad_request"}},
            status_code=400,
        )

    try:
        await svc.start()
        cooldown = cooldown_guard(svc.cooldown_remaining())
        if cooldown is not None:
            return cooldown
        model_obj, _display = svc.resolve_model(req.model)
        reference_urls = list(getattr(req, "image", None) or [])

        collector: MediaCollector = svc._media_collector()
        if reference_urls:
            class _Ref:  # tiny shim so the collector reads role='user' entries
                role = "user"
                content = [
                    {"type": "image_url", "image_url": {"url": url}} for url in reference_urls
                ]

            await collector.collect([_Ref])
        files = collector.files
        prompt = req.prompt if req.n <= 1 else (
            f"{req.prompt}\n\n(Please produce {req.n} distinct images for this request.)"
        )

        chat = svc.client.start_chat(model=model_obj)
        try:
            output = await svc.send_upstream(chat, prompt, files)
        finally:
            svc._cleanup_files(files)

        if notes := collector.notes:
            logger.warning("reference image notes: %s", "; ".join(notes))

        images = list(getattr(output, "images", None) or [])
        if not images:
            raise UpstreamError(
                502,
                "Gemini did not return any images for this prompt. Try asking "
                "explicitly to 'generate' an image rather than find one online.",
            )
        if len(images) > req.n:
            images = images[: req.n]

        data = [ImageData(b64_json=await svc.fetch_image_b64(image)) for image in images]
        return ImagesResponse(data=data)
    except UpstreamError as exc:
        logger.warning("image generation failed: %s", exc.message)
        body: dict = {"message": exc.message, "type": "upstream_error", "code": exc.code}
        if exc.hint:
            body["hint"] = exc.hint
        return JSONResponse({"error": body}, status_code=exc.status)
    except Exception as exc:  # noqa: BLE001
        mapped = map_exception(exc)
        logger.exception("images endpoint crashed")
        return JSONResponse(
            {"error": {"message": mapped.message, "type": "upstream_error", "code": mapped.code}},
            status_code=mapped.status,
        )
