"""Timeout wiring + SSE keepalive."""

from __future__ import annotations

import asyncio

import pytest

from gemini_openai_proxy.config import Settings
from gemini_openai_proxy.errors import UpstreamError
from gemini_openai_proxy.routes import with_keepalive
from gemini_openai_proxy.upstream import GeminiService


async def _slow(delay: float, items: list[str]):
    for item in items:
        await asyncio.sleep(delay)
        yield item


def test_keepalive_interleaves_none_on_silence():
    async def go():
        out = [x async for x in with_keepalive(_slow(0.12, ["a", "b"]), 0.05)]
        assert [x for x in out if x is not None] == ["a", "b"]
        assert out.count(None) >= 2

    asyncio.run(go())


def test_keepalive_disabled_passes_through():
    async def go():
        out = [x async for x in with_keepalive(_slow(0.01, ["a", "b"]), 0)]
        assert out == ["a", "b"]

    asyncio.run(go())


def test_keepalive_closes_source_on_early_exit():
    async def go():
        closed = False

        async def src():
            nonlocal closed
            try:
                while True:
                    await asyncio.sleep(0.01)
                    yield "x"
            finally:
                closed = True

        gen = with_keepalive(src(), 0.05)
        assert await gen.__anext__() == "x"
        await gen.aclose()
        assert closed

    asyncio.run(go())


def test_send_total_timeout_maps_to_504():
    async def go():
        svc = GeminiService(Settings(total_timeout=0.05, send_interval=0))

        class Chat:
            async def send_message(self, *a, **k):
                await asyncio.sleep(1)

        with pytest.raises(UpstreamError) as err:
            await svc._send(Chat(), "hi", [])
        assert err.value.status == 504

    asyncio.run(go())
