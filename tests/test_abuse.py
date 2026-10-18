"""Tests for the abuse-safety layer: send pacing, quota cooldown, instance lock."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse
from gemini_webapi.exceptions import UsageLimitExceededError

from gemini_openai_proxy.routes import cooldown_guard
from gemini_openai_proxy.config import Settings
from gemini_openai_proxy.errors import UpstreamError
from gemini_openai_proxy.upstream import GeminiService


def make_service(**overrides) -> GeminiService:
    settings = Settings(
        send_interval=overrides.pop("send_interval", 0.0),
        quota_cooldown=overrides.pop("quota_cooldown", 60.0),
        data_dir=overrides.pop("data_dir", None) or Path("."),
    )
    assert not overrides, f"unused overrides: {list(overrides)}"
    return GeminiService(settings)


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------- pacing


def test_send_interval_paces_consecutive_sends():
    svc = make_service(send_interval=0.15)

    async def go():
        stamps: list[float] = [time.monotonic()]
        await svc._respect_send_interval()
        stamps.append(time.monotonic())
        await svc._respect_send_interval()
        stamps.append(time.monotonic())
        return stamps

    stamps = run(go())
    gap1 = stamps[1] - stamps[0]
    gap2 = stamps[2] - stamps[1]
    assert gap1 + gap2 >= 0.14  # jittered but bounded below by the interval


def test_send_interval_zero_is_passthrough():
    svc = make_service(send_interval=0.0)

    async def go():
        t0 = time.monotonic()
        await svc._respect_send_interval()
        await svc._respect_send_interval()
        return time.monotonic() - t0

    assert run(go()) < 0.05


# ----------------------------------------------------------------- cooldown


class _RaisingChat:
    async def send_message(self, *a, **k):
        raise UsageLimitExceededError("quota dry")


def test_send_failure_429_arms_cooldown():
    svc = make_service(quota_cooldown=60.0)
    assert svc.cooldown_remaining() == 0.0

    async def go():
        with pytest.raises(UpstreamError) as info:
            await svc._send(_RaisingChat(), "hi", [])
        return info

    info = run(go())
    assert info.value.status == 429
    assert 0 < svc.cooldown_remaining() <= 60.0


def test_send_failure_429_without_cooldown_setting():
    svc = make_service(quota_cooldown=0.0, send_interval=0.0)

    async def go():
        with pytest.raises(UpstreamError):
            await svc._send(_RaisingChat(), "hi", [])

    run(go())
    assert svc.cooldown_remaining() == 0.0  # opt-out honoured


def test_send_upstream_same_discipline_as_chat():
    svc = make_service(quota_cooldown=30.0)

    async def go():
        with pytest.raises(UpstreamError) as info:
            await svc.send_upstream(_RaisingChat(), "draw", [])
        return info.value.status

    assert run(go()) == 429
    assert svc.cooldown_remaining() > 0


def test_send_failure_401_sets_recovery_kick_not_cooldown():
    from gemini_webapi import AuthError

    class _Auth:
        async def send_message(self, *a, **k):
            raise AuthError

    svc = make_service(quota_cooldown=60.0)

    async def go():
        with pytest.raises(UpstreamError) as info:
            await svc._send(_Auth(), "hi", [])
        return info.value.status

    assert run(go()) == 401
    assert svc.cooldown_remaining() == 0.0       # 401 is a re-auth, not a pacing problem
    assert not svc.ready                          # flagged to rebuild on next request


# ------------------------------------------------------------ cooldown guard


def test_cooldown_guard_shapes_a_429():
    resp = cooldown_guard(29.3)
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "30"
    assert cooldown_guard(0) is None


# ----------------------------------------------------------- instance lock


def test_instance_lock_rejects_second_holder(tmp_path):
    one = make_service(data_dir=tmp_path)
    two = make_service(data_dir=tmp_path)

    async def go():
        one.acquire_instance_lock()          # first holder is fine
        try:
            with pytest.raises(UpstreamError) as info:
                two.acquire_instance_lock()  # second holder must abort
            return info.value.status
        finally:
            one.release_instance_lock()

    assert run(go()) == 409

    # after release the slot is free again
    async def go2():
        two.acquire_instance_lock()
        try:
            return True
        finally:
            two.release_instance_lock()

    assert run(go2()) is True


def test_instance_lock_blocks_foreign_descriptor(tmp_path):
    """A lock held elsewhere (same process, other fd) also blocks acquisition."""
    import fcntl

    svc = make_service(data_dir=tmp_path)
    held = open(tmp_path / "gemini_session.lock", "a+")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(UpstreamError):
            svc.acquire_instance_lock()
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()


# ------------------------------------------------------------------- banner


def test_banner_reports_cooldown_seconds():
    svc = make_service(quota_cooldown=120.0)
    svc._cooldown_until = time.monotonic() + 42.0
    assert svc.banner()["quota_cooldown_seconds"] in {41, 42}
