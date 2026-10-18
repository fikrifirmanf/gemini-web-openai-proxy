"""Tests for the auto-recovery loop (cookie rotation + session rebuild)."""

from __future__ import annotations

import asyncio

import pytest

from gemini_openai_proxy.config import Settings
from gemini_openai_proxy.errors import UpstreamError
from gemini_openai_proxy.upstream import GeminiService


def make_service(**overrides) -> GeminiService:
    settings = Settings(
        cookie_1psid=overrides.pop("psid", "test.1psid.value"),
        cookie_1psidts=overrides.pop("psidts", "test.1psidts.value"),
        auto_recovery=overrides.pop("auto_recovery", True),
        recovery_max_attempts=overrides.pop("max_attempts", 3),
        recovery_interval=overrides.pop("interval", 60.0),
    )
    assert not overrides, f"unused overrides: {list(overrides)}"
    return GeminiService(settings)


def run(coro):
    return asyncio.run(coro)


def patch_round(svc: GeminiService, result: bool):
    """Replace the network-touching round with a scripted one."""

    async def round() -> bool:
        return result

    svc._recovery_round = round  # type: ignore[method-assign]


# ------------------------------------------------------------------ config


def test_recovery_settings_defaults_are_sane():
    s = Settings()
    assert s.auto_recovery is True
    assert s.recovery_interval == 60.0
    assert s.recovery_max_attempts == 5


def test_recovery_disabled_via_settings():
    svc = make_service(auto_recovery=False)
    svc.start_recovery()  # disabled -> no-op even without a running loop
    assert svc._recovery_task is None
    assert svc.recovery_info()["enabled"] is False


# ------------------------------------------------------------------- status


def test_banner_includes_recovery_state():
    svc = make_service()
    banner = svc.banner()
    assert banner["recovery"] == {"enabled": True, "attempts": 0, "dormant": False}

    svc._recovery_attempts = 3
    assert svc.recovery_info()["dormant"] is True  # max_attempts = 3


def test_account_ok_requires_client_marked_available():
    svc = make_service()
    assert not svc._account_ok()  # no client at all

    class _Status:
        name = "UNAUTHENTICATED"

    class _Client:
        account_status = _Status()

    svc.client = _Client()
    assert not svc._account_ok()

    _Status.name = "AVAILABLE"
    assert svc._account_ok()


# ---------------------------------------------------------- maybe_recover


def test_maybe_recover_healthy_short_circuit_resets_attempts():
    svc = make_service()
    svc._recovery_attempts = 2  # stale failures linger

    async def go():
        svc.ready = True
        svc._account_ok = lambda: True
        assert await svc.maybe_recover() is True
        assert svc._recovery_attempts == 0  # reset by the cheap health check

    run(go())


def test_round_success_restores_health_and_resets_attempts():
    svc = make_service()
    svc._recovery_attempts = 1

    async def go():
        state = {"healthy": False}

        async def round() -> bool:
            state["healthy"] = True  # rotation rebuilt an authenticated session
            return True

        svc._recovery_round = round  # type: ignore[method-assign]
        svc._account_ok = lambda: state["healthy"]
        svc.ready = True

        assert await svc.maybe_recover() is True
        assert svc._recovery_attempts == 0

    run(go())


def test_maybe_recover_counts_failures_and_goes_dormant():
    svc = make_service(max_attempts=3)

    async def go():
        patch_round(svc, result=False)
        svc._account_ok = lambda: False
        attempts: list[int] = []
        for _ in range(4):
            await svc.maybe_recover()
            svc._recovery_last = 0.0  # bypass the 60s debounce
            attempts.append(svc._recovery_attempts)
        assert attempts == [1, 2, 3, 3]  # stops counting once dormant
        assert svc.recovery_info()["dormant"] is True

    run(go())


def test_maybe_recover_is_debounced_within_60s():
    svc = make_service()

    async def go():
        patch_round(svc, result=False)
        svc._account_ok = lambda: False
        await svc.maybe_recover()
        assert svc._recovery_attempts == 1
        # straight after the failed round the debounce still holds
        await svc.maybe_recover()
        assert svc._recovery_attempts == 1

    run(go())


def test_maybe_recover_handles_upstream_errors():
    svc = make_service(max_attempts=5)

    async def round() -> bool:
        raise UpstreamError(401, "rotation refused")

    svc._recovery_round = round  # type: ignore[method-assign]

    async def go():
        svc._account_ok = lambda: False
        assert await svc.maybe_recover() is False
        assert svc._recovery_attempts == 1  # counted, not crashed

    run(go())


# --------------------------------------------------------------- lifecycle


def test_kick_recovery_noop_without_loop_or_when_disabled():
    svc = make_service(auto_recovery=False)
    svc.kick_recovery()  # disabled -> fine without a running loop
    assert svc._recovery_kick is None

    svc2 = make_service()
    svc2.kick_recovery()  # sync context: must not explode; scheduling skipped
    assert svc2._recovery_kick is None


def test_start_recovery_is_idempotent_and_cancellable():
    svc = make_service()

    async def go():
        svc.start_recovery()
        try:
            task = svc._recovery_task
            assert task is not None
            svc.start_recovery()  # second call keeps the same task
            assert svc._recovery_task is task
        finally:
            svc.stop_recovery()
        assert svc._recovery_task is None
        assert svc._recovery_kick is None

    run(go())


def test_circuit_breaker_rearmed_by_manual_reset():
    svc = make_service(max_attempts=2)

    async def go():
        patch_round(svc, result=False)
        svc._account_ok = lambda: False
        for _ in range(2):
            await svc.maybe_recover()
            svc._recovery_last = 0.0
        assert svc.recovery_info()["dormant"] is True

        # what reinit() does: clears the counter so rounds can run again
        svc._recovery_attempts = 0
        svc._recovery_last = 0.0
        assert svc.recovery_info()["dormant"] is False

    run(go())


# ------------------------------------------------------ rotation session


def test_rotation_session_reuses_live_client():
    class _Live:
        pass

    class _Client:
        @property
        def _live_client(self) -> _Live:
            return _Live()

    svc = make_service()
    svc.client = _Client()
    cookies = svc._resolve_cookies()

    session = svc._rotation_session(cookies)
    assert isinstance(session, _Live)
    assert svc._owns_rotation_session is False  # live session: not ours to close


def test_rotation_session_builds_throwaway_when_client_is_down():
    class _Client:
        @property
        def _live_client(self):
            raise RuntimeError("client down")

    svc = make_service()
    svc.client = _Client()
    cookies = svc._resolve_cookies()

    session = svc._rotation_session(cookies)
    assert svc._owns_rotation_session is True
    names = {c.name for c in session.cookies.jar}
    assert "__Secure-1PSID" in names
    assert "__Secure-1PSIDTS" in names


def test_drop_cookie_cache_removes_lib_cache_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(tmp_path))
    cache_file = tmp_path / ".cached_cookies_test.1psid.value.json"
    cache_file.write_text("{}")
    svc = make_service()
    cookies = svc._resolve_cookies()

    GeminiService._drop_cookie_cache(cookies)
    assert not cache_file.exists()
    GeminiService._drop_cookie_cache(cookies)  # second unlink: silent no-op
