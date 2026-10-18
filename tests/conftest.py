"""Shared fixtures: a stub GeminiService so route tests never need real cookies."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class GeneratedImage:
    url: str = "https://lh3.googleusercontent.com/demo.png"
    alt: str = "a demo generated image"
    title: str = "[Image]"


@dataclass
class FakeOutput:
    text: str = "pong"
    thoughts: str | None = None
    images: list = field(default_factory=list)
    metadata: list = field(default_factory=lambda: ["c1", "r1", "rc1"])
    text_delta: str = "pong"


class _StubCollector:
    files: list = []
    notes: list = []

    async def collect(self, messages, *, only_last: bool = False) -> None:
        return None


class _FakeChat:
    def __init__(self, svc: "StubService"):
        self._svc = svc

    async def send_message(self, prompt, files=None, **kwargs):
        svc = self._svc
        return FakeOutput(text=svc.text, thoughts=svc.thoughts, images=list(svc.generated))

    async def send_message_stream(self, prompt, files=None, **kwargs):
        svc = self._svc
        head = (svc.text or "")[: 1 + len(svc.text) // 2]
        tail = (svc.text or "")[len(head):]
        yield FakeOutput(text=head, text_delta=head)
        yield FakeOutput(text=svc.text, text_delta=tail, images=list(svc.generated))


class _StubClient:
    def __init__(self, svc: "StubService"):
        self._svc = svc

    def start_chat(self, model=None):
        return _FakeChat(self._svc)


class StubService:
    """Mimics gemini_openai_proxy.upstream.GeminiService's API for route tests."""

    ready = True
    started = True
    cookie_source = "env"
    last_init_error = ""

    def __init__(self, settings=None):
        self.settings = settings
        self.text = "pong"
        self.thoughts = None
        self.generated: list[GeneratedImage] = []
        self.calls: list = []
        self.client = _StubClient(self)

    # lifecycle ------------------------------------------------------------
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def reinit(self) -> dict:
        return {"service": "stub", "ready": self.ready, "models": []}

    # introspection --------------------------------------------------------
    def banner(self) -> dict:
        return {
            "service": "gemini-openai-proxy",
            "ready": True,
            "cookie_source": "env",
            "cookie_path": "/tmp",
            "upstream_proxy": None,
            "temporary_chats": True,
            "models": [{"id": "gemini-pro", "display": "Fastest", "aliases": ["flash"]}],
        }

    def status(self) -> dict:
        return self.banner()

    def resolve_model(self, requested: str):
        return (None, requested or "default")

    def openai_models(self) -> list[dict]:
        return [{"id": "gemini-pro", "object": "model", "created": 0, "owned_by": "google"}]

    # plumbing stubs -------------------------------------------------------
    @staticmethod
    def _media_collector():
        return _StubCollector()

    @staticmethod
    def _cleanup_files(files) -> None:
        return None

    def _split(self, request):
        return None, [], "text", request.messages[-1], None, request.model

    # generation -----------------------------------------------------------
    async def chat_non_stream(self, request):
        self.calls.append(request)
        return FakeOutput(text=self.text, thoughts=self.thoughts, images=list(self.generated)), (
            request.model or "default"
        ), []

    async def chat_stream(self, request):
        self.calls.append(request)
        yield FakeOutput(text="pon", text_delta="pon")
        yield FakeOutput(text="pong", text_delta="g", images=list(self.generated))

    async def fetch_image_b64(self, image) -> str:
        return "STUBB64"


@pytest.fixture()
def app():
    from gemini_openai_proxy.config import Settings
    from gemini_openai_proxy.main import create_app

    application = create_app(Settings.load())
    application.state.gemini = StubService(application.state.settings)
    return application


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:  # lifespan runs against the stub
        yield test_client
