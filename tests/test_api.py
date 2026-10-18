"""HTTP behavior of /v1/* endpoints against the stubbed upstream."""

import json


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_list(client):  # noqa: F811
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "gemini-pro" in ids


def test_chat_completions_json(client):  # noqa: F811
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gemini-pro", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "pong"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0


def test_chat_completions_with_thoughts(client):  # noqa: F811
    client.app.state.gemini.thoughts = "step by step..."
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "step by step..."
    assert message["content"] == "pong"


def test_generated_images_markdown_in_chat(client):  # noqa: F811
    from tests.conftest import GeneratedImage as FakeGeneratedImage

    client.app.state.gemini.generated = [FakeGeneratedImage()]
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "draw a cat"}]}
    )
    content = resp.json()["choices"][0]["message"]["content"]
    assert content.startswith("pong")
    assert "![a demo generated image]" in content


def test_stream_sse_format(client):  # noqa: F811
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("data: ") for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"

    parsed = [json.loads(e) for e in events[:-1]]
    assert parsed[0]["choices"][0]["delta"] == {"role": "assistant"}
    deltas = [p["choices"][0]["delta"].get("content") for p in parsed[1:-1]]
    assert "".join(d for d in deltas if d) == "pong"
    usage_chunk = parsed[-1]
    assert usage_chunk["choices"] == [] and usage_chunk["usage"]["total_tokens"] > 0


def test_image_generation_b64(client):  # noqa: F811
    from tests.conftest import GeneratedImage as FakeGeneratedImage

    client.app.state.gemini.generated = [FakeGeneratedImage()]
    resp = client.post("/v1/images/generations", json={"prompt": "a cat", "n": 1})
    assert resp.status_code == 200
    assert resp.json()["data"][0]["b64_json"] == "STUBB64"


def test_bearer_auth_required(monkeypatch):  # noqa: F811
    monkeypatch.setenv("GWOP_API_KEY", "s3cret")
    from fastapi.testclient import TestClient

    from gemini_openai_proxy.config import Settings
    from gemini_openai_proxy.main import create_app
    from tests.conftest import StubService

    application = create_app(Settings.load())
    application.state.gemini = StubService(application.state.settings)
    with TestClient(application) as tc:
        assert tc.get("/health").status_code == 200  # exempt
        assert tc.get("/v1/models").status_code == 401
        auth = {"Authorization": "Bearer wrong"}
        assert tc.get("/v1/models", headers=auth).status_code == 401
        auth["Authorization"] = "Bearer s3cret"
        assert tc.get("/v1/models", headers=auth).status_code == 200


def test_admin_status(client):  # noqa: F811
    body = client.get("/admin/status").json()
    assert body["service"] == "gemini-openai-proxy"


def test_responses_endpoint_non_stream(client):
    payload = {
        "model": "gemini-pro",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ],
        "instructions": "Be terse.",
        "reasoning": {"effort": "high"},
    }
    resp = client.post("/v1/responses", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "gemini-pro"
    assert body["output"][0]["type"] == "message"
    assert "output_text" in body["output"][0]["content"][0]["type"]
    assert body["usage"]["total_tokens"] > 0


def test_responses_endpoint_stream_events(client):
    payload = {"model": "gemini-flash", "input": "hi", "stream": True}
    events = [
        line.split(": ", 1)[1]  # event name
        for line in client.post("/v1/responses", json=payload).text.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "response.created"
    assert "response.output_text.delta" in events
    assert events[-1] == "response.completed"
