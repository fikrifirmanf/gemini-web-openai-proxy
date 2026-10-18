"""Prompt builders and media materialization."""

import base64

from gemini_openai_proxy.content import (
    MediaCollector,
    build_turn_text,
    decode_data_url,
    make_fallback_prompt,
    text_of,
)

PNG_1PX = base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"0" * 32
).decode()


def test_text_of_plain_and_parts():
    assert text_of("hello") == "hello"
    assert text_of(None) == ""
    assert text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert text_of([{"type": "image_url", "image_url": {"url": "x"}}]) == ""


def test_build_turn_text_appends_media():
    text, urls = build_turn_text("User", "look")
    assert (text, urls) == ("[User]: look", [])


def test_decode_data_url_roundtrip():
    raw = decode_data_url(f"data:image/png;base64,{PNG_1PX}", max_bytes=1024)
    assert raw.startswith(b"\x89PNG")


def test_decode_data_url_rejects_oversize():
    big = base64.b64encode(b"1" * 9999).decode()
    try:
        decode_data_url(f"data:image/png;base64,{big}", max_bytes=10)
        raise AssertionError("should have raised MediaError")
    except Exception as exc:
        assert type(exc).__name__ == "MediaError"


def test_fallback_prompt_structure():
    prompt = make_fallback_prompt(
        history=[("User", "hi"), ("Assistant", "hello")],
        final_user="answer me",
        system="Be terse",
        has_files=False,
    )
    assert "System instruction" in prompt
    assert "[User]: hi" in prompt and "[Assistant]: hello" in prompt
    assert prompt.strip().endswith("[User]: answer me")
    assert "Respond only to the final [User] message" in prompt


def test_collector_picks_image_parts_only_from_last_user(monkeypatch):
    async def fake_materialize(self, url):
        return f"/tmp/converted::{url}"

    monkeypatch.setattr(MediaCollector, "_materialize", fake_materialize)

    class M:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    messages = [
        M("user", [{"type": "text", "text": "old turn"}]),
        M("assistant", "assistant text"),
        M("user", [
            {"type": "text", "text": "with pic"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZA=="}},
        ]),
    ]
    collector = MediaCollector(max_bytes=1024, http=object())
    import asyncio

    asyncio.new_event_loop().run_until_complete(collector.collect(messages, only_last=True))
    assert collector.files == ["/tmp/converted::data:image/png;base64,ZA=="]
    assert not any("old turn" in f for f in collector.files)
