"""OpenAI request/response schema tolerance."""

from gemini_openai_proxy.schemas import ChatCompletionRequest, ImagesRequest, make_usage


def _msg(**kwargs):
    base = {"role": "user", "content": "hi"}
    base.update(kwargs)
    return base


def test_minimal_request():
    req = ChatCompletionRequest.model_validate({"messages": [_msg()]})
    assert req.model == "default" and req.stream is False


def test_content_part_lists_and_none():
    req = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]}]}
    )
    assert isinstance(req.messages[0].content, list)


def test_null_content_normalized():
    req = ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": None}]})
    assert req.messages[0].content == ""


def test_extra_fields_ignored():
    noisy = {"messages": [_msg()], "temperature": 0.7, "top_p": 1, "frequency_penalty": 0}
    req = ChatCompletionRequest.model_validate(noisy)
    assert req.messages


def test_reasoning_effort_mapping():
    for value in ("high", "MEDIUM", "enable"):
        req = ChatCompletionRequest.model_validate({"messages": [_msg()], "reasoning_effort": value})
        assert req.wants_extended_thinking
    for value in (None, "low", "minimal"):
        req = ChatCompletionRequest.model_validate({"messages": [_msg()], "reasoning_effort": value})
        assert not req.wants_extended_thinking


def test_stream_options_include_usage():
    req = ChatCompletionRequest.model_validate(
        {"messages": [_msg()], "stream": True, "stream_options": {"include_usage": True}}
    )
    assert req.include_usage


def test_images_request_defaults():
    req = ImagesRequest.model_validate({"prompt": "cat"})
    assert req.n == 1 and req.response_format == "b64_json"


def test_usage_math():
    usage = make_usage(800, 1200)
    assert usage == {"prompt_tokens": 200, "completion_tokens": 300, "total_tokens": 500}
