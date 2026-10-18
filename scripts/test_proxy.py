#!/usr/bin/env python3
"""End-to-end endpoint tester for gemini-openai-proxy (uses `requests`).

Points at a running proxy and verifies the API surface behaves like OpenAI.

Usage:
    uv run python scripts/test_proxy.py                       # standard checks
    uv run python scripts/test_proxy.py --base-url http://127.0.0.1:8011
    uv run python scripts/test_proxy.py --key s3cret          # bearer auth round-trip
    uv run python scripts/test_proxy.py --images              # also image generation
    uv run python scripts/test_proxy.py --model gemini-flash  # pin a model
    uv run python scripts/test_proxy.py --no-stream           # skip SSE test

Exit code 0 when every check passes; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

GREEN, RED, YELLOW, DIM = "\033[32m", "\033[31m", "\033[33m", "\033[2m"
BOLD = "\033[1m"


def c(status: str, text: str) -> str:
    return f"{status}{text}\033[0m"


class Reporter:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        mark = c(GREEN, "PASS") if ok else c(RED, "FAIL")
        line = f"{mark}  {name}"
        if detail:
            line += f"  {c(DIM, detail[:160])}"
        print(line)
        return ok

    def summary(self) -> int:
        failed = [r for r in self.results if not r[1]]
        print()
        if failed:
            print(f"{c(RED, str(len(failed)) + ' FAILED')} of {len(self.results)} checks")
            return 1
        print(f"{c(GREEN, 'ALL PASSED')} ({len(self.results)} checks)")
        return 0


def komplet_payload(model: str, messages: list[dict], **extra) -> dict:
    return {"model": model, "messages": messages, **extra}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("GWOP_TEST_BASE_URL", "http://127.0.0.1:8011"))
    ap.add_argument("--key", default=os.environ.get("GWOP_API_KEY", ""),
                    help="bearer key to use when the proxy requires auth (GWOP_API_KEY)")
    ap.add_argument("--model", default="default", help="model id; falls back to first from /v1/models")
    ap.add_argument("--images", action="store_true", help="also test /v1/images/generations (quota-heavy)")
    ap.add_argument("--no-stream", action="store_true", help="skip the SSE streaming check")
    ap.add_argument("--timeout", type=float, default=180, help="per-request timeout seconds")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    r = Reporter()
    headers = {}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"

    def req(method: str, path: str, **kw):
        url = base + path
        kw.setdefault("timeout", args.timeout)
        kw.setdefault("headers", headers)
        try:
            return requests.request(method, url, **kw)
        except requests.RequestException as exc:
            print(c(RED, f"Cannot reach {url}: {exc.__class__.__name__}: {exc}"))
            print(c(YELLOW, "Is the proxy running?  uv run python -m gemini_openai_proxy"))
            sys.exit(1)

    # ---------------------------------------------------------------- health
    resp = req("GET", "/health")
    ok = resp.status_code == 200 and resp.json().get("status") == "ok"
    r.check("GET /health → 200 ok", ok, f"status={resp.status_code}")

    resp = req("GET", "/readyz")
    body = {}
    try:
        body = resp.json()
    except ValueError:
        pass
    ok = resp.status_code == 200 and body.get("ready") is True
    r.check("GET /readyz → ready", ok, body.get("error", "")[:120])

    # ---------------------------------------------------------------- models
    resp = req("GET", "/v1/models")
    ok = resp.status_code == 200 and resp.json().get("object") == "list"
    if not r.check("GET /v1/models → OpenAI list shape", ok, f"status={resp.status_code}"):
        return r.summary()
    models = [m["id"] for m in resp.json()["data"]]
    r.check(f"models advertised ({len(models)})", len(models) > 0, ", ".join(models[:6]))

    model = args.model
    if model != "default" and models and model not in models:
        fallback = models[0]
        print(c(YELLOW, f"model {model!r} not advertised; using {fallback!r} instead"))
        model = fallback

    # ---------------------------------------------- chat: single-turn strict
    resp = req("POST", "/v1/chat/completions", json=komplet_payload(
        model, [{"role": "user", "content": "Reply with exactly: pong"}]))
    try:
        body = resp.json()
    except ValueError:
        body = {}
    ok = resp.status_code == 200 and body.get("object") == "chat.completion"
    if not r.check("chat/completions → completion envelope", ok,
                   f"status={resp.status_code} body={json.dumps(body)[:120]}"):
        return r.summary()
    choice = (body.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    usage_ok = body.get("usage", {}).get("total_tokens", 0) > 0
    r.check("assistant content non-empty", bool(content.strip()), repr(content[:80]))
    r.check("usage.total_tokens > 0", usage_ok, str(body.get("usage")))
    r.check("finish_reason=stop", choice.get("finish_reason") == "stop", str(choice.get("finish_reason")))

    # ------------------------------------------------------- chat multi-turn
    resp = req("POST", "/v1/chat/completions", json=komplet_payload(
        model,
        [
            {"role": "system", "content": "Answer in one word only."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
            {"role": "user", "content": "And the country directly south of it?"},
        ],
    ))
    try:
        body = resp.json()
    except ValueError:
        body = {}
    followup = ""
    if resp.status_code == 200:
        followup = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    r.check("multi-turn context respected", bool(followup.strip()) and "capital of france" not in followup.lower(),
            repr(followup[:80]))

    # -------------------------------------------------------------- streaming
    if not args.no_stream:
        started = time.monotonic()
        resp = req("POST", "/v1/chat/completions", stream=True, json=komplet_payload(
            model, [{"role": "user", "content": "Count 1 to 3, comma separated."}],
            stream=True, stream_options={"include_usage": True}))
        ct = resp.headers.get("content-type", "")
        ok = resp.status_code == 200 and ct.startswith("text/event-stream")
        if not r.check("stream → text/event-stream", ok, f"status={resp.status_code} ct={ct}"):
            resp.close()
            return r.summary()

        events: list[str] = []
        content_clock: float | None = None
        for raw in resp.iter_lines(decode_unicode=False):
            line = raw.decode("utf-8", errors="replace")
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                events.append("done")
                break
            chunk_meta = json.loads(payload)
            if events and not content_clock:
                for choice_ in chunk_meta.get("choices", []):
                    if choice_["delta"].get("content"):
                        content_clock = time.monotonic() - started
            events.append("chunk")
        resp.close()

        if not r.check("SSE terminated with [DONE]", events and events[-1] == "done",
                       f"{len(events)} events"):
            return r.summary()
        usage_seen = len(events) >= 3  # role chunk + content chunk(s) + final
        r.check("SSE chunk sequence", usage_seen, f"{len(events)} events")
        if content_clock is not None:
            r.check("first content delta latency", content_clock < 60,
                    f"{content_clock:.1f}s")
        else:
            r.check("first content delta latency", False, "no content chunk seen")

    # ------------------------------------------------------------------ auth
    if args.key:
        bad = requests.get(base + "/v1/models", headers={"Authorization": "Bearer NOT-YOUR-KEY"},
                           timeout=args.timeout)
        r.check("wrong bearer → 401", bad.status_code == 401, f"status={bad.status_code}")

    # ---------------------------------------------------------------- images
    if args.images:
        resp = req("POST", "/v1/images/generations", json={
            "prompt": "Generate an image: a tiny watercolor lemon in a white bowl", "n": 1,
        })
        try:
            body = resp.json()
        except ValueError:
            body = {}
        data = body.get("data") if resp.status_code == 200 else None
        ok = bool(data) and bool((data[0].get("b64_json") or ""))
        r.check("images/generations → b64_json", ok,
                json.dumps(body.get("error", {}))[:120] if not ok else f"{len(data)} image(s)")

    return r.summary()


if __name__ == "__main__":
    sys.exit(main())
