# gemini-openai-proxy

**OpenAI-compatible API over your Google Gemini Web session.** One small FastAPI service
that turns `gemini.google.com` (which you already use in a browser, with its free daily
quota) into a drop-in `base_url` for any OpenAI-style client — no official API key needed.

```text
OpenAI client ──HTTP──▶ this proxy ──cookie session──▶ gemini.google.com
                        /v1/chat/completions   (Chat Completions API)
                        /v1/responses          (Responses API — OpenAI Codex CLI)
                        /v1/models
                        /v1/images/generations
```

No API key, no billing. Chat, vision (send images/files), generated-image extraction,
SSE streaming, thinking-mode passthrough, **tool-calling emulation for coding agents
(OpenAI Codex CLI, incl. MCP tools)**, optional bearer auth, Docker, cookie auto-refresh
and self-healing.

## Contents

[Quick start](#quick-start) · [Wiring a client](#wiring-a-client) ·
[Features](#features) · [Configuration](#configuration-env-vars-gwop_-prefix) ·
[Endpoints](#endpoints) · [Using it with Codex CLI](#using-it-with-openai-codex-cli) ·
[Tool-calling emulation](#how-tool-calling-emulation-works) ·
[Debugging](#debugging-a-misbehaving-agent-loop) ·
[Cookie self-healing](#self-healing-cookie-rotation) ·
[Abuse safety](#being-treated-like-a-human-abuse-safety) ·
[Conversation mapping](#how-it-maps-an-openai-conversation-onto-the-web-app) ·
[Troubleshooting](#troubleshooting) · [Development](#development) ·
[Licensing](#licensing)

> ⚠️ **Unofficial & unaffiliated with Google.** It automates the web app with your own
> session and may break when Google changes internals. Attribution: built on the
> excellent [`gemini-webapi`](https://github.com/HanaokaYuzu/Gemini-API) (AGPL-3.0).
> See [NOTICE](NOTICE) for licensing notes.

---

## Quick start

You need Python ≥ 3.11 and a logged-in [gemini.google.com](https://gemini.google.com)
session.

```bash
# 1. Clone & install (uv — recommended)
uv sync
# or plain pip: pip install -e . && pip install "uvicorn[standard]"
```

```
# 2. Grab your cookies
#    gemini.google.com → F12 → Network → refresh → click any request
#    → Request Cookies → copy `__Secure-1PSID` and `__Secure-1PSIDTS`
```

```bash
# 3. Run
GWOP_GEMINI_1PSID='g.a0...' \
GWOP_GEMINI_1PSIDTS='sidts-Cj...' \
gemini-openai-proxy
# or: uv run python -m gemini_openai_proxy
# Server listens on http://127.0.0.1:8011
```

```bash
# 4. Verify
curl -sS http://127.0.0.1:8011/health
curl -sS http://127.0.0.1:8011/v1/models | jq .data[].id
curl -sS http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say pong"}]}'
```

### Or with a `.env` file

```bash
cp .env.example .env   # edit cookie values
uv run python -m gemini_openai_proxy
```

### Or with Docker

```bash
export GWOP_GEMINI_1PSID='g.a0...'
export GWOP_GEMINI_1PSIDTS='sidts-Cj...'
docker compose up -d          # cookies cache persists under ./data
```

The container binds `0.0.0.0`; publish the port on localhost only (or set `GWOP_API_KEY`) unless you
really want it reachable from other machines.

---

## Wiring a client

Any tool that talks OpenAI dialect works: point it at `http://127.0.0.1:8011/v1`
(any non-empty `api_key`, unless you set one via config).

**OpenAI Python SDK** — full runnable versions in [`examples/`](examples/).

**curl (streaming)**

```bash
curl -N http://localhost:8011/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"stream": true, "stream_options": {"include_usage": true}, \
       "messages":[{"role":"user","content":"tell me a haiku about proxies"}]}'
```

Known-compatible chat apps: ChatBox, NextChat, LobeChat, BoltAI / LibreChat — set
provider = "OpenAI", base URL as above, API key = anything.

---

## Features

| Feature | Notes |
| --- | --- |
| Responses API | `POST /v1/responses` (streaming + non-streaming) for Codex CLI ≥ 0.145, which no longer speaks Chat Completions |
| Tool calling | Prompt-based emulation: function tools, Codex `apply_patch` (custom tool) and MCP tools become real `function_call` items; see [below](#how-tool-calling-emulation-works) |
| Chat completions | Multi-turn stateless mapping, system messages flattened into the prompt |
| Streaming (SSE) | True incremental deltas from `send_message_stream`; `stream_options.include_usage` supported; malformed upstream failures surface as an SSE `error` event, then `[DONE]` |
| Vision & files | `[image_url]` content parts: `data:` URL, http(s) URL, or local path; also PDF/text file pairs (`file_url`); sent upstream as attachments |
| Image generation | `/v1/images/generations` returns `b64_json` always — no Google CDN link leaking |
| Thoughts | `reasoning_effort`/`reasoning.effort` → Gemini Web extended thinking. Triggers: `medium`, `high`, `max`, `enabled` (the mapping is `wants_extended_thinking` in `schemas.py`); `none`/`low`/unset = standard mode. Thoughts stream as `reasoning_content` deltas (hide with `GWOP_INCLUDE_THOUGHTS=0`) |
| Session reuse | Stateless requests are mapped onto a small LRU of live web conversations; a cache hit only sends your newest message upstream |
| Cookies | Manual (env / raw header / cookies.txt). The library auto-refreshes `__Secure-1PSIDTS` and caches it under `GWOP_DATA_DIR/gemini_cookies` |
| Auth | `GWOP_API_KEY` triggers Bearer-token enforcement on everything except `/health`/`/readyz` |
| Model list | `/v1/models` mirrors the models *your account* actually has, discovered live |
| Ephemeral chats | `GWOP_TEMPORARY_CHATS=1` (default) so model calls don't flood your Gemini history |

## Configuration (env vars, `GWOP_` prefix)

| Variable | Default | Behavior |
| --- | --- | --- |
| `GWOP_HOST` / `GWOP_PORT` | `127.0.0.1` / `8011` | listen address |
| `GWOP_API_KEY` | *(unset)* | empty → open; set → require `Authorization: Bearer <key>` |
| `GWOP_LOG_LEVEL` | `INFO` | DEBUG gets chatty upstream logs |
| `GWOP_GEMINI_1PSID` / `GWOP_GEMINI_1PSIDTS` | — | cookie pair (recommended) |
| `GWOP_GEMINI_COOKIES_RAW` | — | full cookie header: `k=v; k2=v2` |
| `GWOP_COOKIE_FILE` | — | Netscape cookies.txt path |
| `GWOP_HTTP_PROXY` | — | e.g. `http://127.0.0.1:7897` (also applies to upstream downloads) |
| `GWOP_DATA_DIR` | `./data` | where auto-refreshed cookies cache |
| `GWOP_MAX_CONCURRENCY` | `3` | cap on simultaneous upstream calls |
| `GWOP_SESSION_CACHE_SIZE` | `256` | remembered conversations |
| `GWOP_TEMPORARY_CHATS` | `1` | `0` persists chats in Gemini history |
| `GWOP_STRICT_MODEL` | `0` | `1` → unknown model strings return **400** instead of falling back to default |
| `GWOP_IMAGES_IN_CHAT` | `1` | append generated-image markdown to chat replies |
| `GWOP_INCLUDE_THOUGHTS` | `1` | `0` hides `reasoning_content` |
| `GWOP_REQUEST_TIMEOUT` | `450` | max wait for one generation |
| `GWOP_STALL_TIMEOUT` | `240` | idle stream before the library reconnects and re-sends the prompt |
| `GWOP_TOTAL_TIMEOUT` | `900` | hard cap per non-stream call, library retries included (504) |
| `GWOP_INIT_TIMEOUT` | `120` | session handshake only |
| `GWOP_SSE_KEEPALIVE` | `15` | idle seconds before an SSE `: keepalive` comment (0 = off) |
| `GWOP_AUTO_RECOVERY` | `1` | self-healing: when Google reports the session unauthenticated, rotate `__Secure-1PSIDTS` from the long-lived 1PSID and rebuild the client in the background |
| `GWOP_RECOVERY_INTERVAL` | `60` | seconds between recovery checks |
| `GWOP_RECOVERY_MAX_ATTEMPTS` | `5` | consecutive failed rounds before recovery goes dormant (until `POST /admin/reinit` re-arms it) |
| `GWOP_SEND_INTERVAL` | `1.5` | min jittered gap in seconds between upstream sends (burst pacing) |
| `GWOP_QUOTA_COOLDOWN` | `300` | seconds new requests are locally 429'd (`Retry-After`) after an upstream 429 |
| `GWOP_SINGLE_INSTANCE` | `1` | flock-protect the data dir; a 2nd proxy process aborts instead of racing cookie rotation |
| `GWOP_TOOL_EMULATION` | `1` | `/v1/responses`: turn declared tools into the JSON/raw-block protocol and lift replies into `function_call` items. `0` → tools are ignored and Gemini answers in plain text |
| `GWOP_TOOL_ROUND_THINKING` | `1` | honour the requested reasoning effort on **every** agent round. `0` skips extended thinking on rounds that only answer a tool result (faster, but Gemini Web then returns canned errors / fake summaries about half the time) |
| `GWOP_TOOL_EXCLUDE` | `^(multi_agent\|mcp__codex_apps__)` | regex of Codex tool names/namespaces **not** offered to Gemini (hosted/connector tools it can't use well; keeps the prompt small) |
| `GWOP_DEBUG_DUMP_DIR` | *(unset)* | write every upstream prompt + reply (and the declared tool list) as JSON files for debugging. Contains your prompts and code — keep it local |

Precedence: `cookies.txt` **>** raw header **>** 1PSID pair. You only need one.

## Endpoints

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | liveness (always 200) |
| `GET /readyz` | readiness: 200 when the web session is live, 503 with a reason otherwise |
| `GET /v1/models` | models your account has (live-discovered) |
| `POST /v1/chat/completions` | chat; `stream=true` gives SSE |
| `POST /v1/responses` | OpenAI **Responses API** (Codex ≥0.145 talks this); instructions + input items → Gemini Web; streaming and non-streaming, with emulated tool calling |
| `POST /v1/images/generations` | prompt in, `b64_json` out; reference images accepted via an extra `"image"` field (e.g. OpenAI SDK `extra_body={"image": [...]}`) |
| `POST /admin/reinit` | force a fresh upstream session; **re-reads `.env`** first (refresh cookies without restarting); re-arms the recovery breaker |
| `GET /admin/status` | banner + quota/usage snapshot when the account exposes it (includes the `recovery` state) |

## Self-healing cookie rotation

Google binds Gemini web sessions to a small set of rotating cookies:

- `__Secure-1PSID` — long-lived; identifies your account
- `__Secure-1PSIDTS` — **short-lived; Google rotates it server-side within hours** and
  invalidates the previous value (this is what most often makes a copied session go
  "unauthenticated")

The upstream library already keeps a *healthy* session alive (background rotation every
10 min + activity heartbeat). What this proxy adds is a recovery loop for the *broken*
state:

1. A background loop checks every `GWOP_RECOVERY_INTERVAL` seconds (and any request that
   hits a 401 triggers an immediate attempt).
2. When the session is unauthenticated, it drops the stale cookie cache entry, asks
   Google's `RotateCookies` endpoint for a fresh `__Secure-1PSIDTS` using your
   `__Secure-1PSID` (same browser TLS fingerprint as sign-in), rebuilds the client, and
   verifies the account is fully authenticated again.
3. **Google only grants rotation to sessions it still trusts.** A pair whose TS was
   already rotated away (e.g. your browser invalidated it right after you copied it) gets
   a 401 and cannot self-heal — the loop reports exactly that and stops once its attempt
   budget is gone; the fix is to re-copy fresh values into `.env` and `POST /admin/reinit`
   (which re-reads `.env` immediately, no restart).
4. It never hammers Google: rounds are debounced to >=60 s apart and stop (dormant, logged
   clearly, visible in `/admin/status`) after `GWOP_RECOVERY_MAX_ATTEMPTS` consecutive
   failures. A successful round or manual `POST /admin/reinit` re-arms it.

ℹ️ Because the healthy-session refresh runs every ~10 min from within the proxy, the
usual failure mode ("Chrome rotated the session under the proxy's feet") is rare — it
mostly hits when the proxy was stopped/offline while Chrome kept rotating.

## Being treated like a human (abuse safety)

Google flags accounts on *traffic shape*, not on the proxy itself. The proxy ships the
guard-rails below, on by default:

| Guard | Default | What it does |
| --- | --- | --- |
| Send pacing | `GWOP_SEND_INTERVAL=1.5`s (jittered ×1.0–1.5) | minimum real-time gap between upstream sends, regardless of how hard clients parallelize |
| Concurrency cap | `GWOP_MAX_CONCURRENCY=3` | simultaneous upstream calls, including image generation (which used to be ungated) |
| Quota cooldown | `GWOP_QUOTA_COOLDOWN=300`s | after an upstream 429, new model traffic gets a local 429 + `Retry-After` instead of continuing to poke Google with a dry quota |
| Single instance | `GWOP_SINGLE_INSTANCE=1` | refuses a 2nd proxy on the same `data_dir` (concurrent cookie-rotation races kill sessions and look ugly to Google) |
| Bounded self-heal | ≥60 s apart, ≤ `GWOP_RECOVERY_MAX_ATTEMPTS` | the recovery loop never hammers `/RotateCookies` |
| Temporary chats | `GWOP_TEMPORARY_CHATS=1` | requests never land in Gemini web history |

Extra posture advice:

- **Run it for yourself only.** Bind `GWOP_HOST=127.0.0.1`, or set `GWOP_API_KEY` the
  moment you bind anything wider — an open relay on your account is the single worst
  abuse-safety mistake.
- **Keep one egress IP** (`GWOP_HTTP_PROXY` for steadiness if your home IP churns);
  rotating hosts/VPN hops are a strong automation signal.
- **Watch your abuse status** in `/admin/status` (`Account abuse status: Clean`) and the
  per-model quota lines; stop when credits run dry instead of hot-retrying.
- **Volume discipline**: a chat CLI for a human stays far below the per-day web quota; if
  you wire it into bulk automation (cron loops, bulk pipelines), slow it down on purpose.
- Be aware automating the Gemini web app is itself against Google's ToS — account-level
  risk is yours; the guards above minimize the *pattern* that triggers flags, not the
  policy violation.

## Using it with OpenAI Codex CLI

Codex ≥0.145 dropped support for chat-completions providers — it requires the OpenAI
**Responses API**, so the proxy implements `POST /v1/responses` (translate input items /
instructions into Gemini Web turns; stream as `response.*` events).

`~/.codex/config.toml`:

```toml
[model_providers.gemini-web]
name = "Gemini Web (local proxy)"
base_url = "http://127.0.0.1:8011/v1"
wire_api = "responses"
```

plus one profile file per model, e.g. `~/.codex/gemini-flash.config.toml`:

```toml
model = "gemini-flash"
model_provider = "gemini-web"
```

Then: `codex --profile gemini-flash` (also `gemini-flash-lite` / `gemini-pro` /
`gemini-argon`).

Recommended profile for coding work (`gemini-argon` with high effort works well; see
the [thinking caveats](#thinking-effort-reality-check-measured-live) for `gemini-pro`):

```toml
model = "gemini-argon"
model_provider = "gemini-web"
model_reasoning_effort = "high"
```

Start a **new** Codex session after changing proxy versions — a resumed session replays
old assistant turns (including past refusals) that teach the model bad habits.

### MCP tools in Codex

MCP servers configured in Codex (`[mcp_servers.*]`) are forwarded to the proxy as
namespaced tools. The proxy flattens each namespace into individual function names for
Gemini, and restores the `namespace` on the way back so Codex routes the call to the
right server. Verified live: `McpToolCall` events complete normally. Notes:

- Tools matching `GWOP_TOOL_EXCLUDE` (by default Codex's hosted `multi_agent` and
  `mcp__codex_apps__*` connectors) are never offered.
- If two servers expose the same tool name, the first one wins.

## How tool-calling emulation works

Gemini Web has no function-calling API, so the proxy emulates one when a request declares
tools:

1. **Protocol prompt.** A compact tool catalogue plus the reply protocol is added to the
   system instructions, written in the user's voice (framing it as the user's own harness
   avoids Gemini treating it as a "prompt injection"). A short reminder is appended to
   each outgoing turn at send time.
2. **Two reply formats.** Gemini answers with either JSON,
   `{"tool_calls": [{"name": "...", "arguments": {...}}]}`, or **raw blocks** for
   code-heavy arguments where JSON escaping breaks:

   ```text
   @@tool exec_command
   @@arg cmd
   cat > src/app.js << 'EOF'
   console.log("no escaping needed");
   EOF
   @@end
   ```

3. **Lifting.** The proxy parses the reply (with bare-key/comma repair and tolerance for
   trailing junk), coerces arguments to the declared JSON schema (string → array,
   numeric strings → numbers), and emits real `function_call` / `custom_tool_call`
   items. Codex runs them and sends back `function_call_output`, closing the loop.
4. **Guard rails** — Gemini Web is flaky, so replies are checked and given **one
   corrective retry** when they:
   - are mangled JSON / a failed raw block,
   - call a tool name that was never declared (e.g. an invented `shell`),
   - are a canned Google error ("I'm having trouble…"),
   - refuse with an invented limit ("my tool budget is exhausted") or call the protocol
     a prompt injection,
   - *announce* an action in prose instead of doing it ("I will now create…"),
   - claim the work is done when no file-changing call actually ran this turn
     (reads, `pnpm dev`, `pnpm build` don't count as changes).
   Past assistant turns of those kinds are also scrubbed from the replayed history so
   they are not copied.
5. **Speed.** The proxy re-keys the cached web session under the assistant turn it just
   returned, so round N sends only the newest tool result instead of the whole
   transcript. Tool results are clipped to ~12k characters (head + tail), and the tool
   catalogue is compacted. Tool rounds are buffered (the JSON is never streamed to the
   client as text); plain answers stream incrementally. An SSE keep-alive comment is sent
   while Gemini thinks so Codex doesn't time out.
6. **Final-text integrity.** Gemini Web rewrites text after streaming starts (e.g. wraps
   URLs as `[url](url)`), which can make delta diffs lose characters. The proxy uses the
   final full snapshot, unwraps bare-URL auto-links, and keys the session cache on exactly
   what Codex will send back.
7. **Cheap titles.** Codex's hidden "generate a task title" request is answered without the
   coding-agent system prompt and without extended thinking.

Limits to be aware of: it is prompt-based, so quality depends on the model obeying the
protocol; each agent round is one upstream Gemini turn (and obeys the pacing interval);
very long sessions can fall off the session cache and re-send the history once. Disable
everything with `GWOP_TOOL_EMULATION=0`.

## Debugging a misbehaving agent loop

Set `GWOP_DEBUG_DUMP_DIR=./data/dump` and restart. For every upstream send the proxy
writes `HHMMSS-mmm.json` (`prompt`, `reply`, `reused_session`, `extended_thinking`) and,
per Responses request, `tools-HHMMSS.json` (the tool names Codex declared). Compare them
with Codex's own session rollout in `~/.codex/sessions/YYYY/MM/DD/*.jsonl`:

- `reused_session: false` with a huge prompt on a late turn → session cache miss
  (slower; check that Codex's stored text matches the proxy's last reply).
- A reply that is prose where a tool call was expected → look for the retry log lines
  (`corrective retry`, `unsupported call`).
- `docker`/service logs at `GWOP_LOG_LEVEL=DEBUG` show the upstream watchdog and
  reconnect events.

The dump files hold your prompts and code: they live under the git-ignored `data/` folder
by default, never commit or share them blindly.

## How it maps an OpenAI conversation onto the web app

The web app is session-based; OpenAI's API is stateless. The proxy bridges them:

1. **Fresh conversation** — the full message list is flattened into one transcript prompt
   (`[User]: … [Assistant]: …`) with the system instructions up top, sent to a new web
   chat.
2. **Known conversation** — if this exact history was already sent upstream (LRU cache,
   keyed by normalized messages + model), the proxy reattaches to the same web session and
   sends *only* your newest message. Faster and more natural context.
3. **Streaming** — `chat.send_message_stream` gives true incremental `text_delta`s, which
   become `delta.content` SSE chunks. Generated images arrive on the final chunks as
   markdown (`![alt](url)`); non-stream responses append them after the text.

Everything is `temporary=True` upstream by default so nothing lands in your Gemini web
history.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `401 upstream_auth_error` | cookies expired / logout | re-copy the pair; `POST /admin/reinit` |
| `429 upstream_rate_limited` | daily quota / IP block from parallel calls | wait, lower `GWOP_MAX_CONCURRENCY` |
| `502 … did not return any images` | prompt phrased as "search for images" | ask it to *generate* an image |
| models list only shows `gemini-flash-lite` | session is unauthenticated (anonymous tier) | auto-recovery will attempt `__Secure-1PSIDTS` rotation in the background; **Google only rotates a session it still trusts** — if the copied pair has already been rotated away, re-copy fresh values into `.env` and `POST /admin/reinit` (re-reads `.env`; no restart needed) |
| `503 No Gemini cookies found` | env vars missing/typo'd | see message; use exactly the cookie **values** (not the names) |
| cookies keep expiring although you re-copy them | an interactive Chrome session keeps rotating `__Secure-1PSIDTS` under the proxy's feet | log in to gemini.google.com in a **separate private/incognito window**, export the cookies, close that window so nothing else rotates the session; Firefox cookies also rotate far less (browser-cookie3 path) |
| Streaming clients hang | a proxy in front strips SSE? | server sets `X-Accel-Buffering: no`; bypass intermediate caching proxies |
| Codex says "tool budget exhausted" / "prompt injection" / only narrates actions | old proxy version, or a resumed session that replays old refusals | restart the proxy and start a **new** Codex session; with `GWOP_DEBUG_DUMP_DIR` check the latest dump |
| Logs show `Heartbeat triggered` / `[Watchdog] Connection idle … Attempting recovery` | Gemini produced nothing for `GWOP_STALL_TIMEOUT` (long thinking on high effort) | normal self-recovery; raise `GWOP_STALL_TIMEOUT` / `GWOP_REQUEST_TIMEOUT` if it fires too early |
| Agent rounds randomly return "I'm having trouble…" | Gemini Web canned error, more frequent with thinking off on tool rounds | keep `GWOP_TOOL_ROUND_THINKING=1`; the proxy already retries once |
| behind a corporate proxy | `GWOP_HTTP_PROXY` missing | set it to your HTTP proxy URL |

The runtime cookie cache lives at `GWOP_DATA_DIR/gemini_cookies/.cached_cookies_*.json`;
delete those files and reinit if it ever goes stale mid-session.

## Development

```bash
uv sync --all-groups
uv run pytest tests/ -v          # ~120 tests, offline, no cookies needed
```

### CLI chatbot

```bash
uv run python scripts/chat.py                                   # model + effort pickers, then chat
uv run python scripts/chat.py --model gemini-flash --effort high
uv run python scripts/chat.py --model gemini-argon --effort medium --system "Reply like a pirate"
uv run python scripts/chat.py --prompt "1+1?" --model gemini-flash   # one-shot for pipes/scripts
uv run python scripts/chat.py --prompt "…" --no-stream              # plain print for automation
```

In-chat commands: `/help` `/new` `/model` `/effort` `/status` `/reinit` `/exit`
(`--effort medium|high` enables Gemini thinking; thoughts stream in dim ink).
Requires the proxy to be running (`uv run python -m gemini_openai_proxy`).

#### Thinking-effort reality check (measured live)

- **`gemini-flash` + effort** — works well: private thoughts stream as
  `reasoning_content`, then the answer.
- **`gemini-pro` + effort** — thinking engages (thoughts arrive) but the final
  answer frequently turns into Google's "I seem to be encountering an error"
  grovel, and turns take 40 s to minutes. Leave effort unset for `gemini-pro`
  unless you enjoy waiting.
- **Codex** — the CLI already sends `effort: "low"` by default, which the proxy
  maps to standard (fast) mode. To turn thinking on for a profile, add
  `model_reasoning_effort = "medium"` to its `~/.codex/<name>.config.toml`.

### End-to-end endpoint smoke test (against a running proxy)

```bash
uv run uvicorn gemini_openai_proxy.main:app --port 8011 &   # in another shell
uv run python scripts/test_proxy.py              # health, models, chat, multi-turn, SSE
uv run python scripts/test_proxy.py --images     # + image generation (quota-heavy)
uv run python scripts/test_proxy.py --key s3cret # + bearer auth round-trip
uv run python scripts/test_proxy.py --base-url http://127.0.0.1:8011 --model gemini-flash
```

Exit code 0 when all checks pass (CI-friendly).

Project layout:

```
src/gemini_openai_proxy/
├── config.py       # env settings
├── cookies.py      # 3 cookie sources
├── schemas.py      # OpenAI pydantic models (Chat Completions + Responses → chat request)
├── content.py      # message flattening + attachments
├── errors.py       # upstream exception → HTTP mapping
├── upstream.py     # GeminiClient lifecycle, session cache, timeouts, generation
├── tools_emulation.py  # tool protocol prompt, parsers, refusal/fake-done detectors
├── routes/         # meta | chat | responses | images (+ SSE keep-alive helper)
└── main.py         # app factory, auth middleware, lifespan
```

## Licensing

AGPL-3.0-or-later (the upstream client is AGPL, this derives from it). See
[LICENSE](LICENSE) + [NOTICE](NOTICE). Personal use is unaffected; if you offer this as
a network service, the license requires you to share your modifications.
