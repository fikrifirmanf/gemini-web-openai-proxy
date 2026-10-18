# gemini-openai-proxy

**OpenAI-compatible API over your Google Gemini Web session.** One small FastAPI service
that turns `gemini.google.com` (which you already use in a browser, with its free daily
quota) into a drop-in `base_url` for any OpenAI-style client — no official API key needed.

```text
OpenAI client ──HTTP──▶ this proxy ──cookie session──▶ gemini.google.com
                        /v1/chat/completions
                        /v1/models
                        /v1/images/generations
```

No API key, no billing. Chat, vision (send images/files), generated-image extraction,
SSE streaming, thinking-mode passthrough, optional bearer auth, Docker, cookie
auto-refresh.

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
| Chat completions | Multi-turn stateless mapping, system messages flattened into the prompt |
| Streaming (SSE) | True incremental deltas from `send_message_stream`; `stream_options.include_usage` supported; malformed upstream failures surface as an SSE `error` event, then `[DONE]` |
| Vision & files | `[image_url]` content parts: `data:` URL, http(s) URL, or local path; also PDF/text file pairs (`file_url`); sent upstream as attachments |
| Image generation | `/v1/images/generations` returns `b64_json` always — no Google CDN link leaking |
| Thoughts | `reasoning_effort: "high"`/`"medium"` maps to extended thinking; internal thoughts stream as `reasoning_content` deltas |
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
| `GWOP_REQUEST_TIMEOUT` | `300` | per upstream call |
| `GWOP_INIT_TIMEOUT` | `120` | session handshake |
| `GWOP_AUTO_RECOVERY` | `1` | self-healing: when Google reports the session unauthenticated, rotate `__Secure-1PSIDTS` from the long-lived 1PSID and rebuild the client in the background |
| `GWOP_RECOVERY_INTERVAL` | `60` | seconds between recovery checks |
| `GWOP_RECOVERY_MAX_ATTEMPTS` | `5` | consecutive failed rounds before recovery goes dormant (until `POST /admin/reinit` re-arms it) |

Precedence: `cookies.txt` **>** raw header **>** 1PSID pair. You only need one.

## Endpoints

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | liveness (always 200) |
| `GET /readyz` | readiness: 200 when the web session is live, 503 with a reason otherwise |
| `GET /v1/models` | models your account has (live-discovered) |
| `POST /v1/chat/completions` | chat; `stream=true` gives SSE |
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
| behind a corporate proxy | `GWOP_HTTP_PROXY` missing | set it to your HTTP proxy URL |

The runtime cookie cache lives at `GWOP_DATA_DIR/gemini_cookies/.cached_cookies_*.json`;
delete those files and reinit if it ever goes stale mid-session.

## Development

```bash
uv sync --all-groups
uv run pytest tests/ -v          # 30 tests, offline, no cookies needed
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
├── schemas.py      # OpenAI pydantic models
├── content.py      # message flattening + attachments
├── errors.py       # upstream exception → HTTP mapping
├── upstream.py     # GeminiClient lifecycle, session cache, generation
├── routes/         # meta | chat | images
└── main.py         # app factory, auth middleware, lifespan
```

## Licensing

AGPL-3.0-or-later (the upstream client is AGPL, this derives from it). See
[LICENSE](LICENSE) + [NOTICE](NOTICE). Personal use is unaffected; if you offer this as
a network service, the license requires you to share your modifications.
