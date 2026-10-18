#!/usr/bin/env python3
"""Simple interactive chatbot over gemini-openai-proxy (uses `requests`).

Interactive (default):
    uv run python scripts/chat.py
    uv run python scripts/chat.py --model gemini-flash --effort high

One-shot (pipes / cron / other tools):
    uv run python scripts/chat.py --prompt "1+1?" --model gemini-flash

Commands inside the chat:
    /help                    show the command list
    /new                     reset the conversation
    /model [<id>]            show or switch the model
    /effort [<level>]        show or set thinking effort (none/low/medium/high)
    /status                  upstream status / quota snapshot (admin)
    /reinit                  rebuild the upstream session (after cookie changes)
    /exit (or Ctrl+D)        leave

`--effort medium|high` sends `reasoning_effort` upstream; when the model thinks,
its thoughts stream in dim ink before the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import requests

DIM, BOLD, GREEN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"

# effort -> the value sent as reasoning_effort (proxy maps medium/high to extended thinking)
EFFORTS: dict[str, str | None] = {
    "none": None,
    "low": None,
    "medium": "medium",
    "high": "high",
}

HELP_TEXT = """\
  /help          show this list
  /new           reset the conversation
  /model [<id>]  show or switch the model
  /effort [<l>]  show or set effort (none/low/medium/high)
  /status        upstream status + quota snapshot (admin)
  /reinit        rebuild the upstream session (after cookie changes)
  /exit          leave (Ctrl+D / Ctrl+C on empty prompt also work)
"""


@dataclass
class Shell:
    base_url: str
    headers: dict
    model: str = "default"
    effort: str = "none"
    system: str | None = None
    timeout: float = 300.0
    history: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ http
    def request(self, method: str, path: str, **kw) -> requests.Response:
        kw.setdefault("timeout", self.timeout)
        kw.setdefault("headers", self.headers)
        return requests.request(method, self.base_url + path, **kw)

    def _parse(self, resp: requests.Response) -> dict:
        try:
            return resp.json()
        except ValueError:
            return {}

    def _print_http_error(self, resp: requests.Response) -> None:
        body = self._parse(resp)
        err = body.get("error", {}) if isinstance(body, dict) else {}
        print(f"{RED}✗ HTTP {resp.status_code}: {err.get('message', 'unknown error')}{RESET}")
        if err.get("hint"):
            print(f"{RED}  hint: {err['hint']}{RESET}")

    # --------------------------------------------------------------- helpers
    def system_msg(self) -> list[dict]:
        return [{"role": "system", "content": self.system}] if self.system else []

    # ------------------------------------------------------------ generation
    def complete(self, user_text: str) -> str:
        """Non-streaming single completion; returns the full text."""
        self.history.append({"role": "user", "content": user_text})
        payload = {"model": self.model, "messages": self.system_msg() + self.history}
        if EFFORTS[self.effort]:
            payload["reasoning_effort"] = EFFORTS[self.effort]
        try:
            resp = self.request("POST", "/v1/chat/completions", json=payload)
        except (requests.ConnectionError, requests.Timeout) as exc:
            self.history.pop()
            print(f"{RED}connection error: {exc.__class__.__name__}{RESET}")
            return ""
        if resp.status_code != 200:
            self.history.pop()
            self._print_http_error(resp)
            return ""
        text = (self._parse(resp).get("choices") or [{}])[0].get("message", {}).get("content", "")
        if text.strip():
            self.history.append({"role": "assistant", "content": text})
        else:
            self.history.pop()  # dangling user turn with nothing to anchor it
            print(f"{YELLOW}(empty response — try again or switch model){RESET}")
        return text

    def complete_stream(self, user_text: str) -> str:
        """Streaming completion: prints deltas as they arrive, returns full text."""
        self.history.append({"role": "user", "content": user_text})
        payload = {"model": self.model, "messages": self.system_msg() + self.history, "stream": True}
        if EFFORTS[self.effort]:
            payload["reasoning_effort"] = EFFORTS[self.effort]

        show_thoughts = EFFORTS[self.effort] is not None  # only dim-print thoughts when effort is set
        shown_thought = False
        collected: list[str] = []
        started = time.monotonic()

        try:
            resp = self.request("POST", "/v1/chat/completions", json=payload, stream=True)
        except (requests.ConnectionError, requests.Timeout) as exc:
            self.history.pop()
            print(f"{RED}connection error: {exc.__class__.__name__}{RESET}")
            return ""

        if resp.status_code != 200:
            self.history.pop()
            self._print_http_error(resp)
            return ""

        aborted = False
        try:
            for raw in resp.iter_lines(decode_unicode=False):
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                payload_text = line[len("data: "):]
                if payload_text == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:  # upstream failure mid-stream
                    err = chunk["error"]
                    self.history.pop()
                    print(f"\n{RED}✗ {err.get('message', 'upstream error')}{RESET}")
                    if err.get("hint"):
                        print(f"{RED}  hint: {err['hint']}{RESET}")
                    return ""
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                thoughts = delta.get("reasoning_content")
                if thoughts and show_thoughts:
                    print(f"{DIM}{thoughts}{RESET}", end="", flush=True)
                    shown_thought = True
                piece = delta.get("content")
                if piece:
                    if shown_thought:
                        print()  # close the dimmed thoughts block before the answer
                        shown_thought = False
                    collected.append(piece)
                    print(piece, end="", flush=True)
        except KeyboardInterrupt:
            aborted = True
            print(f"\n{DIM}…(stopped by you){RESET}")
        finally:  # noqa: B012 - close regardless; iter_lines can linger
            resp.close()

        answer = "".join(collected)
        if answer.strip():
            self.history.append({"role": "assistant", "content": answer})
            if not aborted:
                print()
                print(f"{DIM}⏱ {time.monotonic() - started:.1f}s · {len(answer)} chars{RESET}")
        else:
            self.history.pop()
            if not aborted:
                print(f"{YELLOW}(empty response — try again or switch model){RESET}")
        return answer

    # -------------------------------------------------------------- commands
    def list_models(self) -> list[str]:
        resp = self.request("GET", "/v1/models")
        if resp.status_code != 200:
            return []
        return [m["id"] for m in self._parse(resp).get("data", [])]

    def choose_model_interactively(self) -> None:
        models = self.list_models()
        if not models:
            print(f"{YELLOW}no models advertised; using account default{RESET}")
            return
        print(f"\n{BOLD}Choose a model:{RESET}")
        print(f"  [{GREEN}0{RESET}] account default")
        for i, mid in enumerate(models, 1):
            print(f"  [{GREEN}{i}{RESET}] {mid}")
        raw = self.ask("model > ").lower()
        if raw.strip() in {"", "0", "d", "default"}:
            self.model = "default"
        elif raw.isdigit() and 1 <= int(raw) <= len(models):
            self.model = models[int(raw) - 1]
        elif raw in models:
            self.model = raw
        else:
            self.model = "default"
            print(f"{YELLOW}unrecognized choice; using account default{RESET}")

    def choose_effort_interactively(self) -> None:
        print(f"\n{BOLD}Thinking effort:{RESET}")
        levels = list(EFFORTS)
        for i, level in enumerate(levels):
            mark = "  ← current" if level == self.effort else ""
            print(f"  [{GREEN}{i}{RESET}] {level}{mark}")
        raw = self.ask("effort > ").strip().lower()
        if raw == "":
            pass  # keep current
        elif raw.isdigit() and int(raw) < len(levels):
            self.effort = levels[int(raw)]
        elif raw in EFFORTS:
            self.effort = raw
        else:
            print(f"{YELLOW}kept '{self.effort}'{RESET}")

    @staticmethod
    def ask(prompt: str) -> str:
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0) from None

    def status(self) -> None:
        resp = self.request("GET", "/admin/status")
        body = self._parse(resp) if resp.status_code == 200 else {"error": f"HTTP {resp.status_code}"}
        print(json.dumps(body, indent=2)[:1200])

    def reinit(self) -> None:
        print(f"{DIM}reinitializing upstream session…{RESET}")
        resp = self.request("POST", "/admin/reinit")
        body = self._parse(resp)
        if resp.status_code == 200:
            print(f"{GREEN}✓ upstream session rebuilt{RESET}")
        else:
            print(f"{RED}✗ {body.get('error', 'reinit failed')}{RESET}")

    def banner(self) -> None:
        histories = len(self.history)
        bg = f"  · {histories} message(s) in context" if histories else ""
        print(
            f"\n[{GREEN}{self.model}{RESET}] [effort={GREEN}{self.effort}{RESET}]{bg} "
            f"{DIM}— /help for commands{RESET}"
        )

    # ------------------------------------------------------------------ loop
    def slash(self, line: str) -> bool:
        """Handle /command → False means exit."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in {"/exit", "/quit", "/q"}:
            return False
        if cmd == "/help":
            print(HELP_TEXT, end="")
        elif cmd == "/new":
            self.history.clear()
            print(f"{GREEN}✓ conversation cleared{RESET}")
        elif cmd == "/model":
            if arg:
                self.model = arg
                print(f"model → {GREEN}{arg}{RESET}")
            else:
                print(f"model: {GREEN}{self.model}{RESET}\navailable: {', '.join(self.list_models()) or '(none)'}")
        elif cmd == "/effort":
            if arg and arg.lower() in EFFORTS:
                self.effort = arg.lower()
                print(f"effort → {GREEN}{self.effort}{RESET}")
            else:
                print(f"effort: {GREEN}{self.effort}{RESET}  (none, low, medium, high)")
        elif cmd == "/status":
            self.status()
        elif cmd == "/reinit":
            self.reinit()
        else:
            print(f"{YELLOW}unknown command {cmd} — try /help{RESET}")
        return True

    def repl(self) -> None:
        while True:
            try:
                raw = input(f"\n{BOLD}you{RESET} > ").strip()
            except (KeyboardInterrupt, EOFError):  # Ctrl+C on empty prompt / Ctrl+D
                print()
                return
            if not raw:
                continue
            if raw.startswith("/"):
                if not self.slash(raw):
                    return
                self.banner()
                continue
            self.complete_stream(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini Web chatbot over the local proxy")
    ap.add_argument("--base-url", default=os.environ.get("GWOP_BASE_URL", "http://127.0.0.1:8011"))
    ap.add_argument("--key", default=os.environ.get("GWOP_API_KEY", ""),
                    help="bearer key if the proxy requires auth")
    ap.add_argument("--model", help="skip the interactive model picker")
    ap.add_argument("--effort", choices=list(EFFORTS), default=None,
                    help="skip the interactive effort picker")
    ap.add_argument("--system", help="system prompt applied to the conversation")
    ap.add_argument("--prompt", help="one-shot: send prompt, print answer, exit")
    ap.add_argument("--no-stream", action="store_true", help="print the whole answer instead of streaming")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    shell = Shell(base_url=args.base_url.rstrip("/"), headers=headers, model=args.model or "default",
                  effort=args.effort or "none", system=args.system, timeout=args.timeout)

    if args.prompt is None:
        if args.model is None:
            shell.choose_model_interactively()
        if args.effort is None:
            shell.choose_effort_interactively()
        shell.banner()
        try:
            shell.repl()
        except requests.ConnectionError:
            print(f"{RED}cannot reach {args.base_url} — is the proxy running?\n"
                  f"start it with: uv run python -m gemini_openai_proxy{RESET}")
            return 2
        return 0

    # one-shot: never print prompt scaffolding; answer only
    if (shell_model := shell.list_models()) and shell.model not in ("default", shell_model) \
            and shell.model not in shell_model:
        fallback = shell_model[0] if shell_model else "default"
        print(f"{YELLOW}model '{shell.model}' not advertised; using {fallback}{RESET}", file=sys.stderr)
        shell.model = fallback
    answer = shell.complete_stream(args.prompt) if not args.no_stream else shell.complete(args.prompt)
    if args.no_stream and answer:
        print(answer)
    if args.no_stream and not answer:
        return 1
    return 0 if answer else 1


if __name__ == "__main__":
    sys.exit(main())
