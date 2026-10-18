"""Resolve Gemini Web cookies from env vars / raw header / Netscape cookies.txt.

Priority: GWOP_COOKIE_FILE  >  GWOP_GEMINI_COOKIES_RAW  >  GWOP_GEMINI_1PSID(+TS)
Only the auth cookies matter; everything else in the file is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

AUTH_COOKIE_NAMES = ("__Secure-1PSID", "__Secure-1PSIDTS", "__Secure-1PSIDCC", "__Secure-1PSIDCCSA")


@dataclass(slots=True)
class GeminiCookies:
    secure_1psid: str = ""
    secure_1psidts: str = ""
    extra: dict[str, str] = field(default_factory=dict)  # 1PSIDCC etc.

    def __bool__(self) -> bool:
        return bool(self.secure_1psid)

    def as_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"__Secure-1PSID": self.secure_1psid}
        if self.secure_1psidts:
            out["__Secure-1PSIDTS"] = self.secure_1psidts
        out.update(self.extra)
        return out


def _clean_domain(domain: str) -> str:
    if domain.startswith("."):
        domain = domain[1:]
    return domain


def _prefer_google_com(rows: list[tuple[str, str, str]]) -> tuple[str, str]:
    """rows are (domain, name, value) for one cookie name; pick the best row."""
    for domain, _name, value in rows:
        if _clean_domain(domain) == "google.com":
            return value, "google.com"
    domain, _name, value = rows[-1]
    return value, domain


def parse_cookie_file(path: Path) -> GeminiCookies:
    collected: dict[str, list[tuple[str, str, str]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, _flag, _path, _secure, _expiry, name, value = parts[:7]
        if name.startswith("__Secure-1PSID"):
            collected.setdefault(name, []).append((domain, name, value))

    if "__Secure-1PSID" not in collected:
        return GeminiCookies()

    psid, _dom = _prefer_google_com(collected["__Secure-1PSID"])
    psidts = ""
    if "__Secure-1PSIDTS" in collected:
        psidts, _dom = _prefer_google_com(collected["__Secure-1PSIDTS"])

    extra: dict[str, str] = {}
    for name in AUTH_COOKIE_NAMES:
        if name in collected and name not in ("__Secure-1PSID", "__Secure-1PSIDTS"):
            value, _dom = _prefer_google_com(collected[name])
            extra[name] = value

    return GeminiCookies(secure_1psid=psid, secure_1psidts=psidts, extra=extra)


def _parse_raw(header: str) -> GeminiCookies:
    out = GeminiCookies()
    for chunk in header.split(";"):
        if "=" not in chunk:
            continue
        name, _, value = chunk.strip().partition("=")
        name, value = name.strip(), value.strip()
        if name == "__Secure-1PSID":
            out.secure_1psid = value
        elif name == "__Secure-1PSIDTS":
            out.secure_1psidts = value
        elif name.startswith("__Secure-1PSID"):
            out.extra[name] = value
    return out


def resolve_cookies(
    *, cookie_1psid: str, cookie_1psidts: str, cookie_raw: str, cookie_file: str
) -> GeminiCookies:
    """Returns an empty GeminiCookies when nothing usable is present; the caller
    decides how loudly to fail (never raise from a resolver)."""
    if cookie_file:
        return parse_cookie_file(Path(cookie_file).expanduser().resolve())
    if cookie_raw:
        return _parse_raw(cookie_raw)
    return GeminiCookies(secure_1psid=cookie_1psid.strip(), secure_1psidts=cookie_1psidts.strip())
