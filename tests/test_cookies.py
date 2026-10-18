"""Cookie resolution: env pair, raw header, Netscape cookies.txt."""

from pathlib import Path

from gemini_openai_proxy.cookies import GeminiCookies, _parse_raw, parse_cookie_file, resolve_cookies

NETSCAPE = "\n".join(
    [
        "# Netscape HTTP Cookie File",
        ".google.com\tTRUE\t/\tTRUE\t2147483647\t__Secure-1PSID\tgood-value",
        ".google.com\tTRUE\t/\tTRUE\t2147483647\t__Secure-1PSIDTS\tts-value",
        ".google.be\tTRUE\t/\tTRUE\t2147483647\t__Secure-1PSID\tstale-value",
        "NID.example.com\tTRUE\t/\tFALSE\t2147483647\tNID\tignored",
    ]
)


def test_parse_cookie_file_prefers_google_com(tmp_path: Path):
    f = tmp_path / "cookies.txt"
    f.write_text(NETSCAPE)
    cookies = parse_cookie_file(f)
    assert cookies.secure_1psid == "good-value"
    assert cookies.secure_1psidts == "ts-value"


def test_parse_cookie_file_empty(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("# only comments\n")
    assert parse_cookie_file(f) == GeminiCookies()


def test_raw_header_parsing():
    raw = '__Secure-1PSID=abc; __Secure-1PSIDTS=tsz; __Secure-1PSIDCC=cc; NID=junk'
    out = _parse_raw(raw)
    assert out.secure_1psid == "abc"
    assert out.secure_1psidts == "tsz"
    assert out.extra == {"__Secure-1PSIDCC": "cc"}


def test_resolve_priority_file_over_env(tmp_path: Path):
    f = tmp_path / "cookies.txt"
    f.write_text(NETSCAPE)
    got = resolve_cookies(
        cookie_1psid="env-value", cookie_1psidts="", cookie_raw="", cookie_file=str(f)
    )
    assert got.secure_1psid == "good-value"


def test_resolve_pair_ok():
    got = resolve_cookies(cookie_1psid="psid", cookie_1psidts="ts", cookie_raw="", cookie_file="")
    assert got.secure_1psid == "psid" and got.secure_1psidts == "ts"


def test_resolve_nothing_is_empty_object():
    got = resolve_cookies(cookie_1psid="", cookie_1psidts="", cookie_raw="", cookie_file="")
    assert not got
