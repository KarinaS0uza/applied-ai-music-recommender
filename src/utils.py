"""Pure text, URL and link helpers shared across the discovery pipeline."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from urllib.parse import urlsplit, quote


def normalize(value: str) -> str:
    """Lowercase text, strip accents and punctuation, and collapse it to single-spaced words."""
    return " ".join(re.findall(r"\w+", "".join(
        c for c in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(c)
    )))


def clean_text(value: str) -> str:
    """Remove control and replacement characters so a title or artist displays as plain text.

    `normalize` ignores such characters when matching, so a garbled name can still match a
    clean candidate; this keeps the garbage out of what the listener sees. Tabs and newlines
    become spaces, other control characters and U+FFFD are dropped, and spaces are collapsed.
    """
    value = value.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return " ".join("".join(
        c for c in value if c != "�" and unicodedata.category(c) != "Cc"
    ).split())


def clean_excerpt(value: str) -> str:
    """Strip markdown and URLs from a search fragment so it reads as plain text."""
    value = re.sub(r"!\[[^]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"https?://\S+|www\.\S+", "", value)
    value = re.sub(r"[#*_`>|]+", " ", value)
    return " ".join(value.split())


def safe_url(url: str | None) -> bool:
    """Tell whether a URL is a public http(s) address that is safe to link to."""
    if not isinstance(url, str) or len(url) > 2048 or any(c.isspace() for c in url):
        return False
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.scheme not in {"https", "http"} or not host or parts.username or parts.password:
            return False
        if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    except ValueError:
        return False


def platform_search_links(title: str, artist: str) -> dict[str, str]:
    """Build YouTube and Spotify search links for a song title and artist."""
    query = quote(f"{title} {artist}", safe="")
    return {"YouTube search": f"https://www.youtube.com/results?search_query={query}",
            "Spotify search": f"https://open.spotify.com/search/{query}"}
