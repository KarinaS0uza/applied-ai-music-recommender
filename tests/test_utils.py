"""Tests for the helpers in src/utils.py."""

import pytest

from src.utils import clean_excerpt, clean_text, normalize, platform_search_links, safe_url


# --- normalize -----------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("  Águas de MARÇO!! ", "aguas de marco"),       # case, accents, punctuation, outer spaces
    ("AC/DC", "ac dc"),                              # symbols split words instead of joining them
    ("Rock & Roll?", "rock roll"),
    ("Don't Stop", "don t stop"),                    # an apostrophe is a word break
    ("Beyoncé—Halo", "beyonce halo"),
    ("Straße", "strasse"),                           # casefold, not just lower
    ("---", ""),
    ("", ""),
])
def test_normalize_folds_case_accents_and_punctuation_into_single_spaced_words(raw, expected):
    """Normalize folds case accents and punctuation into single spaced words."""
    assert normalize(raw) == expected


def test_normalize_is_idempotent_so_stored_and_incoming_text_compare_equal():
    """Normalize is idempotent so stored and incoming text compare equal."""
    once = normalize("Águas de Março — Elis Regina!")
    assert normalize(once) == once


# --- clean_excerpt -------------------------------------------------------------------------

def test_clean_excerpt_turns_search_markdown_into_plain_text():
    """Clean excerpt turns search markdown into plain text."""
    raw = ("## Best ![cover](http://x/y.png) songs: [Take Five](https://a.b/c) at https://example.org/x "
           "or www.foo.com **bold** `code` > quote | pipe")
    assert clean_excerpt(raw) == "Best songs: Take Five at or bold code quote pipe"


def test_clean_excerpt_collapses_all_whitespace():
    """Clean excerpt collapses all whitespace."""
    assert clean_excerpt("  one\n\n two \t three  ") == "one two three"


# --- safe_url ------------------------------------------------------------------------------
# javascript:, userinfo, 127.0.0.1, localhost and embedded spaces are pinned end to end by
# test_rag.py::test_unsafe_source_urls_are_rejected; the cases below are the remaining rules.

@pytest.mark.parametrize("url", [
    "https://example.org/a",
    "http://example.org",
    "https://sub.domain.example.co.uk/path?q=1#frag",
    "http://8.8.8.8/",                               # a public IP address is fine
])
def test_safe_url_accepts_public_http_and_https_addresses(url):
    """Safe url accepts public http and https addresses."""
    assert safe_url(url)


@pytest.mark.parametrize("url", [
    None,
    123,
    "ftp://example.org/file",
    "file:///etc/passwd",
    "https://intranet/page",                         # a bare host has no dot
    "https://printer.local/",
    "https://api.internal/x",
    "http://10.0.0.5/",                              # private ranges
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data",       # link-local cloud metadata address
    "http://[::1]/",
    "http://[invalid",                               # unparseable
    "https://example.org\t/x",                       # any whitespace, not only spaces
])
def test_safe_url_rejects_non_web_internal_or_malformed_addresses(url):
    """Safe url rejects non web internal or malformed addresses."""
    assert not safe_url(url)


def test_safe_url_length_limit_is_2048_characters():
    """Safe url length limit is 2048 characters."""
    prefix = "https://example.org/"
    assert safe_url(prefix + "a" * (2048 - len(prefix)))
    assert not safe_url(prefix + "a" * (2049 - len(prefix)))


# --- platform_search_links -----------------------------------------------------------------

def test_platform_search_links_encode_the_query():
    """Platform search links encode the query."""
    links = platform_search_links("Rock & Roll?", "AC/DC")
    assert links["YouTube search"].endswith("search_query=Rock%20%26%20Roll%3F%20AC%2FDC")
    assert links["Spotify search"].endswith("/search/Rock%20%26%20Roll%3F%20AC%2FDC")
    assert all(safe_url(url) for url in links.values())


@pytest.mark.parametrize("raw, expected", [
    ("\x10\ufffd\x10\ufffd\ufffd\ufffd\u00c1guas de Mar\u00e7o", "\u00c1guas de Mar\u00e7o"),
    ("So\x00 What\x1f", "So What"),
    ("Take\tFive\n", "Take Five"),
    ("  Blue   in  Green ", "Blue in Green"),
    ("Caf\u00e9 \u6771\u4eac", "Caf\u00e9 \u6771\u4eac"),
])
def test_clean_text_drops_control_and_replacement_characters_but_keeps_real_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent():
    once = clean_text("\x10\ufffdA\tB")
    assert clean_text(once) == once
