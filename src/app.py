"""Display music discovery forms, sourced recommendations, and CSV rankings."""

# Behavior: runs as a Streamlit page with a web-discovery form and a CSV catalog ranking form; provider keys stay on the server.

import logging
import os
import re
import sys
from html import escape
from pathlib import Path
import time
from urllib.parse import urlsplit

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.providers import Settings
from src.composition import find_music
from src.discovery import INTENT_FALLBACK_NOTICE, UNVERIFIED_WARNING_PREFIX
from src.models import DiscoveryError, DiscoveryResult, MusicRequest, ParsedIntent, TasteCriterion
from src.utils import platform_search_links
from src.validation import validate_request
from src.recommender import (RANKING_STRATEGIES, load_songs, match_fraction,
                             recommend_songs, resolve_genres, suggest_genres,
                             trait_match_fractions)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

STYLE = """
<style>
:root {
    /* light-dark(<light>, <dark>) resolves against the color-scheme Streamlit sets on .stApp. */
    --music-ink: light-dark(#1D1D1F, #F7F5FF);
    --music-muted: light-dark(#5A5666, #B7B1C8);
    --music-wash: light-dark(#1D1D1F, #FFFFFF);
    --music-border: light-dark(rgb(29 29 31 / 14%), color-mix(in srgb, var(--music-wash) 11%, transparent));
    --music-pink: light-dark(#D81B60, #FF4D8D);
    --music-violet: light-dark(#6D3FE0, #8B5CF6);
    --music-cyan: light-dark(#007EA3, #39D9FF);
    --music-strong: light-dark(#1D1D1F, #FFFFFF);
    --music-soft: light-dark(#4A4658, #C9C3D7);
    --music-accent-ink: light-dark(#C2185B, #FF91B9);
    --music-violet-ink: light-dark(#5B3BD6, #DCD5FF);
    --music-cyan-ink: light-dark(#0B6E8A, #83E9FF);
    --music-field: light-dark(#FFFFFF, rgb(6 8 17 / 68%));
    --music-card-a: light-dark(#FFFFFF, #161522);
    --music-card-b: light-dark(#F7F5FC, #11111C);
    --music-solid: light-dark(#FFFFFF, #11121E);
    --music-shadow: light-dark(rgb(60 50 110 / 14%), rgb(0 0 0 / 24%));
    --music-shadow-hover: light-dark(rgb(60 50 110 / 22%), rgb(0 0 0 / 34%));
    --music-shadow-soft: light-dark(rgb(60 50 110 / 10%), rgb(0 0 0 / 18%));
}
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    -webkit-font-smoothing: antialiased;
    color: var(--music-ink);
    background:
        radial-gradient(circle at 10% 4%, light-dark(rgb(139 92 246 / 10%), rgb(139 92 246 / 18%)), transparent 28rem),
        radial-gradient(circle at 90% 24%, light-dark(rgb(0 126 163 / 7%), rgb(57 217 255 / 10%)), transparent 26rem),
        linear-gradient(145deg, light-dark(#F5F5F7, #080A12) 0%, light-dark(#EEEDF5, #10101C) 54%, light-dark(#F5F5F7, #090B14) 100%);
}
.stMainBlockContainer { max-width: 70rem; padding: 3.25rem 1.5rem 5rem; }
.stApp h1 {
    letter-spacing: -0.045em; padding-bottom: 0.35rem; font-size: clamp(2.5rem, 6vw, 4.5rem);
    background: linear-gradient(100deg, light-dark(#1D1D1F, #FFFFFF) 8%, light-dark(#5B3BD6, #DCD5FF) 48%, light-dark(#0B7FA6, #78E8FF) 92%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.stApp [data-testid="stText"] { white-space: pre-wrap; line-height: 1.65; overflow-wrap: anywhere; }
.stApp [data-testid="stCaptionContainer"] { color: var(--music-muted); opacity: 1; line-height: 1.55; }
.stApp [data-testid="stWidgetLabel"] p { font-weight: 500; }
.stApp input, .stApp textarea { font-family: inherit; line-height: 1.5; }
.stApp input { min-height: 2.75rem; }
.stApp textarea { min-height: 8rem; }
.stApp [data-testid="stForm"] {
    padding: 1.35rem; margin-bottom: 1.5rem; border: 1px solid var(--music-border);
    border-radius: 1.25rem; background: linear-gradient(145deg, color-mix(in srgb, var(--music-wash) 5%, transparent), color-mix(in srgb, var(--music-wash) 2%, transparent));
    box-shadow: 0 1.5rem 4rem var(--music-shadow-soft); backdrop-filter: blur(18px);
}
.stApp :is(input, textarea) { background: var(--music-field); border-color: color-mix(in srgb, var(--music-wash) 14%, transparent); }
.stApp :is(input, textarea):focus { border-color: var(--music-violet); box-shadow: 0 0 0 1px var(--music-violet); }
.stApp [data-testid="stRadio"] [role="radiogroup"] { gap: 0.5rem; }
.stApp [data-testid="stRadio"] [role="radiogroup"] label {
    min-height: 2.5rem; padding: 0.45rem 0.65rem; border: 1px solid var(--music-border);
    border-radius: 0.75rem; background: color-mix(in srgb, var(--music-wash) 4%, transparent); transition: 150ms ease;
}
.stApp [data-testid="stRadio"] [role="radiogroup"] label:hover { border-color: rgb(139 92 246 / 60%); background: rgb(139 92 246 / 10%); }
.stApp [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    border-color: var(--music-violet); background: rgb(139 92 246 / 18%);
}
.stApp [data-testid="stFormSubmitButton"] button {
    min-height: 3.2rem; border-radius: 0.8rem; font-weight: 750; letter-spacing: 0.01em;
}
.stApp [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(100deg, var(--music-pink), var(--music-violet));
    border: 0; color: #FFFFFF; box-shadow: 0 0.75rem 2rem rgb(139 92 246 / 24%);
}
.stApp [data-testid="stFormSubmitButton"] button:active { transform: translateY(1px); }
.stApp :is(button, a, input, textarea):focus-visible {
    outline: 2px solid var(--music-cyan); outline-offset: 3px;
}
.results-heading { margin: 2rem 0 1rem; }
.results-heading span, .mode-heading span {
    color: var(--music-accent-ink); font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;
}
.results-heading span { display: block; font-size: 0.72rem; margin-bottom: 0.25rem; }
.results-heading strong, .mode-heading strong { color: var(--music-strong); letter-spacing: -0.025em; }
.results-heading strong { font-size: 1.55rem; }
.empty-state {
    margin: 1.5rem 0 0.75rem; padding: 1.25rem; border: 1px dashed color-mix(in srgb, var(--music-wash) 16%, transparent);
    border-radius: 1rem; color: var(--music-muted); background: color-mix(in srgb, var(--music-wash) 2%, transparent);
}
/* Sits under the Discover music form, so it lines up with the form's inner text. */
.request-summary { margin: 0.35rem 0 0.9rem; padding-inline: 1.35rem; }
.request-summary span {
    display: block; color: var(--music-muted); font-size: 0.7rem; font-weight: 750;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.18rem;
}
.request-summary strong { color: var(--music-ink); font-size: 0.9rem; font-weight: 650; line-height: 1.5; }
.discovery-note {
    display: flex; align-items: flex-start; gap: 0.85rem; margin: 1rem 0 1.25rem;
    padding: 1rem 1.1rem; border: 1px solid rgb(139 92 246 / 25%);
    border-radius: 1rem; background: linear-gradient(135deg, rgb(139 92 246 / 12%), rgb(57 217 255 / 5%));
}
.discovery-note-icon {
    display: grid; place-items: center; flex: 0 0 auto; width: 2rem; height: 2rem;
    border-radius: 50%; color: var(--music-violet-ink); background: rgb(139 92 246 / 20%);
    font-size: 0.95rem; font-weight: 800;
}
.discovery-note strong { display: block; color: var(--music-strong); font-size: 0.95rem; line-height: 1.4; }
.discovery-note p { color: var(--music-soft); font-size: 0.86rem; line-height: 1.55; margin: 0.22rem 0 0; }
.mode-heading { margin-bottom: 1rem; }
.mode-heading span { font-size: 0.7rem; }
.mode-heading strong {
    display: block; font-size: 1.45rem; margin-top: 0.2rem;
}
/* Starts level with the Discover music card (below its heading) and matches its height. */
.choice-divider {
    min-height: 22.1rem; margin-top: 5.1rem; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 0.8rem; color: var(--music-muted); font-size: 0.7rem;
    font-weight: 800; letter-spacing: 0.12em;
}
.choice-divider::before, .choice-divider::after {
    content: ""; width: 1px; flex: 1 1 0;
    background: linear-gradient(transparent, color-mix(in srgb, var(--music-wash) 20%, transparent), transparent);
}
.choice-divider span {
    display: grid; place-items: center; width: 2.25rem; height: 2.25rem;
    border: 1px solid var(--music-border); border-radius: 50%; background: var(--music-solid);
}
.stApp [class*="st-key-song_card_"] {
    --card-accent: var(--music-pink); position: relative; overflow: hidden;
    min-height: 19.9rem; /* short cards keep a steady height; long text grows the card */
    border-radius: 1.25rem; padding: 1.15rem; margin-block: 0.35rem;
    border-color: color-mix(in srgb, var(--music-wash) 11%, transparent);
    background: linear-gradient(155deg, color-mix(in srgb, var(--card-accent) 15%, var(--music-card-a)), var(--music-card-b) 54%);
    box-shadow: 0 1.2rem 2.5rem var(--music-shadow);
    transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
}
.stApp [class*="st-key-song_card_"]::before {
    content: ""; position: absolute; inset: 0 0 auto; height: 3px;
    background: linear-gradient(90deg, var(--card-accent), transparent 75%);
}
.stApp [class*="st-key-song_card_"]:hover {
    transform: translateY(-5px); border-color: color-mix(in srgb, var(--card-accent) 55%, transparent);
    box-shadow: 0 1.6rem 3.2rem var(--music-shadow-hover);
}
.stApp [class*="st-key-song_card_2"], .stApp [class*="st-key-song_card_5"] { --card-accent: var(--music-violet); }
.stApp [class*="st-key-song_card_3"] { --card-accent: var(--music-cyan); }
.stApp [class*="st-key-catalog_song_card_"] {
    --card-accent: var(--music-pink); position: relative; overflow: hidden;
    min-height: 19rem;
    border-radius: 1.25rem; padding: 1.15rem; margin-block: 0.35rem;
    border-color: color-mix(in srgb, var(--music-wash) 11%, transparent);
    background: linear-gradient(155deg, color-mix(in srgb, var(--card-accent) 15%, var(--music-card-a)), var(--music-card-b) 54%);
    box-shadow: 0 1.2rem 2.5rem var(--music-shadow);
    transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
}
.stApp [class*="st-key-catalog_song_card_"]::before {
    content: ""; position: absolute; inset: 0 0 auto; height: 3px;
    background: linear-gradient(90deg, var(--card-accent), transparent 75%);
}
.stApp [class*="st-key-catalog_song_card_"]:hover {
    transform: translateY(-5px); border-color: color-mix(in srgb, var(--card-accent) 55%, transparent);
    box-shadow: 0 1.6rem 3.2rem var(--music-shadow-hover);
}
.stApp [class*="st-key-catalog_song_card_2"],
.stApp [class*="st-key-catalog_song_card_5"] { --card-accent: var(--music-violet); }
.stApp [class*="st-key-catalog_song_card_3"] { --card-accent: var(--music-cyan); }
.stApp [class*="st-key-catalog_song_card_"] [data-testid="stCaptionContainer"]:first-of-type p {
    color: var(--card-accent); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em;
}
.stApp [class*="st-key-song_card_"] [data-testid="stCaptionContainer"]:first-of-type p {
    color: var(--card-accent); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.12em;
}
.song-identity { margin: 0.2rem 0 0.8rem; }
.song-title { color: var(--music-strong); font-size: 1.35rem; font-weight: 740; letter-spacing: -0.025em; line-height: 1.15; }
.song-artist { color: var(--music-muted); font-size: 0.95rem; font-weight: 600; line-height: 1.35; margin-top: 0.35rem; }
.catalog-meta { display: flex; flex-wrap: wrap; gap: 0.38rem; margin: 0 0 0.8rem; }
.catalog-meta span {
    display: inline-flex; align-items: center; min-height: 1.55rem; padding: 0.16rem 0.5rem;
    border: 1px solid color-mix(in srgb, var(--music-wash) 9%, transparent); border-radius: 999px;
    background: color-mix(in srgb, var(--music-wash) 6%, transparent); color: var(--music-soft); font-size: 0.68rem;
    font-weight: 650; line-height: 1.2; text-transform: capitalize;
}
.catalog-reason-label {
    color: var(--music-muted); font-size: 0.66rem; font-weight: 750;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.25rem;
}
.catalog-reason { color: var(--music-ink); font-size: 0.86rem; line-height: 1.45; }
/* The links flow after the text instead of being pinned over it: margin-top: auto puts them at
   the bottom of a short card and directly below the text of a long one, however tall they are. */
.stApp :is([class*="st-key-song_card_"], [class*="st-key-catalog_song_card_"]) > [data-testid="stLayoutWrapper"] {
    margin-top: auto;
}
.stApp :is([class*="st-key-song_links_"], [class*="st-key-catalog_song_links_"]) { padding-top: 0.5rem; }
/* Streamlit gives every markdown block margin-bottom: -1rem to cancel a paragraph's margin.
   The logo row is not a paragraph, so that pulled it 1rem past the card's bottom padding. */
.stApp [data-testid="stMarkdownContainer"]:has(> .platform-logos) { margin-bottom: 0; }
.platform-logos { display: flex; align-items: center; gap: 0.5rem; }
.stApp .platform-logos a.platform-logo-link {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2.4rem; height: 2.4rem; border-radius: 999px; text-decoration: none;
    background: color-mix(in srgb, var(--music-wash) 7%, transparent);
    transition: background 0.15s ease, transform 0.15s ease;
}
.stApp .platform-logos a.platform-logo-link:hover {
    background: color-mix(in srgb, var(--music-wash) 13%, transparent); transform: translateY(-1px);
}
.stApp .platform-logos a.platform-logo-link:focus-visible { outline: 2px solid var(--music-violet); outline-offset: 2px; }
.platform-logo-link svg { width: 1.35rem; height: 1.35rem; display: block; }
.catalog-summary {
    margin: 2.2rem 0 1rem; padding: 1rem 1.1rem; border: 1px solid var(--music-border);
    border-radius: 1rem; background: color-mix(in srgb, var(--music-wash) 3%, transparent);
}
.catalog-summary-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.catalog-summary-head strong { color: var(--music-strong); font-size: 1.15rem; letter-spacing: -0.02em; }
.catalog-tags { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.75rem; }
.catalog-tags span {
    display: inline-flex; align-items: center; min-height: 1.8rem; padding: 0.2rem 0.7rem;
    border: 1px solid color-mix(in srgb, var(--music-wash) 12%, transparent); border-radius: 999px;
    background: color-mix(in srgb, var(--music-wash) 6%, transparent); color: var(--music-soft);
    font-size: 0.8rem; font-weight: 650; line-height: 1.2;
}
.catalog-tags span.is-any { border-style: dashed; background: transparent; color: var(--music-muted); font-weight: 550; }
/* An ignored tag is the one warm cue: nothing in the collection matched it. */
.catalog-tags span.is-ignored {
    border-color: light-dark(rgb(166 106 0 / 40%), rgb(245 196 81 / 38%));
    background: light-dark(rgb(166 106 0 / 9%), rgb(245 196 81 / 10%));
    color: light-dark(#7A4B00, #F5C451);
}
.catalog-summary-note { margin-top: 0.7rem; color: var(--music-muted); font-size: 0.84rem; line-height: 1.5; }
.catalog-tags + .catalog-summary-note { margin-top: 0.85rem; }
.catalog-reason-label + .catalog-tags { margin-top: 0; }
.catalog-summary-badge {
    flex: 0 0 auto; padding: 0.4rem 0.65rem; border: 1px solid rgb(57 217 255 / 24%);
    border-radius: 999px; color: var(--music-cyan-ink); background: rgb(57 217 255 / 8%);
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
}
.stApp :is([class*="st-key-song_details_"], [class*="st-key-source_entry_"]) {
    padding: 0.25rem 0 0.85rem; margin-bottom: 0.85rem;
    border-bottom: 1px solid var(--music-border);
}
.stApp :is([class*="st-key-song_details_"], [class*="st-key-source_entry_"]):last-child { border-bottom: 0; margin-bottom: 0; }
.source-note { margin-top: 0.7rem; }
.source-kind {
    color: var(--music-muted); font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.source-title { color: var(--music-ink); font-size: 0.96rem; font-weight: 650; line-height: 1.4; margin-top: 0.15rem; }
.stApp [data-testid="stExpander"] { border-radius: 0.9rem; border-color: var(--music-border); background: color-mix(in srgb, var(--music-wash) 2%, transparent); }
.stApp [data-testid="stAlert"] { border-radius: 0.75rem; }
.stApp [data-testid="stDataFrame"] { border-radius: 0.75rem; overflow: hidden; }
@media (max-width: 640px) {
    .stMainBlockContainer { padding: 2rem 1rem 3rem; }
    .stApp h1 { font-size: 2.7rem; }
    .stApp [data-testid="stForm"] { padding: 1rem; }
    .request-summary { padding-inline: 1rem; }
    .stApp :is([class*="st-key-song_card_"], [class*="st-key-catalog_song_card_"]) { padding: 1rem; }
    .catalog-summary-head { align-items: flex-start; flex-direction: column; }
    .choice-divider { min-height: auto; flex-direction: row; margin: 0.5rem 0 1.5rem; }
    .choice-divider::before, .choice-divider::after { width: 5.5rem; height: 1px; flex: none; }
}
@media (prefers-contrast: more) {
    .stApp :is([class*="st-key-song_card_"], [class*="st-key-catalog_song_card_"]) { border-color: currentColor; box-shadow: none; }
}
@media (prefers-reduced-motion: reduce) {
    .stApp :is([class*="st-key-song_card_"], [class*="st-key-catalog_song_card_"]) { transition: none; }
    .stApp :is([class*="st-key-song_card_"], [class*="st-key-catalog_song_card_"]):hover { transform: none; }
}
</style>
"""


def server_settings():
    """Merge the process environment with Streamlit secrets so provider keys can come from either."""
    values = dict(os.environ)
    try:
        for name in ("TAVILY_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL",
                     "GEMINI_EMBEDDING_MODEL", "SEMANTIC_RETRIEVAL_ENABLED",
                     "MUSIC_SOURCE_FOCUS_ENABLED",
                     "MUSICBRAINZ_CONTACT", "YOUTUBE_API_KEY",
                     "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
            if name in st.secrets:
                values[name] = st.secrets[name]
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return Settings.from_mapping(values)


def compact_text(value: str, limit: int = 145) -> str:
    """Collapse whitespace and cut the text at a word boundary once it passes the limit."""
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit].rsplit(" ", 1)[0] + "…"


def literal_caption(value: str) -> str:
    """Escape markdown so model- and web-sourced text shows exactly as written in a caption."""
    return re.sub(r"([\\`*_\[\]()$<>#~|!])", r"\\\1", value)


def readable_source_excerpt(value: str, limit: int = 180) -> str:
    """Turn a search-result fragment into a quiet, single-line preview."""
    value = re.sub(r"!\[[^]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"#{1,6}\s*", "", value)
    value = re.sub(r"\[\.\.\.\]", " ", value)
    return compact_text(value, limit)


def source_kind_label(kind: str) -> str:
    """Map an evidence kind to the listener-friendly label shown on source links."""
    return {
        "web": "Website",
        "song_card": "Curated song note",
        "guide": "Listening guide",
        "metadata": "Recording details",
    }.get(kind, "Source")


def catalog_preferences(intent) -> dict:
    """Translate discovery intent into the CSV recommender's preference shape."""
    criteria = getattr(intent, "criteria", []) if intent else []
    positive = [criterion for criterion in criteria if criterion.kind != "must_avoid"]

    def values(facet):
        """Return the cleaned, lowercased values of the positive criteria for one facet."""
        return [criterion.value.strip().lower() for criterion in positive
                if criterion.facet == facet and criterion.value.strip()]

    genres = values("genre")
    moods = values("mood")
    energy_words = values("energy")
    energy_text = " ".join(energy_words + moods)
    energy = 0.5
    if any(word in energy_text for word in ("intense", "high", "energetic", "fast", "upbeat")):
        energy = 0.85
    elif any(word in energy_text for word in ("quiet", "low", "calm", "peaceful", "relaxed", "slow")):
        energy = 0.3
    elif any(word in energy_text for word in ("medium", "moderate", "balanced")):
        energy = 0.55

    languages = values("language")
    instrumental_requested = any(
        criterion.facet == "vocals"
        and (criterion.kind == "must_avoid"
             or "instrumental" in criterion.value.lower()
             or "no vocal" in criterion.value.lower())
        for criterion in criteria
    )
    if instrumental_requested:
        languages.append("instrumental")

    return {
        "genre": genres[0] if genres else "",
        "mood": moods[0] if moods else "",
        "energy": energy,
        "mood_tags": moods,
        "instrumentation_tags": values("instrument"),
        "language": languages,
    }


def catalog_song_summary(song: dict) -> str:
    """Describe a catalog song in one sentence using its mood, genre and energy level."""
    energy = song.get("energy")
    energy_label = "balanced"
    if energy is not None:
        energy_label = "gentle" if energy < 0.4 else "high" if energy > 0.75 else "moderate"
    mood = str(song.get("mood", "")).strip()
    genre = str(song.get("genre", "")).strip()
    return f"A {mood} {genre} track with {energy_label} energy.".replace("  ", " ")


CATALOG_FALLBACK_SIZE = 5


def catalog_fallback_picks(intent) -> list:
    """Rank the local CSV catalog for the request: up to five (song, match, traits, reasons) rows.

    Returns [] when the catalog cannot be read or has nothing to offer, so the caller can fall
    through to the next fallback.
    """
    try:
        songs = load_songs(str(ROOT / "data/songs.csv"))
        strategy = RANKING_STRATEGIES["genre-first"]
        preferences = catalog_preferences(intent)
        preferences["genre"] = resolve_genres(preferences["genre"], songs) or preferences["genre"]
        rows = recommend_songs(preferences, songs, k=CATALOG_FALLBACK_SIZE, strategy=strategy)
        return [(song, match_fraction(preferences, song, strategy),
                 trait_match_fractions(preferences, song), reasons)
                for song, _, reasons in rows]
    except (OSError, ValueError, KeyError):
        logger.warning("catalog_fallback_failed", exc_info=True)
        return []


def show_catalog_fallback(picks) -> None:
    """Show the catalog songs found for a request discovery could not verify on the web."""
    st.caption("We could not verify a web recommendation, so here are songs from our local collection.")
    section_heading("results-heading", "Backup picks", "From our collection")
    for start in range(0, len(picks), 3):
        columns = st.columns(3, gap="medium")
        for offset, (song, match, fractions, reasons) in enumerate(picks[start:start + 3]):
            with columns[offset]:
                show_catalog_fallback_card(song, match, fractions, reasons, start + offset + 1)


def show_catalog_fallback_card(song, match, fractions, reasons, rank) -> None:
    """Render one backup catalog song with its match score, reason and platform links."""
    matched_traits = [
        label for marker, label in (
            ("genre match", "genre"),
            ("mood match", "mood"),
            ("energy similarity", "energy"),
            ("mood tag match", "mood"),
            ("instrumentation match", "instrumentation"),
            ("language match", "language"),
        ) if marker in reasons
    ]
    matched_traits = list(dict.fromkeys(matched_traits))
    with st.container(border=True, key=f"song_card_catalog_fallback_{rank}"):
        st.caption(f"COLLECTION FALLBACK {rank:02}  ·  {match:.0%} MATCH SCORE")
        st.markdown(
            f'<div class="song-identity"><div class="song-title">{escape(song["title"])}</div>'
            f'<div class="song-artist">{escape(song["artist"])}</div></div>',
            unsafe_allow_html=True,
        )
        st.text(catalog_song_summary(song))
        st.markdown(
            '<div class="catalog-reason-label">Why it fits</div>'
            f'<div class="catalog-reason">{escape(catalog_reason_text(reasons, fractions))}</div>',
            unsafe_allow_html=True,
        )
        if matched_traits:
            st.caption("Matched on " + ", ".join(matched_traits) + ".")
        show_platform_links(song["title"], song["artist"], f"song_links_catalog_fallback_{rank}")


def intent_from_text(text: str) -> ParsedIntent:
    """Guess a rough intent from the raw request, for when discovery failed before parsing it.

    Catalog genres and moods named in the text become preferences; everything else is left out,
    so an unclear request ranks on energy alone.
    """
    words = set(normalize_words(text))
    criteria = []
    try:
        songs = load_songs(str(ROOT / "data/songs.csv"))
    except (OSError, ValueError, KeyError):
        songs = []
    for facet in ("genre", "mood"):
        names = sorted({str(song.get(facet, "")).strip().lower() for song in songs} - {"", "unknown"})
        criteria += [TasteCriterion(facet, "preference", name, 2, name)
                     for name in names if set(name.split()) <= words]
    criteria.append(TasteCriterion("energy", "preference", " ".join(sorted(words)), 1, text))
    return ParsedIntent(criteria, [text])


def normalize_words(text: str) -> list[str]:
    """Split text into lowercase words."""
    return re.findall(r"[a-z0-9']+", text.lower())


# Brand marks (24x24 viewBox) keyed by the labels platform_search_links() returns.
PLATFORM_LOGOS = {
    "YouTube search": (
        "#FF0000",
        "M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505"
        "A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136"
        "c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12"
        "s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z",
    ),
    "Spotify search": (
        "#1DB954",
        "M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24"
        "-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6"
        " 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38"
        "-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2z"
        "m.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26"
        " 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z",
    ),
}


def platform_links_html(title: str, artist: str, video=None, track=None) -> str:
    """Logo-only links to YouTube and Spotify; each opens the exact video or track when one was matched."""
    links = []
    for label, url in platform_search_links(title, artist).items():
        color, path = PLATFORM_LOGOS[label]
        name = label.removesuffix(" search")
        if video is not None and name == "YouTube":
            url, hint = video.url, "Watch on YouTube"
            aria = f"Watch {video.title} by {video.channel} on YouTube"
        elif track is not None and name == "Spotify":
            url, hint = track.url, "Listen on Spotify"
            aria = f"Listen to {track.title} by {track.artist} on Spotify"
        else:
            hint, aria = f"Search {name}", f"Search {name} for {title} by {artist}"
        links.append(
            f'<a class="platform-logo-link" href="{escape(url)}" target="_blank" '
            f'rel="noopener noreferrer" title="{escape(hint)}" aria-label="{escape(aria)}">'
            f'<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
            f'<path fill="{color}" d="{path}"/></svg></a>'
        )
    return f'<div class="platform-logos">{"".join(links)}</div>'


def show_platform_links(title: str, artist: str, key: str, video=None, track=None) -> None:
    """Render the platform search links, plus the matched video caption when there is one."""
    with st.container(key=key):
        if video is not None:
            st.caption(literal_caption(f"▶ {compact_text(video.title, 60)} · {compact_text(video.channel, 30)}"))
        st.markdown(platform_links_html(title, artist, video, track), unsafe_allow_html=True)


def section_heading(style: str, label: str, title: str) -> None:
    """Render a styled section heading with a small label above its title."""
    st.markdown(
        f'<div class="{style}"><span>{escape(label)}</span>'
        f'<strong>{escape(title)}</strong></div>', unsafe_allow_html=True)


def request_summary(label: str, values: list[str]) -> None:
    """Show interpreted preferences without system-like terminology."""
    st.markdown(
        f'<div class="request-summary"><span>{escape(label)}</span>'
        f'<strong>{escape(" · ".join(values))}</strong></div>',
        unsafe_allow_html=True,
    )


def discovery_note(title: str, message: str) -> None:
    """Render a calm, readable explanation for an imperfect result."""
    st.markdown(
        '<div class="discovery-note" role="status">'
        '<div class="discovery-note-icon" aria-hidden="true">♪</div>'
        f'<div><strong>{escape(title)}</strong><p>{escape(message)}</p></div></div>',
        unsafe_allow_html=True,
    )


def show_song_card(rec, rank, unverified=()):
    """Render one recommendation as a numbered card with its score, explanation and links.

    ``unverified`` lists the requirements no song could be confirmed to meet; the card then says
    it is not verified and names them, instead of looking like a match.
    """
    score = getattr(rec, "fit_score", None)
    scored = bool(score)  # no real score is 0; None (or a stale 0.0) means never validated
    # Unscored cards quote the source, so they grow taller than a one-line explanation needs.
    with st.container(border=True, key=f"song_card_{rank}"):
        st.caption(f"DISCOVERY {rank:02}  ·  "
                   + (f"{score:.0%} MATCH SCORE" if scored else "NOT VERIFIED" if unverified else "NOT SCORED"))
        st.markdown(
            f'<div class="song-identity"><div class="song-title">{escape(rec.title)}</div>'
            f'<div class="song-artist">{escape(rec.artist)}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="catalog-reason-label">{"Why it fits" if scored else "What the source says"}</div>'
            f'<div class="catalog-reason">{escape(compact_text(rec.explanation, 145 if scored else 300))}</div>',
            unsafe_allow_html=True,
        )
        if unverified:
            tags = "".join(f'<span class="is-ignored">{escape(label)}</span>' for label in unverified)
            st.markdown('<div class="catalog-reason-label">Couldn’t confirm</div>'
                        f'<div class="catalog-tags">{tags}</div>', unsafe_allow_html=True)
        elif rec.uncertainties:
            count = len(rec.uncertainties)
            st.caption(f"{count} detail{'s' if count != 1 else ''} couldn’t be confirmed")
        show_platform_links(rec.title, rec.artist, f"song_links_{rank}", getattr(rec, "video", None),
                             getattr(rec, "track", None))


def video_fallback_source(evidence):
    """Return the first relevant YouTube source, if one is available."""
    for source in evidence:
        if source.kind != "web" or not source.url:
            continue
        host = (urlsplit(source.url).hostname or "").casefold()
        if host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com"):
            return source
    return None


def show_video_fallback(evidence, metadata_unverified: bool = False) -> bool:
    """Offer a relevant YouTube source when no individual song was verified."""
    video = video_fallback_source(evidence)
    if video is None:
        return False

    detail = " We also couldn’t confirm every recording detail." if metadata_unverified else ""
    discovery_note(
        "Here’s the closest listen we found",
        "We couldn’t confirm a song that checks every box, but this related video may still "
        f"be worth a listen.{detail}",
    )
    section_heading("results-heading", "Related listening", "A video to try")
    with st.container(border=True, key="video_fallback"):
        st.caption("RELATED VIDEO")
        st.markdown(f'<div class="song-title">{escape(video.title)}</div>', unsafe_allow_html=True)
        excerpt = readable_source_excerpt(video.text)
        if excerpt:
            st.caption(excerpt)
        st.link_button("Watch on YouTube", video.url, type="primary")
    return True


def catalog_reason_text(reasons: str, fractions: dict | None = None) -> str:
    """Turn scoring internals into short, listener-friendly card copy.

    ``fractions`` maps a reason label ("genre match") to how well the song matches it, from
    0 to 1; a reason found there is followed by that percentage instead of its raw points.
    """
    labels = []
    for reason in reasons.split(";"):
        reason = reason.strip()
        if not reason or "penalty" in reason:
            continue
        label = re.sub(r"\s*\([^)]*\)\s*$", "", reason).strip()
        text = label.capitalize()
        if fractions and label in fractions:
            text += f" {fractions[label]:.0%}"
        labels.append(text)
    return " · ".join(labels) or "Closest available catalog match"


def catalog_query_summary(genre, mood, energy, matched, rows, songs, strategy) -> dict:
    """Describe how the typed genre, mood and energy line up with the collection.

    Returns ``{"tags": [(text, kind)], "lines": [str]}``. Each tag is one part of the request,
    and its kind is "ok", "any" (left blank) or "ignored" (nothing in the collection matched,
    so it did not affect the ranking). ``lines`` holds short explanations, usually none.
    """
    tags, lines = [], []
    if not genre:
        tags.append(("any genre", "any"))
    elif matched:
        if [genre.lower()] == matched:
            tags.append((genre, "ok"))
        else:
            more = f" +{len(matched) - 3} more" if len(matched) > 3 else ""
            tags.append((f"{genre} → {', '.join(matched[:3])}{more}", "ok"))
    else:
        tags.append((f"{genre} · no match", "ignored"))
        related = suggest_genres(genre, songs)
        hint = f" Related genres: {', '.join(related)}." if related else ""
        lines.append(f"No genre in our collection matches “{genre}”, so genre was ignored.{hint}")

    moods = {str(song.get("mood", "")).strip().lower() for song in songs}
    if not mood:
        tags.append(("any mood", "any"))
    elif mood.lower() in moods:
        tags.append((mood, "ok"))
    else:
        tags.append((f"{mood} · no match", "ignored"))
        lines.append(f"No song in our collection is labeled “{mood}”, so mood was ignored.")

    tags.append((f"{energy:.0%} target energy", "ok"))

    if matched:
        in_genre = sum(1 for song, _, _ in rows if song["genre"] in matched)
        if in_genre < len(rows) and strategy.key != "genre-first":
            where = ", ".join(matched) if len(matched) <= 3 else "the matched genres"
            lead = "None of these songs are" if not in_genre else f"Only {in_genre} of {len(rows)} songs are"
            ranking = strategy.label.removesuffix(" (Default)")
            lines.append(f"{lead} in {where}. {ranking} ranking weighs mood and energy alongside "
                         "genre; try Genre-First to put genre first.")
    return {"tags": tags, "lines": lines}


def show_catalog_card(song, match, reasons, rank, fractions=None):
    """Present a scored CSV result with the same hierarchy as discovery cards."""
    with st.container(border=True, key=f"catalog_song_card_{rank}"):
        st.caption(f"CATALOG {rank:02}  ·  {match:.0%} MATCH SCORE")
        st.markdown(
            f'<div class="song-identity"><div class="song-title">{escape(song["title"])}</div>'
            f'<div class="song-artist">{escape(song["artist"])}</div></div>',
            unsafe_allow_html=True,
        )
        energy = song.get("energy")
        energy_label = f"{energy:.0%} energy" if energy is not None else "Energy unknown"
        metadata = [song.get("genre", ""), song.get("mood", ""), energy_label]
        pills = "".join(
            f"<span>{escape(str(value))}</span>" for value in metadata if value
        )
        st.markdown(f'<div class="catalog-meta">{pills}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="catalog-reason-label">Why it fits</div>'
            f'<div class="catalog-reason">{escape(catalog_reason_text(reasons, fractions))}</div>',
            unsafe_allow_html=True,
        )
        show_platform_links(song["title"], song["artist"], f"catalog_song_links_{rank}")


def show_catalog_results(rows, matches, summary, traits=None) -> None:
    """Render the request summary and the ranked catalog cards.

    ``matches`` holds each row's match fraction (0 to 1), in the same order as ``rows``.
    ``summary`` is what ``catalog_query_summary`` returned for the request.
    ``traits`` holds each row's per-trait match fractions (genre, mood, energy), in the same order.
    """
    tags = "".join(f'<span class="is-{kind}">{escape(text)}</span>' for text, kind in summary["tags"])
    lines = "".join(f'<div class="catalog-summary-note">{escape(line)}</div>' for line in summary["lines"])
    st.markdown(
        '<div class="catalog-summary">'
        '<div class="catalog-summary-head"><strong>Your catalog matches</strong>'
        f'<div class="catalog-summary-badge">{len(rows)} songs found</div></div>'
        f'<div class="catalog-tags">{tags}</div>{lines}</div>',
        unsafe_allow_html=True,
    )
    for start in range(0, len(rows), 3):
        columns = st.columns(3, gap="medium")
        for offset, (song, _, reasons) in enumerate(rows[start:start + 3]):
            with columns[offset]:
                index = start + offset
                show_catalog_card(song, matches[index], reasons, index + 1,
                                  traits[index] if traits else None)


def show_request_summary(result) -> None:
    """Show how the request was understood: the preferences and the must-haves."""
    intent = getattr(result, "intent", None)
    if not intent:
        return
    for label, criteria in (("Your vibe", intent.preferences),
                            ("Your must-haves", intent.hard_criteria)):
        values = [criterion.value.strip().capitalize() for criterion in criteria]
        if values:
            request_summary(label, values)


def show_result(result):
    """Render a discovery result: the song cards and their sources."""
    sources = {e.id: e for e in result.evidence}
    recommendations = list(result.recommendations)
    intent = getattr(result, "intent", None)
    metadata_messages = [warning for warning in result.warnings
                         if warning.startswith("MusicBrainz ")]
    unverified = getattr(result, "unverified_requirements", [])
    # The catalog comes first: songs no check could verify (or none at all) are replaced by ranked
    # catalog songs. Unverified songs, a related video and the plain message only follow if the
    # catalog gives nothing.
    catalog_picks = catalog_fallback_picks(intent) if (not recommendations or unverified) else []
    if catalog_picks:
        recommendations = []
        unverified = []
    video_available = (video_fallback_source(result.evidence) is not None
                       and not recommendations and not catalog_picks)
    for warning in result.warnings:
        explains_missing_match = (
            warning.startswith("No candidate had evidence")
            or warning == "The retrieved evidence did not establish a supported recommendation."
            or warning.startswith(UNVERIFIED_WARNING_PREFIX)
        )
        if warning in metadata_messages or ((video_available or catalog_picks) and explains_missing_match):
            continue
        if warning == INTENT_FALLBACK_NOTICE:
            st.info(warning)
        else:
            st.warning(warning)
    metadata_messages.extend(getattr(result, "notices", []))
    if metadata_messages and not recommendations and not video_available and not catalog_picks:
        st.caption("Metadata note: Some recording details could not be verified.")
    if recommendations:
        if unverified:
            caption = None  # the warning above already says these songs are not verified
        elif not any(getattr(rec, "fit_score", None) for rec in recommendations):
            caption = "These songs came up in your search, but we couldn’t confirm they match your request."
        elif getattr(result, "used_partial_matches", False):
            caption = "Here are the closest matches we found. Some details may vary by recording."
        else:
            caption = "Here are a few matches to explore."
        if caption and not unverified and any(not getattr(rec, "fit_score", None) for rec in recommendations) \
                and any(getattr(rec, "fit_score", None) for rec in recommendations):
            # Checked songs come first; the rest were only named by their sources.
            caption += " Songs marked NOT SCORED were named by their sources; we couldn’t check their fit."
        if caption:
            st.caption(caption)
        section_heading("results-heading", *(("Not verified", "Songs to check") if unverified
                                             else ("Made for this moment", "Fresh finds")))
    elif catalog_picks:
        show_catalog_fallback(catalog_picks)
    elif not show_video_fallback(result.evidence, bool(metadata_messages)):
        st.caption("No evidence-backed matches are available for this request yet.")
    for start in range(0, len(recommendations), 3):
        columns = st.columns(3, gap="medium")
        for offset, rec in enumerate(recommendations[start:start + 3]):
            with columns[offset]:
                show_song_card(rec, start + offset + 1, unverified)
    if recommendations:
        with st.expander("Why these songs?"):
            st.caption(
                "Why each song was picked, what its source says, and anything we couldn’t confirm."
                + (" Some recording details could not be verified; this does not mean the songs are unavailable."
                   if metadata_messages else "")
            )
            for rank, rec in enumerate(recommendations, 1):
                with st.container(key=f"song_details_{rank}"):
                    st.markdown(
                        f'<div class="source-kind">Song {rank}</div>'
                        f'<div class="source-title">{escape(rec.title)} — {escape(rec.artist)}</div>',
                        unsafe_allow_html=True,
                    )
                    label = "Why it may fit" if getattr(rec, "fit_score", None) else "What the source says"
                    st.markdown(f'<div class="source-kind source-note">{label}</div>', unsafe_allow_html=True)
                    st.caption(literal_caption(rec.explanation))
                    if rec.uncertainties:
                        st.markdown('<div class="source-kind source-note">What we couldn’t confirm</div>',
                                    unsafe_allow_html=True)
                        for uncertainty in rec.uncertainties:
                            st.caption(f"• {literal_caption(uncertainty)}")
                    if rec.claims:
                        st.markdown('<div class="source-kind source-note">From the source</div>',
                                    unsafe_allow_html=True)
                        for claim in rec.claims:
                            st.caption(f"“{literal_caption(compact_text(claim.quote, 260))}”")
                    for id in rec.source_ids:
                        source = sources.get(id)
                        if source and source.url:
                            st.link_button(f"View source: {source.title}", source.url, type="tertiary")
    if result.evidence:
        count = len(result.evidence)
        with st.expander(f"Sources checked ({count})"):
            st.caption(
                "These pages helped verify song names and listening details. "
                "You do not need to read them unless you want to check the recommendations."
            )
            for index, e in enumerate(result.evidence, 1):
                with st.container(key=f"source_entry_{index}"):
                    st.markdown(
                        f'<div class="source-kind">{escape(source_kind_label(e.kind))}</div>'
                        f'<div class="source-title">{escape(e.title)}</div>',
                        unsafe_allow_html=True,
                    )
                    excerpt = readable_source_excerpt(e.text)
                    if excerpt:
                        st.caption(excerpt)
                    if e.url:
                        st.link_button("Read source", e.url, type="tertiary")


def clear_results() -> None:
    """Drop results from either mode so a new search never shows stale cards."""
    for key in ("discovery_result", "catalog_results"):
        st.session_state.pop(key, None)


def keep_catalog_fallback(description: str) -> None:
    """After discovery fails, hand show_result an empty result so it offers catalog songs."""
    st.session_state["discovery_result"] = DiscoveryResult([], [], [], intent_from_text(description))


def main():
    """Build the Streamlit page with the discovery and catalog forms and show their results."""
    st.set_page_config(page_title="Music Discovery", page_icon="🎵", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Discover music with context")
    st.caption("Find music that fits how you want to listen—and explore the sources behind every recommendation.")
    discovery, choice, catalog = st.columns([1, 0.14, 1], gap="small", vertical_alignment="top")
    with discovery:
        section_heading("mode-heading", "Explore beyond our collection", "Discover music")
        with st.form("discovery", border=False):
            description = st.text_area("What would you like to hear?", max_chars=1200,
                placeholder="A mood, a language, an instrument, a moment. For example: quiet piano for studying, without vocals.")
            reference = st.text_input("Reference song and artist (optional)", max_chars=200)
            submitted = st.form_submit_button("Find music", type="primary", width="stretch")
        if submitted:
            clear_results()
            try:
                # The web app always uses the music guide, song cards and MusicBrainz;
                # only the CLI exposes the web-only baseline (no --enhanced flag).
                request = MusicRequest(description, reference or None, enhanced=True)
                validate_request(request)
                if time.monotonic() - st.session_state.get("last_attempt", float("-inf")) < 15:
                    st.warning("Please wait a few seconds before making another discovery request.")
                else:
                    settings = server_settings()
                    st.session_state["last_attempt"] = time.monotonic()
                    with st.spinner("Finding music and checking sources…"):
                        st.session_state["discovery_result"] = find_music(request, settings)
            except DiscoveryError as error:
                display = st.warning if error.code == "invalid_input" else st.error
                display(str(error))
                logger.info("discovery_failed code=%s", error.code)
                if error.code != "invalid_input":
                    keep_catalog_fallback(description)
            except Exception as error:
                logger.warning("discovery_failed code=unexpected type=%s", type(error).__name__)
                st.error("Discovery could not finish. Please try again later.")
                keep_catalog_fallback(description)
        # The collection panel is taller, so the request summary fills the space under the Find
        # music box. It is reserved here but filled in below, once both forms have cleared or
        # replaced the old results.
        summary_area = st.container()
    with choice:
        st.markdown('<div class="choice-divider"><span>OR</span></div>', unsafe_allow_html=True)
    with catalog:
        section_heading("mode-heading", "Use our prepared collection", "Music Collection")
        with st.form("catalog", border=False):
            st.caption("Pick a genre, mood, and energy level. Genre and mood are optional.")
            genre = st.text_input("Preferred genre", placeholder="pop")
            mood = st.text_input("Preferred mood", placeholder="happy")
            energy = st.slider("Target energy", 0.0, 1.0, 0.7)
            strategy = st.radio(
                "What’s most important to you?",
                list(RANKING_STRATEGIES),
                format_func=lambda name: RANKING_STRATEGIES[name].label,
                horizontal=True,
                width="stretch",
            )
            rank = st.form_submit_button("Browse collection", type="primary", width="stretch")
        if rank:
            clear_results()
            try:
                songs = load_songs(str(ROOT / "data/songs.csv"))
                ranking = RANKING_STRATEGIES[strategy]
                matched = resolve_genres(genre, songs)
                preferences = {"genre": matched or genre.strip().lower(),
                               "mood": mood.strip().lower(), "energy": energy}
                rows = recommend_songs(preferences, songs, strategy=ranking)
                st.session_state["catalog_results"] = {
                    "rows": rows,
                    "matches": [match_fraction(preferences, song, ranking) for song, _, _ in rows],
                    "traits": [trait_match_fractions(preferences, song) for song, _, _ in rows],
                    "summary": catalog_query_summary(genre.strip(), mood.strip(), energy, matched,
                                                     rows, songs, ranking),
                }
            except (OSError, ValueError, KeyError):
                st.error("The CSV catalog could not be read. Check its file format and location.")
    if not {"summary", "traits"} <= st.session_state.get("catalog_results", {"summary": 0, "traits": 0}).keys():
        del st.session_state["catalog_results"]  # saved by an older version of this page
    if "catalog_results" in st.session_state:
        catalog_result = st.session_state["catalog_results"]
        show_catalog_results(catalog_result["rows"], catalog_result["matches"],
                             catalog_result["summary"], catalog_result["traits"])
    if "discovery_result" in st.session_state:
        with summary_area:
            show_request_summary(st.session_state["discovery_result"])
        show_result(st.session_state["discovery_result"])
    elif "catalog_results" not in st.session_state:
        st.markdown(
            '<div class="empty-state">Your discoveries will appear here. '
            'Start with a few details about what you\'d like to hear.</div>',
            unsafe_allow_html=True,
        )
    st.caption("Spotify and YouTube links open the exact track or video when we’re confident of the match, "
               "and a search otherwise.")


if __name__ == "__main__":
    main()
