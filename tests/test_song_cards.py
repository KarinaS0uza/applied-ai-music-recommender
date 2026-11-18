"""Integrity and coverage checks for the curated song-card evidence set."""

import json
import re
from pathlib import Path

from src.utils import safe_url


ROOT = Path(__file__).resolve().parents[1]
CARDS_PATH = ROOT / "data/knowledge/song_cards.json"
SOURCES_PATH = ROOT / "data/knowledge/SOURCES.md"


def cards():
    """Load the tracked cards exactly as the knowledge index does."""
    return json.loads(CARDS_PATH.read_text(encoding="utf-8"))["cards"]


def test_expanded_collection_has_56_unique_source_backed_cards():
    """The expansion adds 30 cards without replacing or duplicating the original 26."""
    rows = cards()
    assert len(rows) == 56
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["sources"] and safe_url(row["sources"][0]["url"]) for row in rows)
    assert all(row["retrieved_at"] and row["unknowns"] for row in rows)


def test_expansion_covers_the_documented_catalog_gaps():
    """K-pop, mainstream genres and globally popular dance styles all gained evidence."""
    added = {row["id"]: row for row in cards() if row["retrieved_at"] == "2026-11-17"}
    assert len(added) == 30
    expected = {
        "dynamite", "next-level-aespa", "love-scenario", "ditto",
        "blinding-lights", "shake-it-off", "mr-brightside", "seven-nation-army",
        "lose-yourself", "alright-kendrick", "no-scrubs", "crazy-in-love",
        "titanium", "levels", "gasolina", "calm-down-rema", "water-tyla",
    }
    assert expected <= set(added)


def test_new_cards_establish_tempo_mood_or_activity_only_with_visible_copy():
    """The added retrieval vocabulary is visible in the evidence text, not hidden metadata."""
    added = [row for row in cards() if row["retrieved_at"] == "2026-11-17"]
    enriched = [row for row in added if re.search(
        r"\b(?:\d{2,3} BPM|upbeat|mellow|hope|feel-good|self-love|running|sport|football|World Cup)\b",
        row["description"], re.IGNORECASE)]
    assert len(enriched) >= 18
    for row in enriched:
        if "BPM" in row["description"]:
            assert "exact BPM" not in row["unknowns"]


def test_sources_table_tracks_every_card_and_current_count():
    """Attribution stays one-for-one with the JSON collection."""
    source_text = SOURCES_PATH.read_text(encoding="utf-8")
    assert "All 56 cards" in source_text
    for row in cards():
        assert f"| `{row['id']}` |" in source_text
        assert row["sources"][0]["url"] in source_text
