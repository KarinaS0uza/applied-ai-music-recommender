"""Tests for the command-line interface in src/main.py."""

import re
import sys
from unittest.mock import patch

import pytest

from src import main as cli
from src.models import (DiscoveryError, DiscoveryResult, Evidence, MusicRequest, Recommendation,
                        TrackLink, VideoLink)


def run_cli(monkeypatch, *args):
    """Run the command-line main() with the given arguments."""
    monkeypatch.setattr(sys, "argv", ["main", *args])
    cli.main()


def test_table_wraps_long_cells_within_fixed_width(capsys):
    """Table wraps long cells within fixed width."""
    cli.print_recommendation_table([({"title": "T" * 60, "artist": "A"}, 4.0, "genre match (+1.0); " * 8)])
    lines = capsys.readouterr().out.splitlines()
    assert len({len(line) for line in lines}) == 1
    assert len(lines) > 5


def test_default_run_prints_a_ranked_table_per_profile(monkeypatch, capsys):
    """Default run prints a ranked table per profile."""
    run_cli(monkeypatch, "--strategy", "genre-first")
    out = capsys.readouterr().out
    assert out.count("TOP MUSIC RECOMMENDATIONS") == 4
    assert "Ranking mode: Genre-First" in out
    for profile in ("My Preference — Ballad / Romantic", "Deep Intense Rock", "High-Energy Pop", "Chill Lofi"):
        assert f"TOP MUSIC RECOMMENDATIONS — {profile}" in out


def test_query_prints_sources_search_links_and_notes(monkeypatch, capsys):
    """Query prints sources search links and notes."""
    result = DiscoveryResult(
        [Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.", ["web_1"], ["Version unknown."])],
        [Evidence("web_1", "web", "Source", "text", "https://example.org/music")],
        ["Careful."], notices=["Identity only."])
    with patch("src.composition.find_music", return_value=result) as service:
        run_cli(monkeypatch, "--query", "jazz", "--reference", "So What", "--enhanced")
    assert service.call_args.args[0] == MusicRequest("jazz", "So What", True)
    out = capsys.readouterr().out
    for expected in ("1. Take Five — Dave Brubeck Quartet", "Uncertainty: Version unknown.",
                     "Source [web_1]: https://example.org/music", "YouTube search: https://",
                     "Notice: Careful.", "Metadata note: Identity only."):
        assert expected in out


def test_query_prints_the_exact_video_instead_of_the_youtube_search(monkeypatch, capsys):
    """Query prints the exact video instead of the youtube search."""
    video = VideoLink("dQw4w9WgXcQ", "Take Five", "Dave Brubeck - Topic")
    result = DiscoveryResult([Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.", [], [],
                                             video=video)], [], [])
    with patch("src.composition.find_music", return_value=result):
        run_cli(monkeypatch, "--query", "jazz")
    out = capsys.readouterr().out
    assert "YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ (Take Five — Dave Brubeck - Topic)" in out
    assert "YouTube search" not in out and "Spotify search: https://" in out


def test_query_prints_the_exact_spotify_track_instead_of_the_search(monkeypatch, capsys):
    """Query prints the exact Spotify track instead of the Spotify search."""
    track = TrackLink("11dFghVXANMlKmJXsNCbNl", "Take Five", "Dave Brubeck")
    result = DiscoveryResult([Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.", [], [],
                                             track=track)], [], [])
    with patch("src.composition.find_music", return_value=result):
        run_cli(monkeypatch, "--query", "jazz")
    out = capsys.readouterr().out
    assert "Spotify track: https://open.spotify.com/track/11dFghVXANMlKmJXsNCbNl (Take Five — Dave Brubeck)" in out
    assert "Spotify search" not in out and "YouTube search: https://" in out


def test_query_failure_exits_with_the_safe_message(monkeypatch, capsys):
    """Query failure exits with the safe message."""
    with patch("src.composition.find_music", side_effect=DiscoveryError("quota", "Quota reached.")), \
            pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "--query", "jazz")
    assert exit_info.value.code == 1
    assert "Quota reached." in capsys.readouterr().err


def test_table_prints_headings_ranks_and_two_decimal_scores(capsys):
    """Table prints headings ranks and two decimal scores."""
    cli.print_recommendation_table([({"title": "First", "artist": "A"}, 4.0, "genre match"),
                                    ({"title": "Second", "artist": "B"}, 2.5, "")])
    out = capsys.readouterr().out
    heading, first, second = (line for line in out.splitlines() if line.startswith("|"))
    assert all(word in heading for word in ("#", "Song", "Artist", "Score", "Reasons"))
    assert re.match(r"\| 1 +\| First +\| A +\| 4\.00 +\| genre match", first)
    assert re.match(r"\| 2 +\| Second +\| B +\| 2\.50 +\|", second)

    cli.print_recommendation_table([])
    assert len(capsys.readouterr().out.splitlines()) == 4  # borders and headings only


def test_default_run_uses_the_balanced_strategy_and_lists_five_songs_per_profile(monkeypatch, capsys):
    """Default run uses the balanced strategy and lists five songs per profile."""
    run_cli(monkeypatch)
    out = capsys.readouterr().out
    assert out.count("Ranking mode: Balanced (Default)") == 4
    assert len(re.findall(r"(?m)^\| 5 ", out)) == 4 and not re.search(r"(?m)^\| 6 ", out)


def test_an_unknown_strategy_is_rejected_by_the_argument_parser(monkeypatch, capsys):
    """An unknown strategy is rejected by the argument parser."""
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "--strategy", "loudest")
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [OSError("/private/path/songs.csv"), ValueError("bad id"), KeyError("genre")])
def test_an_unreadable_catalog_exits_with_a_safe_message(monkeypatch, capsys, failure):
    """An unreadable catalog exits with a safe message."""
    with patch("src.main.load_songs", side_effect=failure), pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch)
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "The CSV catalog could not be read" in err and "/private/path" not in err


def test_query_without_optional_flags_asks_for_the_web_only_baseline(monkeypatch):
    """Query without optional flags asks for the web only baseline."""
    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [])) as service:
        run_cli(monkeypatch, "--query", "jazz")
    assert service.call_args.args[0] == MusicRequest("jazz", None, False)


def test_query_prints_only_the_links_that_exist(monkeypatch, capsys):
    """Query prints only the links that exist."""
    result = DiscoveryResult(
        [Recommendation("Take Five", "Dave Brubeck Quartet", "Jazz.", ["guide_1", "web_1"], [])],
        [Evidence("guide_1", "guide", "Pace", "Vocabulary."), Evidence("web_1", "web", "Source", "text", "https://example.org/a")],
        [])
    with patch("src.composition.find_music", return_value=result):
        run_cli(monkeypatch, "--query", "jazz")
    out = capsys.readouterr().out
    assert "Source [web_1]: https://example.org/a" in out and "guide_1" not in out


def test_query_with_no_recommendations_still_prints_the_warnings(monkeypatch, capsys):
    """Query with no recommendations still prints the warnings."""
    result = DiscoveryResult([], [], ["Nothing relevant was found."])
    with patch("src.composition.find_music", return_value=result):
        run_cli(monkeypatch, "--query", "jazz")
    out = capsys.readouterr().out
    assert out.strip() == "Notice: Nothing relevant was found."


def test_blank_query_fails_validation_before_any_provider_is_contacted(monkeypatch, capsys):
    """Blank query fails validation before any provider is contacted."""
    with pytest.raises(SystemExit) as exit_info:  # find_music is not patched: it must stop at validation
        run_cli(monkeypatch, "--query", "  ")
    assert exit_info.value.code == 1
    assert "Describe a genre, mood" in capsys.readouterr().err
