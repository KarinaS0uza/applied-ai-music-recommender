"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

# Behavior: with --query it runs web discovery and prints sources and links; without it, it prints ranked catalog tables for sample profiles.
import argparse
import textwrap
import logging
from pathlib import Path

from src.recommender import RANKING_STRATEGIES, load_songs, recommend_songs


TABLE_COLUMNS = (
    ("#", 3),
    ("Song", 24),
    ("Artist", 18),
    ("Score", 7),
    ("Reasons", 48),
)


def print_recommendation_table(recommendations) -> None:
    """Print recommendations in a wrapped, dependency-free ASCII table."""
    widths = [width for _, width in TABLE_COLUMNS]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def print_row(values) -> None:
        """Print one table row with each cell wrapped to its column width."""
        wrapped = [
            textwrap.wrap(
                str(value),
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for value, width in zip(values, widths)
        ]
        for line_index in range(max(len(cell_lines) for cell_lines in wrapped)):
            cells = [
                cell_lines[line_index] if line_index < len(cell_lines) else ""
                for cell_lines in wrapped
            ]
            print("| " + " | ".join(
                cell.ljust(width) for cell, width in zip(cells, widths)
            ) + " |")

    print(border)
    print_row([heading for heading, _ in TABLE_COLUMNS])
    print(border)
    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        print_row([rank, song["title"], song["artist"], f"{score:.2f}", reasons])
    print(border)


def main() -> None:
    """Parse the command-line arguments and run a discovery query or the sample profiles."""
    parser = argparse.ArgumentParser(description="Run the music recommender.")
    parser.add_argument(
        "--strategy",
        choices=RANKING_STRATEGIES,
        default="current",
        help="Ranking mode: current, genre-first, mood-first, or energy-focused.",
    )
    parser.add_argument("--query", help="Discover music from a natural-language description.")
    parser.add_argument("--reference", help="Optional reference song and artist.")
    parser.add_argument("--enhanced", action="store_true", help="Add custom documents and MusicBrainz metadata.")
    args = parser.parse_args()
    if args.query is not None:
        from src.composition import find_music
        from src.models import DiscoveryError, MusicRequest
        from src.utils import platform_search_links
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("src").setLevel(logging.INFO)
        try:
            result = find_music(MusicRequest(args.query, args.reference, args.enhanced))
        except DiscoveryError as error:
            parser.exit(1, f"{error}\n")
        sources = {e.id: e for e in result.evidence}
        for rank, rec in enumerate(result.recommendations, 1):
            print(f"\n{rank}. {rec.title} — {rec.artist}\n{rec.explanation}")
            for uncertainty in rec.uncertainties:
                print(f"Uncertainty: {uncertainty}")
            for id in rec.source_ids:
                if sources[id].url:
                    print(f"Source [{id}]: {sources[id].url}")
            video = rec.video
            if video:
                print(f"YouTube video: {video.url} ({video.title} — {video.channel})")
            for label, url in platform_search_links(rec.title, rec.artist).items():
                if video and label == "YouTube search":
                    continue
                print(f"{label}: {url}")
        for warning in result.warnings:
            print(f"Notice: {warning}")
        for notice in result.notices:
            print(f"Metadata note: {notice}")
        return
    strategy = RANKING_STRATEGIES[args.strategy]

    try:
        songs = load_songs(str(Path(__file__).resolve().parents[1] / "data/songs.csv"))
    except (OSError, ValueError, KeyError):
        parser.exit(1, "The CSV catalog could not be read. Check its file format and location.\n")

    profiles = [
        (
            "My Preference — Ballad / Romantic",
            {"genre": "ballad", "mood": "romantic", "energy": 0.45},
        ),
        (
            "Deep Intense Rock",
            {"genre": "rock", "mood": "intense", "energy": 0.9},
        ),
        (
            "High-Energy Pop",
            {"genre": "pop", "mood": "happy", "energy": 0.9},
        ),
        (
            "Chill Lofi",
            {"genre": "lofi", "mood": "chill", "energy": 0.3},
        ),
    ]

    for profile_name, user_prefs in profiles:
        recommendations = recommend_songs(
            user_prefs,
            songs,
            k=5,
            strategy=strategy,
        )

        print("\n" + "=" * 50)
        print(f"TOP MUSIC RECOMMENDATIONS — {profile_name}")
        print(f"Ranking mode: {strategy.label}")
        print("=" * 50)

        print_recommendation_table(recommendations)

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
