"""Catalog loading and ranking logic for the music recommender."""

# Behavior: loads the CSV catalog and ranks songs by weighted genre, mood and energy matches, with a penalty for repeated artists.

import csv
import difflib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.utils import normalize

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the songs this recommender can choose from."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs for a user profile (placeholder: currently the first k songs)."""
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a song was recommended (placeholder text for now)."""
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs, preserving unknown numeric and multi-value metadata safely; lines starting with # are skipped."""
    songs = []
    float_fields = {
        "energy",
        "tempo_bpm",
        "valence",
        "danceability",
        "acousticness",
    }

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(line for line in csv_file if not line.startswith("#")):
            row["id"] = int(row["id"])
            for field in float_fields:
                raw_value = row[field].strip()
                row[field] = (
                    float(raw_value)
                    if raw_value and raw_value.lower() != "unknown"
                    else None
                )
            for field in ("subgenre", "lyrical_theme"):
                row[field] = row[field].strip().lower()
            for field in ("mood_tags", "instrumentation_tags", "language"):
                row[field] = [
                    value.strip().lower()
                    for value in row[field].split(";")
                    if value.strip()
                ]
            songs.append(row)

    print(f"Loaded songs: {len(songs)}")
    return songs

def _energy_similarity(song_energy: float, preferred_energy: float) -> float:
    """1.0 for an identical energy level, falling off linearly to 0.0 half a point away."""
    return max(0.0, 1.0 - abs(song_energy - preferred_energy) / 0.5)


def _genre_matches(song_genre: str, preferred) -> bool:
    """A preferred genre is one catalog label or a collection of acceptable labels."""
    if isinstance(preferred, str):
        return song_genre == preferred
    return song_genre in preferred


@dataclass(frozen=True)
class RankingStrategy:
    """A replaceable set of weights for the core preference signals."""
    key: str
    label: str
    genre_weight: float
    mood_weight: float
    energy_weight: float

    def score_core(self, user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
        """Score one song against the user preferences and list the reasons for the score."""
        score = 0.0
        reasons = []

        if _genre_matches(song["genre"], user_prefs["genre"]):
            score += self.genre_weight
            reasons.append(f"genre match (+{self.genre_weight:.1f})")

        if song["mood"] == user_prefs["mood"]:
            score += self.mood_weight
            reasons.append(f"mood match (+{self.mood_weight:.1f})")

        energy = song.get("energy")
        if energy is not None and "energy" in user_prefs:
            energy_similarity = _energy_similarity(energy, user_prefs["energy"])
            weighted_energy = self.energy_weight * energy_similarity
            if weighted_energy > 0:
                score += weighted_energy
                reasons.append(f"energy similarity (+{weighted_energy:.2f})")

        return score, reasons


def trait_match_fractions(user_prefs: Dict, song: Dict) -> Dict[str, float]:
    """How well a song matches each core trait, from 0.0 to 1.0, keyed by the reason label.

    The keys are the labels ``score_core`` writes into its reasons, so a card can pair each
    reason with a percentage. Only traits the listener actually asked for are included.
    """
    fractions = {}
    if _has_preference(user_prefs.get("genre")):
        fractions["genre match"] = 1.0 if _genre_matches(song["genre"], user_prefs["genre"]) else 0.0
    if _has_preference(user_prefs.get("mood")):
        fractions["mood match"] = 1.0 if song["mood"] == user_prefs["mood"] else 0.0
    energy = song.get("energy")
    if energy is not None and user_prefs.get("energy") is not None:
        fractions["energy similarity"] = _energy_similarity(energy, user_prefs["energy"])
    return fractions


RANKING_STRATEGIES = {
    # Keep the existing scoring as the default so existing rankings are stable.
    "current": RankingStrategy("current", "Balanced (Default)", 1.0, 1.0, 2.0),
    "genre-first": RankingStrategy("genre-first", "Genre-First", 3.0, 1.0, 0.5),
    "mood-first": RankingStrategy("mood-first", "Mood-First", 1.0, 3.0, 0.5),
    "energy-focused": RankingStrategy("energy-focused", "Energy-Focused", 0.5, 0.5, 3.0),
}

ARTIST_REPEAT_PENALTY = 1.5
GENRE_REPEAT_PENALTY = 0.5


def _resolve_strategy(strategy: Optional[RankingStrategy]) -> RankingStrategy:
    """Return the given ranking strategy, or the default one when none is given."""
    return strategy or RANKING_STRATEGIES["current"]


def score_song(
    user_prefs: Dict,
    song: Dict,
    strategy: Optional[RankingStrategy] = None,
) -> Tuple[float, List[str]]:
    """Score a song against user preferences and return matching reasons."""
    selected_strategy = _resolve_strategy(strategy)
    score, reasons = selected_strategy.score_core(user_prefs, song)

    # Advanced preferences are optional; omitting them leaves existing profiles
    # and their ranking behavior unchanged.
    for field, preference_key, reason_label in (
        ("subgenre", "subgenre", "subgenre"),
        ("lyrical_theme", "lyrical_theme", "lyrical theme"),
    ):
        preferred = user_prefs.get(preference_key)
        available = song.get(field)
        if (
            preferred
            and available
            and preferred.strip().lower() != "unknown"
            and available != "unknown"
            and available == preferred.strip().lower()
        ):
            score += 1.0
            reasons.append(f"{reason_label} match ({available}) (+1.0)")

    for field, preference_key, reason_label in (
        ("mood_tags", "mood_tags", "mood tag"),
        ("instrumentation_tags", "instrumentation_tags", "instrumentation"),
        ("language", "language", "language"),
    ):
        preferred = user_prefs.get(preference_key)
        if isinstance(preferred, str):
            preferred = preferred.split(";")
        preferred_values = {
            value.strip().lower()
            for value in (preferred or [])
            if value.strip() and value.strip().lower() != "unknown"
        }
        available_values = {
            value for value in song.get(field, []) if value != "unknown"
        }
        matches = sorted(preferred_values.intersection(available_values))
        if matches:
            score += 1.0
            reasons.append(
                f"{reason_label} match ({', '.join(matches)}) (+1.0)"
            )

    return score, reasons

def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    strategy: Optional[RankingStrategy] = None,
) -> List[Tuple[Dict, float, str]]:
    """Greedily build a diverse top-k list using adjusted recommendation scores."""
    selected_strategy = _resolve_strategy(strategy)
    remaining = [
        (song, *score_song(user_prefs, song, selected_strategy))
        for song in songs
    ]
    selected = []
    selected_artist_counts = {}
    selected_genre_counts = {}
    raw_genre = user_prefs.get("genre", "")
    preferred_genres = (
        {raw_genre.strip().lower()}
        if isinstance(raw_genre, str)
        else {str(genre).strip().lower() for genre in raw_genre}
    )

    while remaining and len(selected) < k:
        best_index = 0
        best_adjusted_score = float("-inf")
        best_artist_penalty = 0.0
        best_genre_penalty = 0.0

        for index, (song, base_score, _) in enumerate(remaining):
            artist = str(song.get("artist", "")).strip().lower()
            genre = str(song.get("genre", "")).strip().lower()

            artist_count = (
                selected_artist_counts.get(artist, 0)
                if artist and artist != "unknown"
                else 0
            )
            artist_penalty = ARTIST_REPEAT_PENALTY * artist_count

            genre_count = (
                selected_genre_counts.get(genre, 0)
                if genre and genre != "unknown"
                else 0
            )
            # Keep the user's requested genre eligible for multiple picks.
            genre_penalty = (
                GENRE_REPEAT_PENALTY * genre_count
                if genre_count and genre not in preferred_genres
                else 0.0
            )

            adjusted_score = base_score - artist_penalty - genre_penalty
            if adjusted_score > best_adjusted_score:
                best_index = index
                best_adjusted_score = adjusted_score
                best_artist_penalty = artist_penalty
                best_genre_penalty = genre_penalty

        song, _, reasons = remaining.pop(best_index)
        if best_artist_penalty:
            reasons.append(f"artist repetition penalty (-{best_artist_penalty:.2f})")
        if best_genre_penalty:
            reasons.append(f"genre repetition penalty (-{best_genre_penalty:.2f})")
        selected.append(
            (
                song,
                best_adjusted_score,
                "; ".join(reasons) if reasons else "No preference matches",
            )
        )

        artist = str(song.get("artist", "")).strip().lower()
        if artist and artist != "unknown":
            selected_artist_counts[artist] = selected_artist_counts.get(artist, 0) + 1
        genre = str(song.get("genre", "")).strip().lower()
        if genre and genre != "unknown":
            selected_genre_counts[genre] = selected_genre_counts.get(genre, 0) + 1

    return selected


# Common names that do not appear verbatim in the catalog's genre labels.
_GENRE_ALIASES = {
    "baile funk": ("funk carioca",),
    "favela funk": ("funk carioca",),
    "funk brasileiro": ("funk carioca", "funk melody"),
}
_GENRE_FILLER = {"music", "song", "songs", "genre", "style"}


def _name_tokens(*values) -> set:
    """Collect the normalized word tokens of the given values."""
    return {token for value in values for token in normalize(str(value or "")).split()}


def _catalog_genres(songs: List[Dict]) -> List[str]:
    """Return the sorted, known genre labels present in the catalog."""
    return sorted({
        genre for genre in (str(song.get("genre", "")).strip().lower() for song in songs)
        if genre and genre != "unknown"
    })


def resolve_genres(query: str, songs: List[Dict]) -> List[str]:
    """Map a free-text genre to the catalog genre labels it confidently names.

    Tries an exact label, a known alias, then a song whose genre, subgenre and
    language together contain every word ("brazilian funk" finds the funk songs
    sung in Brazilian Portuguese). Returns [] rather than guessing.
    """
    wanted = normalize(query or "")
    genres = _catalog_genres(songs)
    exact = [genre for genre in genres if normalize(genre) == wanted]
    if not wanted or exact:
        return exact
    aliased = [genre for genre in _GENRE_ALIASES.get(wanted, ()) if genre in genres]
    if aliased:
        return aliased
    tokens = {token for token in wanted.split() if token not in _GENRE_FILLER}
    if not tokens:
        return []
    return sorted({
        str(song.get("genre", "")).strip().lower()
        for song in songs
        if str(song.get("genre", "")).strip().lower() not in {"", "unknown"}
        and tokens <= _name_tokens(song.get("genre"), song.get("subgenre"),
                                   *(song.get("language") or []))
    })


def suggest_genres(query: str, songs: List[Dict], limit: int = 6) -> List[str]:
    """Catalog genres that share a word with, or are spelled like, the query."""
    wanted = normalize(query or "")
    tokens = {token for token in wanted.split() if token not in _GENRE_FILLER}
    overlap = {}
    for song in songs:
        genre = str(song.get("genre", "")).strip().lower()
        if not genre or genre == "unknown":
            continue
        shared = len(tokens & _name_tokens(genre, song.get("subgenre")))
        if shared:
            overlap[genre] = max(overlap.get(genre, 0), shared)
    ranked = sorted(overlap, key=lambda genre: (-overlap[genre], genre))
    for genre in difflib.get_close_matches(wanted, _catalog_genres(songs), n=3, cutoff=0.7):
        if genre not in ranked:
            ranked.insert(0, genre)
    return ranked[:limit]


def _has_preference(value) -> bool:
    """Tell whether a value holds at least one real preference (not blank or unknown)."""
    if isinstance(value, str):
        value = value.split(";")
    return any(
        str(item).strip() and str(item).strip().lower() != "unknown"
        for item in (value or [])
    )


def max_score(user_prefs: Dict, strategy: Optional[RankingStrategy] = None) -> float:
    """The most a song could score for the preferences actually supplied."""
    selected_strategy = _resolve_strategy(strategy)
    total = 0.0
    if _has_preference(user_prefs.get("genre")):
        total += selected_strategy.genre_weight
    if _has_preference(user_prefs.get("mood")):
        total += selected_strategy.mood_weight
    if user_prefs.get("energy") is not None:
        total += selected_strategy.energy_weight
    for key in ("subgenre", "lyrical_theme", "mood_tags", "instrumentation_tags", "language"):
        if _has_preference(user_prefs.get(key)):
            total += 1.0
    return total


def match_fraction(
    user_prefs: Dict,
    song: Dict,
    strategy: Optional[RankingStrategy] = None,
) -> float:
    """How much of the attainable score a song earns, from 0.0 to 1.0.

    Uses the song's own match, before diversity penalties, so repeating an
    artist in the list does not make a song look like a worse fit.
    """
    ceiling = max_score(user_prefs, strategy)
    if ceiling <= 0:
        return 0.0
    score, _ = score_song(user_prefs, song, strategy)
    return max(0.0, min(1.0, score / ceiling))
