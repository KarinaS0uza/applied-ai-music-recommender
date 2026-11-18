"""Tests for catalog loading and ranking in src/recommender.py."""

# Behavior: ranks the real catalog and small in-memory songs to check scoring, strategies and the diversity penalty.

from pathlib import Path

import pytest

from src.recommender import (RANKING_STRATEGIES, Recommender, Song, UserProfile,
                             load_songs, match_fraction, max_score, recommend_songs,
                             resolve_genres, score_song, suggest_genres)

CATALOG = Path(__file__).resolve().parents[1] / "data/songs.csv"

def make_small_recommender() -> Recommender:
    """Build a small Recommender for the tests."""
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    """Recommend returns songs sorted by score."""
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    """Explain recommendation returns non empty string."""
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_diversity_penalty_demotes_repeated_artist():
    """Diversity penalty demotes repeated artist."""
    songs = [
        {"title": "A1", "artist": "X", "genre": "pop", "mood": "happy", "energy": 0.8},
        {"title": "A2", "artist": "X", "genre": "pop", "mood": "happy", "energy": 0.8},
        {"title": "B1", "artist": "Y", "genre": "pop", "mood": "happy", "energy": 0.6},
    ]
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    results = recommend_songs(prefs, songs, k=3)
    assert [s["title"] for s, _, _ in results] == ["A1", "B1", "A2"]
    assert "artist repetition penalty" in results[2][2]


def test_load_songs_normalizes_unknowns_and_multi_value_fields(tmp_path):
    """Load songs normalizes unknowns and multi value fields."""
    path = tmp_path / "songs.csv"
    path.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness,"
        "subgenre,mood_tags,lyrical_theme,instrumentation_tags,language\n"
        "7,Song,Artist,pop,happy,unknown,,0.5,0.5,0.5, Synth-Pop ,Bright; Calm ;,Love,Synth; Drums,English;Spanish\n",
        encoding="utf-8",
    )
    [song] = load_songs(str(path))
    assert (song["id"], song["energy"], song["tempo_bpm"], song["valence"]) == (7, None, None, 0.5)
    assert (song["subgenre"], song["lyrical_theme"]) == ("synth-pop", "love")
    assert song["mood_tags"] == ["bright", "calm"]
    assert song["language"] == ["english", "spanish"]


def test_load_songs_skips_comment_lines(tmp_path):
    """Load songs skips lines that start with a hash, so the catalog can carry a comment."""
    path = tmp_path / "songs.csv"
    path.write_text(
        "# Catalog comment\n"
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness,"
        "subgenre,mood_tags,lyrical_theme,instrumentation_tags,language\n"
        "7,Song,Artist,pop,happy,0.5,100,0.5,0.5,0.5,pop,bright,love,synth,English\n",
        encoding="utf-8",
    )
    assert [song["id"] for song in load_songs(str(path))] == [7]


def test_catalog_is_loadable_and_well_formed():
    """Catalog is loadable and well formed."""
    songs = load_songs(str(CATALOG))
    assert songs and len({song["id"] for song in songs}) == len(songs)
    # The app and CLI index these fields directly, so none may be blank.
    assert all(song[field].strip() for song in songs for field in ("title", "artist", "genre", "mood"))
    for field in ("energy", "valence", "danceability", "acousticness"):
        assert all(song[field] is None or 0 <= song[field] <= 1 for song in songs), field
    assert all(song["tempo_bpm"] is None or song["tempo_bpm"] > 0 for song in songs)


def test_optional_attributes_add_score_and_unknown_values_are_ignored():
    """Optional attributes add score and unknown values are ignored."""
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "subgenre": "synth-pop",
            "lyrical_theme": "love", "mood_tags": ["bright", "calm"],
            "instrumentation_tags": ["synth"], "language": ["english", "unknown"]}
    prefs = {"genre": "rock", "mood": "sad", "subgenre": "Synth-Pop", "lyrical_theme": "love",
             "mood_tags": "bright; calm", "instrumentation_tags": ["synth"],
             "language": ["english", "unknown"]}
    score, reasons = score_song(prefs, song)
    assert score == 5.0
    assert "mood tag match (bright, calm) (+1.0)" in reasons
    unknown = {"genre": "pop", "mood": "happy", "subgenre": "unknown", "language": ["unknown"]}
    assert score_song({"genre": "x", "mood": "y", "subgenre": "unknown", "language": ["unknown"]},
                      unknown)[0] == 0


def test_ranking_strategy_changes_which_song_wins():
    """Ranking strategy changes which song wins."""
    songs = [{"title": "Genre", "artist": "A", "genre": "pop", "mood": "sad", "energy": 0.1},
             {"title": "Mood", "artist": "B", "genre": "rock", "mood": "happy", "energy": 0.1}]
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.9}

    def top(strategy):
        """Return the title of the top-ranked song for a strategy."""
        return recommend_songs(prefs, songs, k=1, strategy=RANKING_STRATEGIES[strategy])[0][0]["title"]

    assert (top("genre-first"), top("mood-first")) == ("Genre", "Mood")


def test_missing_energy_is_skipped_and_k_limits_results():
    """Missing energy is skipped and k limits results."""
    songs = [{"title": f"S{i}", "artist": f"A{i}", "genre": "pop", "mood": "happy", "energy": None}
             for i in range(3)]
    results = recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.5}, songs, k=2)
    assert len(results) == 2
    assert "energy" not in results[0][2]


def test_repeated_non_preferred_genre_is_penalized_but_preferred_genre_is_not():
    """Repeated non preferred genre is penalized but preferred genre is not."""
    songs = [{"title": f"{genre}{i}", "artist": f"{genre}{i}", "genre": genre, "mood": "happy", "energy": 0.5}
             for genre in ("pop", "rock") for i in (1, 2)]
    results = recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.5}, songs, k=4)
    assert [("genre repetition penalty" in reasons) for _, _, reasons in results] == [False, False, False, True]


def test_free_text_genre_resolves_to_the_catalog_labels_it_names():
    """Free text genre resolves to the catalog labels it names."""
    songs = load_songs(str(CATALOG))
    assert resolve_genres("Brazilian Funk", songs) == ["funk carioca", "funk melody"]
    assert resolve_genres("baile funk", songs) == ["funk carioca"]
    assert resolve_genres("pop", songs) == ["pop"]
    assert resolve_genres("funk", songs) == ["funk"]
    assert resolve_genres("brazilian jazz", songs) == []
    assert resolve_genres("", songs) == []
    assert "funk carioca" in suggest_genres("brazilian jazz", songs) + suggest_genres("funk", songs)
    assert suggest_genres("sertaneyo", songs) == ["sertanejo"]


def test_genre_preference_can_name_several_catalog_genres():
    """Genre preference can name several catalog genres."""
    songs = [{"title": "A", "artist": "A", "genre": "funk carioca", "mood": "x", "energy": 0.5},
             {"title": "B", "artist": "B", "genre": "funk melody", "mood": "x", "energy": 0.5},
             {"title": "C", "artist": "C", "genre": "rock", "mood": "x", "energy": 0.5}]
    prefs = {"genre": ["funk carioca", "funk melody"], "mood": "y", "energy": 0.5}
    results = recommend_songs(prefs, songs, k=3)
    assert [song["title"] for song, _, _ in results] == ["A", "B", "C"]
    assert all("genre match" in reasons for _, _, reasons in results[:2])
    assert all("genre repetition penalty" not in reasons for _, _, reasons in results)


def test_match_fraction_is_share_of_the_attainable_score():
    """Match fraction is share of the attainable score."""
    song = {"genre": "pop", "mood": "happy", "energy": 0.8}
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    assert max_score(prefs) == 4.0 and match_fraction(prefs, song) == 1.0
    assert match_fraction({**prefs, "mood": "sad"}, song) == 0.75
    assert match_fraction({**prefs, "genre": "rock", "mood": "sad", "energy": 0.0}, song) == 0.0
    # A blank genre cannot match, so it does not lower what is attainable.
    assert max_score({**prefs, "genre": ""}) == 3.0
    assert max_score({**prefs, "language": ["english"], "mood_tags": []}) == 5.0
    assert match_fraction({"genre": "", "mood": ""}, song) == 0.0


# --- strategies and scoring arithmetic -----------------------------------------------------

PERFECT = {"genre": "pop", "mood": "happy", "energy": 0.8}
PERFECT_SONG = {"title": "T", "artist": "A", "genre": "pop", "mood": "happy", "energy": 0.8}


@pytest.mark.parametrize("name,genre,mood,energy", [
    ("current", 1.0, 1.0, 2.0), ("genre-first", 3.0, 1.0, 0.5),
    ("mood-first", 1.0, 3.0, 0.5), ("energy-focused", 0.5, 0.5, 3.0),
])
def test_each_strategy_weighs_genre_mood_and_energy_as_documented(name, genre, mood, energy):
    """Each strategy weighs genre mood and energy as documented."""
    strategy = RANKING_STRATEGIES[name]
    assert (strategy.genre_weight, strategy.mood_weight, strategy.energy_weight) == (genre, mood, energy)
    assert score_song(PERFECT, PERFECT_SONG, strategy)[0] == genre + mood + energy == max_score(PERFECT, strategy)


def test_energy_focused_ranking_prefers_the_closest_energy_and_the_default_is_balanced():
    """Energy focused ranking prefers the closest energy and the default is balanced."""
    songs = [{"title": "Match", "artist": "A", "genre": "pop", "mood": "happy", "energy": 0.1},
             {"title": "Close", "artist": "B", "genre": "rock", "mood": "sad", "energy": 0.9}]
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.9}
    # Balanced: 2.0 for the genre+mood match against 2.0 for the energy match; the tie goes to the first song.
    assert recommend_songs(prefs, songs, k=1)[0][0]["title"] == "Match"
    assert recommend_songs(prefs, songs, k=1, strategy=RANKING_STRATEGIES["energy-focused"])[0][0]["title"] == "Close"


def test_energy_similarity_falls_off_linearly_and_stops_half_a_point_away():
    """Energy similarity falls off linearly and stops half a point away."""
    song = {"genre": "x", "mood": "y", "energy": 0.8}
    score, reasons = score_song({"genre": "p", "mood": "q", "energy": 0.6}, song)
    assert score == pytest.approx(1.2) and reasons == ["energy similarity (+1.20)"]
    assert score_song({"genre": "p", "mood": "q", "energy": 0.3}, song) == (0.0, [])   # exactly 0.5 away
    assert score_song({"genre": "p", "mood": "q", "energy": 0.1}, song) == (0.0, [])   # never negative
    assert score_song({"genre": "p", "mood": "q"}, song) == (0.0, [])                  # no energy preference given


def test_a_song_that_matches_nothing_gets_a_plain_explanation():
    """A song that matches nothing gets a plain explanation."""
    [(song, score, reasons)] = recommend_songs({"genre": "jazz", "mood": "sad"},
                                               [{"title": "T", "artist": "A", "genre": "pop", "mood": "happy"}])
    assert (score, reasons) == (0.0, "No preference matches")


def test_artist_repetition_penalty_grows_with_each_repeat():
    """Artist repetition penalty grows with each repeat."""
    songs = [{"title": f"S{n}", "artist": "X", "genre": "pop", "mood": "happy", "energy": 0.8} for n in range(3)]
    results = recommend_songs(PERFECT, songs, k=3)
    assert ["artist repetition penalty" in reasons for _, _, reasons in results] == [False, True, True]
    assert "(-1.50)" in results[1][2] and "(-3.00)" in results[2][2]
    assert [score for _, score, _ in results] == [4.0, 2.5, 1.0]


def test_unknown_artists_and_genres_are_never_penalized_as_repeats():
    """Unknown artists and genres are never penalized as repeats."""
    songs = [{"title": f"S{n}", "artist": "unknown", "genre": "unknown", "mood": "happy", "energy": 0.8}
             for n in range(3)]
    results = recommend_songs(PERFECT, songs, k=3)
    assert len(results) == 3  # all three were picked, so there was something to penalize
    assert not any("penalty" in reasons for _, _, reasons in results)


def test_equal_scores_keep_the_catalog_order():
    """Equal scores keep the catalog order."""
    songs = [{"title": title, "artist": title, "genre": "pop", "mood": "happy", "energy": 0.8}
             for title in ("C", "A", "B")]
    assert [song["title"] for song, _, _ in recommend_songs(PERFECT, songs, k=3)] == ["C", "A", "B"]


def test_recommending_from_nothing_or_for_zero_songs_returns_an_empty_list():
    """Recommending from nothing or for zero songs returns an empty list."""
    assert recommend_songs(PERFECT, []) == []
    assert recommend_songs(PERFECT, [PERFECT_SONG], k=0) == []


# --- catalog loading errors and genre suggestions ------------------------------------------

HEADER = ("id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness,"
          "subgenre,mood_tags,lyrical_theme,instrumentation_tags,language\n")


@pytest.mark.parametrize("content,error", [
    (None, OSError),                                               # no such file
    (HEADER + "x,S,A,pop,happy,0.5,100,0.5,0.5,0.5,,,,,\n", ValueError),    # id is not a number
    (HEADER + "1,S,A,pop,happy,high,100,0.5,0.5,0.5,,,,,\n", ValueError),   # energy is not a number
    ("id,title\n1,S\n", KeyError),                                 # columns missing
], ids=["missing-file", "bad-id", "bad-number", "missing-columns"])
def test_a_bad_catalog_raises_the_errors_the_cli_and_app_handle(tmp_path, content, error):
    """A bad catalog raises the errors the cli and app handle."""
    path = tmp_path / "songs.csv"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        load_songs(str(path))


def test_genre_suggestions_honor_the_limit_and_a_blank_query():
    """Genre suggestions honor the limit and a blank query."""
    songs = [{"genre": f"funk {name}"} for name in ("c", "a", "b")]
    assert suggest_genres("funk", songs, limit=2) == ["funk a", "funk b"]  # equal overlap sorts by name
    assert suggest_genres("", songs) == []
