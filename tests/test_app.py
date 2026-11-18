"""Tests for the Streamlit app in src/app.py."""

# Behavior: drives the Streamlit app with AppTest and patched services, so no test makes a real API call.

import re
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import src.app as app_module
from src.app import (catalog_preferences, catalog_query_summary, catalog_reason_text,
                     catalog_song_summary, compact_text, literal_caption, platform_links_html,
                     readable_source_excerpt, server_settings, source_kind_label,
                     video_fallback_source)
from src.discovery import INTENT_FALLBACK_NOTICE
from src.models import (DiscoveryError, DiscoveryResult, Evidence, MusicRequest, ParsedIntent,
                        Recommendation, SupportedClaim, TasteCriterion, VideoLink)
from src.recommender import RANKING_STRATEGIES


@pytest.fixture
def app(monkeypatch):
    """Build the Streamlit app test with fixture API keys set."""
    monkeypatch.setenv("TAVILY_API_KEY", "fixture-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-secret")
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/app.py"), default_timeout=30)


def submit(app, text):
    """Run the app, type a description and click the first button."""
    app.run()
    app.text_area[0].set_value(text)
    app.button[0].click().run()


def test_platform_links_render_logos_with_accessible_names():
    """Platform links render logos with accessible names."""
    html = platform_links_html("Rock & Roll", "AC/DC")
    assert html.count("<svg") == 2 and html.count('class="platform-logo-link"') == 2
    assert 'fill="#FF0000"' in html and 'fill="#1DB954"' in html
    assert 'aria-label="Search YouTube for Rock &amp; Roll by AC/DC"' in html
    assert 'aria-label="Search Spotify for Rock &amp; Roll by AC/DC"' in html
    assert 'rel="noopener noreferrer"' in html
    assert "YouTube search</" not in html  # logos replace the old text labels


def test_matched_video_replaces_only_the_youtube_search_link():
    """Matched video replaces only the youtube search link."""
    video = VideoLink("dQw4w9WgXcQ", 'Take "Five" <b>', "Dave Brubeck - Topic")
    html = platform_links_html("Take Five", "Dave Brubeck", video)
    assert 'href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"' in html
    assert "results?search_query" not in html and 'title="Watch on YouTube"' in html
    assert 'aria-label="Watch Take &quot;Five&quot; &lt;b&gt; by Dave Brubeck - Topic on YouTube"' in html
    assert "open.spotify.com/search/" in html and 'aria-label="Search Spotify for Take Five by Dave Brubeck"' in html


def test_song_card_shows_the_matched_video_with_markup_neutralized(app):
    """Song card shows the matched video with markup neutralized."""
    video = VideoLink("dQw4w9WgXcQ", "Take Five [Official](https://evil.example)", "Dave Brubeck - Topic")
    result = DiscoveryResult([Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.",
                                             ["web_1"], [], fit_score=0.8, video=video)],
                             [Evidence("web_1", "web", "Source", "Evidence", "https://example.org/music")], [])
    with patch("src.composition.find_music", return_value=result):
        submit(app, "instrumental jazz")
    logo_links = [e.value for e in app.markdown if 'class="platform-logo-link"' in e.value]
    assert "watch?v=dQw4w9WgXcQ" in logo_links[0] and "results?search_query" not in logo_links[0]
    caption = next(e.value for e in app.caption if e.value.startswith("▶"))
    assert "Dave Brubeck - Topic" in caption and r"\[Official\]\(https://evil.example\)" in caption
    assert not app.exception


def test_a_mix_of_scored_and_unscored_songs_explains_the_unscored_ones(app):
    """Topped-up songs follow the checked ones, and the page says what NOT SCORED means."""
    checked = Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.", ["web_1"], [], fit_score=1.0)
    named = Recommendation("So What", "Miles Davis", "“So What by Miles Davis.”", ["web_1"],
                           ["We couldn't check this song's fit against the source."])
    result = DiscoveryResult([checked, named],
                             [Evidence("web_1", "web", "Source", "Evidence", "https://example.org/music")], [])
    with patch("src.composition.find_music", return_value=result):
        submit(app, "jazz")
    captions = [element.value for element in app.caption]
    assert any("Here are a few matches to explore." in v and "NOT SCORED were named by their sources" in v
               for v in captions)
    assert any("NOT SCORED" in v and v.startswith("DISCOVERY 02") for v in captions)
    assert not app.exception


def test_blank_input_does_not_call_service(app):
    """Blank input does not call service."""
    with patch("src.composition.find_music") as service:
        submit(app, " ")
    service.assert_not_called()
    assert app.warning
    assert not app.exception


def test_submitted_description_renders_recommendation_and_source(app):
    """Submitted description renders recommendation and source."""
    intent = ParsedIntent([
        TasteCriterion("mood", "preference", "happy", 2, "happy"),
        TasteCriterion("mood", "preference", "peaceful", 3, "very peaceful"),
    ], ["happy peaceful songs"])
    result = DiscoveryResult([Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.",
                            ["web_1"], ["Study fit is subjective."], fit_score=0.8)],
                            [Evidence("web_1", "web", "Source title", "Evidence", "https://example.org/music",
                                      "2026-11-16T04:48:14.876050+00:00")],
                            [], intent)
    with patch("src.composition.find_music", return_value=result) as service:
        submit(app, "instrumental jazz")
    service.assert_called_once()
    rendered = str(app)
    assert "Take Five" in rendered and "Dave Brubeck Quartet" in rendered
    assert "Five-beat jazz." in rendered and "Study fit is subjective." in rendered
    captions = [element.value for element in app.caption]
    assert any("80% MATCH SCORE" in value for value in captions)
    assert any("1 detail couldn’t be confirmed" in value for value in captions)
    assert any("Here are a few matches to explore." in value for value in captions)
    summaries = [element.value for element in app.markdown]
    assert any("What we couldn’t confirm" in value for value in summaries)
    assert any("Song 1" in value and "Take Five — Dave Brubeck Quartet" in value for value in summaries)
    assert any("Happy" in value and "Peaceful" in value for value in summaries)
    assert "https://example.org/music" in rendered
    assert "Sources checked (1)" in rendered
    assert any("You do not need to read them" in value for value in captions)
    visible_copy = [element.value for kind in ("caption", "text", "markdown") for element in app.get(kind)]
    assert any("Why it fits" in value for value in visible_copy)
    assert not any("2026-11-16T04:48:14.876050+00:00" in value for value in visible_copy)
    assert any(element.proto.label == "Read source" for element in app.get("link_button"))
    logo_links = [value for value in summaries if 'class="platform-logo-link"' in value]
    assert logo_links and "youtube.com/results?search_query=" in logo_links[0]
    assert "open.spotify.com/search/" in logo_links[0]
    assert not any(element.proto.label in {"YouTube search", "Spotify search"}
                   for element in app.get("link_button"))
    assert not app.exception


def test_unscored_fallback_card_does_not_claim_a_zero_percent_match(app):
    """Unscored fallback card does not claim a zero percent match."""
    intent = ParsedIntent([
        TasteCriterion("mood", "preference", "happy", 2, "happy"),
        TasteCriterion("energy", "preference", "slower beats", 2, "slower beats"),
    ], ["happy slower beats"])
    rec = Recommendation(
        "A Thousand Years", "Christina Perri",
        "Found in a source for your search, but we couldn't verify that it matches what you asked for (happy, slower beats).",
        ["web_1"], ["We couldn't check this song's fit against the source."])
    result = DiscoveryResult([rec], [Evidence("web_1", "web", "Source title", "Evidence",
                                              "https://example.org/music")], [], intent)
    with patch("src.composition.find_music", return_value=result):
        submit(app, "happy songs with slower beats")
    captions = [element.value for element in app.caption]
    assert any("NOT SCORED" in value for value in captions)
    assert not any("MATCH SCORE" in value for value in captions)
    assert any("we couldn’t confirm they match your request" in value for value in captions)
    assert any("What the source says" in value for value in captions + [e.value for e in app.markdown])
    assert not any("Why it fits" in value for value in [e.value for e in app.markdown])
    assert not app.exception


def test_legacy_session_result_survives_schema_upgrade(app):
    """Legacy session result survives schema upgrade."""
    app.session_state["discovery_result"] = SimpleNamespace(
        recommendations=[Recommendation(
            "Old Song", "Old Artist", "A result from the previous app version.",
            ["web_1"], [],
        )],
        evidence=[Evidence(
            "web_1", "web", "Legacy source", "Legacy evidence",
            "https://example.org/legacy",
        )],
        warnings=[],
    )

    app.run()

    assert "Old Song" in str(app)
    assert not app.exception


def test_metadata_notice_is_not_rendered_as_a_warning(app):
    """Metadata notice is not rendered as a warning."""
    result = DiscoveryResult(
        recommendations=[],
        evidence=[],
        warnings=[],
        notices=["MusicBrainz identity matches do not confirm musical fit."],
    )
    with patch("src.composition.find_music", return_value=result), \
            patch("src.recommender.load_songs", side_effect=OSError("missing")):
        submit(app, "instrumental jazz")

    assert not app.warning
    assert any("Metadata note" in element.value for element in app.caption)
    assert not app.exception


def test_empty_discovery_recommends_csv_catalog_songs_as_the_first_fallback(app):
    """Empty discovery shows every catalog song it can rank, not just one."""
    intent = ParsedIntent([
        TasteCriterion("genre", "preference", "pop", 2, "pop"),
        TasteCriterion("mood", "preference", "happy", 3, "happy"),
        TasteCriterion("energy", "preference", "high energy", 2, "high energy"),
    ], ["happy high energy pop"])
    video = Evidence("web_1", "web", "Happy pop mix", "upbeat", "https://www.youtube.com/watch?v=example")
    result = DiscoveryResult([], [video], ["No supported web match was found."], intent)

    with patch("src.composition.find_music", return_value=result):
        submit(app, "happy high energy pop")

    visible_copy = [element.value for kind in ("caption", "text", "markdown") for element in app.get(kind)]
    rendered = "\n".join(visible_copy)
    assert "COLLECTION FALLBACK 01" in rendered and "COLLECTION FALLBACK 05" in rendered
    assert "From our collection" in rendered and "Backup picks" in rendered
    assert "Sunrise City" in rendered
    assert "Neon Echo" in rendered
    assert "MATCH SCORE" in rendered
    assert "Why it fits" in rendered
    assert re.search(r"Energy similarity \d+%", rendered)
    assert "songs from our local collection" in rendered
    assert "RELATED VIDEO" not in rendered  # the catalog comes before the video
    assert not app.exception


def test_empty_discovery_recommends_related_youtube_video_with_friendly_note(app):
    """Empty discovery recommends related youtube video with friendly note."""
    intent = ParsedIntent([
        TasteCriterion("language", "must_have", "Korean", 3, "korean"),
        TasteCriterion("genre", "must_have", "Ballad", 3, "ballad"),
    ], ["Korean ballad songs"])
    result = DiscoveryResult([], [Evidence(
        "web_1", "web",
        "Romantic Korean Ballads | Korean love songs | K-Heartballads",
        "A range of soulful tracks that capture the essence of love.",
        "https://www.youtube.com/watch?v=example",
    )], ["No candidate had evidence supporting every required trait."], intent)

    with patch("src.composition.find_music", return_value=result), \
            patch("src.recommender.load_songs", side_effect=OSError("missing")):
        submit(app, "I want to hear a korean ballad")

    visible_copy = [element.value for kind in ("caption", "text", "markdown", "warning")
                    for element in app.get(kind)]
    rendered = "\n".join(visible_copy)
    assert "A video to try" in rendered
    assert "Romantic Korean Ballads" in rendered
    assert "RELATED VIDEO" in rendered
    assert "Here’s the closest listen we found" in rendered
    assert "may still be worth a listen" in rendered
    assert "No candidate had evidence" not in rendered
    assert not app.warning
    assert any(element.proto.label == "Watch on YouTube" for element in app.get("link_button"))
    assert "COLLECTION FALLBACK" not in rendered
    assert not app.exception


@pytest.mark.parametrize("error,message", [
    (DiscoveryError("quota", "Quota reached. Try later."), "Quota reached. Try later."),
    (RuntimeError("fixture-secret"), "Discovery could not finish. Please try again later."),
])
def test_safe_error_without_traceback_or_key(app, error, message):
    """Safe error without traceback or key."""
    with patch("src.composition.find_music", side_effect=error):
        submit(app, "Portuguese samba")
    assert app.error[0].value == message
    assert "fixture-secret" not in str(app)
    assert not app.exception


def test_discovery_always_uses_enhanced_retrieval_without_a_toggle(app):
    """Discovery always uses enhanced retrieval without a toggle."""
    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [])) as service:
        submit(app, "instrumental jazz")
    assert service.call_args.args[0].enhanced is True
    assert not app.checkbox
    assert not app.exception


def test_rapid_resubmission_is_throttled(app):
    """Rapid resubmission is throttled."""
    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [])) as service:
        submit(app, "instrumental jazz")
        app.button[0].click().run()
    assert service.call_count == 1
    assert any("wait a few seconds" in warning.value for warning in app.warning)
    assert not app.exception


def test_partial_matches_and_enhancement_warnings_are_explained(app):
    """Partial matches and enhancement warnings are explained."""
    result = DiscoveryResult(
        [Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.", ["web_1"], [],
                        fit_score=0.5)],
        [Evidence("web_1", "web", "Source title", "Evidence", "https://example.org/music")],
        ["Custom documents are unavailable; using external evidence.",
         "MusicBrainz is unavailable; recording versions remain uncertain."],
        used_partial_matches=True)
    with patch("src.composition.find_music", return_value=result):
        submit(app, "instrumental jazz")
    captions = [element.value for element in app.caption]
    assert any("closest matches" in value for value in captions)
    assert any("Some recording details could not be verified" in value for value in captions)
    assert [warning.value for warning in app.warning] == [
        "Custom documents are unavailable; using external evidence."]
    assert not app.exception


def test_intent_fallback_is_a_soft_note_not_a_warning(app):
    """Intent fallback is a soft note not a warning."""
    result = DiscoveryResult(
        [Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.", ["web_1"], [],
                        fit_score=0.9)],
        [Evidence("web_1", "web", "Source title", "Evidence", "https://example.org/music")],
        [INTENT_FALLBACK_NOTICE])
    with patch("src.composition.find_music", return_value=result):
        submit(app, "instrumental jazz")
    assert [info.value for info in app.info] == [INTENT_FALLBACK_NOTICE]
    assert not app.warning and not app.exception


def test_catalog_mode_works_without_api_calls(app):
    """Catalog mode works without api calls."""
    with patch("src.composition.find_music") as service:
        app.run()
        app.button[1].click().run()
    service.assert_not_called()
    assert len(ranked(app)) == 5 and len(match_cards(app)) == 5
    assert not app.exception


def discovery_result():
    """Return a small DiscoveryResult with one jazz recommendation."""
    return DiscoveryResult(
        [Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat jazz.", ["web_1"], [])],
        [Evidence("web_1", "web", "Source title", "Evidence", "https://example.org/music",
                  "2026-11-16T04:48:14.876050+00:00")],
        [], ParsedIntent([], []))


def test_catalog_ranks_on_energy_alone_when_genre_and_mood_are_blank(app):
    """Catalog ranks on energy alone when genre and mood are blank."""
    rank_catalog(app, "", "")
    assert len(ranked(app)) == 5 and len(match_cards(app)) == 5
    assert not app.warning and not app.exception
    assert "any genre" in summary_card(app) and "any mood" in summary_card(app)


def test_ranking_catalog_removes_previous_discovery_results(app):
    """Ranking catalog removes previous discovery results."""
    with patch("src.composition.find_music", return_value=discovery_result()):
        submit(app, "instrumental jazz")
    assert "Take Five" in str(app) and "discovery_result" in app.session_state

    app.button[1].click().run()

    assert "discovery_result" not in app.session_state
    assert "Take Five" not in str(app)
    assert len(ranked(app)) == 5 and len(match_cards(app)) == 5
    assert not app.exception


def test_finding_music_removes_previous_catalog_results(app):
    """Finding music removes previous catalog results."""
    app.run()
    app.button[1].click().run()
    assert match_cards(app) and "catalog_results" in app.session_state

    with patch("src.composition.find_music", return_value=discovery_result()):
        app.text_area[0].set_value("instrumental jazz")
        app.button[0].click().run()

    assert "catalog_results" not in app.session_state
    assert not match_cards(app)
    assert "Take Five" in str(app)
    assert not app.exception


def test_failed_discovery_does_not_leave_old_catalog_results(app):
    """Failed discovery does not leave old catalog results."""
    app.run()
    app.button[1].click().run()
    assert "catalog_results" in app.session_state

    with patch("src.composition.find_music") as service:
        app.text_area[0].set_value(" ")
        app.button[0].click().run()

    service.assert_not_called()
    assert "catalog_results" not in app.session_state
    assert not match_cards(app)
    assert app.warning


def ranked(app):
    """The (song, score, reasons) rows behind the catalog cards currently on the page."""
    return app.session_state["catalog_results"]["rows"]


def match_cards(app):
    """The caption of every catalog song card on the page."""
    return [caption.value for caption in app.caption if "MATCH SCORE" in caption.value]


def rank_catalog(app, genre, mood, strategy="current"):
    """Fill in the catalog ranking form with a genre, mood and strategy."""
    app.run()
    next(box for box in app.text_input if box.label == "Preferred genre").set_value(genre)
    next(box for box in app.text_input if box.label == "Preferred mood").set_value(mood)
    app.radio[0].set_value(strategy)
    next(button for button in app.button if button.label == "Browse collection").click().run()
    return [element.value for element in app.caption]


def test_catalog_cards_show_a_percentage_not_raw_points(app):
    """Catalog cards show a percentage not raw points."""
    captions = rank_catalog(app, "pop", "happy")
    cards = [value for value in captions if "MATCH SCORE" in value]
    assert len(cards) == 5
    assert all(re.search(r"\d+% MATCH SCORE", value) for value in cards)
    assert not any(re.search(r"\d\.\d\d MATCH", value) for value in cards)
    # Like Discover music, the results have no separate scoring table behind an expander.
    assert not app.dataframe
    assert not any("scoring details" in expander.label.lower() for expander in app.expander)


def test_brazilian_funk_finds_the_funk_songs_instead_of_energy_only_matches(app):
    """Brazilian funk finds the funk songs instead of energy only matches."""
    rank_catalog(app, "Brazilian funk", "happy", strategy="genre-first")
    summary = summary_card(app)
    assert "Brazilian funk → funk carioca, funk melody" in summary
    assert not app.warning and not app.info
    assert all("genre match" in reasons for _, _, reasons in ranked(app))


def summary_card(app):
    """The HTML of the catalog summary card shown above the ranked songs."""
    return next(element.value for element in app.markdown if 'class="catalog-summary"' in element.value)


def test_catalog_explains_when_the_genre_or_mood_matches_nothing(app):
    """Catalog explains when the genre or mood matches nothing."""
    rank_catalog(app, "brazilian jazz", "party")
    summary = summary_card(app)
    assert '<span class="is-ignored">brazilian jazz · no match</span>' in summary
    assert '<span class="is-ignored">party · no match</span>' in summary
    assert "No genre in our collection matches “brazilian jazz”" in summary
    assert "No song in our collection is labeled “party”" in summary
    assert not app.warning and not app.info


def test_the_summary_is_one_card_with_no_separate_banners(app):
    """The request, its notes and the song count share one card instead of stacked banners."""
    rank_catalog(app, "Brazilian funk", "calm", strategy="current")
    summary = summary_card(app)
    assert "5 songs found" in summary
    assert "calm · no match" in summary and "target energy" in summary
    assert not app.warning and not app.info and not app.error


def test_missing_server_configuration_is_safe(app, monkeypatch):
    """Missing server configuration is safe."""
    monkeypatch.delenv("TAVILY_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY")
    # A developer's real .streamlit/secrets.toml must not leak into this test.
    app.secrets["TAVILY_API_KEY"] = ""
    app.secrets["GEMINI_API_KEY"] = ""
    with patch("src.composition.find_music") as service:
        submit(app, "instrumental piano")
    service.assert_not_called()
    assert "Configure" in app.error[0].value
    assert not app.exception


def card_blocks(node):
    """Every song-card container on the page, found by its key."""
    found = []
    for child in getattr(node, "children", {}).values():
        if (getattr(child, "key", None) or "").startswith(("song_card_", "catalog_song_card_")):
            found.append(child)
        found.extend(card_blocks(child))
    return found


def test_song_cards_size_to_their_content_so_long_text_cannot_run_under_the_links(app):
    """Song cards size to their content so long text cannot run under the links."""
    # Cards used to have a fixed pixel height with the link buttons pinned to the bottom, so a
    # long title or an extra caption was clipped or drawn over the buttons. Cards must grow instead.
    intent = ParsedIntent([TasteCriterion("genre", "preference", "pop", 2, "pop")], ["pop songs"])
    with patch("src.composition.find_music", return_value=discovery_result()):
        submit(app, "instrumental jazz")
    discovery_cards = card_blocks(app.main)
    assert [card.key for card in discovery_cards] == ["song_card_1"]

    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [], intent)):
        app.session_state["last_attempt"] = float("-inf")  # skip the resubmission cooldown
        app.text_area[0].set_value("happy pop")
        app.button[0].click().run()
    fallback_cards = card_blocks(app.main)
    assert [card.key for card in fallback_cards] == [
        f"song_card_catalog_fallback_{rank}" for rank in range(1, 6)]

    app.button[1].click().run()
    catalog_cards = card_blocks(app.main)
    assert len(catalog_cards) == 5

    for card in discovery_cards + fallback_cards + catalog_cards:
        assert card.proto.height_config.use_content, f"{card.key} has a fixed height"


def test_catalog_preferences_translate_discovery_intent():
    """Catalog preferences translate discovery intent."""
    intent = ParsedIntent([
        TasteCriterion("genre", "must_avoid", "metal", 3, "no metal"),
        TasteCriterion("genre", "must_have", "Pop", 3, "pop"),
        TasteCriterion("mood", "preference", "calm", 2, "calm"),
        TasteCriterion("instrument", "preference", "piano", 1, "piano"),
        TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals"),
    ], ["calm pop piano"])
    assert catalog_preferences(intent) == {
        "genre": "pop", "mood": "calm", "energy": 0.3, "mood_tags": ["calm"],
        "instrumentation_tags": ["piano"], "language": ["instrumental"]}
    assert catalog_preferences(None)["energy"] == 0.5


@pytest.mark.parametrize("url,is_video", [
    ("https://www.youtube.com/watch?v=1", True),
    ("https://youtu.be/1", True),
    ("https://notyoutube.com/watch?v=1", False),
    ("https://youtube.com.evil.example/watch?v=1", False),
])
def test_video_fallback_only_accepts_real_youtube_hosts(url, is_video):
    """Video fallback only accepts real youtube hosts."""
    source = Evidence("web_1", "web", "Video", "text", url)
    assert (video_fallback_source([source]) is source) is is_video


def test_source_and_reason_copy_hide_markup_and_scoring_internals():
    """Source and reason copy hide markup and scoring internals."""
    assert readable_source_excerpt("## Title [link](http://x) ![img](http://y) more [...] text") == "Title link more text"
    assert compact_text("one two three four", 9) == "one two…"
    assert catalog_reason_text(
        "genre match (+1.0); artist repetition penalty (-1.50); energy similarity (+1.80)"
    ) == "Genre match · Energy similarity"
    assert catalog_reason_text("") == "Closest available catalog match"
    assert catalog_reason_text(
        "genre match (+1.0); mood match (+1.0); energy similarity (+1.80); language match (+1.0)",
        {"genre match": 1.0, "mood match": 1.0, "energy similarity": 0.9},
    ) == "Genre match 100% · Mood match 100% · Energy similarity 90% · Language match"


def test_catalog_cards_show_genre_mood_and_energy_as_percentages(app):
    """Each catalog card's "Why it fits" gives the genre, mood and energy match as a percentage."""
    captions = rank_catalog(app, "pop", "happy")
    reasons = [e.value for e in app.markdown if "catalog-reason\">" in e.value]
    assert reasons and all(re.search(r"Energy similarity \d+%", value) for value in reasons)
    assert any("Genre match 100%" in value and "Mood match 100%" in value for value in reasons)
    assert not any(re.search(r"\(\+\d", value) for value in reasons)  # no raw points


# --- text helpers --------------------------------------------------------------------------

def test_literal_caption_escapes_every_markdown_special_character():
    """Literal caption escapes every markdown special character."""
    for char in "\\`*_[]()$<>#~|!":
        assert literal_caption(f"a{char}b") == f"a\\{char}b"
    assert literal_caption("Take Five — Dave Brubeck, 1959.") == "Take Five — Dave Brubeck, 1959."


@pytest.mark.parametrize("value,limit,expected", [
    ("a   b\n c", 145, "a b c"),                         # whitespace collapsed, short text untouched
    ("exactly ten", 11, "exactly ten"),                  # text at the limit is not cut
    ("alpha beta gamma delta", 14, "alpha beta…"),       # cut back to a whole word
    ("x" * 200, 50, "x" * 50 + "…"),                     # nowhere to break, so cut at the limit
])
def test_compact_text_collapses_whitespace_and_cuts_at_a_word(value, limit, expected):
    """Compact text collapses whitespace and cuts at a word."""
    assert compact_text(value, limit) == expected


def test_source_kinds_get_listener_friendly_labels():
    """Source kinds get listener friendly labels."""
    kinds = ("web", "song_card", "guide", "metadata", "something-else")
    assert [source_kind_label(kind) for kind in kinds] == [
        "Website", "Curated song note", "Listening guide", "Recording details", "Source"]


@pytest.mark.parametrize("energy,label", [(0.2, "gentle"), (0.39, "gentle"), (0.4, "moderate"),
                                           (0.75, "moderate"), (0.76, "high"), (None, "balanced")])
def test_catalog_summary_labels_the_energy_level(energy, label):
    """Catalog summary labels the energy level."""
    assert catalog_song_summary({"mood": "happy", "genre": "pop", "energy": energy}) == (
        f"A happy pop track with {label} energy.")


def test_catalog_summary_closes_the_gap_left_by_a_missing_mood():
    """Catalog summary closes the gap left by a missing mood."""
    assert catalog_song_summary({"mood": "", "genre": "pop", "energy": 0.5}) == "A pop track with moderate energy."


# --- intent to catalog translation ---------------------------------------------------------

def one_criterion_intent(facet, value, kind="preference"):
    """Build a ParsedIntent holding a single criterion."""
    return ParsedIntent([TasteCriterion(facet, kind, value, 2, value)], ["x"])


@pytest.mark.parametrize("facet,value,expected", [
    ("energy", "intense", 0.85), ("energy", "High energy", 0.85), ("mood", "upbeat", 0.85),
    ("energy", "quiet", 0.3), ("mood", "peaceful", 0.3), ("energy", "relaxed", 0.3),
    ("energy", "moderate", 0.55), ("energy", "balanced", 0.55),
    ("energy", "unspecified", 0.5),
])
def test_catalog_energy_comes_from_energy_and_mood_words(facet, value, expected):
    """Catalog energy comes from energy and mood words."""
    assert catalog_preferences(one_criterion_intent(facet, value))["energy"] == expected


def test_a_high_energy_word_outranks_a_calm_one():
    """A high energy word outranks a calm one."""
    both = ParsedIntent([TasteCriterion("mood", "preference", "calm", 2, "calm"),
                         TasteCriterion("energy", "preference", "high", 2, "high")], ["x"])
    assert catalog_preferences(both)["energy"] == 0.85


@pytest.mark.parametrize("kind,value,expected", [
    ("preference", "Instrumental only", True),
    ("must_have", "no vocals", True),
    ("preference", "female vocals", False),
])
def test_catalog_language_gains_instrumental_when_the_listener_wants_no_singing(kind, value, expected):
    """Catalog language gains instrumental when the listener wants no singing."""
    languages = catalog_preferences(one_criterion_intent("vocals", value, kind))["language"]
    assert ("instrumental" in languages) is expected


def test_catalog_preferences_pass_languages_through_and_ignore_blank_values():
    """Catalog preferences pass languages through and ignore blank values."""
    intent = ParsedIntent([TasteCriterion("genre", "preference", "  ", 2, "x"),
                           TasteCriterion("language", "must_have", "Korean", 3, "korean")], ["x"])
    preferences = catalog_preferences(intent)
    assert preferences["genre"] == "" and preferences["language"] == ["korean"]


# --- catalog notes -------------------------------------------------------------------------

CATALOG_SONGS = [{"genre": genre, "subgenre": "", "mood": "happy", "language": []}
                 for genre in ("funk carioca", "rock", "pop")]


def summary(genre, mood, matched, rows=(), strategy="current", energy=0.5):
    """Return the catalog summary for a genre, mood and energy request."""
    return catalog_query_summary(genre, mood, energy, matched, list(rows), CATALOG_SONGS,
                                 RANKING_STRATEGIES[strategy])


def test_blank_genre_and_mood_show_as_any_and_the_energy_is_always_listed():
    """Blank genre and mood show as any and the energy is always listed."""
    assert summary("", "", [], energy=0.11) == {"tags": [
        ("any genre", "any"), ("any mood", "any"), ("11% target energy", "ok")], "lines": []}


def test_an_unmatched_genre_is_ignored_and_names_related_catalog_genres():
    """An unmatched genre is ignored and names related catalog genres."""
    result = summary("carioca", "", [])
    assert result["tags"][0] == ("carioca · no match", "ignored")
    assert result["lines"] == ["No genre in our collection matches “carioca”, so genre was ignored. "
                               "Related genres: funk carioca."]
    assert summary("zzz", "", [])["lines"] == [
        "No genre in our collection matches “zzz”, so genre was ignored."]  # nothing related to suggest


def test_a_genre_resolved_to_another_label_shows_the_label_but_an_exact_label_does_not():
    """A genre resolved to another label shows the label but an exact label does not."""
    assert summary("baile funk", "", ["funk carioca"])["tags"][0] == ("baile funk → funk carioca", "ok")
    assert summary("Pop", "", ["pop"])["tags"][0] == ("Pop", "ok")
    assert summary("Pop", "", ["pop"])["lines"] == []


def test_a_long_list_of_matched_genres_is_shortened():
    """A long list of matched genres is shortened."""
    matched = [f"genre {n}" for n in range(8)]
    assert summary("many", "", matched)["tags"][0] == ("many → genre 0, genre 1, genre 2 +5 more", "ok")


def test_results_outside_the_matched_genres_are_explained_unless_genre_already_ranks_first():
    """Results outside the matched genres are explained unless genre already ranks first."""
    rows = [({"genre": "funk carioca"}, 4.0, ""), ({"genre": "rock"}, 2.0, "")]
    assert summary("funk carioca", "", ["funk carioca"], rows)["lines"] == [
        "Only 1 of 2 songs are in funk carioca. Balanced ranking weighs mood and energy alongside "
        "genre; try Genre-First to put genre first."]
    assert summary("funk carioca", "", ["funk carioca"], rows, strategy="genre-first")["lines"] == []


def test_no_results_in_the_matched_genres_says_none_not_only_zero():
    """No results in the matched genres says none, not only zero."""
    rows = [({"genre": "rock"}, 2.0, ""), ({"genre": "pop"}, 1.0, "")]
    [line] = summary("funk", "", ["funk carioca", "funk melody"], rows)["lines"]
    assert line.startswith("None of these songs are in funk carioca, funk melody. ")
    assert "Only 0" not in line


def test_a_mood_no_catalog_song_has_is_ignored_whatever_its_case():
    """A mood no catalog song has is ignored whatever its case."""
    result = summary("", "Party", [])
    assert result["tags"][1] == ("Party · no match", "ignored")
    assert result["lines"] == ["No song in our collection is labeled “Party”, so mood was ignored."]
    assert summary("", "HAPPY", [])["tags"][1] == ("HAPPY", "ok")
    assert summary("", "HAPPY", [])["lines"] == []


def test_video_fallback_picks_the_first_web_page_hosted_on_youtube():
    """Video fallback picks the first web page hosted on youtube."""
    skipped = [Evidence("g", "guide", "Guide", "text"),
               Evidence("m", "metadata", "Meta", "text", "https://www.youtube.com/watch?v=meta"),
               Evidence("w1", "web", "Elsewhere", "text", "https://example.org/a")]
    first = Evidence("w2", "web", "First", "text", "https://www.youtube.com/watch?v=1")
    second = Evidence("w3", "web", "Second", "text", "https://youtu.be/2")
    assert video_fallback_source(skipped) is None
    assert video_fallback_source(skipped + [first, second]) is first


# --- server configuration ------------------------------------------------------------------

def test_streamlit_secrets_override_the_environment(monkeypatch):
    """Streamlit secrets override the environment."""
    monkeypatch.setenv("TAVILY_API_KEY", "env-tavily")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")
    with patch.object(app_module.st, "secrets", {
            "TAVILY_API_KEY": "secret-tavily", "YOUTUBE_API_KEY": "secret-yt",
            "GEMINI_EMBEDDING_MODEL": "secret-embedding",
            "SEMANTIC_RETRIEVAL_ENABLED": "false",
            "MUSIC_SOURCE_FOCUS_ENABLED": "false"}):
        settings = server_settings()
    assert (settings.tavily_api_key, settings.gemini_api_key, settings.youtube_api_key) == (
        "secret-tavily", "env-gemini", "secret-yt")
    assert settings.gemini_embedding_model == "secret-embedding"
    assert settings.semantic_retrieval_enabled is False
    assert settings.music_source_focus_enabled is False


@pytest.mark.parametrize("missing", [FileNotFoundError, app_module.st.errors.StreamlitSecretNotFoundError])
def test_a_missing_secrets_file_falls_back_to_the_environment(monkeypatch, missing):
    """A missing secrets file falls back to the environment."""
    monkeypatch.setenv("TAVILY_API_KEY", "env-tavily")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")

    class NoSecretsFile:
        """Stand-in for st.secrets when no secrets file exists."""
        def __contains__(self, name):
            """Raise the error Streamlit raises when no secrets file exists."""
            raise missing("no secrets")

    with patch.object(app_module.st, "secrets", NoSecretsFile()):
        settings = server_settings()
    assert (settings.tavily_api_key, settings.gemini_api_key) == ("env-tavily", "env-gemini")


# --- the page itself -----------------------------------------------------------------------

def html_of(app):
    """Join the app's markdown elements into one HTML string."""
    return "\n".join(element.value for element in app.markdown)


def test_empty_state_invites_a_first_search_until_something_is_shown(app):
    """Empty state invites a first search until something is shown."""
    app.run()
    assert 'class="empty-state"' in html_of(app)
    app.button[1].click().run()
    assert 'class="empty-state"' not in html_of(app)


def test_footer_explains_what_the_platform_links_do(app):
    """Footer explains what the platform links do."""
    app.run()
    assert any("exact track or video" in caption.value for caption in app.caption)


def test_forms_start_with_the_documented_defaults_and_limits(app):
    """Forms start with the documented defaults and limits."""
    app.run()
    assert (app.text_area[0].max_chars, app.text_input[0].max_chars) == (1200, 200)
    assert (app.text_input[1].value, app.text_input[2].value, app.slider[0].value) == ("", "", 0.7)
    assert (app.text_input[1].placeholder, app.text_input[2].placeholder) == ("pop", "happy")
    assert app.radio[0].options == ["Balanced (Default)", "Genre-First", "Mood-First", "Energy-Focused"]
    assert app.radio[0].value == "current"


def test_reference_song_is_sent_only_when_filled_in(app):
    """Reference song is sent only when filled in."""
    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [])) as service:
        app.run()
        app.text_area[0].set_value("instrumental jazz")
        app.text_input[0].set_value("So What — Miles Davis")
        app.button[0].click().run()
        app.session_state["last_attempt"] = float("-inf")  # skip the resubmission cooldown
        app.text_input[0].set_value("")
        app.button[0].click().run()
    assert [call.args[0] for call in service.call_args_list] == [
        MusicRequest("instrumental jazz", "So What — Miles Davis", True), MusicRequest("instrumental jazz", None, True)]


def test_rejected_input_does_not_start_the_cooldown(app):
    """Rejected input does not start the cooldown."""
    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [])) as service:
        submit(app, " ")
        app.text_area[0].set_value("instrumental jazz")
        app.button[0].click().run()
    assert service.call_count == 1
    assert not any("wait a few seconds" in warning.value for warning in app.warning)


def test_input_errors_from_the_service_are_warnings_not_errors(app):
    """Input errors from the service are warnings not errors."""
    with patch("src.composition.find_music", side_effect=DiscoveryError("invalid_input", "Try different words.")):
        submit(app, "instrumental jazz")
    assert [warning.value for warning in app.warning] == ["Try different words."] and not app.error


def test_an_unreadable_csv_is_reported_instead_of_crashing_the_catalog(app):
    """An unreadable csv is reported instead of crashing the catalog."""
    app.run()
    with patch("src.recommender.load_songs", side_effect=OSError("fixture-secret-csv-path")):
        app.button[1].click().run()
    assert [error.value for error in app.error] == [
        "The CSV catalog could not be read. Check its file format and location."]
    assert not match_cards(app) and "fixture-secret-csv-path" not in str(app) and not app.exception


def test_a_failing_catalog_fallback_degrades_to_the_plain_message(app):
    """A failing catalog fallback degrades to the plain message."""
    intent = ParsedIntent([TasteCriterion("genre", "preference", "pop", 2, "pop")], ["pop songs"])
    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [], intent)), \
            patch("src.recommender.load_songs", side_effect=OSError("missing")):
        submit(app, "happy pop")
    captions = [caption.value for caption in app.caption]
    assert "No evidence-backed matches are available for this request yet." in captions
    assert not any("COLLECTION FALLBACK" in caption for caption in captions) and not app.exception


def test_catalog_summary_names_the_request_and_counts_the_songs(app):
    """Catalog summary names the request and counts the songs."""
    rank_catalog(app, "", "")
    summary = summary_card(app)
    assert all(tag in summary for tag in ('<span class="is-any">any genre</span>',
                                          '<span class="is-any">any mood</span>',
                                          '<span class="is-ok">70% target energy</span>'))
    assert "5 songs found" in summary
    assert not app.warning and not app.exception


def test_catalog_input_is_trimmed_and_case_insensitive(app):
    """Catalog input is trimmed and case insensitive."""
    rank_catalog(app, "pop", "happy")
    expected = [song["title"] for song, _, _ in ranked(app)]
    rank_catalog(app, "  POP ", " Happy ")
    assert [song["title"] for song, _, _ in ranked(app)] == expected


def test_catalog_results_saved_by_an_older_version_are_discarded(app):
    """Catalog results saved by an older version are discarded."""
    app.session_state["catalog_results"] = {"rows": [], "genre": "pop", "mood": "happy", "energy": 0.7}
    app.run()
    assert "catalog_results" not in app.session_state and not app.exception
    assert 'class="empty-state"' in html_of(app)


def test_summary_lists_preferences_and_must_haves_separately_and_escapes_markup(app):
    """Summary lists preferences and must haves separately and escapes markup."""
    intent = ParsedIntent([TasteCriterion("mood", "preference", "<i>calm</i>", 2, "calm"),
                           TasteCriterion("language", "must_have", "korean", 3, "korean")], ["korean calm"])
    result = DiscoveryResult(
        [Recommendation("<script>alert(1)</script>", "A & B", "Why.", ["web_1"], [], fit_score=0.5)],
        [Evidence("web_1", "web", "<img src=x onerror=1>", "text", "https://example.org/a")], [], intent)
    with patch("src.composition.find_music", return_value=result):
        submit(app, "korean calm")
    html = html_of(app)
    assert "Your vibe" in html and "Your must-haves" in html and "Korean" in html
    assert "&lt;i&gt;calm&lt;/i&gt;" in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "A &amp; B" in html and "&lt;img src=x onerror=1&gt;" in html
    assert not any(raw in html for raw in ("<script>", "<img", "<i>"))
    assert not app.exception


def test_the_summary_sits_under_the_discover_form_not_with_the_results(app):
    """Your vibe and Your must-haves fill the space under Find music, in the left column."""
    intent = ParsedIntent([TasteCriterion("mood", "preference", "calm", 2, "calm"),
                           TasteCriterion("language", "must_have", "korean", 3, "korean")], ["korean calm"])
    result = DiscoveryResult(
        [Recommendation("Take Five", "Dave Brubeck Quartet", "Why.", ["web_1"], [], fit_score=0.5)],
        [Evidence("web_1", "web", "Source", "text", "https://example.org/a")], [], intent)
    with patch("src.composition.find_music", return_value=result):
        submit(app, "korean calm")
    discover, _, collection = app.columns[:3]
    assert "Your vibe" in html_of(discover) and "Your must-haves" in html_of(discover)
    assert "Your vibe" not in html_of(collection)
    assert html_of(app).count("Your vibe") == 1 and html_of(app).count("Your must-haves") == 1
    assert not app.exception


def test_the_summary_disappears_when_the_discovery_results_do(app):
    """A catalog search clears the discovery results, so their summary goes with them."""
    intent = ParsedIntent([TasteCriterion("mood", "preference", "calm", 2, "calm")], ["calm"])
    with patch("src.composition.find_music", return_value=DiscoveryResult([], [], [], intent)):
        submit(app, "calm")
    assert "Your vibe" in html_of(app)
    app.button[1].click().run()
    assert "Your vibe" not in html_of(app) and not app.exception


def test_unverified_songs_are_labeled_and_name_what_could_not_be_confirmed(app):
    """Songs no check could verify never look like matches: the card, the tags and the heading all say so."""
    intent = ParsedIntent([TasteCriterion("vocals", "must_have", "clean singing", 3, "clean singing"),
                           TasteCriterion("vocals", "must_avoid", "screaming", 3, "no screaming")], ["metal"])
    song = Recommendation("Du Hast", "Rammstein", "“Du Hast by Rammstein.”", ["web_1"],
                          ["We couldn't confirm clean singing.", "We couldn't confirm no screaming."],
                          [SupportedClaim("The source names it.", "web_1", "Du Hast by Rammstein")])
    warning = ("We couldn't confirm that any song meets every requirement (clean singing; no screaming). "
               "These songs are only named by their sources, so check them before you rely on them.")
    result = DiscoveryResult([song], [Evidence("web_1", "web", "Metal", "Du Hast by Rammstein", "https://example.org/a")],
                             [warning], intent, unverified_requirements=["clean singing", "no screaming"])
    with patch("src.composition.find_music", return_value=result), \
            patch("src.recommender.load_songs", side_effect=OSError("missing")):
        submit(app, "clean singing metal with no screaming")
    captions = [caption.value for caption in app.caption]
    assert any("NOT VERIFIED" in caption for caption in captions)
    assert not any("MATCH SCORE" in caption or "NOT SCORED" in caption for caption in captions)
    assert not any("details couldn’t be confirmed" in caption for caption in captions)  # the tags say it instead
    html = html_of(app)
    assert '<span class="is-ignored">clean singing</span>' in html
    assert '<span class="is-ignored">no screaming</span>' in html
    assert "Songs to check" in html and "Fresh finds" not in html
    assert [element.value for element in app.warning] == [warning]
    assert not app.exception


def test_catalog_songs_replace_unverified_songs_when_the_catalog_works(app):
    """Unverified songs only appear if the catalog fallback fails."""
    intent = ParsedIntent([TasteCriterion("vocals", "must_have", "clean singing", 3, "clean singing")], ["metal"])
    song = Recommendation("Du Hast", "Rammstein", "“Du Hast by Rammstein.”", ["web_1"], [])
    warning = "We couldn't confirm that any song meets every requirement (clean singing). Check them."
    result = DiscoveryResult([song], [Evidence("web_1", "web", "Metal", "Du Hast by Rammstein", "https://example.org/a")],
                             [warning], intent, unverified_requirements=["clean singing"])
    with patch("src.composition.find_music", return_value=result):
        submit(app, "clean singing metal")
    captions = [caption.value for caption in app.caption]
    assert any("COLLECTION FALLBACK 01" in caption for caption in captions)
    assert not any("NOT VERIFIED" in caption for caption in captions)
    assert "Du Hast" not in html_of(app) and not app.warning and not app.exception


def test_a_failed_discovery_falls_back_to_catalog_songs(app):
    """When Discover music errors out, the error is shown and so are catalog songs."""
    app.run()
    app.text_area[0].set_value("happy pop")
    with patch("src.composition.find_music", side_effect=DiscoveryError("quota", "Quota reached. Try later.")):
        app.button[0].click().run()
    assert [error.value for error in app.error] == ["Quota reached. Try later."]
    captions = [caption.value for caption in app.caption]
    assert any("COLLECTION FALLBACK 01" in caption for caption in captions)
    assert not app.exception


def test_invalid_input_does_not_trigger_the_catalog_fallback(app):
    """A request that fails validation is the listener's to fix, so no backup songs appear."""
    app.run()
    app.text_area[0].set_value("   ")
    app.button[0].click().run()
    assert not any("COLLECTION FALLBACK" in caption.value for caption in app.caption)
    assert not app.exception


def test_cards_pluralise_what_could_not_be_confirmed_and_trim_long_explanations(app):
    """Cards pluralise what could not be confirmed and trim long explanations."""
    long_text = " ".join(f"word{n}" for n in range(200))
    scored = Recommendation("Scored", "A", long_text, ["web_1"], ["one", "two"], fit_score=0.9)
    unscored = Recommendation("Unscored", "B", long_text, ["web_1"], [])
    result = DiscoveryResult([scored, unscored], [Evidence("web_1", "web", "Source", "text", "https://example.org/a")], [])
    with patch("src.composition.find_music", return_value=result):
        submit(app, "instrumental jazz")
    assert "2 details couldn’t be confirmed" in [caption.value for caption in app.caption]
    reasons = re.findall(r'class="catalog-reason">([^<]*)</div>', html_of(app))
    assert len(reasons) == 2
    assert 100 < len(reasons[0]) <= 146 and reasons[0].endswith("…")  # scored cards show about a line and a half
    assert 250 < len(reasons[1]) <= 301 and reasons[1].endswith("…")  # unscored cards quote the source at more length


def test_recommendations_are_numbered_in_order_with_their_own_scores(app):
    """Recommendations are numbered in order with their own scores."""
    recs = [Recommendation(f"Song {n}", "A", "Why.", ["web_1"], [], fit_score=1 - n / 10) for n in range(1, 5)]
    result = DiscoveryResult(recs, [Evidence("web_1", "web", "Source", "text", "https://example.org/a")], [])
    with patch("src.composition.find_music", return_value=result):
        submit(app, "instrumental jazz")
    headings = [caption.value for caption in app.caption if caption.value.startswith("DISCOVERY")]
    assert headings == [f"DISCOVERY 0{n}  ·  {100 - 10 * n}% MATCH SCORE" for n in range(1, 5)]


def test_why_these_songs_shows_claims_uncertainties_and_only_real_source_links(app):
    """Why these songs shows claims uncertainties and only real source links."""
    rec = Recommendation("Take Five", "Dave Brubeck Quartet", "Five-beat *jazz*.", ["guide_1", "web_1"],
                         ["Version unknown."], [SupportedClaim("Five beats", "web_1", "five-beat jazz")], fit_score=0.8)
    evidence = [Evidence("guide_1", "guide", "Pace", "Vocabulary only."),
                Evidence("web_1", "web", "Brubeck page", "five-beat jazz in 5/4", "https://example.org/brubeck")]
    with patch("src.composition.find_music", return_value=DiscoveryResult([rec], evidence, [])):
        submit(app, "instrumental jazz")
    captions, html = [caption.value for caption in app.caption], html_of(app)
    assert "• Version unknown." in captions and "“five-beat jazz”" in captions
    assert r"Five-beat \*jazz\*." in captions  # markup in model text is shown literally
    assert all(label in html for label in ("Why it may fit", "What we couldn’t confirm", "From the source",
                                           "Listening guide", "Website"))
    labels = [button.proto.label for button in app.get("link_button")]
    assert labels.count("View source: Brubeck page") == 1 and not any("Pace" in label for label in labels)
    assert labels.count("Read source") == 1  # the guide section has no page to open


def test_related_video_note_mentions_unconfirmed_recording_details(app):
    """Related video note mentions unconfirmed recording details."""
    video = Evidence("web_1", "web", "Romantic Korean Ballads", "soulful", "https://www.youtube.com/watch?v=example")
    result = DiscoveryResult([], [video], ["MusicBrainz is unavailable; recording versions remain uncertain."])
    with patch("src.composition.find_music", return_value=result), \
            patch("src.recommender.load_songs", side_effect=OSError("missing")):
        submit(app, "korean ballad")
    assert "We also couldn’t confirm every recording detail." in html_of(app)
    assert not app.warning and not app.exception
