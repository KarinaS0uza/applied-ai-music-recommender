"""Tests for the YouTube video lookup and its daily budget."""

import json
import logging
from unittest.mock import Mock, patch

import pytest
import requests

from src.composition import attach_videos, find_music
from src.models import DiscoveryResult, MusicRequest, Recommendation, VideoLink
from src.providers import (SearchBudget, Settings, TTLCache, YouTubeProvider,
                           pick_video)

KEYS = {"TAVILY_API_KEY": "tavily-secret", "GEMINI_API_KEY": "gemini-secret"}


class Response:
    """Fake HTTP response with canned data and a status code."""
    def __init__(self, data, status=200):
        """Store the data and status code."""
        self.data, self.status_code = data, status

    def raise_for_status(self):
        """Raise an HTTP error for failure status codes."""
        if self.status_code >= 400:
            raise requests.HTTPError("fixture-secret-do-not-log", response=self)

    def iter_content(self, size):
        """Yield the response body as a single chunk."""
        yield self.data if isinstance(self.data, bytes) else json.dumps(self.data).encode()

    def close(self):
        """Do nothing; the fake response holds no resources."""
        pass


class Session:
    """Fake HTTP session that records its calls and returns the queued responses in order."""
    def __init__(self, *responses):
        """Store the queued responses and start with no recorded calls."""
        self.responses, self.calls = list(responses), []

    def get(self, *args, **kwargs):
        """Record the call and return the next queued response."""
        self.calls.append((args, kwargs))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def item(video_id="dQw4w9WgXcQ", title="Take Five", channel="Dave Brubeck - Topic"):
    """Build a YouTube search result item."""
    return {"id": {"videoId": video_id}, "snippet": {"title": title, "channelTitle": channel}}


def video(id="dQw4w9WgXcQ", title="Take Five", channel="Dave Brubeck - Topic"):
    """Build a video dictionary with an id, title and channel."""
    return {"id": id, "title": title, "channel": channel}


def provider(*responses, budget=None):
    """Build a YouTubeProvider on a fake session and return both."""
    session = Session(*(responses or (Response({"items": [item()]}),)))
    return YouTubeProvider("yt-secret", session=session, cache=TTLCache(),
                           budget=budget or SearchBudget()), session


# --- settings ------------------------------------------------------------------------------

def test_youtube_key_is_optional_and_placeholders_disable_it():
    """Youtube key is optional and placeholders disable it."""
    assert Settings.from_mapping(KEYS).youtube_api_key == ""
    assert Settings.from_mapping({**KEYS, "YOUTUBE_API_KEY": "YOUR_YOUTUBE_API_KEY"}).youtube_api_key == ""
    settings = Settings.from_mapping({**KEYS, "YOUTUBE_API_KEY": " yt-secret "})
    assert settings.youtube_api_key == "yt-secret"
    assert "yt-secret" not in repr(settings)


# --- choosing the right video --------------------------------------------------------------

def test_pick_video_prefers_the_studio_upload_over_lyric_videos():
    """Pick video prefers the studio upload over lyric videos."""
    videos = [video("aaaaaaaaaaa", "Dave Brubeck Quartet - Take Five (Lyric Video)", "Some Channel"),
              video("bbbbbbbbbbb", "Take Five", "Dave Brubeck Quartet - Topic"),
              video("ccccccccccc", "Take Five", "Dave Brubeck")]
    assert pick_video("Take Five", "Dave Brubeck Quartet", videos)["id"] == "bbbbbbbbbbb"


@pytest.mark.parametrize("title", [
    "Take Five - Dave Brubeck Quartet (Cover)", "Take Five Dave Brubeck Quartet karaoke",
    "Take Five Dave Brubeck Quartet live in Berlin", "Take Five Dave Brubeck Quartet (Slowed + Reverb)",
    "Take Five Dave Brubeck Quartet - 1 hour loop", "Best of Dave Brubeck Quartet: Take Five mix",
])
def test_pick_video_rejects_other_recordings(title):
    """Pick video rejects other recordings."""
    assert pick_video("Take Five", "Dave Brubeck Quartet", [video(title=title, channel="Random")]) is None


def test_pick_video_keeps_variant_words_the_song_itself_contains():
    """Pick video keeps variant words the song itself contains."""
    videos = [video(title="Live and Let Die", channel="Paul McCartney & Wings - Topic")]
    assert pick_video("Live and Let Die", "Paul McCartney & Wings", videos) is not None


def test_pick_video_requires_both_the_title_and_the_artist():
    """Pick video requires both the title and the artist."""
    assert pick_video("Take Five", "Dave Brubeck Quartet", [video(title="Take Five", channel="Other Band")]) is None
    assert pick_video("Take Five", "Dave Brubeck Quartet",
                      [video(title="Dave Brubeck Quartet - So What", channel="Dave Brubeck Quartet")]) is None


def test_pick_video_accepts_the_artist_in_the_title_and_ignores_featured_credits():
    """Pick video accepts the artist in the title and ignores featured credits."""
    videos = [video(title="AC/DC - Thunderstruck (Official Video)", channel="acdcVEVO")]
    assert pick_video("Thunderstruck", "AC/DC", videos) is not None
    assert pick_video("Hello", "Adele feat. Someone", [video(title="Adele - Hello", channel="Adele")]) is not None


def test_pick_video_returns_none_when_nothing_matches():
    """Pick video returns none when nothing matches."""
    assert pick_video("Take Five", "Dave Brubeck Quartet", []) is None
    assert pick_video("", "", [video()]) is None


# --- the provider --------------------------------------------------------------------------

def test_provider_searches_music_videos_and_builds_the_url_from_the_id():
    """Provider searches music videos and builds the url from the id."""
    yt, session = provider(Response({"items": [item(title="Take Five &amp; Friends", channel="Dave Brubeck - Topic")]}))
    link = yt.find_videos([("Take Five &amp; Friends".replace("&amp;", "&"), "Dave Brubeck")])[0]
    assert isinstance(link, VideoLink) and link.title == "Take Five & Friends"
    assert link.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    (args, kwargs), = session.calls
    assert args == ("https://www.googleapis.com/youtube/v3/search",)
    assert kwargs["params"]["type"] == "video" and kwargs["params"]["videoCategoryId"] == "10"
    assert kwargs["params"]["maxResults"] == 5 and kwargs["params"]["key"] == "yt-secret"
    assert kwargs["timeout"] == (5, 20) and kwargs["allow_redirects"] is False


def test_matches_and_misses_are_both_cached():
    """Matches and misses are both cached."""
    yt, session = provider(Response({"items": [item()]}))
    assert yt.find_videos([("Take Five", "Dave Brubeck")])[0]
    assert yt.find_videos([("take five", "DAVE BRUBECK")])[0]
    miss, miss_session = provider(Response({"items": []}))
    assert miss.find_videos([("Nothing", "Nobody")]) == [None]
    assert miss.find_videos([("Nothing", "Nobody")]) == [None]
    assert len(session.calls) == 1 and len(miss_session.calls) == 1


def test_malformed_items_and_unsafe_video_ids_are_dropped():
    """Malformed items and unsafe video ids are dropped."""
    items = [item("<script>alert(1)</script>"), {"id": {}}, "junk", item("short"), item("bbbbbbbbbbb")]
    yt, _ = provider(Response({"items": items}))
    assert yt.find_videos([("Take Five", "Dave Brubeck")])[0].video_id == "bbbbbbbbbbb"


@pytest.mark.parametrize("data", [[], {"items": "x"}, {"no": "items"}, b"not json"])
def test_unreadable_responses_degrade_to_no_video(data):
    """Unreadable responses degrade to no video."""
    yt, _ = provider(Response(data))
    assert yt.find_videos([("Take Five", "Dave Brubeck")]) == [None]


def test_lookup_is_limited_and_results_stay_aligned_with_the_songs():
    """Lookup is limited and results stay aligned with the songs."""
    yt, session = provider(Response({"items": [item()]}))
    songs = [(f"Song {n}", "Artist") for n in range(8)] + [("Take Five", "Dave Brubeck")]
    results = yt.find_videos(songs)
    assert len(results) == len(songs) and len(session.calls) == 5
    assert all(result is None for result in results[5:])


def test_a_failure_stops_new_calls_but_cached_songs_still_resolve():
    """A failure stops new calls but cached songs still resolve."""
    yt, session = provider(Response({"items": [item()]}), Response({}, status=500))
    assert yt.find_videos([("Take Five", "Dave Brubeck")])[0]
    results = yt.find_videos([("Other", "Someone"), ("Take Five", "Dave Brubeck"), ("Third", "Song")])
    assert results[0] is None and results[1] is not None and results[2] is None
    assert len(session.calls) == 2  # the 500 ended API calls for the rest of that run


@pytest.mark.parametrize("status", [403, 429])
def test_quota_or_key_rejection_blocks_further_calls(status):
    """Quota or key rejection blocks further calls."""
    now = [1_000_000.0]
    budget = SearchBudget(clock=lambda: now[0])
    yt, session = provider(Response({}, status=status), budget=budget)
    assert yt.find_videos([("Take Five", "Dave Brubeck")]) == [None]
    assert yt.find_videos([("Another", "Song")]) == [None]
    assert len(session.calls) == 1
    now[0] += 3601
    yt.find_videos([("Third", "Song")])
    assert len(session.calls) == 2


def test_failures_never_expose_the_api_key(caplog):
    """Failures never expose the api key."""
    class Broken:
        """Fake session whose requests fail with an error that mentions the key."""
        def get(self, *args, **kwargs):
            """Raise a connection error whose message contains the fake key."""
            raise requests.ConnectionError("https://example/?key=yt-secret-do-not-log")

    yt = YouTubeProvider("yt-secret-do-not-log", session=Broken(), cache=TTLCache(), budget=SearchBudget())
    with caplog.at_level(logging.INFO):
        assert yt.find_videos([("Take Five", "Dave Brubeck")]) == [None]
        yt.session = Session(Response({}, status=403))
        yt.find_videos([("Another", "Song")])
    assert "secret" not in caplog.text and "youtube_lookup_failed" in caplog.text


def test_an_overrun_time_budget_skips_the_remaining_songs():
    """An overrun time budget skips the remaining songs."""
    ticks = iter([0, 0, 99, 99, 99])
    yt, session = provider()
    results = yt.find_videos([("Take Five", "Dave Brubeck"), ("So What", "Miles Davis")],
                             time_budget=10, clock=lambda: next(ticks))
    assert results[0] is not None and results[1] is None and len(session.calls) == 1


# --- the daily search budget ---------------------------------------------------------------

def test_budget_stops_before_the_daily_limit_and_resets_each_pacific_day():
    """Budget stops before the daily limit and resets each pacific day."""
    now = [10 * 86400 + 8 * 3600 + 60.0]  # just after midnight Pacific
    budget = SearchBudget(limit=2, clock=lambda: now[0])
    assert [budget.take() for _ in range(3)] == [True, True, False]
    now[0] += 86400
    assert budget.take()


def test_budget_block_expires():
    """Budget block expires."""
    now = [0.0]
    budget = SearchBudget(clock=lambda: now[0])
    budget.block(60)
    assert not budget.take()
    now[0] = 61
    assert budget.take()


# --- composition ---------------------------------------------------------------------------

def recs(*names):
    """Build a bare recommendation for each given title."""
    return [Recommendation(name, "Artist", "Why.", [], []) for name in names]


def test_attach_videos_only_changes_songs_with_a_confident_match():
    """Attach videos only changes songs with a confident match."""
    link = VideoLink("dQw4w9WgXcQ", "Take Five", "Dave Brubeck")
    youtube = Mock(find_videos=Mock(return_value=[link, None]))
    result = attach_videos(DiscoveryResult(recs("Take Five", "So What"), [], []), youtube)
    assert [rec.video for rec in result.recommendations] == [link, None]
    youtube.find_videos.assert_called_once_with([("Take Five", "Artist"), ("So What", "Artist")])


def test_attach_videos_never_costs_the_user_their_results():
    """Attach videos never costs the user their results."""
    original = recs("Take Five")
    result = attach_videos(DiscoveryResult(original, [], []), Mock(find_videos=Mock(side_effect=ValueError)))
    assert result.recommendations == original


def run_find_music(settings, discovered=None):
    """Run find_music with the providers patched out."""
    discovered = discovered or DiscoveryResult(recs("Take Five"), [], [])
    link = VideoLink("dQw4w9WgXcQ", "Take Five", "Dave Brubeck")
    with patch("src.composition.GeminiProvider"), patch("src.composition.TavilyProvider"), \
            patch("src.composition.discover_music", return_value=discovered), \
            patch("src.composition.YouTubeProvider") as youtube:
        youtube.return_value.find_videos.return_value = [link]
        result = find_music(MusicRequest("jazz"), settings)
    return result, youtube


def test_find_music_attaches_videos_when_a_youtube_key_is_configured():
    """Find music attaches videos when a youtube key is configured."""
    result, youtube = run_find_music(Settings.from_mapping({**KEYS, "YOUTUBE_API_KEY": "yt-secret"}))
    youtube.assert_called_once_with("yt-secret")
    assert result.recommendations[0].video.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    youtube.return_value.session.close.assert_called_once()


def test_find_music_skips_youtube_without_a_key():
    """Find music skips youtube without a key."""
    result, youtube = run_find_music(Settings.from_mapping(KEYS))
    youtube.assert_not_called()
    assert result.recommendations[0].video is None


def test_find_music_spends_no_youtube_quota_when_there_are_no_recommendations():
    """Find music spends no youtube quota when there are no recommendations."""
    empty = DiscoveryResult([], [], ["Nothing relevant was found."])
    result, youtube = run_find_music(Settings.from_mapping({**KEYS, "YOUTUBE_API_KEY": "yt-secret"}), empty)
    youtube.return_value.find_videos.assert_not_called()
    assert result is empty
    youtube.return_value.session.close.assert_called_once()  # the provider is still cleaned up
