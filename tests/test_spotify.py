"""Tests for the Spotify track lookup, its settings and how links are rendered."""

import json
from unittest.mock import Mock, patch

import requests

from src.app import platform_links_html
from src.composition import attach_tracks, find_music
from src.models import DiscoveryResult, MusicRequest, Recommendation, TrackLink
from src.providers import (Settings, SearchBudget, SpotifyProvider, TTLCache, pick_track)

KEYS = {"TAVILY_API_KEY": "tavily-secret", "GEMINI_API_KEY": "gemini-secret"}
SPOTIFY = {"SPOTIFY_CLIENT_ID": "client-id", "SPOTIFY_CLIENT_SECRET": "client-secret"}
TRACK_ID = "11dFghVXANMlKmJXsNCbNl"


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
        yield json.dumps(self.data).encode()

    def close(self):
        """Do nothing; the fake response holds no resources."""


class Session:
    """Fake HTTP session: token requests get a token, searches get the queued responses."""
    def __init__(self, *searches, token=None):
        """Queue the search responses and the token response."""
        self.searches = list(searches)
        self.token = token or Response({"access_token": "token-value", "expires_in": 3600})
        self.calls = []

    def post(self, url, **kwargs):
        """Record the token request and return the token response."""
        self.calls.append(("post", url, kwargs))
        return self.token

    def get(self, url, **kwargs):
        """Record the search and return the next queued response."""
        self.calls.append(("get", url, kwargs))
        return self.searches.pop(0) if len(self.searches) > 1 else self.searches[0]


def item(track_id=TRACK_ID, name="Take Five", artists=("The Dave Brubeck Quartet",)):
    """Build a Spotify search result item."""
    return {"id": track_id, "name": name, "artists": [{"name": a} for a in artists]}


def search(*items):
    """Build a Spotify search response."""
    return Response({"tracks": {"items": list(items)}})


def provider(*searches, **session_args):
    """Build a SpotifyProvider on a fake session with fresh caches, and return both."""
    session = Session(*(searches or (search(item()),)), **session_args)
    return SpotifyProvider("client-id", "client-secret", session=session, cache=TTLCache(),
                           token_cache=TTLCache(), budget=SearchBudget()), session


# --- settings ------------------------------------------------------------------------------

def test_spotify_needs_both_halves_of_the_credential_pair():
    """Spotify needs both halves of the credential pair."""
    settings = Settings.from_mapping({**KEYS, **SPOTIFY})
    assert (settings.spotify_client_id, settings.spotify_client_secret) == ("client-id", "client-secret")
    for partial in ({"SPOTIFY_CLIENT_ID": "client-id"}, {"SPOTIFY_CLIENT_SECRET": "client-secret"},
                    {**SPOTIFY, "SPOTIFY_CLIENT_SECRET": "YOUR_SPOTIFY_CLIENT_SECRET"}, {}):
        settings = Settings.from_mapping({**KEYS, **partial})
        assert (settings.spotify_client_id, settings.spotify_client_secret) == ("", "")


def test_spotify_credentials_never_appear_in_the_settings_repr():
    """Spotify credentials never appear in the settings repr."""
    assert "client-secret" not in repr(Settings.from_mapping({**KEYS, **SPOTIFY}))


# --- matching ------------------------------------------------------------------------------

def track(id=TRACK_ID, title="Take Five", artists=("The Dave Brubeck Quartet",)):
    """Build a parsed track dictionary."""
    return {"id": id, "title": title, "artists": list(artists)}


def test_pick_track_matches_the_song_and_artist():
    """Pick track matches the song and artist."""
    assert pick_track("Take Five", "Dave Brubeck Quartet", [track()])["id"] == TRACK_ID


def test_pick_track_ignores_reissue_suffixes_but_not_other_versions():
    """Pick track ignores reissue suffixes but not other versions."""
    remaster = track(title="Take Five - Remastered 2013")
    assert pick_track("Take Five", "Dave Brubeck Quartet", [remaster]) == remaster
    for title in ("Take Five - Live", "Take Five (Remix)", "Take Five - Karaoke Version"):
        assert pick_track("Take Five", "Dave Brubeck Quartet", [track(title=title)]) is None


def test_pick_track_rejects_the_wrong_artist_and_prefers_the_exact_title():
    """Pick track rejects the wrong artist and prefers the exact title."""
    assert pick_track("Take Five", "Dave Brubeck Quartet", [track(artists=("Someone Else",))]) is None
    exact = track(id="2" * 22)
    assert pick_track("Take Five", "Dave Brubeck Quartet",
                      [track(title="Take Five Again"), exact]) == exact


def test_pick_track_ignores_a_featured_credit_in_the_request():
    """Pick track ignores a featured credit in the request."""
    found = track(title="Song", artists=("Main Artist", "Guest"))
    assert pick_track("Song", "Main Artist feat. Guest", [found]) == found


# --- provider ------------------------------------------------------------------------------

def test_find_tracks_uses_client_credentials_and_returns_a_rebuilt_url():
    """Find tracks uses client credentials and returns a rebuilt url."""
    spotify, session = provider()
    [link] = spotify.find_tracks([("Take Five", "Dave Brubeck Quartet")])
    assert link == TrackLink(TRACK_ID, "Take Five", "The Dave Brubeck Quartet")
    assert link.url == f"https://open.spotify.com/track/{TRACK_ID}"
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("post", "https://accounts.spotify.com/api/token")
    assert kwargs["auth"] == ("client-id", "client-secret")
    assert kwargs["data"] == {"grant_type": "client_credentials"}
    _, url, kwargs = session.calls[1]
    assert url == "https://api.spotify.com/v1/search"
    assert kwargs["headers"] == {"Authorization": "Bearer token-value"}
    assert kwargs["params"]["type"] == "track" and kwargs["params"]["limit"] == 5


def test_find_tracks_requests_a_token_once_and_caches_answers():
    """Find tracks requests a token once and caches answers."""
    spotify, session = provider()
    spotify.find_tracks([("Take Five", "Dave Brubeck Quartet"), ("Take Five", "Dave Brubeck Quartet"),
                         ("So What", "Miles Davis")])
    assert [call[0] for call in session.calls].count("post") == 1
    assert [call[0] for call in session.calls].count("get") == 2  # the repeated song came from the cache


def test_find_tracks_leaves_an_unsure_song_without_a_link():
    """Find tracks leaves an unsure song without a link."""
    spotify, _ = provider(search(item(name="Something Else")))
    assert spotify.find_tracks([("Take Five", "Dave Brubeck Quartet")]) == [None]


def test_find_tracks_skips_malformed_items():
    """Find tracks skips malformed items."""
    bad = [{"id": "short", "name": "Take Five", "artists": [{"name": "Dave Brubeck Quartet"}]},
           {"id": TRACK_ID}, "nonsense", item()]
    spotify, _ = provider(search(*bad))
    [link] = spotify.find_tracks([("Take Five", "Dave Brubeck Quartet")])
    assert link.track_id == TRACK_ID


def test_a_rejected_secret_never_raises_leaks_or_retries_every_song(caplog):
    """A rejected secret never raises, leaks or retries every song."""
    spotify, session = provider(token=Response({}, 401))
    with caplog.at_level("INFO"):
        assert spotify.find_tracks([("A", "X"), ("B", "Y")]) == [None, None]
    assert [call[0] for call in session.calls] == ["post"]  # no more calls after the first failure
    assert "client-secret" not in caplog.text and "fixture-secret-do-not-log" not in caplog.text
    assert spotify.find_tracks([("C", "Z")]) == [None]  # and the budget stays blocked afterwards
    assert len(session.calls) == 1


def test_rate_limiting_blocks_further_searches():
    """Rate limiting blocks further searches."""
    spotify, session = provider(Response({}, 429))
    assert spotify.find_tracks([("A", "X")]) == [None]
    assert spotify.find_tracks([("B", "Y")]) == [None]
    assert [call[0] for call in session.calls].count("get") == 1


def test_an_unreadable_search_response_gives_no_link():
    """An unreadable search response gives no link."""
    spotify, _ = provider(Response({"unexpected": True}))
    assert spotify.find_tracks([("Take Five", "Dave Brubeck Quartet")]) == [None]


def test_find_tracks_only_looks_up_the_first_five_songs():
    """Find tracks only looks up the first five songs."""
    spotify, session = provider()
    results = spotify.find_tracks([(f"Song {n}", "Artist") for n in range(8)])
    assert len(results) == 8 and [call[0] for call in session.calls].count("get") == 5


# --- composition ---------------------------------------------------------------------------

def recs(*names):
    """Build a bare recommendation for each given title."""
    return [Recommendation(name, "Artist", "Why.", [], []) for name in names]


def test_attach_tracks_only_changes_songs_with_a_confident_match():
    """Attach tracks only changes songs with a confident match."""
    link = TrackLink(TRACK_ID, "Take Five", "Dave Brubeck")
    spotify = Mock(find_tracks=Mock(return_value=[link, None]))
    result = attach_tracks(DiscoveryResult(recs("Take Five", "So What"), [], []), spotify)
    assert [rec.track for rec in result.recommendations] == [link, None]


def test_attach_tracks_never_costs_the_user_their_results():
    """Attach tracks never costs the user their results."""
    original = recs("Take Five")
    result = attach_tracks(DiscoveryResult(original, [], []), Mock(find_tracks=Mock(side_effect=ValueError)))
    assert result.recommendations == original


def run_find_music(settings, discovered=None):
    """Run find_music with the providers patched out."""
    discovered = discovered or DiscoveryResult(recs("Take Five"), [], [])
    link = TrackLink(TRACK_ID, "Take Five", "Dave Brubeck")
    with patch("src.composition.GeminiProvider"), patch("src.composition.TavilyProvider"), \
            patch("src.composition.discover_music", return_value=discovered), \
            patch("src.composition.SpotifyProvider") as spotify:
        spotify.return_value.find_tracks.return_value = [link]
        result = find_music(MusicRequest("jazz"), settings)
    return result, spotify


def test_find_music_attaches_tracks_when_spotify_is_configured():
    """Find music attaches tracks when spotify is configured."""
    result, spotify = run_find_music(Settings.from_mapping({**KEYS, **SPOTIFY}))
    spotify.assert_called_once_with("client-id", "client-secret")
    assert result.recommendations[0].track.url == f"https://open.spotify.com/track/{TRACK_ID}"
    spotify.return_value.session.close.assert_called_once()


def test_find_music_skips_spotify_without_credentials():
    """Find music skips spotify without credentials."""
    result, spotify = run_find_music(Settings.from_mapping(KEYS))
    spotify.assert_not_called()
    assert result.recommendations[0].track is None


def test_find_music_makes_no_spotify_calls_when_there_are_no_recommendations():
    """Find music makes no spotify calls when there are no recommendations."""
    empty = DiscoveryResult([], [], ["Nothing relevant was found."])
    result, spotify = run_find_music(Settings.from_mapping({**KEYS, **SPOTIFY}), empty)
    spotify.return_value.find_tracks.assert_not_called()
    assert result is empty
    spotify.return_value.session.close.assert_called_once()


# --- rendering -----------------------------------------------------------------------------

def test_matched_track_replaces_only_the_spotify_search_link():
    """Matched track replaces only the spotify search link."""
    link = TrackLink(TRACK_ID, 'Take "Five" <b>', "Dave Brubeck")
    html = platform_links_html("Take Five", "Dave Brubeck", None, link)
    assert f'href="https://open.spotify.com/track/{TRACK_ID}"' in html
    assert "open.spotify.com/search/" not in html and 'title="Listen on Spotify"' in html
    assert 'aria-label="Listen to Take &quot;Five&quot; &lt;b&gt; by Dave Brubeck on Spotify"' in html
    assert "youtube.com/results?search_query=" in html  # YouTube keeps its own fallback
