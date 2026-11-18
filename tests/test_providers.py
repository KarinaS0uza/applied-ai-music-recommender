"""Tests for the Tavily, Gemini and MusicBrainz providers and their helpers."""

# Behavior: replaces HTTP sessions and the Gemini client with fakes, so no test touches the network.

import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests

from src.providers import (DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, EMBEDDING_DIMENSIONS,
                           GEMINI_RETRY_PAUSE, GeminiEmbeddingProvider, GeminiProvider,
                           MUSIC_SOURCE_DOMAINS, MusicBrainzProvider, RateLimiter, Settings,
                           TavilyProvider, TTLCache,
                           http_json, optional_text_field, provider_error, strings, text_field)
from src.models import (CHECK_STATUSES, CRITERION_KINDS, INTENT_FACETS, MAX_CANDIDATES, MAX_JUDGED, Candidate,
                        ConstraintCheck, DiscoveryError, Evidence, MusicRequest, ParsedIntent,
                        PreferenceCheck, Recommendation, SupportedClaim, TasteCriterion)


class Response:
    """Fake HTTP response with canned data and a status code."""
    def __init__(self, data, status=200):
        """Store the data and status code."""
        self.data, self.status_code = data, status
    def raise_for_status(self):
        """Raise an HTTP error for failure status codes."""
        if self.status_code >= 400:
            raise requests.HTTPError("fixture-secret", response=self)
    def iter_content(self, size):
        """Yield the response body as a single chunk."""
        yield self.data if isinstance(self.data, bytes) else json.dumps(self.data).encode()
    def close(self):
        """Do nothing; the fake response holds no resources."""
        pass


class Session:
    """Fake HTTP session that records its calls and returns the same response."""
    def __init__(self, data):
        """Store the response data and start with no recorded calls."""
        self.data, self.calls = data, []
    def get(self, *args, **kwargs):
        """Record the call and return the canned response."""
        self.calls.append((args, kwargs))
        return Response(self.data)
    post = get


def jazz_intent():
    """Return a ParsedIntent with a jazz preference."""
    return ParsedIntent(
        [TasteCriterion("genre", "preference", "jazz", 2, "jazz")],
        ["jazz songs", "essential jazz recordings"],
    )


def test_tavily_is_bounded_and_cached():
    """Tavily is bounded and cached."""
    session = Session({"results": [{"title": "Take Five", "content": "Dave Brubeck Quartet jazz",
                                    "url": "https://example.org/music"}]})
    provider = TavilyProvider("fixture", session=session, cache=TTLCache())
    request = MusicRequest("jazz", "So What Miles Davis")
    intent = jazz_intent()
    assert len(provider.search_web(request, intent)) == 1
    provider.search_web(request, intent)
    assert len(session.calls) == 2
    assert all(c[1]["json"]["search_depth"] == "basic" and not c[1]["json"]["include_answer"]
               for c in session.calls)
    assert all(c[1]["timeout"] == (5, 20) for c in session.calls)


def test_cache_expiry_and_capacity():
    """Cache expiry and capacity."""
    now = [0]
    cache = TTLCache(ttl=10, capacity=1, clock=lambda: now[0])
    cache.put("a", "first")
    cache.put("b", "second")
    assert cache.get("a") is None
    assert cache.get("b") == "second"
    now[0] = 11
    assert cache.get("b") is None


@pytest.mark.parametrize("status,code", [(401, "authentication"), (403, "authentication"), (429, "quota"), (503, "provider_unavailable"), (302, "invalid_response")])
def test_safe_provider_status_mapping(status, code):
    """Safe provider status mapping."""
    class FailingSession:
        """Fake session that answers with a failing status."""
        def post(self, *args, **kwargs):
            """Return a response with the failing status."""
            return Response({"results": []}, status)
    with pytest.raises(DiscoveryError) as exc:
        TavilyProvider("fixture-secret", session=FailingSession(), cache=TTLCache()).search_web(
            MusicRequest("jazz"), jazz_intent())
    assert exc.value.code == code
    assert "fixture-secret" not in str(exc.value)


@pytest.mark.parametrize("outcome,code", [
    (Response({"results": "oops"}), "invalid_response"),
    (Response(b"x" * 1_000_001), "invalid_response"),
    (Response(b"not json"), "provider_unavailable"),
    (requests.Timeout(), "timeout"),
], ids=["no-result-list", "oversized", "not-json", "timeout"])
def test_tavily_maps_unusable_responses_to_safe_errors(outcome, code):
    """Tavily maps unusable responses to safe errors."""
    class OneShotSession:
        """Fake session that returns or raises one fixed outcome."""
        def post(self, *args, **kwargs):
            """Raise the outcome if it is an error, otherwise return it."""
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
    with pytest.raises(DiscoveryError) as exc:
        TavilyProvider("fixture", session=OneShotSession(), cache=TTLCache()).search_web(
            MusicRequest("jazz"), jazz_intent())
    assert exc.value.code == code


def recording(id="10000000-0000-0000-0000-000000000001", version="studio"):
    """Build a MusicBrainz recording payload for Take Five."""
    return {"id": id, "title": "Take Five", "disambiguation": version,
            "artist-credit": [{"name": "Dave Brubeck Quartet", "artist": {"id": "artist-id"}}],
            "releases": [{"title": "Time Out", "date": "1959"}], "length": 328000}


def test_metadata_keeps_recording_versions_distinct_and_does_not_prove_tone():
    """Metadata keeps recording versions distinct and does not prove tone."""
    session = Session({"recordings": [recording(), recording("10000000-0000-0000-0000-000000000002", "live")]})
    provider = MusicBrainzProvider(session=session, cache=TTLCache(), limiter=RateLimiter())
    evidence = provider.lookup_recordings([Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])])
    data = json.loads(evidence[0].text)
    assert data["status"] == "ambiguous"
    assert len({r["id"] for r in data["recordings"]}) == 2
    assert {r["disambiguation"] for r in data["recordings"]} == {"live", "studio"}
    assert not {"mood", "bpm", "vocal_tone"}.intersection(data["recordings"][0])
    assert "MusicDiscovery/1.0" in session.calls[0][1]["headers"]["User-Agent"]


def test_missing_recording_is_unknown_and_wrong_artist_not_confirmed():
    """Missing recording is unknown and wrong artist not confirmed."""
    provider = MusicBrainzProvider(session=Session({"recordings": [recording()]}),
                                  cache=TTLCache(), limiter=RateLimiter())
    assert provider.lookup_recordings([Candidate("Take Five", "Unrelated Artist", ["web_1"])]) == []


def test_musicbrainz_rate_limit_across_instances_and_cache():
    """Musicbrainz rate limit across instances and cache."""
    now, starts = [0.0], []
    def sleep(delay):
        """Advance the fake clock instead of sleeping."""
        now[0] += delay
    class TimedSession(Session):
        """Session that records the fake time of every call."""
        def get(self, *args, **kwargs):
            """Record the current fake time, then answer as usual."""
            starts.append(now[0])
            return super().get(*args, **kwargs)
    limiter = RateLimiter(clock=lambda: now[0], sleep=sleep)
    cache = TTLCache()
    first = MusicBrainzProvider(session=TimedSession({"recordings": [recording()]}), limiter=limiter, cache=cache)
    second = MusicBrainzProvider(session=TimedSession({"recordings": []}), limiter=limiter, cache=cache)
    candidate = Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])
    first.lookup_recordings([candidate])
    second.lookup_recordings([candidate])
    second.lookup_recordings([Candidate("So What", "Miles Davis", ["web_2"])])
    assert len(starts) == 2
    assert starts[1] - starts[0] >= 1.0


@pytest.mark.parametrize("contact", ["", "not a contact", "me@example.org\nX-Injected: 1"])
def test_musicbrainz_requires_a_valid_contact(contact):
    """Musicbrainz requires a valid contact."""
    with pytest.raises(DiscoveryError) as exc:
        MusicBrainzProvider(contact)
    assert exc.value.code == "missing_configuration"
    assert "me@example.org" in MusicBrainzProvider("me@example.org").user_agent


def test_musicbrainz_query_escapes_lucene_syntax():
    """Musicbrainz query escapes lucene syntax."""
    session = Session({"recordings": []})
    provider = MusicBrainzProvider(session=session, cache=TTLCache(), limiter=RateLimiter())
    provider.lookup_recordings([Candidate('Say "Yes" (live)', "AC/DC", ["web_1"])])
    assert session.calls[0][1]["params"]["query"] == r'recording:"Say \"Yes\" \(live\)" AND artist:"AC\/DC"'


@pytest.mark.parametrize("data", [{"recordings": "oops"}, {"recordings": [recording(id="not-a-uuid")]}],
                         ids=["no-recording-list", "bad-recording-id"])
def test_musicbrainz_rejects_unusable_payloads(data):
    """Musicbrainz rejects unusable payloads."""
    provider = MusicBrainzProvider(session=Session(data), cache=TTLCache(), limiter=RateLimiter())
    with pytest.raises(DiscoveryError) as exc:
        provider.lookup_recordings([Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])])
    assert exc.value.code == "invalid_response"


@pytest.mark.parametrize("values", [
    {},
    {"TAVILY_API_KEY": "YOUR_TAVILY_KEY", "GEMINI_API_KEY": "gemini-secret"},
    {"TAVILY_API_KEY": "tavily-secret", "GEMINI_API_KEY": "replace-me"},
], ids=["missing", "tavily-placeholder", "gemini-placeholder"])
def test_missing_or_placeholder_configuration_is_rejected(values):
    """Missing or placeholder configuration is rejected."""
    with pytest.raises(DiscoveryError) as exc:
        Settings.from_mapping(values)
    assert exc.value.code == "missing_configuration"


def test_settings_apply_defaults_and_hide_secrets():
    """Settings apply defaults and hide secrets."""
    settings = Settings.from_mapping({"TAVILY_API_KEY": " tavily-secret ", "GEMINI_API_KEY": "gemini-secret"})
    assert settings.tavily_api_key == "tavily-secret"
    assert settings.gemini_model == DEFAULT_MODEL
    assert settings.gemini_embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.semantic_retrieval_enabled is True
    assert settings.music_source_focus_enabled is True
    assert "secret" not in repr(settings)


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "off"])
def test_settings_can_disable_semantic_retrieval(value):
    """The environment provides a small kill switch without changing other discovery settings."""
    settings = Settings.from_mapping({"TAVILY_API_KEY": "t", "GEMINI_API_KEY": "g",
                                      "SEMANTIC_RETRIEVAL_ENABLED": value,
                                      "GEMINI_EMBEDDING_MODEL": "custom-embedding"})
    assert settings.semantic_retrieval_enabled is False
    assert settings.gemini_embedding_model == "custom-embedding"


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "off"])
def test_settings_can_disable_music_source_focus(value):
    """The focused search can be disabled without affecting the other provider settings."""
    settings = Settings.from_mapping({"TAVILY_API_KEY": "t", "GEMINI_API_KEY": "g",
                                      "MUSIC_SOURCE_FOCUS_ENABLED": value})
    assert settings.music_source_focus_enabled is False


def test_provider_failure_never_exposes_secret(caplog):
    """Provider failure never exposes secret."""
    class BrokenSession:
        """Fake session whose requests always fail with a connection error."""
        def post(self, *args, **kwargs):
            """Raise a connection error that contains a fake secret."""
            raise requests.ConnectionError("fixture-secret-do-not-log")

    with caplog.at_level(logging.INFO), pytest.raises(DiscoveryError) as exc:
        TavilyProvider("fixture-secret-do-not-log", session=BrokenSession()).search_web(
            MusicRequest("instrumental piano"), ParsedIntent([], ["instrumental piano songs"])
        )
    assert "provider_failure" in caplog.text  # the failure was logged, so the secret check below inspects real log output
    assert "fixture-secret" not in str(exc.value) + caplog.text


def gemini_client(responses):
    """Build a fake Gemini client that returns the given responses in order."""
    return SimpleNamespace(models=SimpleNamespace(generate_content=Mock(
        side_effect=[SimpleNamespace(text=json.dumps(r)) for r in responses])))


def test_gemini_analyzes_intent_and_has_four_call_limit():
    """Gemini analyzes intent and has four call limit."""
    client = gemini_client([
        {"criteria": [{"facet": "genre", "kind": "preference", "value": "jazz",
                       "weight": 2, "source_text": "jazz"}],
         "search_queries": ["jazz songs"]},
        {"candidates": [{"title": "Take Five", "artist": "Dave Brubeck Quartet", "entity_type": "song",
                          "source_ids": ["web_1"]}]},
        {"recommendations": []},
        {"recommendations": []},
    ])
    provider = GeminiProvider(Settings("fixture", "fixture"), client=client)
    req = MusicRequest("jazz")
    evidence = [Evidence("web_1", "web", "Take Five", "Dave Brubeck Quartet distinctive-source", "https://example.org")]
    intent = provider.analyze_request(req)
    candidates = provider.extract_candidates(req, intent, evidence)
    assert candidates == [Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])]
    provider.generate_recommendations(req, intent, candidates, evidence)
    assert "distinctive-source" not in client.models.generate_content.call_args_list[0].kwargs["contents"]
    assert all("distinctive-source" in call.kwargs["contents"]
               for call in client.models.generate_content.call_args_list[1:])
    assert all(call.kwargs["config"]["response_json_schema"] for call in client.models.generate_content.call_args_list)
    provider.generate_recommendations(req, intent, candidates, evidence)
    with pytest.raises(DiscoveryError) as exc:
        provider.generate_recommendations(req, intent, candidates, evidence)
    assert exc.value.code == "request_limit"


@pytest.mark.parametrize("outcome,code", [
    (RuntimeError("fixture-secret"), "provider_unavailable"),
    (TimeoutError(), "timeout"),
    (SimpleNamespace(text=""), "invalid_response"),
    (SimpleNamespace(text="not json"), "invalid_response"),
    (SimpleNamespace(text="x" * 40_001), "invalid_response"),
], ids=["api-error", "timeout", "empty", "not-json", "oversized"])
def test_gemini_failures_map_to_safe_errors(outcome, code):
    """Gemini failures map to safe errors."""
    effect = outcome if isinstance(outcome, Exception) else [outcome]
    client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(side_effect=effect)))
    with pytest.raises(DiscoveryError) as exc:
        GeminiProvider(Settings("fixture", "fixture"), client=client).analyze_request(MusicRequest("jazz"))
    assert exc.value.code == code
    assert "fixture-secret" not in str(exc.value)


INTENT_ANSWER = {"criteria": [{"facet": "genre", "kind": "preference", "value": "jazz",
                               "weight": 2, "source_text": "jazz"}], "search_queries": ["jazz songs"]}


def flaky_client(*outcomes):
    """Build a fake Gemini client that raises or answers in order; an int is a failure with that HTTP status."""
    effects = [failure(code=o) if isinstance(o, int) else SimpleNamespace(text=json.dumps(o)) for o in outcomes]
    return SimpleNamespace(models=SimpleNamespace(generate_content=Mock(side_effect=effects)))


def test_a_gemini_503_is_retried_once_after_a_pause_and_uses_one_of_the_four_calls(monkeypatch, caplog):
    """A 503 is retried once after the pause, and the retry counts toward the four-call limit."""
    sleeps = []
    monkeypatch.setattr("src.providers.time.sleep", sleeps.append)  # the test never actually waits
    client = flaky_client(503, INTENT_ANSWER)
    provider = GeminiProvider(Settings("fixture", "fixture"), client=client)
    with caplog.at_level(logging.INFO, logger="src.providers"):
        intent = provider.analyze_request(MusicRequest("jazz"))
    assert intent.search_queries == ["jazz songs"]
    assert client.models.generate_content.call_count == 2 and provider.calls == 2
    assert sleeps == [GEMINI_RETRY_PAUSE]
    assert "provider_retry provider=Gemini call=1 http_status=503" in caplog.text
    assert "fixture-secret" not in caplog.text


def test_a_second_gemini_503_is_reported_not_retried_again(monkeypatch):
    """After the one retry a second 503 becomes a safe provider_unavailable error."""
    monkeypatch.setattr("src.providers.time.sleep", lambda seconds: None)
    client = flaky_client(503, 503, INTENT_ANSWER)
    with pytest.raises(DiscoveryError) as exc:
        GeminiProvider(Settings("fixture", "fixture"), client=client).analyze_request(MusicRequest("jazz"))
    assert exc.value.code == "provider_unavailable" and "fixture-secret" not in str(exc.value)
    assert client.models.generate_content.call_count == 2


@pytest.mark.parametrize("status,code", [(429, "quota"), (403, "authentication"), (500, "provider_unavailable"),
                                         (404, "provider_unavailable")])
def test_other_gemini_failures_are_not_retried(status, code, monkeypatch):
    """Only a 503 is worth retrying; quota, authentication and other errors fail at once."""
    monkeypatch.setattr("src.providers.time.sleep", lambda seconds: pytest.fail("must not wait"))
    client = flaky_client(status, INTENT_ANSWER)
    with pytest.raises(DiscoveryError) as exc:
        GeminiProvider(Settings("fixture", "fixture"), client=client).analyze_request(MusicRequest("jazz"))
    assert exc.value.code == code
    assert client.models.generate_content.call_count == 1


def test_a_gemini_503_on_the_last_allowed_call_is_not_retried(monkeypatch):
    """The retry never pushes a request past the four-call limit."""
    monkeypatch.setattr("src.providers.time.sleep", lambda seconds: pytest.fail("must not wait"))
    client = flaky_client(503, INTENT_ANSWER)
    provider = GeminiProvider(Settings("fixture", "fixture"), client=client)
    provider.calls = 3
    with pytest.raises(DiscoveryError) as exc:
        provider.analyze_request(MusicRequest("jazz"))
    assert exc.value.code == "provider_unavailable"
    assert client.models.generate_content.call_count == 1 and provider.calls == 4


def test_gemini_parses_recommendations_and_sends_check_ids():
    """Gemini parses recommendations and sends check ids."""
    row = {"title": "Take Five", "artist": "Dave Brubeck Quartet", "explanation": "Instrumental jazz.",
           "source_ids": ["web_1"], "uncertainties": ["Version unknown."],
           "claims": [{"text": "Instrumental", "source_id": "web_1", "quote": "instrumental jazz"}],
           "constraint_checks": [{"constraint": "vocals:vocals:must_avoid", "status": "supported",
                                  "source_id": "web_1", "quote": "instrumental jazz"}],
           "preference_checks": [{"criterion": "genre:jazz:preference", "status": "unknown",
                                  "source_id": "", "quote": ""}]}
    client = gemini_client([{"recommendations": [row]}])
    candidate = Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])
    evidence = [Evidence("web_1", "web", "Take Five", "instrumental jazz", "https://example.org")]
    recommendations = GeminiProvider(Settings("fixture", "fixture"), client=client).generate_recommendations(
        MusicRequest("jazz"), jazz_intent(), [candidate], evidence)
    assert recommendations == [Recommendation(
        "Take Five", "Dave Brubeck Quartet", "Instrumental jazz.", ["web_1"], ["Version unknown."],
        [SupportedClaim("Instrumental", "web_1", "instrumental jazz")],
        [ConstraintCheck("vocals:vocals:must_avoid", "supported", "web_1", "instrumental jazz")],
        [PreferenceCheck("genre:jazz:preference", "unknown", "", "")])]
    assert "genre:jazz:preference" in client.models.generate_content.call_args.kwargs["contents"]


GOOD_ROW = {"title": "Take Five", "artist": "Dave Brubeck Quartet", "explanation": "Instrumental jazz.",
            "source_ids": ["web_1"], "uncertainties": [],
            "claims": [{"text": "Instrumental", "source_id": "web_1", "quote": "instrumental jazz"}],
            "constraint_checks": [], "preference_checks": []}


def test_gemini_sends_validation_feedback_only_on_a_repair_retry():
    """Gemini sends validation feedback only on a repair retry."""
    client = gemini_client([{"recommendations": []}, {"recommendations": []}])
    provider = GeminiProvider(Settings("fixture", "fixture"), client=client)
    candidate = Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])
    evidence = [Evidence("web_1", "web", "Take Five", "instrumental jazz", "https://example.org")]
    provider.generate_recommendations(MusicRequest("jazz"), jazz_intent(), [candidate], evidence)
    provider.generate_recommendations(MusicRequest("jazz"), jazz_intent(), [candidate], evidence,
                                      feedback="quote for Take Five is not verbatim")
    first, repair = (call.kwargs["contents"] for call in client.models.generate_content.call_args_list)
    assert "validation_feedback" not in first
    assert "quote for Take Five is not verbatim" in repair


def test_gemini_keeps_well_formed_rows_when_another_row_is_malformed():
    """Gemini keeps well formed rows when another row is malformed."""
    client = gemini_client([{"recommendations": [GOOD_ROW, {"title": "So What"}]}])
    recommendations = GeminiProvider(Settings("fixture", "fixture"), client=client).generate_recommendations(
        MusicRequest("jazz"), jazz_intent(), [], [])
    assert [rec.title for rec in recommendations] == ["Take Five"]


def test_gemini_rejects_omitted_explicit_exclusion():
    """Gemini rejects omitted explicit exclusion."""
    client = gemini_client([{"criteria": [], "search_queries": ["piano instrumental songs"]}])
    with pytest.raises(DiscoveryError):
        GeminiProvider(Settings("fixture", "fixture"), client=client).analyze_request(
            MusicRequest("piano no vocals"))


@pytest.mark.parametrize("payload", [[], {"criteria": "broken", "search_queries": []},
                                      {"criteria": [{}], "search_queries": ["jazz songs"]}])
def test_malformed_gemini_output(payload):
    """Malformed gemini output."""
    with pytest.raises(DiscoveryError) as exc:
        GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload])).analyze_request(
            MusicRequest("jazz"))
    assert exc.value.code == "invalid_response"


@pytest.mark.parametrize("call,payload", [
    (lambda p: p.extract_candidates(MusicRequest("jazz"), jazz_intent(), []),
     {"candidates": [{"title": "So What", "artist": "Miles Davis", "entity_type": "album", "source_ids": []}]}),
    (lambda p: p.generate_recommendations(MusicRequest("jazz"), jazz_intent(), [], []),
     {"recommendations": [{"title": "So What"}]}),
], ids=["candidates", "recommendations"])
def test_malformed_candidates_and_recommendations_are_invalid_responses(call, payload):
    """Malformed candidates and recommendations are invalid responses."""
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    with pytest.raises(DiscoveryError) as exc:
        call(provider)
    assert exc.value.code == "invalid_response"


def test_happy_peaceful_intent_is_soft_and_generates_focused_queries():
    """Happy peaceful intent is soft and generates focused queries."""
    case = json.loads((Path(__file__).parent / "fixtures/intent_cases.json").read_text())["cases"][0]
    payload = case["intent"]
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    intent = provider.analyze_request(MusicRequest(case["description"]))
    assert not intent.hard_criteria
    assert [(c.value, c.weight) for c in intent.preferences] == [
        ("happy", 2), ("peaceful", 3), ("low", 2)]
    assert intent.search_queries == payload["search_queries"]


def test_subjective_trait_is_coerced_to_preference_without_explicit_mandatory_language():
    """Subjective trait is coerced to preference without explicit mandatory language."""
    payload = {"criteria": [{"facet": "mood", "kind": "must_have", "value": "happy",
                              "weight": 3, "source_text": "happy"}],
               "search_queries": ["happy songs"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    intent = provider.analyze_request(MusicRequest("I want a happy song"))
    assert intent.criteria[0].kind == "preference"


@pytest.mark.parametrize("query", ["I want happy songs", "https://example.org/happy-songs"])
def test_intent_rejects_filler_and_urls_in_search_queries(query):
    """Intent rejects filler and urls in search queries."""
    payload = {"criteria": [{"facet": "mood", "kind": "preference", "value": "happy",
                              "weight": 2, "source_text": "happy"}],
               "search_queries": [query]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    with pytest.raises(DiscoveryError):
        provider.analyze_request(MusicRequest("I want happy songs"))


def test_tavily_filters_irrelevant_unsafe_and_duplicate_results():
    """Tavily filters irrelevant unsafe and duplicate results."""
    gentle = {"title": "Gentle Joy — Example Artist", "content": "A peaceful uplifting song",
              "url": "https://example.org/gentle-joy"}
    session = Session({"results": [
        {"title": "Voice Memos App", "content": "Record personal notes on your phone",
         "url": "https://example.org/voice-memos"},
        {**gentle, "url": "http://localhost/gentle-joy"},
        "not-a-result",
        gentle,
        gentle,
    ]})
    provider = TavilyProvider("fixture", session=session, cache=TTLCache())
    intent = ParsedIntent([
        TasteCriterion("mood", "preference", "happy", 2, "happy"),
        TasteCriterion("mood", "preference", "peaceful", 3, "very peaceful"),
    ], ["happy peaceful songs"])
    evidence = provider.search_web(MusicRequest("I want a happy song but very peaceful"), intent)
    assert [(e.id, e.title) for e in evidence] == [("web_1", "Gentle Joy — Example Artist")]


def test_candidate_extraction_drops_albums_and_playlists():
    """Candidate extraction drops albums and playlists."""
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([{
        "candidates": [
            {"title": "Kind of Blue", "artist": "Miles Davis", "entity_type": "not_song",
             "source_ids": ["web_1"]},
            {"title": "So What", "artist": "Miles Davis", "entity_type": "song",
             "source_ids": ["web_1"]},
        ]
    }]))
    evidence = [Evidence("web_1", "web", "Miles Davis discography",
                         "Kind of Blue by Miles Davis includes So What by Miles Davis.",
                         "https://example.org/miles")]
    candidates = provider.extract_candidates(MusicRequest("jazz"), jazz_intent(), evidence)
    assert candidates == [Candidate("So What", "Miles Davis", ["web_1"])]


def test_candidate_extraction_carries_the_models_fit_judgment():
    """The model's fits_request verdict is kept on each candidate; a missing one defaults to fitting."""
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([{
        "candidates": [
            {"title": "So What", "artist": "Miles Davis", "entity_type": "song", "fits_request": True,
             "source_ids": ["web_1"]},
            {"title": "Africa", "artist": "Toto", "entity_type": "song", "fits_request": False,
             "source_ids": ["web_1"]},
        ]
    }]))
    evidence = [Evidence("web_1", "web", "Songs", "So What by Miles Davis. Africa by Toto.", "https://example.org/x")]
    fits = [c.fits_request for c in provider.extract_candidates(MusicRequest("jazz"), jazz_intent(), evidence)]
    assert fits == [True, False]


# --- configuration and caching -------------------------------------------------------------

def test_settings_read_the_process_environment_when_no_mapping_is_given(monkeypatch):
    """Settings read the process environment when no mapping is given."""
    for name, value in {"TAVILY_API_KEY": "tavily-env", "GEMINI_API_KEY": "gemini-env",
                        "GEMINI_MODEL": "custom-model", "MUSICBRAINZ_CONTACT": "me@example.org"}.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_mapping()
    assert (settings.tavily_api_key, settings.gemini_api_key, settings.gemini_model,
            settings.musicbrainz_contact) == ("tavily-env", "gemini-env", "custom-model", "me@example.org")


def test_cache_evicts_the_least_recently_used_entry_not_the_oldest():
    """Cache evicts the least recently used entry not the oldest."""
    cache = TTLCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # reading "a" makes "b" the least recently used
    cache.put("c", 3)
    assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)


# --- error mapping and the shared HTTP helper ----------------------------------------------

def failure(kind=RuntimeError, **attributes):
    """Create an error carrying a fake secret and the given attributes."""
    error = kind("fixture-secret")
    for name, value in attributes.items():
        setattr(error, name, value)
    return error


@pytest.mark.parametrize("error,code", [
    (failure(response=SimpleNamespace(status_code=432)), "quota"),
    (failure(response=SimpleNamespace(status_code=433)), "quota"),
    (failure(code=429), "quota"),                     # google-genai errors carry .code, not .response
    (failure(code=403), "authentication"),
    (failure(response=SimpleNamespace(status_code=500)), "provider_unavailable"),
    (failure(type("ReadTimeoutError", (Exception,), {})), "timeout"),   # any error class named for a timeout
    (failure(), "provider_unavailable"),
], ids=["tavily-432", "tavily-433", "sdk-429", "sdk-403", "http-500", "named-timeout", "unknown"])
def test_provider_errors_map_to_a_safe_code_and_never_echo_the_original_message(error, code):
    """Provider errors map to a safe code and never echo the original message."""
    safe = provider_error("Example", error)
    assert safe.code == code and str(safe).startswith("Example ")
    assert "fixture-secret" not in str(safe)


def test_provider_failure_log_records_the_status_and_error_type_without_the_message(caplog):
    """The failure log names the HTTP status, status name and error class so a coarse code can be diagnosed."""
    with caplog.at_level(logging.WARNING, logger="src.providers"):
        provider_error("Gemini", failure(code=503, status="UNAVAILABLE"))
    assert "provider_failure provider=Gemini code=provider_unavailable " \
           "http_status=503 status_name=UNAVAILABLE error_type=RuntimeError" in caplog.text
    assert "fixture-secret" not in caplog.text


@pytest.mark.parametrize("error", [
    failure(code="fixture-secret", status="fixture-secret leaked"),  # neither is a plain status
    failure(response=SimpleNamespace(status_code="fixture-secret")),
], ids=["sdk-attributes", "response-status"])
def test_provider_failure_log_drops_status_values_that_are_not_plain_statuses(error, caplog):
    """A status that is not an integer, or a name that is not upper-case words, is logged as None."""
    with caplog.at_level(logging.WARNING, logger="src.providers"):
        provider_error("Gemini", error)
    assert "http_status=None status_name=None error_type=RuntimeError" in caplog.text
    assert "fixture-secret" not in caplog.text


class TrackedResponse(Response):
    """Fake response that remembers whether it was closed."""
    closed = False

    def close(self):
        """Mark the response as closed."""
        self.closed = True


class SingleResponseSession:
    """Fake session that always returns one given response."""
    def __init__(self, response):
        """Store the response to return."""
        self.response = response

    def get(self, *args, **kwargs):
        """Return the stored response."""
        return self.response


def test_http_json_closes_the_response_whether_it_succeeds_or_fails():
    """Http json closes the response whether it succeeds or fails."""
    good, bad = TrackedResponse({"a": 1}), TrackedResponse({}, status=500)
    assert http_json(SingleResponseSession(good), "get", "https://example.org") == {"a": 1}
    assert good.closed
    with pytest.raises(requests.HTTPError):
        http_json(SingleResponseSession(bad), "get", "https://example.org")
    assert bad.closed


def test_http_json_gives_up_when_the_total_time_budget_is_spent():
    """Http json gives up when the total time budget is spent."""
    class TwoChunks(TrackedResponse):
        """Fake response whose body arrives in two chunks."""
        def iter_content(self, size):
            """Yield the JSON body in two chunks."""
            yield b'{"a":'
            yield b" 1}"

    response = TwoChunks({})
    with patch("src.providers.time") as clock:
        clock.monotonic.side_effect = [0, 1, 26]  # start, first chunk in time, second chunk past the 25 s deadline
        with pytest.raises(requests.Timeout):
            http_json(SingleResponseSession(response), "get", "https://example.org")
    assert response.closed


# --- Tavily request contract ---------------------------------------------------------------

def test_tavily_sends_at_most_two_trimmed_queries_and_keeps_bounded_results():
    """Tavily sends at most two trimmed queries and keeps bounded results."""
    results = [{"title": "jazz " + "t" * 400, "content": "jazz " + "c" * 4000, "url": f"https://example.org/{n}"}
               for n in range(12)]
    session = Session({"results": results})
    intent = ParsedIntent([], ["jazz " + "x" * 500, "second jazz query", "third jazz query"])
    evidence = TavilyProvider("fixture-key", session=session, cache=TTLCache()).search_web(MusicRequest("jazz"), intent)
    assert len(session.calls) == 2  # the third query is never sent
    args, kwargs = session.calls[0]
    assert args == ("https://api.tavily.com/search",) and kwargs["headers"] == {"Authorization": "Bearer fixture-key"}
    assert len(kwargs["json"]["query"]) == 390 and kwargs["json"]["max_results"] == 10
    assert kwargs["json"]["include_raw_content"] is False
    assert "include_domains" not in kwargs["json"]
    assert session.calls[1][1]["json"]["query"] == "second jazz query"
    assert session.calls[1][1]["json"]["include_domains"] == list(MUSIC_SOURCE_DOMAINS)
    assert [e.id for e in evidence] == [f"web_{n}" for n in range(1, 11)]  # ten of twelve kept; the repeat query adds none
    assert all(e.kind == "web" and len(e.title) == 300 and len(e.text) == 3000 for e in evidence)
    assert evidence[0].retrieved_at and evidence[0].retrieved_at.endswith("+00:00")


def test_tavily_skips_results_without_text_and_filters_nothing_when_the_query_has_no_search_terms():
    """Tavily skips results without text and filters nothing when the query has no search terms."""
    session = Session({"results": [
        {"title": "A", "content": "   ", "url": "https://example.org/a"},
        {"title": "B", "content": None, "url": "https://example.org/b"},
        {"title": "C", "content": 42, "url": "https://example.org/c"},
        {"title": "Unrelated app", "content": "Takes voice memos", "url": "https://example.org/d"},
    ]})
    evidence = TavilyProvider("fixture", session=session, cache=TTLCache()).search_web(
        MusicRequest("jazz"), ParsedIntent([], ["a to"]))  # a one-letter word and a stop word leave no terms to filter on
    assert [e.title for e in evidence] == ["Unrelated app"]


def test_tavily_cache_is_not_shared_between_api_keys():
    """Tavily cache is not shared between api keys."""
    session = Session({"results": [{"title": "Take Five", "content": "jazz", "url": "https://example.org/music"}]})
    cache, intent = TTLCache(), ParsedIntent([], ["jazz songs"])
    one = TavilyProvider("key-one", session=session, cache=cache)
    two = TavilyProvider("key-two", session=session, cache=cache)
    one.search_web(MusicRequest("jazz"), intent)
    one.search_web(MusicRequest("jazz"), intent)
    two.search_web(MusicRequest("jazz"), intent)
    assert len(session.calls) == 4  # each key gets one general and one focused entry; the first key reuses both


def test_tavily_reuses_one_query_for_the_focused_search_and_can_disable_focus():
    """A single analyzed query still gets one general and one focused attempt within the two-call cap."""
    session = Session({"results": []})
    intent = ParsedIntent([], ["instrumental jazz songs"])
    TavilyProvider("fixture", session=session, cache=TTLCache()).search_web(MusicRequest("jazz"), intent)
    assert [call[1]["json"]["query"] for call in session.calls] == ["instrumental jazz songs"] * 2
    assert "include_domains" not in session.calls[0][1]["json"]
    assert session.calls[1][1]["json"]["include_domains"] == list(MUSIC_SOURCE_DOMAINS)

    disabled = Session({"results": []})
    TavilyProvider("fixture", session=disabled, cache=TTLCache(),
                   music_source_focus_enabled=False).search_web(MusicRequest("jazz"), intent)
    assert len(disabled.calls) == 1
    assert "include_domains" not in disabled.calls[0][1]["json"]


def test_tavily_prioritizes_real_music_hosts_then_score_and_reissues_ids():
    """Specialist hosts win the merge, but lookalike domains remain ordinary general-web evidence."""
    class SequencedSession:
        def __init__(self):
            self.calls = []
            self.responses = [
                {"results": [
                    {"title": "General", "content": "jazz song", "score": 0.99,
                     "url": "https://example.org/general"},
                    {"title": "Lookalike", "content": "jazz song", "score": 1.0,
                     "url": "https://pitchfork.com.example.org/review"},
                ]},
                {"results": [
                    {"title": "Pitchfork", "content": "jazz song", "score": 0.2,
                     "url": "https://reviews.pitchfork.com/albums/example"},
                    {"title": "Rolling Stone", "content": "jazz song", "score": 0.8,
                     "url": "https://www.rollingstone.com/music/example"},
                ]},
            ]

        def post(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response(self.responses.pop(0))

    evidence = TavilyProvider("fixture", session=SequencedSession(), cache=TTLCache()).search_web(
        MusicRequest("jazz"), ParsedIntent([], ["jazz songs"]))
    assert [(item.id, item.title) for item in evidence] == [
        ("web_1", "Rolling Stone"), ("web_2", "Pitchfork"),
        ("web_3", "Lookalike"), ("web_4", "General"),
    ]


def test_tavily_keeps_general_results_when_the_focused_search_fails(caplog):
    """A focused-source outage degrades safely after a successful general search."""
    class FocusFailureSession:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Response({"results": [{"title": "General", "content": "jazz song",
                                                "url": "https://example.org/general"}]})
            return Response({"message": "fixture-secret"}, 503)

    with caplog.at_level(logging.INFO):
        evidence = TavilyProvider("fixture-secret", session=FocusFailureSession(),
                                  cache=TTLCache()).search_web(
            MusicRequest("jazz"), ParsedIntent([], ["jazz songs"]))
    assert [item.title for item in evidence] == ["General"]
    assert "retrieval_degraded provider=Tavily mode=music code=provider_unavailable" in caplog.text
    assert "fixture-secret" not in caplog.text


# --- Gemini embeddings --------------------------------------------------------------------

def embedding_vector(first=1.0, second=0.0):
    """Build one nonzero vector with the configured output size."""
    return [first, second, *([0.0] * (EMBEDDING_DIMENSIONS - 2))]


def embedding_response(*vectors):
    """Build the part of an SDK embedding response used by the provider."""
    return SimpleNamespace(embeddings=[SimpleNamespace(values=vector) for vector in vectors])


def embedding_provider(*responses):
    """Build an embedding provider with isolated caches and a recording fake client."""
    models = SimpleNamespace(embed_content=Mock(side_effect=responses))
    client = SimpleNamespace(models=models)
    provider = GeminiEmbeddingProvider(Settings("fixture", "fixture"), client=client,
        document_cache=TTLCache(), query_cache=TTLCache())
    return provider, models.embed_content


def test_embedding_provider_uses_asymmetric_retrieval_tasks_and_cosine_scores():
    """Documents and the query use their matching retrieval task types and 768 dimensions."""
    provider, embed = embedding_provider(
        embedding_response(embedding_vector(1, 0), embedding_vector(0, 1)),
        embedding_response(embedding_vector(1, 0)),
    )
    documents = [("card_one", "One", "first document"),
                 ("card_two", "Two", "second document")]
    assert provider.score("first meaning", documents) == {"card_one": 1.0, "card_two": 0.0}
    document_call, query_call = embed.call_args_list
    assert document_call.kwargs["model"] == DEFAULT_EMBEDDING_MODEL
    assert document_call.kwargs["contents"] == [
        "title: One | text: first document", "title: Two | text: second document"]
    assert document_call.kwargs["config"] == {
        "task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": EMBEDDING_DIMENSIONS}
    assert query_call.kwargs["contents"] == "first meaning"
    assert query_call.kwargs["config"]["task_type"] == "RETRIEVAL_QUERY"


def test_embedding_provider_caches_queries_and_document_fingerprints():
    """Repeated content uses both caches; changed document text invalidates only the document batch."""
    provider, embed = embedding_provider(
        embedding_response(embedding_vector()), embedding_response(embedding_vector()),
        embedding_response(embedding_vector()),
    )
    documents = [("card_one", "One", "first version")]
    provider.score("same query", documents)
    provider.score("same query", documents)
    provider.score("same query", [("card_one", "One", "changed version")])
    assert embed.call_count == 3
    assert embed.call_args_list[-1].kwargs["config"]["task_type"] == "RETRIEVAL_DOCUMENT"


@pytest.mark.parametrize("response", [
    SimpleNamespace(embeddings=[]),
    embedding_response([1.0]),
    embedding_response([0.0] * EMBEDDING_DIMENSIONS),
    embedding_response([float("nan"), *([0.0] * (EMBEDDING_DIMENSIONS - 1))]),
], ids=["missing", "wrong-dimension", "zero", "non-finite"])
def test_embedding_provider_rejects_malformed_vectors(response):
    """Malformed vectors never reach the hybrid ranker."""
    provider, _ = embedding_provider(response)
    with pytest.raises(DiscoveryError) as exc:
        provider.score("query", [("card_one", "One", "document")])
    assert exc.value.code == "invalid_response"


def test_embedding_provider_maps_failures_without_logging_query_or_secret(caplog):
    """A provider failure is safe and leaves the caller free to use lexical retrieval."""
    provider, embed = embedding_provider()
    embed.side_effect = RuntimeError("fixture-secret query words")
    with caplog.at_level(logging.WARNING), pytest.raises(DiscoveryError) as exc:
        provider.score("private query words", [("card_one", "One", "document")])
    assert exc.value.code == "provider_unavailable"
    assert "fixture-secret" not in caplog.text and "private query words" not in caplog.text


# --- Gemini client, request contract and parsing -------------------------------------------

def test_gemini_client_is_built_with_a_timeout_and_no_sdk_retries():
    """Gemini client is built with a timeout and no sdk retries."""
    with patch("google.genai.Client") as client:
        provider = GeminiProvider(Settings("tavily-key", "gemini-key"))
    client.assert_called_once_with(api_key="gemini-key",
                                   http_options={"timeout": 30000, "retry_options": {"attempts": 1}})
    assert provider.client is client.return_value


def test_gemini_initialisation_failures_are_reported_without_the_original_message():
    """Gemini initialisation failures are reported without the original message."""
    with patch("google.genai.Client", side_effect=RuntimeError("fixture-secret")), \
            pytest.raises(DiscoveryError) as exc:
        GeminiProvider(Settings("tavily-key", "fixture-secret"))
    assert exc.value.code == "missing_configuration" and "fixture-secret" not in str(exc.value)


def test_a_missing_gemini_sdk_is_reported_as_a_missing_dependency(monkeypatch):
    """A missing gemini sdk is reported as a missing dependency."""
    import google
    monkeypatch.delattr(google, "genai", raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", None)  # makes "from google import genai" fail
    with pytest.raises(DiscoveryError) as exc:
        GeminiProvider(Settings("tavily-key", "gemini-key"))
    assert exc.value.code == "missing_dependency"


def test_closing_the_gemini_provider_closes_its_client():
    """Closing the gemini provider closes its client."""
    client = SimpleNamespace(close=Mock())
    GeminiProvider(Settings("fixture", "fixture"), client=client).close()
    client.close.assert_called_once_with()


def test_gemini_requests_are_deterministic_json_with_an_untrusted_evidence_guard():
    """Gemini requests are deterministic json with an untrusted evidence guard."""
    client = gemini_client([{"criteria": [{"facet": "language", "kind": "preference", "value": "Portuguese",
                                           "weight": 2, "source_text": "português"}],
                             "search_queries": ["Portuguese music"]}])
    GeminiProvider(Settings("fixture", "fixture", gemini_model="custom-model"), client=client).analyze_request(
        MusicRequest("música em português"))
    call = client.models.generate_content.call_args.kwargs
    config = call["config"]
    assert call["model"] == "custom-model"
    assert (config["temperature"], config["max_output_tokens"], config["response_mime_type"]) == (
        0, 4096, "application/json")
    assert config["automatic_function_calling"] == {"disable": True}
    assert "untrusted" in config["system_instruction"] and "evidence" in config["system_instruction"]
    sent = json.loads(call["contents"])
    assert set(sent) == {"task", "data"} and sent["data"]["request"]["description"] == "música em português"
    assert "português" in call["contents"]  # sent as written, not as \u00ea escapes


@pytest.mark.parametrize("request_text", [
    "instrumental piano", "piano sem voz", "heavy guitar with clean singing", "no screaming please",
    "without vocals", "sem vocais por favor", "without screaming", "sem gritos",
])
def test_every_explicit_exclusion_phrase_must_come_back_as_a_hard_criterion(request_text):
    """Every explicit exclusion phrase must come back as a hard criterion."""
    payload = {"criteria": [{"facet": "genre", "kind": "preference", "value": "x", "weight": 2,
                             "source_text": request_text.split()[0]}],
               "search_queries": ["piano songs"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    with pytest.raises(DiscoveryError) as exc:
        provider.analyze_request(MusicRequest(request_text))
    assert exc.value.code == "invalid_response"


def test_an_explicit_exclusion_is_accepted_once_a_hard_criterion_quotes_it():
    """An explicit exclusion is accepted once a hard criterion quotes it."""
    payload = {"criteria": [{"facet": "vocals", "kind": "must_avoid", "value": "vocals", "weight": 3,
                             "source_text": "no vocals"}], "search_queries": ["piano instrumental songs"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    intent = provider.analyze_request(MusicRequest("piano no vocals"))
    assert [(c.kind, c.value) for c in intent.criteria] == [("must_avoid", "vocals")]


def test_clean_singing_graded_as_a_preference_is_promoted_instead_of_rejecting_the_request():
    """Gemini's real answer to "clean singing metal with no screaming" is kept, with clean singing made a requirement."""
    payload = {"criteria": [
        {"facet": "genre", "kind": "preference", "value": "metal", "weight": 2, "source_text": "metal"},
        {"facet": "vocals", "kind": "preference", "value": "clean singing", "weight": 2,
         "source_text": "clean singing"},
        {"facet": "vocals", "kind": "must_avoid", "value": "screaming", "weight": 3, "source_text": "no screaming"}],
        "search_queries": ["metal clean vocals no screaming", "melodic metal clean singing"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    intent = provider.analyze_request(MusicRequest("clean singing metal with no screaming"))
    assert [(c.value, c.kind) for c in intent.criteria] == [
        ("metal", "preference"), ("clean singing", "must_have"), ("screaming", "must_avoid")]
    assert [c.value for c in intent.hard_criteria] == ["clean singing", "screaming"]
    assert intent.search_queries == payload["search_queries"]


@pytest.mark.parametrize("request_text,said,expected", [
    ("piano with no vocals", "no vocals", "must_avoid"),
    ("piano sem voz", "sem voz", "must_avoid"),
    ("rock without screaming please", "without screaming", "must_avoid"),
    ("instrumental piano", "instrumental", "must_have"),
    ("clean singing please", "clean singing", "must_have"),
])
def test_a_soft_vocals_criterion_is_promoted_to_the_polarity_of_its_phrase(request_text, said, expected):
    """A soft vocals criterion becomes must_avoid for "no ..." phrases and must_have for the others."""
    payload = {"criteria": [{"facet": "vocals", "kind": "preference", "value": said, "weight": 2,
                             "source_text": said}], "search_queries": ["some songs"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    assert [c.kind for c in provider.analyze_request(MusicRequest(request_text)).criteria] == [expected]


def test_only_a_vocals_criterion_can_be_promoted():
    """A soft criterion in another facet that quotes the phrase is still rejected, as before."""
    payload = {"criteria": [{"facet": "genre", "kind": "preference", "value": "clean singing", "weight": 2,
                             "source_text": "clean singing"}], "search_queries": ["some songs"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    with pytest.raises(DiscoveryError) as exc:
        provider.analyze_request(MusicRequest("clean singing please"))
    assert exc.value.code == "invalid_response"


def test_a_phrase_the_model_dropped_is_still_rejected_even_when_another_one_is_promotable():
    """Promotion never covers for a phrase that no criterion mentions."""
    payload = {"criteria": [{"facet": "vocals", "kind": "preference", "value": "clean singing", "weight": 2,
                             "source_text": "clean singing"}], "search_queries": ["some songs"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    with pytest.raises(DiscoveryError) as exc:
        provider.analyze_request(MusicRequest("clean singing metal with no screaming"))
    assert exc.value.code == "invalid_response"


def test_the_analysis_prompt_names_the_vocal_phrases_that_must_be_hard_criteria():
    """The prompt tells Gemini these phrases are explicit, so they come back as hard criteria in the first place."""
    payload = {"criteria": [], "search_queries": ["some songs"]}
    client = gemini_client([payload])
    GeminiProvider(Settings("fixture", "fixture"), client=client).analyze_request(MusicRequest("some songs"))
    task = json.loads(client.models.generate_content.call_args.kwargs["contents"])["task"]
    assert "clean singing" in task and "no screaming" in task and "never preference" in task


@pytest.mark.parametrize("facet,said,expected", [
    ("energy", "very slow", "preference"),
    ("activity", "for running", "preference"),
    ("mood", "must be happy", "must_have"),
    ("energy", "only slow", "must_have"),
    ("mood", "sem tristeza", "must_have"),
    ("genre", "jazz", "must_have"),    # only mood, energy and activity are softened
], ids=["energy-softened", "activity-softened", "must-keeps-mood", "only-keeps-energy", "sem-keeps-mood",
        "genre-never-softened"])
def test_subjective_facets_become_preferences_unless_the_listener_made_them_mandatory(facet, said, expected):
    """Subjective facets become preferences unless the listener made them mandatory."""
    payload = {"criteria": [{"facet": facet, "kind": "must_have", "value": said, "weight": 2, "source_text": said}],
               "search_queries": ["some songs"]}
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([payload]))
    assert provider.analyze_request(MusicRequest(said)).criteria[0].kind == expected


@pytest.mark.parametrize("rows", [
    [{"title": f"Song {n}", "artist": "A", "entity_type": "song", "source_ids": ["web_1"]}
     for n in range(MAX_CANDIDATES + 1)],
    [{"title": "So What", "artist": "Miles Davis", "entity_type": "song",
      "source_ids": [f"web_{n}" for n in range(13)]}],
    [{"title": " ", "artist": "Miles Davis", "entity_type": "song", "source_ids": ["web_1"]}],
    [{"title": "So What", "artist": "Miles Davis", "entity_type": "song", "source_ids": "web_1"}],
], ids=["too-many-candidates", "thirteen-sources", "blank-title", "sources-not-a-list"])
def test_candidate_rows_must_respect_size_and_shape_limits(rows):
    """Candidate rows must respect size and shape limits."""
    provider = GeminiProvider(Settings("fixture", "fixture"), client=gemini_client([{"candidates": rows}]))
    with pytest.raises(DiscoveryError) as exc:
        provider.extract_candidates(MusicRequest("jazz"), jazz_intent(), [])
    assert exc.value.code == "invalid_response"


def test_more_than_the_judged_limit_of_recommendation_rows_is_invalid():
    """More recommendation rows than MAX_JUDGED are invalid."""
    provider = GeminiProvider(Settings("fixture", "fixture"),
                              client=gemini_client([{"recommendations": [GOOD_ROW] * (MAX_JUDGED + 1)}]))
    with pytest.raises(DiscoveryError) as exc:
        provider.generate_recommendations(MusicRequest("jazz"), jazz_intent(), [], [])
    assert exc.value.code == "invalid_response"


def test_when_every_recommendation_row_is_malformed_the_error_tells_the_model_what_is_required():
    """When every recommendation row is malformed the error tells the model what is required."""
    provider = GeminiProvider(Settings("fixture", "fixture"),
                              client=gemini_client([{"recommendations": [{"title": "So What"}]}]))
    with pytest.raises(DiscoveryError) as exc:
        provider.generate_recommendations(MusicRequest("jazz"), jazz_intent(), [], [])
    assert exc.value.code == "invalid_response"
    assert "claims, constraint_checks and preference_checks" in exc.value.feedback


def test_repair_feedback_is_capped_at_1500_characters_and_announced_in_the_task():
    """Repair feedback is capped at 1500 characters and announced in the task."""
    client = gemini_client([{"recommendations": []}])
    GeminiProvider(Settings("fixture", "fixture"), client=client).generate_recommendations(
        MusicRequest("jazz"), jazz_intent(), [], [], feedback="x" * 3000)
    sent = json.loads(client.models.generate_content.call_args.kwargs["contents"])
    assert sent["data"]["validation_feedback"] == "x" * 1500
    assert sent["task"].startswith("Your previous answer was rejected.")


def test_response_schemas_are_strict_and_match_the_validators():
    """Response schemas are strict and match the validators."""
    client = gemini_client([
        {"criteria": [{"facet": "genre", "kind": "preference", "value": "jazz", "weight": 2, "source_text": "jazz"}],
         "search_queries": ["jazz songs"]},
        {"candidates": []},
        {"recommendations": []},
    ])
    provider = GeminiProvider(Settings("fixture", "fixture"), client=client)
    provider.analyze_request(MusicRequest("jazz"))
    provider.extract_candidates(MusicRequest("jazz"), jazz_intent(), [])
    provider.generate_recommendations(MusicRequest("jazz"), jazz_intent(), [], [])
    intent_schema, candidate_schema, recommendation_schema = (
        call.kwargs["config"]["response_json_schema"] for call in client.models.generate_content.call_args_list)

    criterion = intent_schema["properties"]["criteria"]["items"]["properties"]
    assert set(criterion["facet"]["enum"]) == INTENT_FACETS
    assert set(criterion["kind"]["enum"]) == CRITERION_KINDS
    assert (criterion["weight"]["minimum"], criterion["weight"]["maximum"]) == (1, 3)
    assert set(candidate_schema["properties"]["candidates"]["items"]["properties"]["entity_type"]["enum"]) == {
        "song", "not_song"}
    row = recommendation_schema["properties"]["recommendations"]["items"]["properties"]
    for name in ("constraint_checks", "preference_checks"):
        assert set(row[name]["items"]["properties"]["status"]["enum"]) == CHECK_STATUSES

    def assert_strict(schema):
        """Check that a schema and all its nested objects are strict and fully required."""
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False and schema["required"] == list(schema["properties"])
            for child in schema["properties"].values():
                assert_strict(child)
        elif schema.get("type") == "array":
            assert_strict(schema["items"])

    for schema in (intent_schema, candidate_schema, recommendation_schema):
        assert_strict(schema)


@pytest.mark.parametrize("value,limit", [(None, 5), (5, 5), ("", 5), ("   ", 5), ("abcdef", 5)],
                         ids=["none", "number", "empty", "blank", "too-long"])
def test_text_field_rejects_non_text_blank_text_and_text_over_the_limit(value, limit):
    """Text field rejects non text blank text and text over the limit."""
    with pytest.raises(ValueError):
        text_field(value, limit)


def test_text_helpers_accept_good_values_unchanged():
    """Text helpers accept good values unchanged."""
    assert text_field(" ok ", 5) == " ok "
    assert optional_text_field("", 5) == ""
    assert strings(["a", "b"], 2) == ["a", "b"]


@pytest.mark.parametrize("bad", [None, 5, "abcdef"], ids=["none", "number", "too-long"])
def test_optional_text_field_allows_empty_text_but_nothing_else_invalid(bad):
    """Optional text field allows empty text but nothing else invalid."""
    with pytest.raises(ValueError):
        optional_text_field(bad, 5)


@pytest.mark.parametrize("value,limit", [("ab", 2), (["a", "b", "c"], 2), (["a", ""], 2), (["x" * 601], 2)],
                         ids=["not-a-list", "too-many", "blank-item", "item-too-long"])
def test_strings_needs_a_bounded_list_of_non_blank_texts(value, limit):
    """Strings needs a bounded list of non blank texts."""
    with pytest.raises(ValueError):
        strings(value, limit)


# --- rate limiter and MusicBrainz details --------------------------------------------------

def test_rate_limiter_waits_only_for_the_rest_of_the_interval():
    """Rate limiter waits only for the rest of the interval."""
    now, slept = [0.0], []
    limiter = RateLimiter(clock=lambda: now[0], sleep=slept.append)
    assert limiter.run(lambda: "first") == "first" and slept == []  # nothing to wait for on the first call
    now[0] = 0.5
    limiter.run(lambda: None)
    now[0] = 5.0
    limiter.run(lambda: None)
    assert slept == [pytest.approx(0.55), 0.0]


def test_rate_limiter_reports_busy_instead_of_waiting_forever():
    """Rate limiter reports busy instead of waiting forever."""
    limiter = RateLimiter()
    limiter._lock = Mock(acquire=Mock(return_value=False))
    operation = Mock()
    with pytest.raises(DiscoveryError) as exc:
        limiter.run(operation)
    assert exc.value.code == "provider_busy"
    assert limiter._lock.acquire.call_args.kwargs == {"timeout": 5}
    operation.assert_not_called()


def test_rate_limiter_is_released_when_the_operation_fails():
    """Rate limiter is released when the operation fails."""
    limiter = RateLimiter(clock=lambda: 0.0, sleep=lambda delay: None)
    with pytest.raises(ZeroDivisionError):
        limiter.run(lambda: 1 / 0)
    assert limiter.run(lambda: "still usable") == "still usable"


def test_a_single_exact_match_is_identity_only_evidence_with_its_details():
    """A single exact match is identity only evidence with its details."""
    provider = MusicBrainzProvider(session=Session({"recordings": [recording()]}), cache=TTLCache(),
                                   limiter=RateLimiter())
    [evidence] = provider.lookup_recordings([Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])])
    data = json.loads(evidence.text)
    assert (evidence.id, evidence.kind, evidence.title) == ("mb_1", "metadata", "Take Five — Dave Brubeck Quartet")
    assert evidence.url == "https://musicbrainz.org/recording/10000000-0000-0000-0000-000000000001"
    assert evidence.retrieved_at
    assert data["status"] == "identity_match_version_unconfirmed" and "Identity metadata only" in data["scope"]
    [match] = data["recordings"]
    assert (match["duration_ms"], match["releases"]) == (328000, [{"title": "Time Out", "date": "1959"}])


def test_musicbrainz_looks_up_five_candidates_and_numbers_evidence_by_match():
    """Musicbrainz looks up five candidates and numbers evidence by match."""
    session = Session({"recordings": [recording()]})
    provider = MusicBrainzProvider(session=session, cache=TTLCache(),
                                   limiter=RateLimiter(clock=lambda: 0.0, sleep=lambda delay: None))
    candidates = ([Candidate(f"Other {n}", "Nobody", ["web_1"]) for n in range(3)]
                  + [Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])]
                  + [Candidate(f"Late {n}", "Nobody", ["web_1"]) for n in range(3)])
    evidence = provider.lookup_recordings(candidates)
    assert len(session.calls) == 5  # the sixth and seventh candidates are never looked up
    assert [e.id for e in evidence] == ["mb_1"] and evidence[0].title.startswith("Take Five")


def test_artist_credits_are_joined_with_their_join_phrases():
    """Artist credits are joined with their join phrases."""
    row = {**recording(), "artist-credit": [{"name": "Dave Brubeck Quartet", "joinphrase": " feat. "},
                                            {"artist": {"name": "Paul Desmond"}}]}
    provider = MusicBrainzProvider(session=Session({"recordings": [row]}), cache=TTLCache(), limiter=RateLimiter())
    [evidence] = provider.lookup_recordings(
        [Candidate("Take Five", "Dave Brubeck Quartet feat. Paul Desmond", ["web_1"])])
    assert json.loads(evidence.text)["recordings"][0]["artist"] == "Dave Brubeck Quartet feat. Paul Desmond"


def test_long_disambiguations_and_release_lists_are_trimmed():
    """Long disambiguations and release lists are trimmed."""
    row = {**recording(), "disambiguation": "d" * 500,
           "releases": [{"title": f"R{n}", "date": "1959"} for n in range(5)]}
    provider = MusicBrainzProvider(session=Session({"recordings": [row]}), cache=TTLCache(), limiter=RateLimiter())
    [evidence] = provider.lookup_recordings([Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])])
    match = json.loads(evidence.text)["recordings"][0]
    assert len(match["disambiguation"]) == 300 and [r["title"] for r in match["releases"]] == ["R0", "R1", "R2"]


@pytest.mark.parametrize("error,code", [
    (requests.ConnectionError("fixture-secret"), "provider_unavailable"),
    (requests.Timeout(), "timeout"),
])
def test_musicbrainz_network_failures_map_to_safe_errors(error, code):
    """Musicbrainz network failures map to safe errors."""
    class Broken:
        """Fake session whose get raises the given error."""
        def get(self, *args, **kwargs):
            """Raise the configured error."""
            raise error

    provider = MusicBrainzProvider(session=Broken(), cache=TTLCache(), limiter=RateLimiter())
    with pytest.raises(DiscoveryError) as exc:
        provider.lookup_recordings([Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])])
    assert exc.value.code == code and "MusicBrainz" in str(exc.value) and "fixture-secret" not in str(exc.value)
