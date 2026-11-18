"""Tests for the discovery pipeline and how it is wired to the providers."""

# Behavior: runs the discovery pipeline with mocked providers to check validation, fallbacks and the single-retry limit.

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.providers import Settings
from src.composition import find_music
from src.discovery import _requirement_label, discover_music
from src.models import (MAX_JUDGED, MAX_RECOMMENDATIONS, MAX_WEB_SOURCES, Candidate, ConstraintCheck,
                        DiscoveryError, Evidence,
                        MusicRequest, ParsedIntent, PreferenceCheck, Recommendation,
                        SupportedClaim, TasteCriterion)


def source(title="Take Five", artist="Dave Brubeck Quartet", id="web_1"):
    """Build a web evidence item for a song."""
    return Evidence(id, "web", f"{title} — {artist}",
                    f"{title} by {artist}: instrumental jazz with saxophone.", "https://example.org/music")


def intent(*criteria):
    """Build a ParsedIntent from the given criteria."""
    return ParsedIntent(list(criteria), ["jazz songs"])


def recommendation(e, *, preference_checks=None, constraint_checks=None):
    """Build a valid recommendation for an evidence item."""
    title, artist = e.title.split(" — ")
    return Recommendation(title, artist, "The source describes instrumental jazz.", [e.id],
                          ["Suitability for studying is subjective."],
                          [SupportedClaim("Instrumental jazz", e.id, "instrumental jazz")],
                          constraint_checks or [], preference_checks or [])


def run(request=MusicRequest("instrumental jazz"), evidence=None, **overrides):
    """Run the discovery pipeline with mocked dependencies."""
    e = source() if evidence is None else evidence
    parsed = intent()
    dependencies = dict(analyze=Mock(return_value=parsed),
                        search=Mock(return_value=[e]),
                        extract=Mock(return_value=[Candidate(*e.title.split(" — "), [e.id])]),
                        generate=Mock(return_value=[recommendation(e)]))
    dependencies.update(overrides)
    return discover_music(request, **dependencies), dependencies


def test_discovery_uses_retrieved_evidence():
    """Discovery uses retrieved evidence."""
    for e in [source(), source("So What", "Miles Davis")]:
        result, deps = run(evidence=e)
        assert result.recommendations[0].title == e.title.split(" — ")[0]
        parsed = deps["analyze"].return_value
        assert deps["search"].call_args.args == (deps["search"].call_args.args[0], parsed)
        assert deps["extract"].call_args.args[1:] == (parsed, [e])
        assert deps["generate"].call_args.args[1:] == (
            parsed, deps["extract"].return_value, [e])
        assert result.intent == parsed


def test_empty_input_skips_providers():
    """Empty input skips providers."""
    analyze, search = Mock(), Mock()
    with pytest.raises(DiscoveryError, match="Describe"):
        run(MusicRequest("  "), analyze=analyze, search=search)
    analyze.assert_not_called()
    search.assert_not_called()


def test_no_evidence_skips_generation():
    """No evidence skips generation."""
    generate = Mock()
    result, _ = run(search=Mock(return_value=[]), generate=generate)
    assert not result.recommendations
    assert result.evidence == [] and len(result.warnings) == 1  # the early "no sources" exit, not some other empty result
    generate.assert_not_called()


def test_unknown_source_is_rejected():
    """Unknown source is rejected."""
    rec = replace(recommendation(source()), source_ids=["invented"])
    with pytest.raises(DiscoveryError) as exc:
        run(generate=Mock(return_value=[rec]))
    assert exc.value.code == "invalid_response"


def test_invalid_model_output_is_retried_once():
    """Invalid model output is retried once."""
    malformed = ParsedIntent([
        TasteCriterion("genre", "preference", "jazz", 2, "not in the request")
    ], ["jazz songs"])
    valid = intent()
    analyze = Mock(side_effect=[malformed, valid])

    result, _ = run(analyze=analyze, retry_invalid=True)

    assert result.recommendations
    assert result.intent == valid
    assert analyze.call_count == 2


def test_invalid_model_retry_is_bounded():
    """Invalid model retry is bounded."""
    malformed = ParsedIntent([
        TasteCriterion("genre", "preference", "jazz", 2, "not in the request")
    ], ["jazz songs"])
    analyze = Mock(return_value=malformed)

    with pytest.raises(DiscoveryError) as exc:
        run(analyze=analyze, retry_invalid=True)

    assert exc.value.code == "invalid_response"
    assert analyze.call_count == 2


@pytest.mark.parametrize("description,reference", [
    ("instrumental piano with no vocals", None),
    ("música em português com percussão", None),
    ("heavy guitar with clean singing, no screaming", None),
    ("similar rhythm, different artist", "Take Five — Dave Brubeck Quartet"),
])
def test_constraints_and_reference_are_passed_unchanged(description, reference):
    """Constraints and reference are passed unchanged."""
    req = MusicRequest(description, reference)
    _, deps = run(req)
    assert deps["analyze"].call_args.args[0] == req
    assert deps["generate"].call_args.args[0] == req


def test_base_skips_enhancements_and_enhanced_receives_them():
    """Base skips enhancements and enhanced receives them."""
    doc = Evidence("guide_1", "guide", "Pace", "Tempo and energy differ.")
    meta = Evidence("mb_1", "metadata", "Take Five — Dave Brubeck Quartet",
                    "Recording version unknown", "https://musicbrainz.org/recording/123")
    documents, metadata = Mock(return_value=[doc]), Mock(return_value=[meta])
    _, deps = run(documents=documents, metadata=metadata)
    documents.assert_not_called()
    metadata.assert_not_called()
    _, deps = run(MusicRequest("instrumental jazz", enhanced=True), documents=documents, metadata=metadata)
    parsed = deps["analyze"].return_value
    documents.assert_called_once_with(MusicRequest("instrumental jazz", enhanced=True), parsed)
    assert doc in deps["generate"].call_args.args[3]
    assert meta in deps["generate"].call_args.args[3]


def test_metadata_only_reports_missing_matches_for_candidates_it_checked():
    """Metadata only reports missing matches for candidates it checked."""
    evidence = [source(f"Song {index}", f"Artist {index}", f"web_{index}")
                for index in range(1, 7)]
    candidates = [Candidate(*item.title.split(" — "), [item.id]) for item in evidence]
    metadata_evidence = [
        Evidence(f"mb_{index}", "metadata", item.title, "Recording identity metadata",
                 f"https://musicbrainz.org/recording/10000000-0000-0000-0000-00000000000{index}")
        for index, item in enumerate(evidence[:5], 1)
    ]
    metadata = Mock(return_value=metadata_evidence)

    result, _ = run(
        MusicRequest("instrumental jazz", enhanced=True),
        evidence=evidence[0],
        search=Mock(return_value=evidence),
        extract=Mock(return_value=candidates),
        metadata=metadata,
    )

    assert metadata.call_args.args == (candidates[:5],)
    assert not any("did not match" in warning for warning in result.warnings)
    assert any("do not confirm" in notice for notice in result.notices)


def test_metadata_warns_when_a_checked_candidate_has_no_match():
    """Metadata warns when a checked candidate has no match."""
    meta = Evidence("mb_1", "metadata", "Take Five — Dave Brubeck Quartet",
                    "Recording identity metadata",
                    "https://musicbrainz.org/recording/10000000-0000-0000-0000-000000000001")
    result, _ = run(
        MusicRequest("instrumental jazz", enhanced=True),
        metadata=Mock(return_value=[]),
    )
    matched_result, _ = run(
        MusicRequest("instrumental jazz", enhanced=True),
        metadata=Mock(return_value=[meta]),
    )

    assert any("did not match every checked candidate" in warning for warning in result.warnings)
    assert not matched_result.warnings


def test_identity_hallucination_is_not_recommended():
    """Identity hallucination is not recommended."""
    generate = Mock()
    result, _ = run(extract=Mock(return_value=[Candidate("Invented song", "Invented artist", ["web_1"])]),
                    generate=generate)
    assert result.recommendations == []
    assert result.warnings == ["Sources were found, but no song and artist identities could be verified."]
    generate.assert_not_called()


def test_unrelated_citation_and_fabricated_quote_rejected():
    """Unrelated citation and fabricated quote rejected."""
    rec = recommendation(source())
    for bad in [replace(rec, source_ids=["guide_1"]),
                replace(rec, claims=[SupportedClaim("piano", "web_1", "piano")])]:
        with pytest.raises(DiscoveryError):
            run(generate=Mock(return_value=[bad]))


def test_metadata_failure_is_visible():
    """Metadata failure is visible."""
    result, _ = run(MusicRequest("instrumental jazz", enhanced=True),
                    metadata=Mock(side_effect=DiscoveryError("timeout", "safe error")))
    assert result.recommendations
    assert "MusicBrainz is unavailable" in result.warnings[0]


def test_duplicate_recommendations_are_removed():
    """Duplicate recommendations are removed."""
    rec = recommendation(source())
    result, _ = run(generate=Mock(return_value=[rec, rec]))
    assert len(result.recommendations) == 1


def test_soft_preferences_are_scored_and_partial_matches_are_kept():
    """Soft preferences are scored and partial matches are kept."""
    e = Evidence("web_1", "web", "Gentle Joy — Example Artist",
                 "Gentle Joy by Example Artist is peaceful, uplifting, and softly performed.",
                 "https://example.org/gentle-joy")
    happy = TasteCriterion("mood", "preference", "happy", 2, "happy")
    peaceful = TasteCriterion("mood", "preference", "peaceful", 3, "very peaceful")
    energy = TasteCriterion("energy", "preference", "low", 2, "very peaceful")
    parsed = ParsedIntent([happy, peaceful, energy],
                          ["happy peaceful songs", "gentle uplifting low-energy songs"])
    checks = [
        PreferenceCheck("mood:happy:preference", "partial", e.id, "uplifting"),
        PreferenceCheck("mood:peaceful:preference", "supported", e.id, "peaceful"),
        PreferenceCheck("energy:low:preference", "partial", e.id, "softly performed"),
    ]
    rec = Recommendation("Gentle Joy", "Example Artist", "Peaceful and uplifting.", [e.id], [],
                         [SupportedClaim("Peaceful and uplifting", e.id, "peaceful, uplifting")],
                         [], checks)
    result, _ = run(MusicRequest("I want a happy song but very peaceful"), evidence=e,
                    analyze=Mock(return_value=parsed),
                    extract=Mock(return_value=[Candidate("Gentle Joy", "Example Artist", [e.id])]),
                    generate=Mock(return_value=[rec]))
    assert result.recommendations[0].fit_score == pytest.approx(5 / 7)
    assert result.used_partial_matches is True


def test_candidate_without_any_preference_match_is_listed_unscored_not_as_a_match():
    """A song with no supported preference gets no score, but its source still names it, so it is listed."""
    e = source()
    preference = TasteCriterion("mood", "preference", "peaceful", 3, "peaceful")
    parsed = ParsedIntent([preference], ["peaceful songs"])
    rec = recommendation(e, preference_checks=[
        PreferenceCheck("mood:peaceful:preference", "unknown", "", "")
    ])
    result, _ = run(MusicRequest("peaceful"), evidence=e, analyze=Mock(return_value=parsed),
                    generate=Mock(return_value=[rec]))
    assert [(r.title, r.fit_score) for r in result.recommendations] == [("Take Five", None)]
    assert result.recommendations[0].uncertainties == ["We couldn't check this song's fit against the source."]
    assert result.warnings == []


def test_a_song_a_source_contradicts_on_a_preference_is_not_listed_at_all():
    """Topping up never brings back a song whose verdict says the source contradicts a preference."""
    e = source()
    preference = TasteCriterion("mood", "preference", "peaceful", 3, "peaceful")
    rec = recommendation(e, preference_checks=[
        PreferenceCheck("mood:peaceful:preference", "contradicted", "web_1", "instrumental jazz")])
    result, _ = run(MusicRequest("peaceful"), evidence=e,
                    analyze=Mock(return_value=ParsedIntent([preference], ["peaceful songs"])),
                    generate=Mock(return_value=[rec]))
    assert result.recommendations == []
    assert "matching any requested preference" in result.warnings[-1]


def test_checked_songs_come_first_and_the_rest_are_topped_up_from_the_sources():
    """A broad request keeps its checked songs and fills the list with the other songs the sources name."""
    pages = [source(f"Song {n}", f"Artist {n}", f"web_{n}") for n in range(1, 8)]
    candidates = [Candidate(f"Song {n}", f"Artist {n}", [f"web_{n}"]) for n in range(1, 8)]
    result, deps = run(search=Mock(return_value=pages), extract=Mock(return_value=candidates),
                       generate=Mock(return_value=[recommendation(pages[4])]))  # the model judged one song
    assert [(r.title, r.fit_score) for r in result.recommendations] == [
        ("Song 5", 1.0), ("Song 1", None), ("Song 2", None), ("Song 3", None), ("Song 4", None),
        ("Song 6", None), ("Song 7", None)]
    assert deps["generate"].call_count == 1  # topping up costs no extra model call
    assert not result.warnings


def test_topping_up_skips_songs_the_model_judged_merely_named_not_tied_to_the_request():
    """A rock request is not padded with songs a generic list names but the model says it never ties to rock."""
    pages = [Evidence("web_1", "web", "The 200 Best Rock Songs", "Seven Nation Army by The White Stripes.", "https://example.org/a"),
             Evidence("web_2", "web", "500 Best Songs of All Time", "Africa by Toto is a classic.", "https://example.org/b")]
    candidates = [Candidate("Seven Nation Army", "The White Stripes", ["web_1"], fits_request=True),
                  Candidate("Africa", "Toto", ["web_2"], fits_request=False)]
    result, _ = run(MusicRequest("rock"), search=Mock(return_value=pages), extract=Mock(return_value=candidates),
                    generate=Mock(return_value=[]))
    assert [r.title for r in result.recommendations] == ["Seven Nation Army"]


def test_topping_up_stops_at_the_display_limit():
    """The checked and topped-up songs together never exceed MAX_RECOMMENDATIONS."""
    total = MAX_RECOMMENDATIONS + 4
    pages = [source(f"Song {n}", f"Artist {n}", f"web_{n}") for n in range(1, total + 1)]
    candidates = [Candidate(f"Song {n}", f"Artist {n}", [f"web_{n}"]) for n in range(1, total + 1)]
    result, _ = run(search=Mock(return_value=pages), extract=Mock(return_value=candidates),
                    generate=Mock(return_value=[recommendation(pages[0])]))
    assert len(result.recommendations) == MAX_RECOMMENDATIONS
    assert result.recommendations[0].fit_score == 1.0


def test_a_request_with_hard_criteria_is_never_topped_up():
    """Source-named songs cannot stand in for a requirement nothing established."""
    hard = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
    pages = [source(f"Song {n}", f"Artist {n}", f"web_{n}") for n in range(1, 4)]
    candidates = [Candidate(f"Song {n}", f"Artist {n}", [f"web_{n}"]) for n in range(1, 4)]
    checked = recommendation(pages[0], constraint_checks=[
        ConstraintCheck("vocals:vocals:must_avoid", "supported", "web_1", "instrumental jazz")])
    result, _ = run(MusicRequest("jazz with no vocals"), search=Mock(return_value=pages),
                    extract=Mock(return_value=candidates),
                    analyze=Mock(return_value=ParsedIntent([hard], ["instrumental jazz songs"])),
                    generate=Mock(return_value=[checked]))
    assert [r.title for r in result.recommendations] == ["Song 1"]


def test_unknown_hard_constraint_lists_the_song_as_unverified_instead_of_matching_it():
    """A hard criterion nothing established never becomes a match: the song is listed, labeled unverified."""
    e = source()
    hard = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
    parsed = ParsedIntent([hard], ["instrumental jazz songs"])
    rec = recommendation(e, constraint_checks=[
        ConstraintCheck("vocals:vocals:must_avoid", "unknown", "", "")
    ])
    result, _ = run(MusicRequest("jazz with no vocals"), evidence=e,
                    analyze=Mock(return_value=parsed), generate=Mock(return_value=[rec]))
    assert [r.title for r in result.recommendations] == ["Take Five"]
    song = result.recommendations[0]
    assert song.fit_score is None  # never presented as a scored match
    assert song.uncertainties == ["We couldn't confirm no vocals."]
    assert result.unverified_requirements == ["no vocals"]
    assert "couldn't confirm that any song meets every requirement (no vocals)" in result.warnings[-1]
    assert "required or excluded trait" not in " ".join(result.warnings)


def test_a_song_a_check_found_to_break_a_requirement_is_never_listed_even_as_unverified():
    """A contradicted song is dropped; only songs nothing ruled out are listed as unverified."""
    first = source("Song 1", "Artist 1", "web_1")
    breaks = recommendation(first, constraint_checks=[
        ConstraintCheck(HARD_KEY, "contradicted", first.id, "instrumental jazz")])
    result, *_ = run_two_songs(Mock(return_value=[breaks]))  # Song 2 was never judged
    assert [rec.title for rec in result.recommendations] == ["Song 2"]
    assert result.unverified_requirements == ["no vocals"]


def test_when_every_candidate_breaks_a_requirement_nothing_is_listed():
    """With every candidate contradicted there is nothing to show, so the old warning stands."""
    e = source()
    hard = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
    rec = recommendation(e, constraint_checks=[
        ConstraintCheck("vocals:vocals:must_avoid", "contradicted", e.id, "instrumental jazz")])
    result, _ = run(MusicRequest("jazz with no vocals"), evidence=e,
                    analyze=Mock(return_value=ParsedIntent([hard], ["instrumental jazz songs"])),
                    generate=Mock(return_value=[rec]))
    assert result.recommendations == [] and result.unverified_requirements == []
    assert "required or excluded trait" in result.warnings[-1]


@pytest.mark.parametrize("kind,value,label", [
    ("must_have", "clean singing", "clean singing"),
    ("must_avoid", "screaming", "no screaming"),
    ("must_avoid", "no vocals", "no vocals"),         # already says it
    ("must_avoid", "Without  Vocals", "Without Vocals"),
    ("must_avoid", "sem voz", "sem voz"),
])
def test_requirement_labels_read_the_way_the_listener_said_them(kind, value, label):
    """Requirement labels read the way the listener said them."""
    assert _requirement_label(TasteCriterion("vocals", kind, value, 3, value)) == label


def test_cross_song_citation_cannot_support_traits():
    """Cross song citation cannot support traits."""
    e = source()
    other = source("Unrelated", "Other", "web_2")
    rec = replace(recommendation(e), source_ids=[e.id, other.id],
                  claims=[SupportedClaim("Instrumental jazz", other.id, "instrumental jazz")])
    with pytest.raises(DiscoveryError):
        run(search=Mock(return_value=[e, other]), generate=Mock(return_value=[rec]))


@pytest.mark.parametrize("url", ["javascript:alert(1)", "https://user:password@example.org", "http://127.0.0.1/a", "http://localhost", "https://example.org/ bad"])
def test_unsafe_source_urls_are_rejected(url):
    """Unsafe source urls are rejected."""
    with pytest.raises(DiscoveryError):
        run(evidence=replace(source(), url=url))


@pytest.mark.parametrize("request_", [MusicRequest("a" * 1201), MusicRequest("jazz", "a" * 201)])
def test_oversized_input_is_rejected_before_providers_run(request_):
    """Oversized input is rejected before providers run."""
    analyze = Mock()
    with pytest.raises(DiscoveryError) as exc:
        run(request_, analyze=analyze)
    assert exc.value.code == "invalid_input"
    analyze.assert_not_called()


def criterion(**overrides):
    """Build a genre preference criterion with optional overrides."""
    return TasteCriterion(**{"facet": "genre", "kind": "preference", "value": "jazz",
                             "weight": 2, "source_text": "jazz", **overrides})


def test_provider_errors_are_not_retried_or_degraded():
    """Provider errors are not retried or degraded."""
    analyze = Mock(side_effect=DiscoveryError("timeout", "slow"))
    with pytest.raises(DiscoveryError) as exc:
        run(analyze=analyze, retry_invalid=True, degrade_invalid=True)
    assert exc.value.code == "timeout"
    assert analyze.call_count == 1


def test_invalid_intent_degrades_to_the_original_wording():
    """Invalid intent degrades to the original wording."""
    malformed = ParsedIntent([criterion(source_text="not in the request")], ["jazz songs"])
    result, deps = run(MusicRequest("instrumental jazz"), analyze=Mock(return_value=malformed),
                       degrade_invalid=True)
    assert result.intent == ParsedIntent([], ["instrumental jazz"])
    assert deps["search"].call_args.args[1] == result.intent
    assert any("searched using your exact words" in warning for warning in result.warnings)
    assert result.recommendations


def test_invalid_candidates_degrade_to_no_results():
    """Invalid candidates degrade to no results."""
    generate = Mock()
    unknown_source = Candidate("Take Five", "Dave Brubeck Quartet", ["invented"])
    result, _ = run(extract=Mock(return_value=[unknown_source]), generate=generate,
                    degrade_invalid=True)
    assert result.recommendations == []
    assert any("couldn’t confirm any specific songs" in warning for warning in result.warnings)
    generate.assert_not_called()


def test_invalid_recommendations_degrade_to_source_identified_songs():
    """Invalid recommendations degrade to source identified songs."""
    e = source()
    invalid = replace(recommendation(e), source_ids=["invented"])
    result, _ = run(evidence=e, generate=Mock(return_value=[invalid]), degrade_invalid=True)
    [rec] = result.recommendations
    assert (rec.title, rec.source_ids, rec.fit_score) == ("Take Five", ["web_1"], None)
    assert rec.uncertainties
    assert rec.claims[0].quote in e.text
    assert rec.explanation.startswith("“")
    assert "instrumental jazz with saxophone" in rec.explanation
    assert any("without a match score" in warning for warning in result.warnings)


def run_fallback(text, songs, *wanted, source_title="Best slow songs for weddings"):
    """Make validation fail so the source-identified fallback builds the cards."""
    e = Evidence("web_1", "web", source_title, text, "https://example.org/music")
    invalid = [replace(recommendation(source(title, artist)), source_ids=["invented"])
               for title, artist in songs]
    result, _ = run(search=Mock(return_value=[e]),
                    extract=Mock(return_value=[Candidate(t, a, ["web_1"]) for t, a in songs]),
                    generate=Mock(return_value=invalid), degrade_invalid=True,
                    analyze=Mock(return_value=intent(*wanted)))
    return result.recommendations


def test_fallback_quotes_the_source_and_says_which_requested_traits_it_mentions():
    """Fallback quotes the source and says which requested traits it mentions."""
    text = ("1. Perfect by Ed Sheeran is a gentle love song. 2. Take Five by Dave Brubeck Quartet is a "
            "slow ballad with an upbeat saxophone. See https://example.org/more or [more](https://example.org/x).")
    [rec] = run_fallback(text, [("Take Five", "Dave Brubeck Quartet")],
                         criterion(value="slower beats", source_text="jazz"),
                         criterion(facet="mood", value="happy", source_text="instrumental"),
                         source_title="Weddings")
    assert rec.fit_score is None
    assert "Take Five by Dave Brubeck Quartet is a slow ballad" in rec.explanation
    assert "slower beats (“slow”)" in rec.explanation
    assert "happy (“upbeat”)" in rec.explanation
    assert "Perfect" not in rec.explanation and "http" not in rec.explanation


def test_fallback_does_not_count_the_songs_own_title_as_evidence():
    """Fallback does not count the songs own title as evidence."""
    [rec] = run_fallback("Happy by Pharrell Williams is a song.", [("Happy", "Pharrell Williams")],
                         criterion(facet="mood", value="happy", source_text="jazz"), source_title="Playlist")
    assert "don’t mention happy" in rec.explanation


def test_fallback_puts_songs_whose_source_mentions_the_request_first():
    """Fallback puts songs whose source mentions the request first."""
    text = "Song One by Artist One is loud and fast. Song Two by Artist Two is a slow track."
    recs = run_fallback(text, [("Song One", "Artist One"), ("Song Two", "Artist Two")],
                        criterion(value="slower beats", source_text="jazz"), source_title="Playlist")
    assert [rec.title for rec in recs] == ["Song Two", "Song One"]


def with_fabricated_quote(rec):
    """Return a copy of the recommendation whose claim quotes text the source lacks."""
    return replace(rec, claims=[SupportedClaim("piano", rec.source_ids[0], "piano")])


def test_one_invalid_song_does_not_discard_the_valid_ones(caplog):
    """One invalid song does not discard the valid ones."""
    first, second = source("Song 1", "Artist 1", "web_1"), source("Song 2", "Artist 2", "web_2")
    candidates = [Candidate(*e.title.split(" — "), [e.id]) for e in (first, second)]
    caplog.set_level("INFO", logger="src.validation")
    result, deps = run(search=Mock(return_value=[first, second]), extract=Mock(return_value=candidates),
                       generate=Mock(return_value=[recommendation(first), with_fabricated_quote(recommendation(second))]),
                       retry_invalid=True, degrade_invalid=True)
    # Song 2's invented quote costs it its score, but its source still names it, so it follows unscored.
    assert [(rec.title, rec.fit_score) for rec in result.recommendations] == [("Song 1", 1.0), ("Song 2", None)]
    assert deps["generate"].call_count == 1
    assert not result.warnings
    assert "recommendation_rejected reason=claim_quote_not_in_source position=1" in caplog.text


def test_all_invalid_songs_raise_with_feedback_for_the_model():
    """All invalid songs raise with feedback for the model."""
    e = source()
    with pytest.raises(DiscoveryError) as exc:
        run(evidence=e, generate=Mock(return_value=[with_fabricated_quote(recommendation(e))]))
    assert exc.value.code == "invalid_response"
    assert "Take Five by Dave Brubeck Quartet" in exc.value.feedback
    assert "verbatim" in exc.value.feedback


def test_invalid_recommendations_are_repaired_on_the_retry():
    """Invalid recommendations are repaired on the retry."""
    e = source()
    bad, good = with_fabricated_quote(recommendation(e)), recommendation(e)
    generate = Mock(side_effect=[[bad], [good]])
    result, _ = run(evidence=e, generate=generate, retry_invalid=True, degrade_invalid=True)
    assert [rec.title for rec in result.recommendations] == ["Take Five"]
    assert result.recommendations[0].fit_score == 1.0
    assert not result.warnings
    first_call, retry_call = generate.call_args_list
    assert "feedback" not in first_call.kwargs
    assert "verbatim" in retry_call.kwargs["feedback"]


def test_failed_repair_still_degrades_after_a_single_retry():
    """Failed repair still degrades after a single retry."""
    e = source()
    generate = Mock(return_value=[with_fabricated_quote(recommendation(e))])
    result, _ = run(evidence=e, generate=generate, retry_invalid=True, degrade_invalid=True)
    assert generate.call_count == 2
    assert [rec.fit_score for rec in result.recommendations] == [None]
    assert any("without a match score" in warning for warning in result.warnings)


def test_degraded_recommendations_never_match_a_hard_criterion():
    """Degraded recommendations never match a hard criterion; they are listed as unverified."""
    hard = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
    invalid = replace(recommendation(source()), source_ids=["invented"])
    result, _ = run(MusicRequest("jazz with no vocals"),
                    analyze=Mock(return_value=ParsedIntent([hard], ["instrumental jazz songs"])),
                    generate=Mock(return_value=[invalid]), degrade_invalid=True)
    assert [(rec.title, rec.fit_score) for rec in result.recommendations] == [("Take Five", None)]
    assert result.unverified_requirements == ["no vocals"]
    assert "couldn't confirm that any song meets every requirement" in result.warnings[-1]


def test_unestablished_recommendation_gets_a_generic_warning():
    """Unestablished recommendation gets a generic warning when not even a source-named song remains."""
    with patch("src.discovery._source_identified", return_value=[]):
        result, _ = run(generate=Mock(return_value=[]))
    assert result.warnings == ["The retrieved evidence did not establish a supported recommendation."]


HARD = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
HARD_KEY = "vocals:vocals:must_avoid"


def run_two_songs(generate, **overrides):
    """Two candidates and a hard requirement, so a song can be judged and excluded."""
    first, second = source("Song 1", "Artist 1", "web_1"), source("Song 2", "Artist 2", "web_2")
    overrides.setdefault("analyze", Mock(return_value=ParsedIntent([HARD], ["instrumental jazz songs"])))
    result, deps = run(MusicRequest("jazz with no vocals"),
                       search=Mock(return_value=[first, second]),
                       extract=Mock(return_value=[Candidate(*e.title.split(" — "), [e.id]) for e in (first, second)]),
                       generate=generate, **overrides)
    return result, deps, first, second


def excluded(e):
    """Build a recommendation whose hard criterion is unknown, so it gets excluded."""
    return recommendation(e, constraint_checks=[ConstraintCheck(HARD_KEY, "unknown", "", "")])


def established(e):
    """Build a recommendation whose hard criterion is supported by the source."""
    return recommendation(e, constraint_checks=[ConstraintCheck(HARD_KEY, "supported", e.id, "instrumental jazz")])


def test_candidates_the_model_skipped_are_checked_when_none_survived():
    """Candidates the model skipped are checked when none survived."""
    # The model judged only Song 1 and it was excluded; Song 2 used to be dropped unseen.
    second = source("Song 2", "Artist 2", "web_2")
    generate = Mock(side_effect=[[excluded(source("Song 1", "Artist 1", "web_1"))], [established(second)]])
    result, deps, first, second = run_two_songs(generate, retry_invalid=True)
    assert [rec.title for rec in result.recommendations] == ["Song 2"]
    first_call, follow_up = generate.call_args_list
    assert [c.title for c in first_call.args[2]] == ["Song 1", "Song 2"]
    assert [c.title for c in follow_up.args[2]] == ["Song 2"]  # only what was not judged
    assert follow_up.args[3] == first_call.args[3]  # same evidence
    assert "feedback" not in follow_up.kwargs
    assert not result.warnings


def test_follow_up_check_is_skipped_when_a_song_survived():
    """Follow up check is skipped when a song survived."""
    generate = Mock(return_value=[established(source("Song 1", "Artist 1", "web_1"))])
    result, _, *_ = run_two_songs(generate, retry_invalid=True)
    assert [rec.title for rec in result.recommendations] == ["Song 1"]
    assert generate.call_count == 1


def test_follow_up_check_is_skipped_when_the_model_judged_nothing():
    """Follow up check is skipped when the model judged nothing."""
    # At temperature 0 asking again about the same candidates would give the same empty answer.
    generate = Mock(return_value=[])
    result, *_ = run_two_songs(generate, retry_invalid=True)
    assert all(rec.fit_score is None for rec in result.recommendations)  # nothing verified or scored
    assert result.unverified_requirements == ["no vocals"]
    assert generate.call_count == 1


def test_follow_up_check_needs_the_shared_retry_so_the_call_limit_holds():
    """Follow up check needs the shared retry so the call limit holds."""
    skipped = Mock(side_effect=[[excluded(source("Song 1", "Artist 1", "web_1"))], [established(source("Song 2", "Artist 2", "web_2"))]])
    result, *_ = run_two_songs(skipped)  # retry_invalid is off
    assert result.unverified_requirements == ["no vocals"] and skipped.call_count == 1

    # An earlier stage already spent the retry, so there is no call left for a follow-up.
    malformed = ParsedIntent([TasteCriterion("genre", "preference", "jazz", 2, "not in the request")], ["jazz"])
    spent = Mock(return_value=[excluded(source("Song 1", "Artist 1", "web_1"))])
    result, *_ = run_two_songs(spent, retry_invalid=True,
                               analyze=Mock(side_effect=[malformed, ParsedIntent([HARD], ["instrumental jazz songs"])]))
    assert result.unverified_requirements == ["no vocals"] and spent.call_count == 1


def test_failed_follow_up_check_keeps_the_result_and_warns():
    """Failed follow up check keeps the result and warns."""
    generate = Mock(side_effect=[[excluded(source("Song 1", "Artist 1", "web_1"))],
                                 DiscoveryError("quota", "Gemini is rate limited or out of quota.")])
    result, *_ = run_two_songs(generate, retry_invalid=True)
    assert result.unverified_requirements == ["no vocals"]
    assert all(rec.fit_score is None for rec in result.recommendations)
    assert any("couldn’t check every song" in warning for warning in result.warnings)
    assert "couldn't confirm that any song meets every requirement" in result.warnings[-1]


def test_unavailable_documents_fall_back_to_external_evidence():
    """Unavailable documents fall back to external evidence."""
    result, _ = run(MusicRequest("instrumental jazz", enhanced=True),
                    documents=Mock(side_effect=DiscoveryError("knowledge_unavailable", "safe error")))
    assert result.recommendations
    assert "Custom documents are unavailable" in result.warnings[0]


def test_duplicate_evidence_ids_are_rejected():
    """Duplicate evidence ids are rejected."""
    duplicate = source("So What", "Miles Davis")
    with pytest.raises(DiscoveryError):
        run(search=Mock(return_value=[source(), duplicate]))


def test_supported_hard_constraint_keeps_the_candidate():
    """Supported hard constraint keeps the candidate."""
    e = source()
    hard = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
    rec = recommendation(e, constraint_checks=[
        ConstraintCheck("vocals:vocals:must_avoid", "supported", e.id, "instrumental jazz")
    ])
    result, _ = run(MusicRequest("jazz with no vocals"), evidence=e,
                    analyze=Mock(return_value=ParsedIntent([hard], ["instrumental jazz songs"])),
                    generate=Mock(return_value=[rec]))
    assert [r.title for r in result.recommendations] == ["Take Five"]
    assert result.recommendations[0].fit_score == 1.0
    assert result.unverified_requirements == []  # a verified song is a real match, so nothing is flagged


@pytest.mark.parametrize("checks", [
    [],
    [PreferenceCheck("mood:peaceful:preference", "bogus", "web_1", "instrumental jazz")],
    [PreferenceCheck("mood:peaceful:preference", "unknown", "web_1", "instrumental jazz")],
    [PreferenceCheck("mood:peaceful:preference", "supported", "web_1", "soft ballad")],
], ids=["missing", "bad-status", "unknown-with-quote", "fabricated-quote"])
def test_incomplete_or_unquoted_preference_checks_cost_the_score_not_the_song(checks):
    """A preference check that is missing or does not quote the source counts as unknown: no score, no rejection."""
    e = source()
    peaceful = TasteCriterion("mood", "preference", "peaceful", 3, "peaceful")
    result, deps = run(MusicRequest("peaceful"), evidence=e,
                       analyze=Mock(return_value=ParsedIntent([peaceful], ["peaceful songs"])),
                       generate=Mock(return_value=[recommendation(e, preference_checks=checks)]))
    assert [(rec.title, rec.fit_score) for rec in result.recommendations] == [("Take Five", None)]
    assert deps["generate"].call_count == 1  # nothing was rejected, so there was no retry


def test_results_are_ranked_by_fit_and_capped_at_the_display_limit():
    """Results are ranked by fit and capped at MAX_RECOMMENDATIONS."""
    evidence = [source(f"Song {i}", f"Artist {i}", f"web_{i}") for i in range(1, MAX_JUDGED + 1)]
    parsed = ParsedIntent([criterion(weight=1)], ["jazz songs"])

    def rec(e, status):
        """Build a recommendation with one preference check of the given status."""
        return recommendation(e, preference_checks=[
            PreferenceCheck("genre:jazz:preference", status, e.id, "instrumental jazz")])

    recs = [rec(e, "partial") for e in evidence[:-1]] + [rec(evidence[-1], "supported")]
    result, _ = run(MusicRequest("jazz"), analyze=Mock(return_value=parsed),
                    search=Mock(return_value=evidence),
                    extract=Mock(return_value=[Candidate(*e.title.split(" — "), [e.id]) for e in evidence]),
                    generate=Mock(return_value=recs))
    assert len(result.recommendations) == MAX_RECOMMENDATIONS
    assert (result.recommendations[0].title, result.recommendations[0].fit_score) == (f"Song {MAX_JUDGED}", 1.0)
    assert result.used_partial_matches is True


@pytest.fixture
def providers():
    """Patch the providers and the discovery function that find_music builds."""
    with patch("src.composition.GeminiProvider") as gemini, \
            patch("src.composition.GeminiEmbeddingProvider") as embedding, \
            patch("src.composition.TavilyProvider") as tavily, \
            patch("src.composition.MusicBrainzProvider") as musicbrainz, \
            patch("src.composition.discover_music") as discover:
        yield SimpleNamespace(gemini=gemini, embedding=embedding, tavily=tavily,
                              musicbrainz=musicbrainz, discover=discover)


@pytest.mark.parametrize("enhanced", [False, True])
def test_find_music_wires_providers_and_always_closes_them(providers, enhanced):
    """Find music wires providers and always closes them."""
    settings = Settings("fixture", "fixture")
    providers.gemini.return_value.close.side_effect = RuntimeError("cleanup failure is swallowed")
    result = find_music(MusicRequest("jazz", enhanced=enhanced), settings)

    kwargs = providers.discover.call_args.kwargs
    assert result is providers.discover.return_value
    assert kwargs["retry_invalid"] and kwargs["degrade_invalid"]
    assert (kwargs["documents"] is not None) is enhanced
    assert (kwargs["metadata"] is not None) is enhanced
    assert providers.musicbrainz.called is enhanced
    assert providers.embedding.called is enhanced
    providers.tavily.assert_called_once_with(
        settings.tavily_api_key,
        music_source_focus_enabled=settings.music_source_focus_enabled,
    )
    if enhanced:
        providers.musicbrainz.assert_called_once_with(settings.musicbrainz_contact)
        providers.embedding.assert_called_once_with(settings, client=providers.gemini.return_value.client)
        providers.musicbrainz.return_value.session.close.assert_called_once()
    providers.tavily.return_value.session.close.assert_called_once()


def test_find_music_can_disable_semantic_retrieval_without_disabling_local_documents(providers):
    """The kill switch uses the original lexical callback and keeps enhanced metadata enabled."""
    settings = Settings("fixture", "fixture", semantic_retrieval_enabled=False)
    find_music(MusicRequest("jazz", enhanced=True), settings)
    kwargs = providers.discover.call_args.kwargs
    assert kwargs["documents"] is not None and kwargs["metadata"] is not None
    providers.embedding.assert_not_called()


@pytest.mark.parametrize("error,code", [
    (DiscoveryError("quota", "limit"), "quota"),
    (RuntimeError("fixture-secret"), "provider_unavailable"),
])
def test_find_music_surfaces_only_safe_errors(providers, error, code):
    """Find music surfaces only safe errors."""
    providers.discover.side_effect = error
    with pytest.raises(DiscoveryError) as exc:
        find_music(MusicRequest("jazz"), Settings("fixture", "fixture"))
    assert exc.value.code == code
    assert "fixture-secret" not in str(exc.value)
    providers.gemini.return_value.close.assert_called_once()


# --- discovery pipeline: input handling and stage behavior ---------------------------------

def test_degraded_intent_searches_the_whole_request_collapsed_and_capped():
    """Degraded intent searches the whole request collapsed and capped."""
    malformed = ParsedIntent([criterion(source_text="not in the request")], ["jazz songs"])
    long_request = MusicRequest("  instrumental   jazz\npiano " + "x" * 300)
    result, _ = run(long_request, analyze=Mock(return_value=malformed), degrade_invalid=True)
    [query] = result.intent.search_queries
    assert query.startswith("instrumental jazz piano xxx") and len(query) == 200

    with_reference, _ = run(MusicRequest("jazz  piano", "So What — Miles Davis"),
                            analyze=Mock(return_value=malformed), degrade_invalid=True)
    assert with_reference.intent.search_queries == ["jazz piano So What — Miles Davis"]


def test_only_the_first_max_web_sources_search_results_are_used():
    """Only the first MAX_WEB_SOURCES search results are used."""
    pages = [source(f"Song {n}", f"Artist {n}", f"web_{n}") for n in range(1, MAX_WEB_SOURCES + 4)]
    extract = Mock(return_value=[])
    result, _ = run(search=Mock(return_value=pages), extract=extract)
    assert extract.call_args.args[2] == pages[:MAX_WEB_SOURCES] and result.evidence == pages[:MAX_WEB_SOURCES]


@pytest.mark.parametrize("bad", [
    replace(source(), id=""),
    replace(source(), kind="metadata"),   # only MusicBrainz lookups may produce this kind
    replace(source(), kind="other"),
    replace(source(), url=None),          # web pages need a public link; only guide sections have none
], ids=["blank-id", "metadata-from-search", "unknown-kind", "web-without-url"])
def test_search_evidence_needs_an_id_a_known_kind_and_a_public_url(bad):
    """Search evidence needs an id a known kind and a public url."""
    with pytest.raises(DiscoveryError) as exc:
        run(evidence=bad)
    assert exc.value.code == "invalid_response"


def test_guide_sections_alone_are_not_music_sources():
    """Guide sections alone are not music sources."""
    guide = Evidence("guide_1", "guide", "Pace", "Tempo and energy differ.")
    extract = Mock()
    result, _ = run(search=Mock(return_value=[guide]), extract=extract)
    assert result.recommendations == [] and result.evidence == [guide]
    assert result.warnings == ["No relevant music sources were found for the interpreted preferences."]
    extract.assert_not_called()


def test_invalid_candidates_are_retried_once_before_giving_up():
    """Invalid candidates are retried once before giving up."""
    e = source()
    unknown = Candidate("Take Five", "Dave Brubeck Quartet", ["invented"])
    known = Candidate("Take Five", "Dave Brubeck Quartet", [e.id])
    extract = Mock(side_effect=[[unknown], [known]])
    result, _ = run(evidence=e, extract=extract, retry_invalid=True)
    assert extract.call_count == 2
    assert [rec.title for rec in result.recommendations] == ["Take Five"] and not result.warnings


def test_a_provider_error_during_the_retry_is_surfaced_not_degraded():
    """A provider error during the retry is surfaced not degraded."""
    malformed = ParsedIntent([criterion(source_text="not in the request")], ["jazz songs"])
    analyze = Mock(side_effect=[malformed, DiscoveryError("quota", "limit")])
    with pytest.raises(DiscoveryError) as exc:
        run(analyze=analyze, retry_invalid=True, degrade_invalid=True)
    assert exc.value.code == "quota" and analyze.call_count == 2


@pytest.mark.parametrize("unusable", [
    Evidence("mb_1", "web", "Take Five", "text", "https://example.org/recording"),
    Evidence("mb_1", "metadata", "Take Five", "text", "http://localhost/recording"),
], ids=["wrong-kind", "unsafe-url"])
def test_unusable_musicbrainz_evidence_is_discarded_with_a_warning(unusable):
    """Unusable musicbrainz evidence is discarded with a warning."""
    result, deps = run(MusicRequest("instrumental jazz", enhanced=True), metadata=Mock(return_value=[unusable]))
    assert result.recommendations
    assert result.warnings == ["MusicBrainz is unavailable; recording versions remain uncertain."]
    assert unusable not in result.evidence and unusable not in deps["generate"].call_args.args[3]
    assert result.notices == []


def test_musicbrainz_evidence_reusing_an_existing_id_is_rejected():
    """Musicbrainz evidence reusing an existing id is rejected."""
    clash = Evidence("web_1", "metadata", "Take Five", "text",
                     "https://musicbrainz.org/recording/10000000-0000-0000-0000-000000000001")
    with pytest.raises(DiscoveryError) as exc:
        run(MusicRequest("instrumental jazz", enhanced=True), metadata=Mock(return_value=[clash]))
    assert exc.value.code == "invalid_response"


# --- the source-quoting fallback, in more detail -------------------------------------------

def test_fallback_says_which_requested_traits_the_source_mentions_and_which_it_does_not():
    """Fallback says which requested traits the source mentions and which it does not."""
    [rec] = run_fallback("Song One by Artist One is a slow track.", [("Song One", "Artist One")],
                         criterion(value="slower beats", source_text="jazz"),
                         criterion(facet="mood", value="happy", source_text="instrumental"),
                         source_title="Playlist")
    assert rec.explanation.endswith("The excerpt and title mention slower beats (“slow”), but not happy.")


@pytest.mark.parametrize("wanted,word", [("sad", "melancholy"), ("calm", "soothing"),
                                         ("fast", "uptempo"), ("energetic", "lively")])
def test_fallback_recognises_a_few_close_synonyms(wanted, word):
    """Fallback recognises a few close synonyms."""
    [rec] = run_fallback(f"Song One by Artist One is {word}.", [("Song One", "Artist One")],
                         criterion(facet="mood", value=wanted, source_text="jazz"), source_title="Playlist")
    assert f"mention {wanted} (“{word}”)" in rec.explanation


@pytest.mark.parametrize("wanted", ["beats", "some songs", "ok"])
def test_fallback_never_counts_generic_or_two_letter_words_as_evidence(wanted):
    """Fallback never counts generic or two letter words as evidence."""
    [rec] = run_fallback("Song One by Artist One has great beats, songs and an ok vibe.",
                         [("Song One", "Artist One")],
                         criterion(facet="mood", value=wanted, source_text="jazz"), source_title="Playlist")
    assert f"don’t mention {wanted}" in rec.explanation


def test_fallback_marks_a_quote_cut_from_a_longer_page_with_ellipses():
    """Fallback marks a quote cut from a longer page with ellipses."""
    page = (" ".join(f"before{n}" for n in range(30))
            + " Song One by Artist One is a slow track that goes on and on without a full stop for many words "
            + " ".join(f"after{n}" for n in range(30)))
    [rec] = run_fallback(page, [("Song One", "Artist One")],
                         criterion(value="slower beats", source_text="jazz"), source_title="Playlist")
    assert rec.explanation.startswith("“…before25 before26 before27 before28 before29 Song One by")
    assert "goes on and on without…”" in rec.explanation
    assert "before24" not in rec.explanation and "after0" not in rec.explanation


def test_fallback_names_the_source_page_and_shortens_a_long_title_at_a_word():
    """Fallback names the source page and shortens a long title at a word."""
    long_title = "The ultimate collection of slow and mellow songs for every quiet evening at home tonight"
    [rec] = run_fallback("Song One by Artist One is a slow track.", [("Song One", "Artist One")],
                         source_title=long_title)
    assert "(from “The ultimate collection of slow and mellow songs for every quiet…”)" in rec.explanation
    [untitled] = run_fallback("Song One by Artist One is a slow track.", [("Song One", "Artist One")],
                              source_title="")
    assert "(from" not in untitled.explanation


def test_fallback_lists_at_most_the_display_limit_in_source_order_when_none_matches_the_request():
    """Fallback lists at most MAX_RECOMMENDATIONS songs in source order when none matches the request."""
    total = MAX_RECOMMENDATIONS + 2
    songs = [(f"Song {n}", f"Artist {n}") for n in range(1, total + 1)]
    page = " ".join(f"Song {n} by Artist {n}." for n in range(1, total + 1))
    recs = run_fallback(page, songs, criterion(value="slower beats", source_text="jazz"),
                        source_title="Playlist")
    assert [rec.title for rec in recs] == [f"Song {n}" for n in range(1, MAX_RECOMMENDATIONS + 1)]


def many_songs(count, source_id, prefix):
    """Build a source naming ``count`` songs, and the candidates it supports."""
    songs = [(f"{prefix} {n}", f"Artist {prefix}") for n in range(count)]
    page = Evidence(source_id, "web", f"{prefix} list", " ".join(f"{t} by {a}." for t, a in songs),
                    f"https://example.org/{source_id}")
    return page, [Candidate(t, a, [source_id]) for t, a in songs]


def test_the_model_judges_only_the_best_few_candidates_spread_over_the_sources():
    """The model gets at most MAX_JUDGED candidates, with the sources taking turns."""
    album, album_songs = many_songs(12, "web_1", "Lullaby")
    chart, chart_songs = many_songs(12, "web_2", "Nocturne")
    generate = Mock(return_value=[])
    run(search=Mock(return_value=[album, chart]), extract=Mock(return_value=album_songs + chart_songs),
        generate=generate)
    sent = generate.call_args.args[2]
    assert len(sent) == MAX_JUDGED
    assert [c.source_ids[0] for c in sent] == ["web_1", "web_2"] * (MAX_JUDGED // 2)


def test_candidates_tied_to_the_request_are_judged_before_songs_merely_named():
    """A song whose evidence ties it to the request goes ahead of one a generic list merely names."""
    page, songs = many_songs(3, "web_1", "Song")
    songs[0] = replace(songs[0], fits_request=False)
    generate = Mock(return_value=[])
    run(search=Mock(return_value=[page]), extract=Mock(return_value=songs), generate=generate)
    assert [c.title for c in generate.call_args.args[2]] == ["Song 1", "Song 2", "Song 0"]


def test_the_follow_up_judges_the_next_batch_not_every_remaining_song():
    """When the first batch holds nothing, the follow-up asks about the next MAX_JUDGED songs only."""
    pages, songs = [], []
    for n in range(3):
        page, found = many_songs(9, f"web_{n + 1}", f"List{n}")
        pages.append(replace(page, text=page.text + " instrumental jazz"))
        songs += found
    first = Evidence("web_1", "web", f"{songs[0].title} — {songs[0].artist}", pages[0].text, pages[0].url)
    generate = Mock(side_effect=[[excluded(first)], []])
    run(MusicRequest("jazz with no vocals"), search=Mock(return_value=pages),
        extract=Mock(return_value=songs), generate=generate,
        analyze=Mock(return_value=ParsedIntent([HARD], ["instrumental jazz songs"])), retry_invalid=True)
    assert generate.call_count == 2
    follow_up = generate.call_args_list[1].args[2]
    assert len(follow_up) == MAX_JUDGED and songs[0] not in follow_up


def test_fallback_cites_the_source_that_actually_identifies_the_song():
    """Fallback cites the source that actually identifies the song."""
    unrelated = Evidence("web_1", "web", "Unrelated page", "Nothing about the song here.", "https://example.org/one")
    naming = source(id="web_2")
    candidate = Candidate("Take Five", "Dave Brubeck Quartet", ["web_1", "web_2"])
    invalid = replace(recommendation(source()), source_ids=["invented"])
    result, _ = run(search=Mock(return_value=[unrelated, naming]), extract=Mock(return_value=[candidate]),
                    generate=Mock(return_value=[invalid]), degrade_invalid=True)
    assert [rec.source_ids for rec in result.recommendations] == [["web_2"]]


# --- find_music: request and configuration handling ----------------------------------------

def test_find_music_rejects_an_invalid_request_before_building_any_provider(providers):
    """Find music rejects an invalid request before building any provider."""
    with pytest.raises(DiscoveryError) as exc:
        find_music(MusicRequest("  "), Settings("fixture", "fixture"))
    assert exc.value.code == "invalid_input"
    for provider in (providers.gemini, providers.tavily, providers.musicbrainz, providers.discover):
        provider.assert_not_called()


def test_find_music_without_settings_reads_the_server_environment(providers, monkeypatch):
    """Find music without settings reads the server environment."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(DiscoveryError) as exc:
        find_music(MusicRequest("jazz"))
    assert exc.value.code == "missing_configuration"
    providers.gemini.assert_not_called()

    monkeypatch.setenv("TAVILY_API_KEY", "tavily-env-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key")
    find_music(MusicRequest("jazz"))
    assert providers.tavily.call_args.args == ("tavily-env-key",)
    assert providers.tavily.call_args.kwargs == {"music_source_focus_enabled": True}
    assert providers.gemini.call_args.args[0].gemini_api_key == "gemini-env-key"
