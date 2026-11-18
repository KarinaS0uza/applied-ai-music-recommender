"""Direct tests of the validators. The end-to-end pipeline checks live in test_rag.py."""

import pytest

from src.models import (Candidate, ConstraintCheck, DiscoveryError, Evidence, MusicRequest,
                        ParsedIntent, PreferenceCheck, Recommendation, SupportedClaim, TasteCriterion)
from src.validation import (criterion_key, identity_supported, validate_candidates,
                            validate_parsed_intent, validate_recommendations, validate_request)

TEXT = "Take Five by Dave Brubeck Quartet: instrumental jazz with saxophone."
LONG_TEXT = "Take Five by Dave Brubeck Quartet: " + " ".join(f"w{i}" for i in range(30))
QUOTE_25 = " ".join(f"w{i}" for i in range(25))
QUOTE_26 = " ".join(f"w{i}" for i in range(26))

HARD = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
HARD_KEY = "vocals:vocals:must_avoid"
PREF = TasteCriterion("mood", "preference", "calm", 2, "calm")
PREF_KEY = "mood:calm:preference"


def web(id="web_1", kind="web", title="Take Five — Dave Brubeck Quartet", text=TEXT,
        url="https://example.org/music"):
    """Build a web evidence item."""
    return Evidence(id, kind, title, text, url)


GUIDE = Evidence("guide_1", "guide", "Pace", "Tempo and energy differ.")
MUSICBRAINZ = Evidence("mb_1", "metadata", "Take Five — Dave Brubeck Quartet",
                       "Take Five Dave Brubeck Quartet recording identity metadata",
                       "https://musicbrainz.org/recording/10000000-0000-0000-0000-000000000001")
CANDIDATE = Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])
CLAIM = SupportedClaim("Instrumental jazz", "web_1", "instrumental jazz")


def rec(**overrides):
    """Build a valid recommendation with optional overrides."""
    fields = dict(title="Take Five", artist="Dave Brubeck Quartet", explanation="Instrumental jazz.",
                  source_ids=["web_1"], uncertainties=[], claims=[CLAIM],
                  constraint_checks=[], preference_checks=[])
    return Recommendation(**{**fields, **overrides})


def keep(recommendations, evidence=None, intent=None, candidates=None):
    """Validate recommendations against default evidence and return the kept ones."""
    return validate_recommendations(recommendations, candidates or [CANDIDATE],
                                    evidence or [web()], intent or ParsedIntent([], ["jazz"]))


def supported(constraint=HARD_KEY):
    """Build a supported constraint check."""
    return ConstraintCheck(constraint, "supported", "web_1", "instrumental jazz")


def pref_check(status="supported", source="web_1", quote="instrumental jazz", criterion=PREF_KEY):
    """Build a preference check with optional overrides."""
    return PreferenceCheck(criterion, status, source, quote)


# --- validate_request ----------------------------------------------------------------------
# Blank, 1,201-character and 201-character-reference requests are pinned end to end by
# test_rag.py; these are the exact boundaries and the non-text case.

@pytest.mark.parametrize("description,reference", [
    ("abc", None), ("  abc  ", ""), ("a" * 1200, None), ("jazz", "r" * 200),
], ids=["three-characters", "padded-three", "limit-1200", "reference-limit-200"])
def test_request_limits_are_inclusive(description, reference):
    """Request limits are inclusive."""
    assert validate_request(MusicRequest(description, reference)) is None  # returns quietly instead of raising


@pytest.mark.parametrize("description", ["ab", "  ab  ", None, 42])
def test_description_must_be_text_of_at_least_three_characters(description):
    """Description must be text of at least three characters."""
    with pytest.raises(DiscoveryError) as exc:
        validate_request(MusicRequest(description))
    assert exc.value.code == "invalid_input"


# --- criterion_key -------------------------------------------------------------------------

def test_criterion_key_is_facet_normalized_value_and_kind():
    """Criterion key is facet normalized value and kind."""
    assert criterion_key(TasteCriterion("language", "must_have", "  Português! ", 3, "x")) == "language:portugues:must_have"


# --- validate_parsed_intent ----------------------------------------------------------------

def criterion(**overrides):
    """Build a genre preference criterion with optional overrides."""
    return TasteCriterion(**{"facet": "genre", "kind": "preference", "value": "jazz",
                             "weight": 2, "source_text": "jazz", **overrides})


def test_valid_intent_is_returned_unchanged_and_matches_the_reference_song_too():
    """Valid intent is returned unchanged and matches the reference song too."""
    request = MusicRequest("calm piano, no vocals", "Nuvole Bianche — Ludovico Einaudi")
    intent = ParsedIntent([
        criterion(facet="mood", value="calm", source_text="CALM"),                       # case never matters
        criterion(facet="vocals", kind="must_avoid", value="vocals", weight=3, source_text="no vocals"),
        criterion(facet="reference", value="Nuvole Bianche", weight=1, source_text="nuvole bianche"),
    ], ["calm piano instrumental"])
    assert validate_parsed_intent(request, intent) is intent


@pytest.mark.parametrize("criteria,queries", [
    ([criterion(weight=4)], ["jazz songs"]),
    ([criterion(facet="tempo")], ["jazz songs"]),
    ([criterion(), criterion()], ["jazz songs"]),
    ([criterion()], ["jazz songs", "Jazz  songs"]),
    ([criterion()], ["jazz songs", "jazz tracks", "jazz music"]),
    ([criterion()], ["jazz\nsongs"]),
    # added with the direct validator tests
    ([criterion(kind="avoid")], ["jazz songs"]),
    ([criterion(weight=0)], ["jazz songs"]),
    ([criterion(weight=True)], ["jazz songs"]),            # a bool is not a weight
    ([criterion(weight=2.0)], ["jazz songs"]),
    ([criterion(value="   ")], ["jazz songs"]),
    ([criterion(source_text="  ")], ["jazz songs"]),
    ([criterion(source_text="blues")], ["jazz songs"]),    # not something the listener wrote
    ([criterion(value=f"v{n}") for n in range(21)], ["jazz songs"]),
    ([criterion()], []),
    ([criterion()], ["ab"]),
    ([criterion()], ["j" * 201]),
    ([criterion()], ["jazz\rsongs"]),
    ([criterion()], [123]),
    ([criterion(), "jazz"], ["jazz songs"]),
    ("jazz", ["jazz songs"]),
    (None, None),
], ids=["weight-above-3", "unknown-facet", "duplicate-criterion", "duplicate-query", "three-queries",
        "newline-in-query", "unknown-kind", "weight-zero", "weight-bool", "weight-float", "blank-value",
        "blank-source-text", "source-text-not-in-request", "twenty-one-criteria", "no-queries",
        "query-too-short", "query-too-long", "carriage-return", "query-not-text", "criterion-not-a-criterion",
        "criteria-not-a-list", "not-an-intent"])
def test_malformed_intent_is_rejected(criteria, queries):
    """Malformed intent is rejected."""
    intent = ParsedIntent(criteria, queries) if queries is not None else None
    with pytest.raises(DiscoveryError) as exc:
        validate_parsed_intent(MusicRequest("jazz"), intent)
    assert exc.value.code == "invalid_response"


def test_intent_text_limits_are_inclusive():
    """Intent text limits are inclusive."""
    description = " ".join(["jazz"] * 60)  # 239 characters
    request = MusicRequest(description)
    at_limit = ParsedIntent([criterion(value="v" * 100, source_text=description[:200])], ["jazz songs"])
    assert validate_parsed_intent(request, at_limit) is at_limit
    for too_long in (criterion(value="v" * 101), criterion(source_text=description[:201])):
        with pytest.raises(DiscoveryError):
            validate_parsed_intent(request, ParsedIntent([too_long], ["jazz songs"]))


# --- identity_supported --------------------------------------------------------------------

def test_identity_needs_title_and_artist_in_a_public_web_or_song_card_source():
    """Identity needs title and artist in a public web or song card source."""
    assert identity_supported(CANDIDATE, web())
    assert identity_supported(CANDIDATE, web(kind="song_card"))
    assert identity_supported(CANDIDATE, web(title="Take Five", text="by Dave Brubeck Quartet"))  # split across title and text
    assert identity_supported(Candidate("Águas de Março", "Elis Regina", []),
                              web(text="Aguas de MARCO, sung by elis regina."))                    # accents and case are folded


@pytest.mark.parametrize("source", [
    web(text="Take Five is a jazz standard.", title="Jazz"),                  # artist missing
    web(text="Dave Brubeck Quartet play jazz.", title="Quartet"),             # title missing
    web(kind="guide"), web(kind="metadata"),                                  # only web pages and song cards identify songs
    web(url=None), web(url="http://localhost/take-five"),
], ids=["no-artist", "no-title", "guide", "metadata", "no-url", "unsafe-url"])
def test_identity_is_not_established_by_partial_or_unsuitable_sources(source):
    """Identity is not established by partial or unsuitable sources."""
    assert not identity_supported(CANDIDATE, source)


# --- validate_candidates -------------------------------------------------------------------

PAGE = web(text="Take Five by Dave Brubeck Quartet. So What by Miles Davis. Perfect by Ed Sheeran.")


def test_candidates_keep_their_order_and_drop_identities_the_sources_do_not_support():
    """Candidates keep their order and drop identities the sources do not support."""
    candidates = [Candidate("So What", "Miles Davis", ["web_1"]),
                  Candidate("Invented Song", "Nobody", ["web_1"]),
                  Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])]
    assert [c.title for c in validate_candidates(candidates, [PAGE])] == ["So What", "Take Five"]


def test_duplicate_identities_are_collapsed_ignoring_case_and_punctuation():
    """Duplicate identities are collapsed ignoring case and punctuation."""
    first = Candidate("Take Five", "Dave Brubeck Quartet", ["web_1"])
    again = Candidate("TAKE FIVE!", "dave brubeck quartet", ["web_1"])
    assert validate_candidates([first, again], [PAGE]) == [first]


def test_only_the_first_eight_candidates_are_considered():
    """Only the first eight candidates are considered."""
    page = web(text=" ".join(f"Song {n} by Artist {n}." for n in range(10)))
    candidates = [Candidate(f"Song {n}", f"Artist {n}", ["web_1"]) for n in range(10)]
    assert [c.title for c in validate_candidates(candidates, [page])] == [f"Song {n}" for n in range(8)]


@pytest.mark.parametrize("candidate", [
    Candidate("", "Dave Brubeck Quartet", ["web_1"]),
    Candidate("Take Five", "   ", ["web_1"]),
    Candidate("!!!", "Dave Brubeck Quartet", ["web_1"]),       # nothing left once normalized
    Candidate("T" * 201, "Dave Brubeck Quartet", ["web_1"]),
    Candidate("Take Five", "A" * 201, ["web_1"]),
    Candidate("Take Five", "Dave Brubeck Quartet", []),
    Candidate("Take Five", "Dave Brubeck Quartet", ["web_1", "web_9"]),  # one unknown id spoils the row
], ids=["blank-title", "blank-artist", "punctuation-title", "long-title", "long-artist", "no-sources",
        "one-unknown-source"])
def test_malformed_candidate_rows_are_invalid_responses(candidate):
    """Malformed candidate rows are invalid responses."""
    with pytest.raises(DiscoveryError) as exc:
        validate_candidates([candidate], [PAGE])
    assert exc.value.code == "invalid_response"


# --- validate_recommendations: every reason a song is rejected -----------------------------
# Unknown source ids, a source about another song, fabricated claim or check quotes, bad check
# statuses and missing preference checks are pinned end to end by test_rag.py.

TWO_PAGES = [web(), web("web_2")]  # web_2 names the song too, but never identified the candidate
CLAIM_ON_GUIDE = SupportedClaim("Pace", "guide_1", "tempo and energy differ")
INTENT_HARD = ParsedIntent([HARD], ["q"])
INTENT_PREF = ParsedIntent([PREF], ["q"])
LONG_CLAIM = [SupportedClaim("Words", "web_1", "w1 w2")]


@pytest.mark.parametrize("song,evidence,intent,hint", [
    (rec(title="So What", artist="Miles Davis"), [web()], None, "not one of the supplied candidates"),
    (rec(explanation="  "), [web()], None, "explanation must be non-empty and at most 1500 characters"),
    (rec(explanation="x" * 1501), [web()], None, "explanation must be non-empty and at most 1500 characters"),
    (rec(source_ids=[]), [web()], None, "source_ids must be ids that exist"),
    (rec(source_ids=["web_2"]), TWO_PAGES, None, "must include a source that identified this candidate"),
    (rec(explanation="Listen at https://example.org/x"), [web()], None, "must not contain URLs"),
    (rec(uncertainties=["see www.example.org"]), [web()], None, "must not contain URLs"),
    (rec(claims=[]), [web()], None, "at least one claim"),
    (rec(claims=[SupportedClaim("t", "web_9", "instrumental jazz")]), [web()], None, "missing from this song's source_ids"),
    (rec(claims=[SupportedClaim("t", "web_1", QUOTE_26)]), [web(text=LONG_TEXT)], None, "at most 25 words"),
    (rec(claims=[SupportedClaim("t", "web_1", "?!")]), [web()], None, "exact, verbatim passage"),
    (rec(source_ids=["web_1", "guide_1"], claims=[CLAIM_ON_GUIDE]), [web(), GUIDE], None, "cannot cite a guide source"),
    (rec(), [web()], INTENT_HARD, "constraint_checks needs exactly one check"),
    (rec(constraint_checks=[supported(), supported()]), [web()], INTENT_HARD, "constraint_checks needs exactly one check"),
    (rec(constraint_checks=[supported(), ConstraintCheck("genre:jazz:must_have", "unknown", "", "")]),
     [web()], INTENT_HARD, "constraint_checks needs exactly one check"),
    (rec(preference_checks=[pref_check(), pref_check()]), [web()], INTENT_PREF, "preference_checks needs exactly one check"),
    (rec(preference_checks=[pref_check(), PreferenceCheck("mood:happy:preference", "unknown", "", "")]),
     [web()], INTENT_PREF, "preference_checks needs exactly one check"),
    (rec(preference_checks=[pref_check(source="web_2")]), TWO_PAGES, INTENT_PREF, f"check {PREF_KEY}:"),
    (rec(source_ids=["web_1", "mb_1"], preference_checks=[pref_check(source="mb_1", quote="identity metadata")]),
     [web(), MUSICBRAINZ], INTENT_PREF, f"check {PREF_KEY}:"),
    (rec(claims=LONG_CLAIM, preference_checks=[pref_check(quote=QUOTE_26)]), [web(text=LONG_TEXT)], INTENT_PREF,
     f"check {PREF_KEY}:"),
], ids=["not-a-candidate", "blank-explanation", "long-explanation", "no-source-ids", "not-the-identifying-source",
        "url-in-explanation", "url-in-uncertainty", "no-claims", "claim-source-not-cited", "claim-quote-too-long",
        "claim-quote-blank", "claim-on-guide", "constraint-check-missing", "constraint-check-duplicated",
        "constraint-check-extra", "preference-check-duplicated", "preference-check-extra",
        "check-source-not-cited", "check-on-metadata", "check-quote-too-long"])
def test_rejected_song_comes_back_with_a_fix_the_model_can_act_on(song, evidence, intent, hint):
    """Rejected song comes back with a fix the model can act on."""
    with pytest.raises(DiscoveryError) as exc:
        keep([song], evidence, intent)
    assert exc.value.code == "invalid_response"
    assert hint in exc.value.feedback


@pytest.mark.parametrize("song,evidence,reason", [
    (rec(preference_checks=[pref_check(source="web_9")]), [web()], "check_source_unknown"),
    (rec(preference_checks=[pref_check(source="web_2")]), TWO_PAGES, "check_source_not_cited"),
    (rec(source_ids=["web_1", "mb_1"], preference_checks=[pref_check(source="mb_1", quote="identity metadata")]),
     [web(), MUSICBRAINZ], "check_source_not_quotable"),
    (rec(preference_checks=[pref_check(quote="  ")]), [web()], "check_quote_empty"),
    (rec(claims=LONG_CLAIM, preference_checks=[pref_check(quote=QUOTE_26)]), [web(text=LONG_TEXT)],
     "check_quote_too_long"),
    (rec(preference_checks=[pref_check(quote="mellow lo-fi vibes")]), [web()], "check_quote_not_in_source"),
], ids=["unknown-source", "source-not-cited", "metadata-source", "empty-quote", "quote-too-long", "quote-not-verbatim"])
def test_each_way_a_check_quote_can_fail_logs_its_own_reason(song, evidence, reason, caplog):
    """Each way a check quote can fail is logged with its own reason, with the model-facing hint unchanged."""
    with caplog.at_level("INFO", logger="src.validation"), pytest.raises(DiscoveryError) as exc:
        keep([song], evidence, INTENT_PREF)
    assert f"recommendation_rejected reason={reason} position=0" in caplog.text
    assert f"check {PREF_KEY}: the quote must be an exact, verbatim passage" in exc.value.feedback


# --- validate_recommendations: what is accepted --------------------------------------------

def test_a_25_word_quote_is_allowed():
    """A 25 word quote is allowed."""
    assert len(keep([rec(claims=[SupportedClaim("Words", "web_1", QUOTE_25)])], [web(text=LONG_TEXT)])) == 1


def test_guide_and_musicbrainz_pages_naming_the_song_may_be_cited_but_unrelated_ones_may_not():
    """Guide and musicbrainz pages naming the song may be cited but unrelated ones may not."""
    assert len(keep([rec(source_ids=["web_1", "guide_1"])], [web(), GUIDE])) == 1
    assert len(keep([rec(source_ids=["web_1", "mb_1"])], [web(), MUSICBRAINZ])) == 1
    other = Evidence("mb_2", "metadata", "So What — Miles Davis", "So What Miles Davis identity metadata",
                     "https://musicbrainz.org/recording/10000000-0000-0000-0000-000000000002")
    with pytest.raises(DiscoveryError) as exc:
        keep([rec(source_ids=["web_1", "mb_2"])], [web(), other])
    assert "does not mention this song and artist" in exc.value.feedback


@pytest.mark.parametrize("status", ["partial", "contradicted"])
def test_a_hard_criterion_that_is_not_fully_supported_excludes_the_song_quietly(status):
    """A hard criterion that is not fully supported excludes the song quietly."""
    check = ConstraintCheck(HARD_KEY, status, "web_1", "instrumental jazz")
    assert keep([rec(constraint_checks=[check])], intent=INTENT_HARD) == []


def test_a_song_whose_every_preference_is_contradicted_is_excluded_quietly():
    """A song whose every preference is contradicted is excluded quietly."""
    assert keep([rec(preference_checks=[pref_check("contradicted")])], intent=INTENT_PREF) == []


def test_fit_counts_supported_fully_and_unknown_or_contradicted_as_zero():
    """Fit counts supported fully and unknown or contradicted as zero."""
    loud = TasteCriterion("mood", "preference", "loud", 1, "loud")
    intent = ParsedIntent([PREF, loud], ["q"])
    song = rec(preference_checks=[pref_check(), pref_check("contradicted", criterion="mood:loud:preference")])
    [kept] = keep([song], intent=intent)
    assert kept.fit_score == pytest.approx(2 / 3)
    assert song.fit_score is None  # the validator returns a scored copy and leaves the input alone


def test_equally_good_songs_keep_the_order_the_model_gave():
    """Equally good songs keep the order the model gave."""
    names = [("Song 3", "Artist 3"), ("Song 1", "Artist 1"), ("Song 2", "Artist 2")]  # deliberately unsorted
    pages = [web(f"web_{n}", text=f"{t} by {a}: instrumental jazz.") for n, (t, a) in enumerate(names, 1)]
    candidates = [Candidate(t, a, [f"web_{n}"]) for n, (t, a) in enumerate(names, 1)]
    songs = [rec(title=t, artist=a, source_ids=[f"web_{n}"],
                 claims=[SupportedClaim("Instrumental jazz", f"web_{n}", "instrumental jazz")])
             for n, (t, a) in enumerate(names, 1)]
    assert [r.title for r in keep(songs, pages, candidates=candidates)] == ["Song 3", "Song 1", "Song 2"]


# --- validate_recommendations: feedback and limits -----------------------------------------

def test_feedback_names_every_rejected_song_and_how_to_fix_it():
    """Feedback names every rejected song and how to fix it."""
    pages = [web(), web("web_2", title="So What — Miles Davis", text="So What by Miles Davis.")]
    candidates = [CANDIDATE, Candidate("So What", "Miles Davis", ["web_2"])]
    bad = [rec(explanation=""), rec(title="So What", artist="Miles Davis", source_ids=["web_2"],
                                    explanation="Listen at https://example.org")]
    with pytest.raises(DiscoveryError) as exc:
        keep(bad, pages, candidates=candidates)
    lines = exc.value.feedback.splitlines()
    assert lines[0].startswith("Your previous answer was rejected by validation.")
    assert lines[1].startswith("- Take Five by Dave Brubeck Quartet: the explanation must be non-empty")
    assert lines[2].startswith("- So What by Miles Davis: the explanation and uncertainties must not contain URLs")


def test_feedback_shortens_long_titles_and_caps_the_list_of_problems_at_1500_characters():
    """Feedback shortens long titles and caps the list of problems at 1500 characters."""
    rows = [rec(title="T" * 200, artist="A" * 200) for _ in range(8)]
    with pytest.raises(DiscoveryError) as exc:
        keep(rows)
    header, problems = exc.value.feedback.split("\n", 1)
    assert f"- {'T' * 80} by {'A' * 80}:" in problems
    assert "T" * 81 not in problems
    assert len(problems) == 1500


def test_a_quietly_excluded_song_does_not_hide_a_rejected_one():
    """A quietly excluded song does not hide a rejected one."""
    pages = [web(), web("web_2", title="So What — Miles Davis", text="So What by Miles Davis.")]
    candidates = [CANDIDATE, Candidate("So What", "Miles Davis", ["web_2"])]
    excluded = rec(constraint_checks=[ConstraintCheck(HARD_KEY, "unknown", "", "")])
    rejected = rec(title="So What", artist="Miles Davis", source_ids=["web_2"], explanation="")
    with pytest.raises(DiscoveryError) as exc:
        keep([excluded, rejected], pages, INTENT_HARD, candidates)
    assert "So What by Miles Davis" in exc.value.feedback and "Take Five" not in exc.value.feedback


def test_only_the_first_eight_recommendations_are_considered():
    """Only the first eight recommendations are considered."""
    nonsense = [rec(title=f"Invented {n}", artist="Nobody") for n in range(8)]
    with pytest.raises(DiscoveryError):  # the valid ninth song is never reached
        keep(nonsense + [rec()])


# --- control characters in names -------------------------------------------------------------


def test_a_garbled_title_is_shown_clean_even_though_it_still_matches_the_candidate():
    """normalize() ignores control and replacement characters, so such a title matches; it must not display."""
    kept = keep([rec(title="\x10\ufffd\x10\ufffd\ufffd\ufffdTake Five", artist="Dave\x00 Brubeck Quartet")])
    assert [(r.title, r.artist) for r in kept] == [("Take Five", "Dave Brubeck Quartet")]


def test_a_garbled_candidate_name_is_cleaned_before_it_is_kept():
    garbled = Candidate("Take\ufffd Five\x10", "Dave Brubeck Quartet\ufffd", ["web_1"])
    kept = validate_candidates([garbled], [web()])
    assert [(c.title, c.artist) for c in kept] == [("Take Five", "Dave Brubeck Quartet")]
