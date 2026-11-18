"""Tests for the data models in src/models.py."""

import dataclasses

import pytest

from src.models import (DiscoveryError, DiscoveryResult, MusicRequest, ParsedIntent,
                        Recommendation, TasteCriterion)


def test_discovery_error_shows_only_its_message_and_keeps_model_feedback_separate():
    """Discovery error shows only its message and keeps model feedback separate."""
    error = DiscoveryError("quota", "Try again later.", feedback="fix the quote for Take Five")
    assert (error.code, str(error), error.feedback) == ("quota", "Try again later.", "fix the quote for Take Five")
    assert "Take Five" not in str(error)  # feedback is for the model and must never reach users
    assert DiscoveryError("timeout", "Slow.").feedback is None


def test_parsed_intent_splits_soft_preferences_from_hard_criteria_in_order():
    """Parsed intent splits soft preferences from hard criteria in order."""
    like = TasteCriterion("mood", "preference", "calm", 2, "calm")
    must = TasteCriterion("language", "must_have", "Korean", 3, "korean")
    avoid = TasteCriterion("vocals", "must_avoid", "vocals", 3, "no vocals")
    other = TasteCriterion("genre", "preference", "jazz", 1, "jazz")
    intent = ParsedIntent([like, must, avoid, other], ["calm jazz"])
    assert intent.preferences == [like, other]
    assert intent.hard_criteria == [must, avoid]
    assert ParsedIntent([], ["x"]).preferences == ParsedIntent([], ["x"]).hard_criteria == []


def test_recommendation_defaults_mean_never_scored_and_no_video_and_are_immutable():
    """Recommendation defaults mean never scored and no video and are immutable."""
    rec = Recommendation("Take Five", "Dave Brubeck Quartet", "Why.", ["web_1"], [])
    assert (rec.fit_score, rec.video, rec.claims, rec.constraint_checks, rec.preference_checks) == (
        None, None, [], [], [])
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.fit_score = 1.0
    assert dataclasses.replace(rec, fit_score=0.5).fit_score == 0.5 and rec.fit_score is None


def test_requests_default_to_the_web_only_baseline_and_are_immutable():
    """Requests default to the web only baseline and are immutable."""
    request = MusicRequest("jazz")
    assert (request.reference_song, request.enhanced) == (None, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.description = "other"


def test_each_discovery_result_gets_its_own_default_lists():
    """Each discovery result gets its own default lists."""
    first, second = DiscoveryResult([], [], []), DiscoveryResult([], [], [])
    first.notices.append("only on the first result")
    assert second.notices == [] and first.used_partial_matches is False and first.intent is None
