"""Validate requests, parsed intent and generated output against source evidence."""

from __future__ import annotations

from dataclasses import replace
import logging
import re

from src.models import (CHECK_STATUSES, CRITERION_KINDS, INTENT_FACETS, Candidate,
                        DiscoveryError, Evidence, MusicRequest, ParsedIntent,
                        TasteCriterion)
from src.utils import normalize, safe_url

logger = logging.getLogger(__name__)


def invalid_response(feedback: str | None = None) -> DiscoveryError:
    """Build the error raised when a model response is not supported by the sources."""
    return DiscoveryError("invalid_response", "The response could not be supported by the retrieved sources. Try a more specific request.",
                          feedback)


class _Rejected(Exception):
    """One recommendation failed a check: a log-safe slug plus a hint the model can act on."""

    def __init__(self, reason: str, hint: str):
        """Store why a recommendation was rejected and a hint on how to fix it."""
        super().__init__(reason)
        self.reason = reason
        self.hint = hint


def validate_request(request: MusicRequest) -> None:
    """Raise an input error when the description or reference song is missing or too long."""
    if not isinstance(request.description, str) or len(request.description.strip()) < 3:
        raise DiscoveryError("invalid_input", "Describe a genre, mood, activity, instrument, or sound to search for.")
    if len(request.description) > 1200 or (request.reference_song and len(request.reference_song) > 200):
        raise DiscoveryError("invalid_input", "Keep the description under 1,200 characters and the reference under 200.")


def criterion_key(criterion: TasteCriterion) -> str:
    """Build the stable identifier of a criterion from its facet, value and kind."""
    return f"{criterion.facet}:{normalize(criterion.value)}:{criterion.kind}"


def validate_parsed_intent(request: MusicRequest, intent: ParsedIntent) -> ParsedIntent:
    """Check the parsed intent's shape and quotes and return it, or raise an invalid-response error."""
    if (not isinstance(intent, ParsedIntent) or not isinstance(intent.criteria, list)
            or len(intent.criteria) > 20 or not isinstance(intent.search_queries, list)
            or not 1 <= len(intent.search_queries) <= 2):
        raise invalid_response()
    original = normalize(request.description + " " + (request.reference_song or ""))
    keys = []
    for criterion in intent.criteria:
        if (not isinstance(criterion, TasteCriterion) or criterion.facet not in INTENT_FACETS
                or criterion.kind not in CRITERION_KINDS or type(criterion.weight) is not int
                or not 1 <= criterion.weight <= 3 or not criterion.value.strip()
                or len(criterion.value) > 100 or not criterion.source_text.strip()
                or len(criterion.source_text) > 200
                or normalize(criterion.source_text) not in original):
            raise invalid_response()
        keys.append(criterion_key(criterion))
    if len(keys) != len(set(keys)):
        raise invalid_response()
    normalized_queries = []
    for query in intent.search_queries:
        if (not isinstance(query, str) or not 3 <= len(query.strip()) <= 200
                or "\n" in query or "\r" in query or re.search(r"https?://|www\.", query)
                or re.search(r"\b(?:i want|i would like|please recommend|looking for)\b",
                             normalize(query))):
            raise invalid_response()
        normalized_queries.append(normalize(query))
    if len(normalized_queries) != len(set(normalized_queries)):
        raise invalid_response()
    return intent


def identity_supported(candidate: Candidate, source: Evidence) -> bool:
    """Tell whether a source names both the title and the artist of the candidate."""
    content = normalize(source.title + " " + source.text)
    return (source.kind in {"web", "song_card"} and safe_url(source.url)
            and normalize(candidate.title) in content and normalize(candidate.artist) in content)


def validate_candidates(candidates: list[Candidate], evidence: list[Evidence]) -> list[Candidate]:
    """Keep candidates whose identity is supported by their cited sources, dropping the rest."""
    by_id = {e.id: e for e in evidence}
    valid, seen = [], set()
    for candidate in candidates[:8]:
        key = (normalize(candidate.title), normalize(candidate.artist))
        if not all(key) or len(candidate.title) > 200 or len(candidate.artist) > 200:
            raise invalid_response()
        if not candidate.source_ids or any(s not in by_id for s in candidate.source_ids):
            raise invalid_response()
        if not any(identity_supported(candidate, by_id[s]) for s in candidate.source_ids):
            logger.info("candidate_rejected reason=identity_not_in_evidence")
            continue
        if key not in seen:
            seen.add(key)
            valid.append(candidate)
    return valid


def _validate_evidence_check(status, source_id, quote, rec, by_id, label):
    """Validate one constraint or preference check's status, source and quote."""
    if status not in CHECK_STATUSES:
        raise _Rejected("bad_check_status", f"check {label}: status must be one of {sorted(CHECK_STATUSES)}")
    if status == "unknown":
        if source_id or quote:
            raise _Rejected("unknown_check_has_evidence",
                            f"check {label}: an unknown status needs an empty source_id and quote")
        return
    if (source_id not in rec.source_ids or source_id not in by_id
            or by_id[source_id].kind in {"guide", "metadata"}
            or not quote.strip() or not normalize(quote)
            or normalize(quote) not in normalize(by_id[source_id].text)
            or len(quote.split()) > 25):
        raise _Rejected("check_quote_not_in_source",
                        f"check {label}: the quote must be an exact, verbatim passage of at most 25 words from a "
                        "web or song_card source listed in this song's source_ids; otherwise use status unknown "
                        "with an empty source_id and quote")


def _check_recommendation(rec, allowed, by_id, hard, preferences):
    """Return (fit_score, rec), None when the song is legitimately excluded, or raise _Rejected."""
    key = (normalize(rec.title), normalize(rec.artist))
    candidate = allowed.get(key)
    if candidate is None:
        raise _Rejected("not_a_candidate", "this is not one of the supplied candidates; use only their exact titles and artists")
    if not rec.explanation.strip() or len(rec.explanation) > 1500:
        raise _Rejected("bad_explanation", "the explanation must be non-empty and at most 1500 characters")
    if not rec.source_ids or any(s not in by_id for s in rec.source_ids):
        raise _Rejected("unknown_source", "source_ids must be ids that exist in the supplied evidence")
    if not set(rec.source_ids).intersection(candidate.source_ids):
        raise _Rejected("source_not_candidate_source",
                        f"source_ids must include a source that identified this candidate: {candidate.source_ids}")
    for id in rec.source_ids:
        source = by_id[id]
        if id not in candidate.source_ids and source.kind != "guide":
            content = normalize(source.title + " " + source.text)
            if source.kind != "metadata" or not all(part in content for part in key):
                raise _Rejected("unrelated_source",
                                f"source {id} does not mention this song and artist, so it cannot be cited")
    if re.search(r"https?://|www\.", rec.explanation + " ".join(rec.uncertainties)):
        raise _Rejected("url_in_text", "the explanation and uncertainties must not contain URLs")
    if not rec.claims:
        raise _Rejected("no_claims", "at least one claim with a verbatim source quote is required")
    for claim in rec.claims:
        if claim.source_id not in rec.source_ids:
            raise _Rejected("claim_source_not_cited",
                            f"a claim cites {claim.source_id!r}, which is missing from this song's source_ids")
        if (not claim.quote.strip() or not normalize(claim.quote)
                or normalize(claim.quote) not in normalize(by_id[claim.source_id].text)
                or len(claim.quote.split()) > 25):
            raise _Rejected("claim_quote_not_in_source",
                            f"a claim quote must be an exact, verbatim passage of at most 25 words from source "
                            f"{claim.source_id}; do not paraphrase or join separate passages")
        if by_id[claim.source_id].kind == "guide":
            raise _Rejected("claim_on_guide", "claims cannot cite a guide source")
    if ([check.constraint for check in rec.constraint_checks] != list(dict.fromkeys(
            check.constraint for check in rec.constraint_checks))
            or {check.constraint for check in rec.constraint_checks} != set(hard)):
        raise _Rejected("constraint_checks_mismatch",
                        f"constraint_checks needs exactly one check for each of {sorted(hard)}")
    hard_failed = False
    for check in rec.constraint_checks:
        _validate_evidence_check(check.status, check.source_id, check.quote, rec, by_id, check.constraint)
        if check.status != "supported":
            hard_failed = True
    if hard_failed:
        logger.info("recommendation_rejected reason=hard_criterion_unestablished")
        return None
    if ([check.criterion for check in rec.preference_checks] != list(dict.fromkeys(
            check.criterion for check in rec.preference_checks))
            or {check.criterion for check in rec.preference_checks} != set(preferences)):
        raise _Rejected("preference_checks_mismatch",
                        f"preference_checks needs exactly one check for each of {sorted(preferences)}")
    matched = False
    weighted_fit = 0.0
    total_weight = sum(c.weight for c in preferences.values())
    for check in rec.preference_checks:
        _validate_evidence_check(check.status, check.source_id, check.quote, rec, by_id, check.criterion)
        if check.status in {"supported", "partial"}:
            matched = True
        factor = 1.0 if check.status == "supported" else 0.5 if check.status == "partial" else 0.0
        weighted_fit += preferences[check.criterion].weight * factor
    if preferences and not matched:
        logger.info("recommendation_rejected reason=no_preference_match")
        return None
    fit_score = weighted_fit / total_weight if total_weight else 1.0
    return fit_score, replace(rec, fit_score=fit_score)


def validate_recommendations(recommendations, candidates, evidence, intent):
    """Keep every recommendation that passes; one bad song no longer discards the others.

    Raises invalid_response, carrying model-facing feedback, only when songs were
    rejected for malformed or unsupported output and none survived.
    """
    by_id = {e.id: e for e in evidence}
    allowed = {(normalize(c.title), normalize(c.artist)): c for c in candidates}
    hard = {criterion_key(c): c for c in intent.hard_criteria}
    preferences = {criterion_key(c): c for c in intent.preferences}
    scored, rejected = [], []
    for position, rec in enumerate(recommendations[:8]):
        try:
            outcome = _check_recommendation(rec, allowed, by_id, hard, preferences)
        except _Rejected as failure:
            logger.info("recommendation_rejected reason=%s position=%d", failure.reason, position)
            rejected.append(f"- {str(rec.title)[:80]} by {str(rec.artist)[:80]}: {failure.hint}")
            continue
        if outcome is not None:
            scored.append((outcome[0], position, outcome[1]))
    if rejected and not scored:
        raise invalid_response(
            "Your previous answer was rejected by validation. Return a complete corrected answer that fixes "
            "every problem below:\n" + "\n".join(rejected)[:1500])
    scored.sort(key=lambda row: (-row[0], row[1]))
    valid, seen = [], set()
    for _, _, rec in scored:
        key = (normalize(rec.title), normalize(rec.artist))
        if key not in seen:
            seen.add(key)
            valid.append(rec)
        if len(valid) == 5:
            break
    return valid
