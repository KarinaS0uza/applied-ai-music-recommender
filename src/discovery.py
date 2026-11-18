"""Run the staged discovery workflow: analyze, retrieve, extract, enrich and recommend."""

from __future__ import annotations

from dataclasses import replace
import logging
import re
import time

from src.models import (MAX_RECOMMENDATIONS, MAX_WEB_SOURCES, Candidate, DiscoveryError,
                        DiscoveryResult, Evidence, MusicRequest, ParsedIntent, Recommendation,
                        SupportedClaim)
from src.utils import clean_excerpt, normalize, safe_url
from src.validation import (identity_supported, invalid_response, validate_candidates,
                            validate_parsed_intent, validate_recommendations,
                            validate_request)

logger = logging.getLogger(__name__)

# Shown as a soft note rather than a warning: results are still usable.
UNVERIFIED_WARNING_PREFIX = "We couldn't confirm that any song meets every requirement"
INTENT_FALLBACK_NOTICE = ("We couldn’t fully interpret the details of your request, "
                          "so we searched using your exact words instead.")


def _fallback_intent(request: MusicRequest) -> ParsedIntent:
    """Keep retrieval usable when generated intent details fail semantic validation."""
    query = " ".join(request.description.split())
    if request.reference_song:
        query = f"{query} {request.reference_song}"
    return ParsedIntent([], [query[:200]])


_GENERIC_WORDS = {"beat", "beats", "song", "songs", "music", "sound", "sounds", "track", "tracks",
                  "vibe", "vibes", "feel", "feeling", "tempo", "very", "more", "with", "and", "the",
                  "for", "some", "that", "like"}
_SUFFIXES = ("iest", "ier", "ily", "est", "er", "ly", "ing", "y", "s")


def _stem(word: str) -> str:
    """Crude stem so slow, slower and slowly (or happy and happier) compare equal."""
    word = word.casefold()
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


# A few close synonyms, so "happy" also finds "upbeat". Deliberately small and tight.
_RELATED_WORDS = {_stem(word): related for word, related in {
    "happy": {"upbeat", "joyful", "cheerful", "uplifting", "feel-good", "sunny"},
    "sad": {"melancholy", "heartbreak", "heartbroken", "sorrowful"},
    "slow": {"ballad", "mellow", "downtempo", "unhurried"},
    "fast": {"uptempo", "up-tempo"},
    "calm": {"soothing", "peaceful", "relaxing"},
    "energetic": {"high-energy", "energizing", "lively"},
}.items()}


def _trait_mention(trait: str, tokens: list[str]) -> str | None:
    """Return the word in the text that mentions the requested trait, if any."""
    for word in re.findall(r"[\w'-]+", trait.casefold()):
        if word in _GENERIC_WORDS or len(word) < 3:
            continue
        stem = _stem(word)
        related = _RELATED_WORDS.get(stem, set())
        for token in tokens:
            if _stem(token) == stem or token in related:
                return token
    return None


def _passage(text: str, title: str, max_words: int) -> str:
    """Words from the text, starting a few words before where the song's title appears."""
    words = text.split()
    size = len(title.split())
    start, end = 0, max_words
    for index in range(len(words) - size + 1):
        if normalize(" ".join(words[index:index + size])) == normalize(title):
            start = max(0, index - 5)
            for back in range(index - 1, start - 1, -1):
                if words[back].endswith((".", "!", "?")):  # begin after the previous sentence or list number
                    start = back + 1
                    break
            end = start + max_words
            for forward in range(index + size, min(end, len(words))):
                if words[forward].endswith((".", "!", "?")):  # stop where this song's sentence ends
                    end = forward + 1
                    break
            break
    passage = words[start:end]
    if len(passage) > size and re.fullmatch(r"\d+[.)]", passage[-1]):  # the next item's list number
        passage.pop()
    return " ".join(passage)


def _source_says(source: Evidence, candidate: Candidate, intent: ParsedIntent) -> tuple[str, int]:
    """Quote what the source says about the song and which requested traits that text mentions."""
    title = clean_excerpt(source.title)
    shown_title = title if len(title) <= 70 else title[:70].rsplit(" ", 1)[0] + "…"
    page = clean_excerpt(source.text or source.title)
    passage = _passage(page, candidate.title, 20)
    # The song's own title and artist must not count as evidence for a trait ("Happy" is not happy).
    own = set(re.findall(r"[\w'-]+", f"{candidate.title} {candidate.artist}".casefold()))
    tokens = [token for token in re.findall(r"[\w'-]+", f"{title} {passage}".casefold())
              if token not in own]
    found, missing = [], []
    for criterion in intent.preferences:
        label = criterion.value.strip().lower()
        word = _trait_mention(criterion.value, tokens)
        if word:
            found.append(label if word == label else f"{label} (“{word}”)")
        else:
            missing.append(label)
    start = "" if page.startswith(passage) else "…"
    end = "" if page.endswith(passage) or passage.endswith((".", "!", "?")) else "…"
    text = f"“{start}{passage}{end}”" + (f" (from “{shown_title}”)." if shown_title else ".")
    if found and missing:
        text += f" The excerpt and title mention {', '.join(found)}, but not {' or '.join(missing)}."
    elif found:
        text += f" The excerpt and title mention {', '.join(found)}."
    elif missing:
        text += f" The excerpt and title don’t mention {' or '.join(missing)}."
    return text, len(found)


def _source_identified(candidates, evidence, intent, skip=frozenset(), limit=MAX_RECOMMENDATIONS,
                       require_fit=False):
    """Return up to ``limit`` songs a source names, each showing what that source says about it.

    Songs whose source text mentions more of the requested traits come first. A song whose
    normalized (title, artist) is in ``skip`` is left out. With ``require_fit``, a song the model
    judged merely named, not tied to the request by its evidence, is left out too, so a request for
    rock is not padded with songs a generic "best songs of all time" list happens to name.
    """
    by_id = {item.id: item for item in evidence}
    ranked = []
    for index, candidate in enumerate(candidates):
        if (normalize(candidate.title), normalize(candidate.artist)) in skip:
            continue
        if require_fit and not candidate.fits_request:
            continue
        source = next((by_id[source_id] for source_id in candidate.source_ids
                       if source_id in by_id and identity_supported(candidate, by_id[source_id])), None)
        if source is None:
            continue
        explanation, mentioned = _source_says(source, candidate, intent)
        quote = _passage(source.text or source.title, candidate.title, 25)
        ranked.append((-mentioned, index, Recommendation(
            candidate.title,
            candidate.artist,
            explanation,
            [source.id],
            ["We couldn't check this song's fit against the source."],
            [SupportedClaim("The source identifies this song and artist.", source.id, quote)],
        )))
    return [rec for _, _, rec in sorted(ranked, key=lambda row: row[:2])[:limit]]


def _fallback_recommendations(candidates, evidence, intent):
    """Return only source-identified songs when generated recommendation details are invalid.

    A request with hard criteria gets none here: a song cannot be listed as meeting a requirement
    that nothing established. Those requests get labeled unverified songs instead.
    """
    if intent.hard_criteria:
        return []
    return _source_identified(candidates, evidence, intent)


def _requirement_label(criterion) -> str:
    """Name a hard criterion the way the listener said it: "clean singing", "no screaming"."""
    value = " ".join(criterion.value.split())
    if criterion.kind == "must_avoid" and not normalize(value).startswith(("no ", "without ", "sem ")):
        return f"no {value}"
    return value


def _unverified_recommendations(candidates, evidence, intent, contradicted):
    """Return source-identified songs for a request whose hard criteria nothing could verify.

    Songs a check found to break a requirement are never shown. Each song says which requirements
    could not be confirmed, and carries no match score.
    """
    labels = [_requirement_label(criterion) for criterion in intent.hard_criteria]
    return [replace(rec, uncertainties=[f"We couldn't confirm {label}." for label in labels])
            for rec in _source_identified(candidates, evidence, intent, skip=contradicted)]


def discover_music(request: MusicRequest, *, analyze, search, extract, generate,
                   documents=None, metadata=None, retry_invalid=False,
                   degrade_invalid=False) -> DiscoveryResult:
    """Run the discovery pipeline (analyze, search, extract, generate) and return a checked result."""
    validate_request(request)
    started = time.monotonic()
    warnings = []
    notices = []
    retry_available = retry_invalid

    def model_stage(name, operation, fallback=None, fallback_warning=None):
        """Run a model stage; one retry per request, told what to fix when the failure says so."""
        nonlocal retry_available
        try:
            return operation()
        except DiscoveryError as error:
            if error.code != "invalid_response":
                raise
            if retry_available:
                retry_available = False
                logger.info("generation_retry stage=%s reason=invalid_response repair=%s",
                            name, bool(error.feedback))
                try:
                    return operation(error.feedback)
                except DiscoveryError as retry_error:
                    if retry_error.code != "invalid_response":
                        raise
                    error = retry_error
            if not degrade_invalid or fallback is None:
                raise error
            logger.warning("generation_degraded stage=%s reason=invalid_response", name)
            if fallback_warning:
                warnings.append(fallback_warning)
            return fallback()

    intent = model_stage(
        "intent", lambda feedback=None: validate_parsed_intent(request, analyze(request)),
        lambda: _fallback_intent(request),
        INTENT_FALLBACK_NOTICE)
    evidence = list(search(request, intent))[:MAX_WEB_SOURCES]
    if request.enhanced and documents:
        try:
            evidence.extend(documents(request, intent))
        except DiscoveryError:
            warnings.append("Custom documents are unavailable; using external evidence.")
    if any(not e.id or e.kind not in {"web", "song_card", "guide"} or
           (e.kind != "guide" and not safe_url(e.url)) for e in evidence):
        raise invalid_response()
    if len({e.id for e in evidence}) != len(evidence):
        raise invalid_response()
    if not any(e.kind in {"web", "song_card"} for e in evidence):
        return DiscoveryResult([], evidence, warnings + [
            "No relevant music sources were found for the interpreted preferences."], intent)
    candidates = model_stage(
        "candidates",
        lambda feedback=None: validate_candidates(extract(request, intent, evidence), evidence),
        lambda: [],
        "We couldn’t confirm any specific songs from the sources we found.")
    if not candidates:
        return DiscoveryResult([], evidence, warnings + [
            "Sources were found, but no song and artist identities could be verified."], intent)
    if request.enhanced and metadata:
        try:
            checked_candidates = candidates[:5]
            extra = metadata(checked_candidates)
            if any(e.kind != "metadata" or not safe_url(e.url) for e in extra):
                raise invalid_response()
            evidence.extend(extra)
            if len(extra) < len(checked_candidates):
                warnings.append("MusicBrainz did not match every checked candidate; missing metadata does not mean a song is nonexistent.")
            if extra:
                notices.append("MusicBrainz matches identify recordings, but do not confirm the intended version or musical fit.")
        except DiscoveryError:
            warnings.append("MusicBrainz is unavailable; recording versions remain uncertain.")
    if len({e.id for e in evidence}) != len(evidence):
        raise invalid_response()
    judged = set()  # candidates the model returned a verdict row for, whether or not it held up
    contradicted = set()  # candidates whose verdict says they break a hard criterion
    disfavored = set()  # candidates whose verdict says a source contradicts a soft preference

    def record_verdicts(generated):
        """Remember which candidates the model judged, and which of them break a hard criterion."""
        for rec in generated:
            key = (normalize(rec.title), normalize(rec.artist))
            judged.add(key)
            if any(check.status == "contradicted" for check in rec.constraint_checks):
                contradicted.add(key)
            if any(check.status == "contradicted" for check in rec.preference_checks):
                disfavored.add(key)

    def generate_checked(feedback=None):
        """Generate recommendations, optionally with repair feedback, and validate them against the evidence."""
        # The first call is unchanged; a retry also tells the model which checks failed.
        extra = {"feedback": feedback} if feedback else {}
        generated = generate(request, intent, candidates, evidence, **extra)
        record_verdicts(generated)
        return validate_recommendations(generated, candidates, evidence, intent)

    recommendations = model_stage(
        "recommendations", generate_checked,
        lambda: _fallback_recommendations(candidates, evidence, intent),
        "We couldn't verify how well these songs fit your request, so they are listed without a match score.")
    # The model is asked for its best candidates, so it may return just one, and if that one is
    # excluded (say a required trait is unestablished) nothing is left although the other
    # candidates were never judged. Ask about those once, using the request's shared retry so
    # the Gemini call limit still holds. If the model judged nothing, asking again would only
    # repeat an empty answer.
    skipped = [c for c in candidates if (normalize(c.title), normalize(c.artist)) not in judged]
    if not recommendations and judged and skipped and retry_available:
        retry_available = False
        logger.info("generation_follow_up stage=recommendations judged=%d skipped=%d",
                    len(candidates) - len(skipped), len(skipped))
        try:
            generated = generate(request, intent, skipped, evidence)
            record_verdicts(generated)
            recommendations = validate_recommendations(generated, skipped, evidence, intent)
        except DiscoveryError as error:
            logger.warning("generation_follow_up_failed code=%s", error.code)
            warnings.append("Some candidates could not be checked against your request.")
    if not intent.hard_criteria and len(recommendations) < MAX_RECOMMENDATIONS:
        # A broad request ("rock") has dozens of valid answers, but the model only judges a few
        # candidates and strict checks drop some of those. The other songs the sources name are still
        # real answers, so fill the list with them, after the checked songs and without a score. A
        # request with hard criteria never does this: a song cannot be offered as meeting a
        # requirement nothing established.
        shown = {(normalize(rec.title), normalize(rec.artist)) for rec in recommendations}
        extra = _source_identified(candidates, evidence, intent, skip=shown | contradicted | disfavored,
                                   limit=MAX_RECOMMENDATIONS - len(recommendations), require_fit=True)
        if extra:
            logger.info("results_topped_up checked=%d added=%d", len(recommendations), len(extra))
            recommendations = recommendations + extra
    unverified = []
    if not recommendations and intent.hard_criteria:
        # Nothing could be confirmed to meet every requirement. Rather than show nothing, list the
        # songs a source names that no check ruled out, labeled so nobody mistakes them for matches.
        recommendations = _unverified_recommendations(candidates, evidence, intent, contradicted)
        if recommendations:
            unverified = [_requirement_label(criterion) for criterion in intent.hard_criteria]
            warnings.append(
                f"{UNVERIFIED_WARNING_PREFIX} ({'; '.join(unverified)}). "
                "These songs are only named by their sources, so check them before you rely on them.")
    if not recommendations:
        if intent.hard_criteria:
            warnings.append("No candidate had evidence supporting every required or excluded trait.")
        elif intent.preferences:
            warnings.append("No candidate had evidence matching any requested preference.")
        else:
            warnings.append("The retrieved evidence did not establish a supported recommendation.")
    used_partial = any(check.status == "partial" for rec in recommendations
                       for check in rec.preference_checks)
    logger.info("discovery_complete enhanced=%s criteria=%d queries=%d evidence=%d candidates=%d recommendations=%d partial=%s unverified=%s latency_ms=%d",
                request.enhanced, len(intent.criteria), len(intent.search_queries), len(evidence),
                len(candidates), len(recommendations), used_partial, bool(unverified),
                (time.monotonic() - started) * 1000)
    return DiscoveryResult(recommendations, evidence, warnings, intent, used_partial, notices, unverified)
