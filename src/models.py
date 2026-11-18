"""Domain types shared by the discovery pipeline, the providers and both interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field


class DiscoveryError(Exception):
    """Error with a safe user-facing message and an internal code."""
    def __init__(self, code: str, message: str, feedback: str | None = None):
        """Store the error code and optional model feedback, keeping only the message user-visible."""
        self.code = code
        # Never shown to users: tells the model what to fix when an invalid response is retried.
        self.feedback = feedback
        super().__init__(message)


@dataclass(frozen=True)
class MusicRequest:
    """What the listener asked for: a description, an optional reference song and the enhanced flag."""
    description: str
    reference_song: str | None = None
    enhanced: bool = False


INTENT_FACETS = {"genre", "mood", "energy", "instrument", "vocals", "language",
                 "era", "activity", "version", "reference", "other"}
CRITERION_KINDS = {"must_have", "must_avoid", "preference"}
CHECK_STATUSES = {"supported", "partial", "unknown", "contradicted"}

# How many songs each discovery stage may carry. A broad request such as "rock" should still fill a
# few rows of cards, so these are wider than one row: web pages -> candidates -> songs the model
# judges -> songs shown (three columns of cards, so a multiple of three).
MAX_WEB_SOURCES = 16
MAX_CANDIDATES = 30
MAX_JUDGED = 10
MAX_RECOMMENDATIONS = 9


@dataclass(frozen=True)
class TasteCriterion:
    """One taste requirement or preference parsed from the request."""
    facet: str
    kind: str
    value: str
    weight: int
    source_text: str


@dataclass(frozen=True)
class ParsedIntent:
    """The criteria and search queries parsed from a request."""
    criteria: list[TasteCriterion]
    search_queries: list[str]

    @property
    def preferences(self) -> list[TasteCriterion]:
        """Return the soft criteria that only affect ranking."""
        return [criterion for criterion in self.criteria if criterion.kind == "preference"]

    @property
    def hard_criteria(self) -> list[TasteCriterion]:
        """Return the must-have and must-avoid criteria."""
        return [criterion for criterion in self.criteria if criterion.kind != "preference"]


@dataclass(frozen=True)
class Evidence:
    """A source (web page, song card, guide section or recording metadata) that can be cited."""
    id: str
    kind: str
    title: str
    text: str
    url: str | None = None
    retrieved_at: str | None = None


@dataclass(frozen=True)
class Candidate:
    """A song and artist pair identified by the sources.

    ``fits_request`` is the model's judgment, made while extracting, of whether the cited evidence ties
    the song to what was asked for (say, the requested genre) rather than merely naming it.
    """
    title: str
    artist: str
    source_ids: list[str]
    fits_request: bool = True


@dataclass(frozen=True)
class SupportedClaim:
    """A statement about a song with the source quote that supports it."""
    text: str
    source_id: str
    quote: str


@dataclass(frozen=True)
class ConstraintCheck:
    """The checked status of one hard criterion for a song, with its supporting quote."""
    constraint: str
    status: str
    source_id: str
    quote: str


@dataclass(frozen=True)
class PreferenceCheck:
    """The checked status of one soft preference for a song, with its supporting quote."""
    criterion: str
    status: str
    source_id: str
    quote: str


@dataclass(frozen=True)
class VideoLink:
    """A YouTube video matched to a recommendation; the URL is always rebuilt from the id."""
    video_id: str
    title: str
    channel: str

    @property
    def url(self) -> str:
        """Return the YouTube watch URL for this video."""
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass(frozen=True)
class TrackLink:
    """A Spotify track matched to a recommendation; the URL is always rebuilt from the id."""
    track_id: str
    title: str
    artist: str

    @property
    def url(self) -> str:
        """Return the Spotify page for this track."""
        return f"https://open.spotify.com/track/{self.track_id}"


@dataclass(frozen=True)
class Recommendation:
    """A recommended song with its explanation, sources and validation checks."""
    title: str
    artist: str
    explanation: str
    source_ids: list[str]
    uncertainties: list[str]
    claims: list[SupportedClaim] = field(default_factory=list)
    constraint_checks: list[ConstraintCheck] = field(default_factory=list)
    preference_checks: list[PreferenceCheck] = field(default_factory=list)
    fit_score: float | None = None  # None: the fit was never validated, so no score exists
    video: VideoLink | None = None  # None: no confident YouTube match, the UI falls back to a search link
    track: TrackLink | None = None  # None: no confident Spotify match, the UI falls back to a search link


@dataclass
class DiscoveryResult:
    """The recommendations plus the evidence, warnings and notices behind them."""
    recommendations: list[Recommendation]
    evidence: list[Evidence]
    warnings: list[str]
    intent: ParsedIntent | None = None
    used_partial_matches: bool = False
    notices: list[str] = field(default_factory=list)
    # Not empty when no song could be verified against the request's hard criteria: the songs shown
    # are only named by their sources, and this lists the requirements none of them could confirm.
    unverified_requirements: list[str] = field(default_factory=list)
