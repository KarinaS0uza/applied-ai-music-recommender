"""Rank local guide sections and song cards by matching query terms.

Retrieval is deliberately selective, because every song card that reaches the model can
become a recommended candidate. Two filters keep weak matches out:

* Specificity: a document must match at least one distinctive term. A word found in a large
  share of the documents ("vocal") cannot tell them apart, so matching only that word is noise.
* Autocut: song cards that score far below the best card are the noise tail and are dropped.
  A song the listener named is always kept and never sets the bar for the other cards.
"""

# Behavior: hybrid-ranks song_cards.json with lexical terms and optional semantic scores, while guide sections stay lexical; returns at most four cards and two guides.

from collections import Counter
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re

from src.models import DiscoveryError, Evidence, MusicRequest, ParsedIntent
from src.utils import normalize, safe_url

logger = logging.getLogger(__name__)
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge"
# Words found in more than this share of the documents are generic. In the shipped collection
# "vocal" is in 47% of them while the most common useful term, "piano", is in 25%.
GENERIC_TERM_SHARE = 0.35
# Mood, energy, pace, and broad style words are useful supporting signals, but they should not
# outrank a genre, instrument, or language match. This matters more as the collection grows:
# otherwise several merely "upbeat" cards can crowd out the requested electronic songs.
BROAD_TRAIT_WEIGHT = 0.15
_BROAD_TRAITS = {
    "calm", "dance", "energetic", "fast", "happy", "mellow", "moderate", "pop", "upbeat",
}
_STRICT_GENRES = {"metal"}
# Song cards whose word-match score is under this share of the best card's are dropped. In the
# shipped collection a one-word genre match that is noise ("pop" for a Korean pop request) scores
# about 0.30 of the best card, and a one-word match that still helps ("piano" for an instrumental
# piano request) about 0.39. Revisit this if the collection grows.
AUTOCUT_SHARE = 0.35
SEMANTIC_MIN_COSINE = 0.65
SEMANTIC_ONLY_LIMIT = 2
RRF_K = 60
# Negations ("no screaming", "sem voz") say what to exclude; matching them finds nothing useful.
_STOP = set("a an and as at be by for from i in is it like me music of on or please similar some song songs that the to want with would language recording track version original another other de da do em e para uma um com eu musica musicas no not without sem nao".split())
_ALIASES = {"instrumentais": "instrumental", "instrumentals": "instrumental", "vocals": "vocal",
            "voices": "vocal", "voice": "vocal", "singing": "vocal", "pianist": "piano",
            "portugues": "portuguese", "percussao": "percussion", "calma": "calm",
            "eletronica": "electronic", "guitars": "guitar", "drums": "drum",
            "peaceful": "calm", "gentle": "calm", "soothing": "calm",
            "relaxed": "calm", "relaxing": "calm", "serene": "calm", "tranquil": "calm",
            "joyful": "happy", "uplifting": "happy", "dreamy": "ethereal",
            "sorrowful": "mournful", "melancholy": "mournful", "melancholic": "mournful",
            "synth": "synthesizer", "synths": "synthesizer", "sax": "saxophone",
            "percussive": "percussion", "edm": "electronic", "heavy": "metal"}


def tokens(text):
    """Normalize text into search tokens, dropping stop words and applying aliases."""
    return [_ALIASES.get(t, t) for t in normalize(text).split() if t not in _STOP and len(t) > 1]


@dataclass(frozen=True)
class _Match:
    """A retrieval hit with its scores and whether the listener pinned it."""
    doc: Evidence
    content: float  # word-match score alone
    score: float  # content plus the language and song-name boosts, used for ordering
    named: bool  # the listener named this song or its artist
    pinned: bool  # named, or in a language the listener asked for


class KnowledgeIndex:
    """Local index of song cards and guide sections that can be searched for evidence."""
    def __init__(self, directory=KNOWLEDGE_DIR):
        """Load the song cards and listening guide from the knowledge directory and index them."""
        try:
            raw = json.loads((directory / "song_cards.json").read_text(encoding="utf-8"))
            guide = (directory / "music_guide.md").read_text(encoding="utf-8")
            self.documents = []
            self.search_text = []
            self.semantic_documents = []
            self.identities = {}
            self.languages = {}
            for card in raw["cards"]:
                url = card["sources"][0]["url"]
                if not safe_url(url) or not card["retrieved_at"] or not card["description"]:
                    raise ValueError()
                text = (f"{card['title']} by {card['artist']}. {card['description']} "
                        f"Version: {card['version']}. Language: {card['language']}. "
                        f"Unknown: {', '.join(card['unknowns'])}.")
                doc_id = "card_" + card["id"]
                self.identities[doc_id] = (normalize(card["title"]), normalize(card["artist"]))
                self.languages[doc_id] = set(tokens(card["language"])) - {"unknown"}
                self.documents.append(Evidence(doc_id, "song_card",
                    f"{card['title']} — {card['artist']}", text, url, card["retrieved_at"]))
                self.search_text.append(" ".join([card["language"], *card.get("tags", [])]))
                # Unknowns are deliberately excluded: a card must not semantically match a requested
                # trait merely because that trait appears in its list of unestablished facts.
                semantic_text = (f"{card['title']} by {card['artist']}. {card['description']} "
                                 f"Version: {card['version']}. Language: {card['language']}. "
                                 f"Tags: {', '.join(card.get('tags', []))}.")
                self.semantic_documents.append((doc_id, f"{card['title']} — {card['artist']}",
                                                semantic_text))
            for section in re.split(r"(?m)^## ", guide)[1:]:
                title, _, text = section.partition("\n")
                self.documents.append(Evidence(f"guide_{len(self.documents)}", "guide", title.strip(), text.strip()))
                self.search_text.append(title + " " + text)
            self.counts = [Counter(tokens(text)) for text in self.search_text]
            self.count_by_id = {doc.id: count for doc, count in zip(self.documents, self.counts)}
            self.df = Counter(t for count in self.counts for t in count)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            raise DiscoveryError("knowledge_unavailable", "The custom music documents could not be read.") from None

    def _is_generic(self, term: str) -> bool:
        """Tell whether a term appears in too many documents to be a useful search word."""
        # The floor of 3 keeps a very small collection from calling every word generic.
        return self.df[term] > max(3, GENERIC_TERM_SHARE * len(self.documents))

    def _term_weight(self, term: str) -> float:
        """Give distinctive terms full weight and retain broad traits as supporting evidence."""
        if self._is_generic(term):
            return 0.0
        if term in _BROAD_TRAITS:
            return BROAD_TRAIT_WEIGHT
        return 1.0

    @staticmethod
    def semantic_query(request: MusicRequest, intent: ParsedIntent | None = None) -> str:
        """Build semantic query text from the listener's words and validated intent expansion."""
        parts = [request.description, request.reference_song or ""]
        if intent:
            parts.extend(criterion.value for criterion in intent.criteria)
            parts.extend(intent.search_queries)
        return " ".join(part.strip() for part in parts if part and part.strip())

    def retrieve(self, request: MusicRequest, intent: ParsedIntent | None = None,
                 semantic_scores: dict[str, float] | None = None) -> list[Evidence]:
        """Return the song cards and guide sections that best match the request and intent."""
        expanded = ""
        if intent:
            expanded = " ".join([criterion.value for criterion in intent.criteria]
                                + intent.search_queries)
        query = set(tokens(request.description + " " + (request.reference_song or "") + " " + expanded))
        # "Instrumental" occupies the language field for non-vocal recordings in the source data,
        # but unknown-language piano pieces can still be valid instrumental candidates. Handle that
        # preference below as a vocal-evidence constraint instead of a language equality check.
        known_languages = ((set().union(*self.languages.values()) - {"instrumental"})
                           if self.languages else set())
        requested_languages = query.intersection(known_languages)
        instrumental_requested = "instrumental" in query
        requested_strict_genres = query.intersection(_STRICT_GENRES)
        # Padded so a song or artist name must match whole words ("gee" is not in "geeky").
        original = f" {normalize(request.description + ' ' + (request.reference_song or ''))} "
        matches = []
        for doc, count in zip(self.documents, self.counts):
            shared = query.intersection(count)
            content = sum(self._term_weight(t) * (1 + math.log(1 + count[t]))
                          * math.log(1 + len(self.documents) / self.df[t])
                          for t in shared)
            specific = any(not self._is_generic(t) for t in shared)
            named = requested = 0
            if doc.kind == "song_card":
                requested = len(self.languages[doc.id].intersection(query))
                named = sum(f" {identity} " in original for identity in self.identities[doc.id])
            if specific or requested or named:  # matching only generic words is noise
                matches.append(_Match(doc, content, content + 15 * requested + 30 * named,
                                      named=bool(named), pinned=bool(named or requested)))
        matches.sort(key=lambda m: (-m.score, m.doc.id))
        cards = [m for m in matches if m.doc.kind == "song_card"]
        # Language and instrumental-format requests are hard retrieval constraints when the local
        # collection can satisfy them. A Portuguese request should not spend half its four-card
        # budget on English songs that happen to share "percussion" or "energetic".
        language_matches = [m for m in cards
                            if self.languages[m.doc.id].intersection(requested_languages)]
        language_filter_active = bool(requested_languages and language_matches)
        if language_filter_active:
            cards = [m for m in cards if m.named or m in language_matches]
        instrumental_filter_active = False
        if instrumental_requested:
            non_vocal_cards = [m for m in cards
                               if "vocal" not in self.count_by_id[m.doc.id]
                               or "instrumental" in self.count_by_id[m.doc.id]]
            if non_vocal_cards:
                instrumental_filter_active = True
                cards = [m for m in cards if m.named or m in non_vocal_cards]
        genre_matches = [m for m in cards
                         if requested_strict_genres.intersection(self.count_by_id[m.doc.id])]
        genre_filter_active = bool(requested_strict_genres and genre_matches)
        if genre_filter_active:
            cards = [m for m in cards if m.named or m in genre_matches]
        # Autocut. The bar is the best word-match score, ignoring a named song (an anchor, not a
        # yardstick) and the language boost (a flag, not a similarity). A card in a requested
        # language, or one the listener named, is never cut.
        bar = max((m.content for m in cards if not m.named), default=0.0)
        cards = [m for m in cards if m.pinned or m.content >= AUTOCUT_SHARE * bar]
        guide = [m for m in matches if m.doc.kind == "guide"]
        if not semantic_scores:
            return [m.doc for m in cards[:4]] + [m.doc for m in guide[:2]]

        card_docs = {doc.id: doc for doc in self.documents if doc.kind == "song_card"}

        def allowed_semantic_card(doc_id):
            """Apply active lexical hard filters to candidates found only by embeddings."""
            count = self.count_by_id[doc_id]
            if language_filter_active and not self.languages[doc_id].intersection(requested_languages):
                return False
            if instrumental_filter_active and "vocal" in count and "instrumental" not in count:
                return False
            if genre_filter_active and not requested_strict_genres.intersection(count):
                return False
            return True

        eligible = [(doc_id, score) for doc_id, score in semantic_scores.items()
                    if doc_id in card_docs and isinstance(score, (int, float))
                    and math.isfinite(score) and score >= SEMANTIC_MIN_COSINE
                    and allowed_semantic_card(doc_id)]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        if not eligible:
            return [m.doc for m in cards[:4]] + [m.doc for m in guide[:2]]
        semantic_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(eligible, 1)}
        lexical_rank = {match.doc.id: rank for rank, match in enumerate(cards, 1)}

        def fused(match):
            """Rank lexical cards by lexical evidence first, with semantic agreement as a boost."""
            score = 2 / (RRF_K + lexical_rank[match.doc.id])
            if match.doc.id in semantic_rank:
                score += 1 / (RRF_K + semantic_rank[match.doc.id])
            return (not match.named, not match.pinned, -score, match.doc.id)

        cards.sort(key=fused)
        selected = cards[:4]
        selected_ids = {match.doc.id for match in selected}
        lexical_ids = set(lexical_rank)
        additions = 0
        for doc_id, _ in eligible:
            if len(selected) == 4 or additions == SEMANTIC_ONLY_LIMIT:
                break
            if doc_id in selected_ids or doc_id in lexical_ids:
                continue
            selected.append(_Match(card_docs[doc_id], 0.0, 0.0, False, False))
            selected_ids.add(doc_id)
            additions += 1
        return [m.doc for m in selected] + [m.doc for m in guide[:2]]


def retrieve_documents(request: MusicRequest, intent: ParsedIntent | None = None,
                       semantic_ranker=None) -> list[Evidence]:
    """Retrieve local evidence, falling back exactly to lexical ranking if semantics fail."""
    index = KnowledgeIndex()
    if semantic_ranker is None:
        return index.retrieve(request, intent)
    try:
        scores = semantic_ranker(index.semantic_query(request, intent), index.semantic_documents)
        if not isinstance(scores, dict):
            raise DiscoveryError("invalid_response", "Semantic retrieval returned an unreadable response.")
        return index.retrieve(request, intent, scores)
    except DiscoveryError as error:
        logger.info("semantic_retrieval_fallback code=%s", error.code)
    except Exception:
        logger.warning("semantic_retrieval_fallback code=unexpected")
    return index.retrieve(request, intent)
