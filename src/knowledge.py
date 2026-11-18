"""Rank local guide sections and song cards by matching query terms.

Retrieval is deliberately selective, because every song card that reaches the model can
become a recommended candidate. Two filters keep weak matches out:

* Specificity: a document must match at least one distinctive term. A word found in a large
  share of the documents ("vocal") cannot tell them apart, so matching only that word is noise.
* Autocut: song cards that score far below the best card are the noise tail and are dropped.
  A song the listener named is always kept and never sets the bar for the other cards.
"""

# Behavior: loads song_cards.json and music_guide.md into an index and returns at most four song cards and two guide sections per request.

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from src.models import DiscoveryError, Evidence, MusicRequest, ParsedIntent
from src.utils import normalize, safe_url

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge"
# Words found in more than this share of the documents are generic. In the shipped collection
# "vocal" is in 47% of them while the most common useful term, "piano", is in 25%.
GENERIC_TERM_SHARE = 0.35
# Song cards whose word-match score is under this share of the best card's are dropped. In the
# shipped collection a one-word genre match that is noise ("pop" for a Korean pop request) scores
# about 0.30 of the best card, and a one-word match that still helps ("piano" for an instrumental
# piano request) about 0.39. Revisit this if the collection grows.
AUTOCUT_SHARE = 0.35
# Negations ("no screaming", "sem voz") say what to exclude; matching them finds nothing useful.
_STOP = set("a an and as at be by for from i in is it like me music of on or please similar some song songs that the to want with would language recording track version original another other de da do em e para uma um com eu musica musicas no not without sem nao".split())
_ALIASES = {"instrumentais": "instrumental", "instrumentals": "instrumental", "vocals": "vocal",
            "voices": "vocal", "voice": "vocal", "singing": "vocal", "pianist": "piano",
            "portugues": "portuguese", "percussao": "percussion", "calma": "calm",
            "eletronica": "electronic", "guitars": "guitar", "drums": "drum",
            "peaceful": "calm", "gentle": "calm", "soothing": "calm",
            "joyful": "happy", "uplifting": "happy"}


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
            for section in re.split(r"(?m)^## ", guide)[1:]:
                title, _, text = section.partition("\n")
                self.documents.append(Evidence(f"guide_{len(self.documents)}", "guide", title.strip(), text.strip()))
                self.search_text.append(title + " " + text)
            self.counts = [Counter(tokens(text)) for text in self.search_text]
            self.df = Counter(t for count in self.counts for t in count)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            raise DiscoveryError("knowledge_unavailable", "The custom music documents could not be read.") from None

    def _is_generic(self, term: str) -> bool:
        """Tell whether a term appears in too many documents to be a useful search word."""
        # The floor of 3 keeps a very small collection from calling every word generic.
        return self.df[term] > max(3, GENERIC_TERM_SHARE * len(self.documents))

    def retrieve(self, request: MusicRequest, intent: ParsedIntent | None = None) -> list[Evidence]:
        """Return the song cards and guide sections that best match the request and intent."""
        expanded = ""
        if intent:
            expanded = " ".join([criterion.value for criterion in intent.criteria]
                                + intent.search_queries)
        query = set(tokens(request.description + " " + (request.reference_song or "") + " " + expanded))
        # Padded so a song or artist name must match whole words ("gee" is not in "geeky").
        original = f" {normalize(request.description + ' ' + (request.reference_song or ''))} "
        matches = []
        for doc, count in zip(self.documents, self.counts):
            shared = query.intersection(count)
            content = sum((1 + math.log(1 + count[t])) * math.log(1 + len(self.documents) / self.df[t])
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
        # Autocut. The bar is the best word-match score, ignoring a named song (an anchor, not a
        # yardstick) and the language boost (a flag, not a similarity). A card in a requested
        # language, or one the listener named, is never cut.
        bar = max((m.content for m in cards if not m.named), default=0.0)
        cards = [m for m in cards if m.pinned or m.content >= AUTOCUT_SHARE * bar]
        guide = [m for m in matches if m.doc.kind == "guide"]
        return [m.doc for m in cards[:4]] + [m.doc for m in guide[:2]]


def retrieve_documents(request: MusicRequest, intent: ParsedIntent | None = None) -> list[Evidence]:
    """Retrieve local evidence for a request using a fresh knowledge index."""
    return KnowledgeIndex().retrieve(request, intent)
