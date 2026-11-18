"""Tests for local retrieval of song cards and guide sections."""

# Behavior: reads the real data/knowledge files and temporary copies to check retrieval limits, filtering and safe errors.

import json
from pathlib import Path
import re

import pytest

from src.knowledge import KNOWLEDGE_DIR, KnowledgeIndex, retrieve_documents, tokens
from src.models import DiscoveryError, MusicRequest, ParsedIntent, TasteCriterion


def test_retrieves_relevant_sections():
    """Retrieves relevant sections."""
    results = KnowledgeIndex().retrieve(MusicRequest("instrumental piano studying no vocals"))
    assert any("piano" in e.text.lower() for e in results)
    assert len(results) <= 6
    assert len([e for e in results if e.kind == "song_card"]) <= 4


def test_no_match_returns_no_documents():
    """No match returns no documents."""
    assert retrieve_documents(MusicRequest("zzzzqxxvv")) == []


def test_song_card_preserves_source_url():
    """Song card preserves source url."""
    results = retrieve_documents(MusicRequest("Take Five Dave Brubeck jazz"))
    assert any(e.url == "https://en.wikipedia.org/wiki/Take_Five" for e in results)
    assert all(e.retrieved_at for e in results if e.kind == "song_card")


def test_collection_is_diverse_sourced_and_not_changed_by_retrieval():
    """Collection is diverse sourced and not changed by retrieval."""
    path = KNOWLEDGE_DIR / "song_cards.json"
    before = path.read_bytes()
    cards = json.loads(before)["cards"]
    assert len(cards) >= 26
    assert len({c["artist"] for c in cards}) >= 20
    assert len({c["language"] for c in cards}) >= 6
    assert all(c["sources"] and c["retrieved_at"] and c["unknowns"] for c in cards)
    retrieve_documents(MusicRequest("Portuguese percussion samba"))
    assert before == path.read_bytes()


def test_fast_activity_does_not_match_fast_car_title():
    """Fast activity does not match fast car title."""
    results = retrieve_documents(MusicRequest("fast upbeat electronic music for running"))
    assert "card_fast-car" in {d.id for d in KnowledgeIndex().documents}  # so its absence below is a real decision
    assert any(e.kind == "song_card" for e in results)                    # and retrieval did return cards
    assert not any(e.id == "card_fast-car" for e in results)


def test_explicit_language_outweighs_generic_language_word():
    """Explicit language outweighs generic language word."""
    results = retrieve_documents(MusicRequest("Portuguese-language music with percussion"))
    cards = [e for e in results if e.kind == "song_card"]
    assert {e.id for e in cards[:2]} == {"card_aguas", "card_mas-que-nada"}


def test_intent_queries_expand_local_retrieval_without_replacing_original_request():
    """Intent queries expand local retrieval without replacing original request."""
    request = MusicRequest("I want something gentle")
    intent = ParsedIntent([
        TasteCriterion("mood", "preference", "peaceful", 3, "gentle")
    ], ["peaceful calm ambient songs"])
    original = {e.id for e in retrieve_documents(request)}
    expanded = {e.id for e in retrieve_documents(request, intent)}
    assert original and original.intersection(expanded)
    assert any(doc_id.startswith("card_") for doc_id in expanded - original)


def card_ids(description, reference=None, intent=None):
    """Return the retrieved song-card ids without their card_ prefix."""
    docs = retrieve_documents(MusicRequest(description, reference), intent)
    return [e.id.removeprefix("card_") for e in docs if e.kind == "song_card"]


def guide_titles(description):
    """Return the titles of the guide sections retrieved for a description."""
    return [e.title for e in retrieve_documents(MusicRequest(description)) if e.kind == "guide"]


def test_generic_word_alone_retrieves_no_song_cards():
    """Generic word alone retrieves no song cards."""
    # "vocal" is in nearly half the documents. Matching only it used to return four arbitrary
    # cards (Águas de Março, Daft Punk, Underworld, Despacito), tied and ordered by id.
    assert card_ids("female vocals") == []
    assert card_ids("Slow-paced music with emotionally expressive vocals") == []


def test_generic_word_does_not_add_unrelated_cards_to_a_specific_request():
    """Generic word does not add unrelated cards to a specific request."""
    assert set(card_ids("Heavy guitar music with clean singing and no screaming")) == {
        "du-hast", "nothing-else-matters"}


def test_negation_words_are_not_search_terms():
    """Negation words are not search terms."""
    assert tokens("heavy guitar, no screaming, without vocals, sem gritos, não") == [
        "metal", "guitar", "screaming", "vocal", "gritos"]
    # "no" used to match the piano guide's "no vocals" and pull it into a metal request.
    assert "Instrumental piano and concentration" not in guide_titles(
        "Heavy guitar music with clean singing and no screaming")


def test_explicit_language_returns_the_expanded_korean_pop_coverage():
    """An explicit language keeps the result budget focused on matching-language cards."""
    assert set(card_ids("upbeat Korean pop for a morning run")) == {
        "ditto", "gee", "love-scenario", "next-level-aespa"}


def test_autocut_keeps_cards_that_still_help():
    """Autocut keeps cards that still help."""
    # Einaudi and Yiruma match only "piano", but that is what was asked for.
    assert set(card_ids("Calm instrumental piano for studying with no vocals")) == {
        "merry-christmas", "miss-clare", "nuvole", "river-flows"}


def test_named_reference_is_kept_and_does_not_set_the_bar():
    """Named reference is kept and does not set the bar."""
    ids = card_ids("Modal jazz with a piano and bass introduction and a moderate pace, by another artist",
                   "Take Five — Dave Brubeck Quartet")
    assert ids == ["take-five", "so-what"]  # not Superstition or Daft Punk, matched only by "bass"


def test_song_and_artist_names_match_whole_words_only():
    """Song and artist names match whole words only."""
    assert "gee" not in card_ids("geeky synth music")
    assert card_ids("something like Gee by Girls' Generation")[0] == "gee"


def test_expanded_query_still_reaches_cards_the_original_words_miss():
    """Expanded query still reaches cards the original words miss."""
    # The live analysis step supplies synonyms such as "mournful"; the offline words alone cannot.
    request = MusicRequest("Slow-paced music with emotionally expressive vocals")
    intent = ParsedIntent([TasteCriterion("vocals", "preference", "mournful ethereal vocals", 3, "expressive vocals")],
                          ["mournful ethereal trip hop vocals"])
    assert "teardrop" in [e.id.removeprefix("card_") for e in retrieve_documents(request, intent)]


def test_small_collection_does_not_treat_every_word_as_generic(tmp_path):
    """Small collection does not treat every word as generic."""
    card = json.loads((KNOWLEDGE_DIR / "song_cards.json").read_text())["cards"][0]
    (tmp_path / "song_cards.json").write_text(json.dumps({"cards": [card]}))
    (tmp_path / "music_guide.md").write_text("## Guide\nVocabulary about saxophone.")
    results = KnowledgeIndex(tmp_path).retrieve(MusicRequest("saxophone"))
    assert {e.id for e in results} == {"card_take-five", "guide_1"}


def test_unreadable_or_unsafe_documents_raise_a_safe_error(tmp_path):
    """Unreadable or unsafe documents raise a safe error."""
    with pytest.raises(DiscoveryError) as exc:
        KnowledgeIndex(tmp_path)
    assert exc.value.code == "knowledge_unavailable"

    card = json.loads((KNOWLEDGE_DIR / "song_cards.json").read_text())["cards"][0]
    card["sources"] = [{"url": "javascript:alert(1)"}]
    (tmp_path / "song_cards.json").write_text(json.dumps({"cards": [card]}))
    (tmp_path / "music_guide.md").write_text("## Guide\nVocabulary only.")
    with pytest.raises(DiscoveryError) as exc:
        KnowledgeIndex(tmp_path)
    assert exc.value.code == "knowledge_unavailable"


# --- tokens, document construction and retrieval limits ------------------------------------

def test_tokens_fold_accents_drop_stop_words_and_apply_aliases():
    """Tokens fold accents drop stop words and apply aliases."""
    assert tokens("Peaceful gentle SOOTHING piano") == ["calm", "calm", "calm", "piano"]
    assert tokens("Músicas instrumentais com voz") == ["instrumental", "voz"]
    assert tokens("guitars drums voices Singing") == ["guitar", "drum", "vocal", "vocal"]
    assert tokens("Eletrônica e Portuguès") == ["electronic", "portuguese"]
    assert tokens("a I x the and") == []


@pytest.mark.parametrize(("variants", "canonical"), [
    ("relaxed relaxing serene tranquil", "calm"),
    ("dreamy", "ethereal"),
    ("sorrowful melancholy melancholic", "mournful"),
    ("synth synths", "synthesizer"),
    ("sax", "saxophone"),
    ("percussive", "percussion"),
    ("edm", "electronic"),
])
def test_added_aliases_use_existing_canonical_vocabulary(variants, canonical):
    """New spellings and close synonyms normalize to vocabulary already used by the cards and guide."""
    assert set(tokens(variants)) == {canonical}


@pytest.mark.parametrize(("variant", "canonical", "required_card"), [
    ("dreamy", "ethereal", "teardrop"),
    ("melancholic", "mournful", "teardrop"),
    ("synth", "synthesizer", "du-hast"),
    ("sax jazz", "saxophone jazz", "take-five"),
    ("edm house", "electronic house", "around-the-world"),
])
def test_alias_requests_retrieve_the_same_required_card_as_canonical_words(
        variant, canonical, required_card):
    """An alias reaches the same relevant song card without changing the canonical request's result."""
    variant_ids = set(card_ids(variant))
    canonical_ids = set(card_ids(canonical))
    assert required_card in variant_ids == canonical_ids


def test_retrieval_benchmark_keeps_precision_and_recall_above_ninety_percent():
    """The six documented requests measure local retrieval without Gemini or network providers."""
    fixture = Path(__file__).parent / "fixtures" / "retrieval_cases.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
    true_positives = retrieved_total = relevant_total = 0
    for case in cases:
        request = MusicRequest(case["description"], case["reference_song"])
        intent = ParsedIntent([], case["search_queries"])
        retrieved = {doc.id.removeprefix("card_") for doc in retrieve_documents(request, intent)
                     if doc.kind == "song_card"}
        relevant = set(case["relevant_card_ids"])
        assert retrieved.intersection(relevant), f"{case['id']} retrieved no relevant card"
        true_positives += len(retrieved.intersection(relevant))
        retrieved_total += len(retrieved)
        relevant_total += len(relevant)
    precision = true_positives / retrieved_total
    recall = true_positives / relevant_total
    assert precision >= 0.90 and recall >= 0.90, (precision, recall)


# --- hybrid semantic + lexical retrieval --------------------------------------------------

def ids(documents, kind="song_card"):
    """Return document ids of one kind for concise hybrid assertions."""
    return [document.id for document in documents if document.kind == kind]


def test_no_semantic_scores_preserve_the_exact_lexical_result():
    """The existing ordering is the fallback contract, including guide sections."""
    index = KnowledgeIndex()
    request = MusicRequest("electronic dance music")
    assert index.retrieve(request) == index.retrieve(request, semantic_scores={})


def test_semantic_agreement_can_rerank_lexically_qualified_cards():
    """Semantic agreement boosts a card that already passed the lexical safety gates."""
    index = KnowledgeIndex()
    request = MusicRequest("electronic dance music")
    lexical = ids(index.retrieve(request))
    hybrid = ids(index.retrieve(request, semantic_scores={"card_born-slippy": 0.95}))
    assert lexical[:2] == ["card_around-the-world", "card_born-slippy"]
    assert hybrid[:2] == ["card_born-slippy", "card_around-the-world"]


def test_high_similarity_semantic_card_fills_an_unused_slot():
    """A paraphrase-only match can be retrieved when lexical matching found nothing."""
    index = KnowledgeIndex()
    request = MusicRequest("otherworldly sorrow in a hovering voice")
    assert ids(index.retrieve(request)) == []
    assert ids(index.retrieve(request, semantic_scores={"card_teardrop": 0.91})) == [
        "card_teardrop"]


def test_semantic_threshold_and_addition_limit_prevent_unbounded_matches():
    """Low similarities are ignored and at most two semantic-only cards enter the context."""
    index = KnowledgeIndex()
    request = MusicRequest("zzzzqxxvv")
    low = {"card_teardrop": 0.649}
    assert ids(index.retrieve(request, semantic_scores=low)) == []
    lexical_request = MusicRequest("calm piano")
    assert index.retrieve(lexical_request, semantic_scores=low) == index.retrieve(lexical_request)
    high = {"card_teardrop": 0.95, "card_du-hast": 0.90, "card_take-five": 0.85}
    assert ids(index.retrieve(request, semantic_scores=high)) == [
        "card_teardrop", "card_du-hast"]


def test_semantic_only_card_does_not_displace_four_lexical_cards():
    """The first rollout preserves a full lexical card set even when another card scores highly."""
    index = KnowledgeIndex()
    request = MusicRequest("calm piano")
    lexical = ids(index.retrieve(request))
    hybrid = ids(index.retrieve(request, semantic_scores={"card_du-hast": 0.99}))
    assert hybrid == lexical and len(hybrid) == 4


def test_named_reference_keeps_priority_over_semantic_ranking():
    """A listener-named song remains the anchor even when another card has the strongest similarity."""
    index = KnowledgeIndex()
    request = MusicRequest("modal jazz piano and bass", "Take Five — Dave Brubeck Quartet")
    hybrid = ids(index.retrieve(request, semantic_scores={"card_so-what": 0.99,
                                                           "card_take-five": 0.70}))
    assert hybrid[:2] == ["card_take-five", "card_so-what"]


def test_semantic_only_cards_obey_hard_filters_while_named_references_remain_anchors():
    """Embeddings cannot bypass language or genre filters, but a named comparison stays visible."""
    index = KnowledgeIndex()
    korean = ids(index.retrieve(
        MusicRequest("Korean pop", "Take Five — Dave Brubeck Quartet"),
        semantic_scores={"card_no-scrubs": 0.99},
    ))
    assert korean[0] == "card_take-five"
    assert "card_no-scrubs" not in korean

    metal = ids(index.retrieve(
        MusicRequest("heavy guitar"), semantic_scores={"card_ditto": 0.99}
    ))
    assert set(metal) == {"card_du-hast", "card_nothing-else-matters"}


def test_semantic_retrieval_does_not_change_guide_selection():
    """Guide sections remain lexical vocabulary rather than semantic song evidence."""
    index = KnowledgeIndex()
    request = MusicRequest("electronic dance music")
    lexical = ids(index.retrieve(request), "guide")
    hybrid = ids(index.retrieve(request, semantic_scores={"card_born-slippy": 0.95}), "guide")
    assert hybrid == lexical


@pytest.mark.parametrize("failure", [
    DiscoveryError("provider_unavailable", "safe"), RuntimeError("fixture-secret")
])
def test_semantic_ranker_failure_falls_back_to_exact_lexical_results(failure):
    """Embedding errors cannot remove or reorder the current local evidence."""
    request = MusicRequest("electronic dance music")
    expected = retrieve_documents(request)

    def fail(query, documents):
        raise failure

    assert retrieve_documents(request, semantic_ranker=fail) == expected


def test_semantic_text_excludes_unknowns_and_query_includes_validated_expansion():
    """Unknown traits do not attract cards, while the request, reference and intent reach embeddings."""
    index = KnowledgeIndex()
    assert all("Unknown:" not in text for _, _, text in index.semantic_documents)
    request = MusicRequest("expressive voice", "Teardrop — Massive Attack")
    intent = ParsedIntent([TasteCriterion("vocals", "preference", "mournful vocals", 3,
                                         "expressive voice")], ["ethereal trip hop"])
    query = index.semantic_query(request, intent)
    assert all(part in query for part in ["expressive voice", "Teardrop", "mournful vocals",
                                          "ethereal trip hop"])


def test_a_request_made_only_of_stop_words_retrieves_nothing():
    """A request made only of stop words retrieves nothing."""
    assert retrieve_documents(MusicRequest("the and of please")) == []


def test_song_cards_become_evidence_built_from_the_card_fields():
    """Song cards become evidence built from the card fields."""
    cards = json.loads((KNOWLEDGE_DIR / "song_cards.json").read_text())["cards"]
    documents = [d for d in KnowledgeIndex().documents if d.kind == "song_card"]
    assert len(documents) == len(cards)
    for card, doc in zip(cards, documents):
        assert doc.id == "card_" + card["id"]
        assert doc.title == f"{card['title']} — {card['artist']}"
        assert doc.text.startswith(f"{card['title']} by {card['artist']}. {card['description']}")
        assert f"Version: {card['version']}." in doc.text and f"Language: {card['language']}." in doc.text
        assert all(unknown in doc.text for unknown in card["unknowns"])  # what the card cannot say stays visible


def test_guide_sections_become_vocabulary_documents_without_links():
    """Guide sections become vocabulary documents without links."""
    headings = re.findall(r"(?m)^## (.+)$", (KNOWLEDGE_DIR / "music_guide.md").read_text())
    guides = [d for d in KnowledgeIndex().documents if d.kind == "guide"]
    assert [d.title for d in guides] == [heading.strip() for heading in headings]
    assert all(d.text and d.url is None and d.retrieved_at is None for d in guides)


def test_every_document_id_is_unique_and_names_its_kind():
    """Every document id is unique and names its kind."""
    documents = KnowledgeIndex().documents
    ids = [d.id for d in documents]
    assert len(ids) == len(set(ids))  # a clash would make the discovery pipeline reject the evidence
    assert all(d.id.startswith("card_" if d.kind == "song_card" else "guide_") for d in documents)


REAL_CARD = json.loads((KNOWLEDGE_DIR / "song_cards.json").read_text())["cards"][0]


def test_retrieval_returns_at_most_four_cards_then_two_guide_sections_with_ties_ordered_by_id(tmp_path):
    """Retrieval returns at most four cards then two guide sections with ties ordered by id."""
    cards = [{**REAL_CARD, "id": f"c{n}", "title": f"Song {n}", "artist": f"Artist {n}",
              "language": "Portuguese", "tags": []} for n in range(6)]
    (tmp_path / "song_cards.json").write_text(json.dumps({"cards": cards}))
    (tmp_path / "music_guide.md").write_text("\n".join(f"## Section {n}\nSaxophone vocabulary {n}." for n in range(3)))
    results = KnowledgeIndex(tmp_path).retrieve(MusicRequest("Portuguese saxophone"))
    # Six cards fill document slots 0-5, so the three guide sections are guide_6 to guide_8.
    assert [e.id for e in results] == ["card_c0", "card_c1", "card_c2", "card_c3", "guide_6", "guide_7"]


@pytest.mark.parametrize("cards", [
    "{not json",
    {"songs": []},
    {"cards": [{**REAL_CARD, "retrieved_at": ""}]},
    {"cards": [{**REAL_CARD, "description": ""}]},
    {"cards": [{**REAL_CARD, "sources": []}]},
    {"cards": [{key: value for key, value in REAL_CARD.items() if key != "unknowns"}]},
], ids=["bad-json", "no-cards-list", "undated-card", "empty-description", "no-source", "missing-field"])
def test_unusable_card_files_raise_the_safe_error(tmp_path, cards):
    """Unusable card files raise the safe error."""
    (tmp_path / "song_cards.json").write_text(cards if isinstance(cards, str) else json.dumps(cards))
    (tmp_path / "music_guide.md").write_text("## Guide\nVocabulary only.")
    with pytest.raises(DiscoveryError) as exc:
        KnowledgeIndex(tmp_path)
    assert (exc.value.code, str(exc.value)) == ("knowledge_unavailable", "The custom music documents could not be read.")


def test_a_missing_guide_file_raises_the_safe_error(tmp_path):
    """A missing guide file raises the safe error."""
    (tmp_path / "song_cards.json").write_text(json.dumps({"cards": [REAL_CARD]}))
    with pytest.raises(DiscoveryError) as exc:
        KnowledgeIndex(tmp_path)
    assert exc.value.code == "knowledge_unavailable"
