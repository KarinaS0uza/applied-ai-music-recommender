# 🎧 Model Card: Music Discovery (built on VibeFinder 1.0)

<!-- Behavior: documents the intended use, data, limits and evaluation of both recommenders. -->

This card covers two systems in one repository: the original content-based
recommender, VibeFinder 1.0 (sections 1–9), and the retrieval-augmented music
discovery service built on it (sections 10–14). Limitations, misuse, reliability
findings and notes on working with AI are in sections 11–14.

## 1. Model Name

VibeFinder 1.0 is a small, content-based music recommender.

## 2. Goal / Task

VibeFinder ranks songs for a listener based on preferred genre, mood, and energy.
It returns the top songs and explains which preferences contributed to each
score.

## 3. Data Used

The current catalog contains 1,000 songs (the first version had 28). Each record
includes an id, title, artist, genre, mood, energy, tempo, valence,
danceability, acousticness, and five added attributes: `subgenre`, `mood_tags`, `lyrical_theme`, `instrumentation_tags`,
and `language`. The five added attributes can contribute to a score when the
user supplies matching preferences. Tempo, valence, danceability, and
acousticness are stored but are not currently used for scoring. Some descriptive
metadata is subjective or has not been independently verified.

## 4. Algorithm Summary

The Strategy pattern provides four selectable ranking modes: `current`,
`genre-first`, `mood-first`, and `energy-focused`. The default `current` mode
weights genre and mood matches at 1 point each and energy similarity at up to 2
points. The other modes change these weights. Each matching advanced preference
adds 1 point. The system then greedily selects the top results, subtracting 1.5
points for each previously selected song by the same artist and 0.5 for each
previously selected song from the same non-preferred genre. The returned table
includes match reasons and any diversity deductions.

The Streamlit panel adds two layers around this scoring without changing it. A
typed genre is first resolved to the catalog labels it names (an exact label, a
short alias list, or a song whose genre, subgenre and language together contain
every word), and a preference can now list several acceptable genres. Results are
then shown as a match percentage: the song's own score divided by the best score
attainable for the preferences supplied, before diversity penalties.

## 5. Observed Behavior / Biases

Core labels and advanced tags require exact normalized matches, so related
labels may not receive credit. Genre is a scoring signal rather than a filter.
The artist and non-preferred-genre penalties encourage variety but can move a
candidate with a strong base score down the list. Different strategies can
produce different rankings. The system does not learn from listening history or
user feedback, and metadata quality affects the result.

Genre resolution is a small hand-written alias list plus word containment, so a
genre outside it (for example "brazilian jazz") matches nothing; the panel says so
instead of guessing. A match percentage is relative to the preferences supplied: 100%
means every supplied preference matched, and a request with fewer preferences can
reach 100% more easily. It is not a measure of how well a listener will like a song.

## 6. Evaluation Process

The rankings below are historical evaluation results from the original
20-song, single-strategy version, before the advanced metadata, selectable
strategies, diversity penalties, and table output were added. They document the
earlier weight-shift experiment and should not be read as current program output.

### My Preference — Ballad / Romantic

```text
TOP MUSIC RECOMMENDATIONS — My Preference — Ballad / Romantic
==================================================

1. All of My Life — Park Won
   Final score: 3.80
   Reasons:
     • genre match (+2.0)
     • mood match (+1.0)
     • energy similarity (+0.80)

2. Goodbye — Park Hyo Shin
   Final score: 2.76
   Reasons:
     • genre match (+2.0)
     • energy similarity (+0.76)

3. Slow — Jackson Wang & Ciara
   Final score: 1.72
   Reasons:
     • mood match (+1.0)
     • energy similarity (+0.72)

4. Baby Powder — Jenevieve
   Final score: 1.00
   Reasons:
     • energy similarity (+1.00)

5. Midnight Coding — LoRoom
   Final score: 0.94
   Reasons:
     • energy similarity (+0.94)
```

### Deep Intense Rock

```text
TOP MUSIC RECOMMENDATIONS — Deep Intense Rock
==================================================

1. Storm Runner — Voltline
   Final score: 3.98
   Reasons:
     • genre match (+2.0)
     • mood match (+1.0)
     • energy similarity (+0.98)

2. The Emptiness Machine — Linkin Park
   Final score: 3.94
   Reasons:
     • genre match (+2.0)
     • mood match (+1.0)
     • energy similarity (+0.94)

3. The Drug In Me Is Reimagined — Falling In Reverse
   Final score: 2.00
   Reasons:
     • genre match (+2.0)

4. Gym Hero — Max Pulse
   Final score: 1.94
   Reasons:
     • mood match (+1.0)
     • energy similarity (+0.94)

5. BIBI Vengeance — BIBI
   Final score: 1.70
   Reasons:
     • mood match (+1.0)
     • energy similarity (+0.70)
```

### High-Energy Pop

```text
TOP MUSIC RECOMMENDATIONS — High-Energy Pop
==================================================

1. Sunrise City — Neon Echo
   Final score: 3.84
   Reasons:
     • genre match (+2.0)
     • mood match (+1.0)
     • energy similarity (+0.84)

2. To the Bone — Pamungkas
   Final score: 3.66
   Reasons:
     • genre match (+2.0)
     • mood match (+1.0)
     • energy similarity (+0.66)

3. Gym Hero — Max Pulse
   Final score: 2.94
   Reasons:
     • genre match (+2.0)
     • energy similarity (+0.94)

4. Rooftop Lights — Indigo Parade
   Final score: 1.72
   Reasons:
     • mood match (+1.0)
     • energy similarity (+0.72)

5. Perspective — Cafuné
   Final score: 1.56
   Reasons:
     • mood match (+1.0)
     • energy similarity (+0.56)
```

### Chill Lofi

```text
TOP MUSIC RECOMMENDATIONS — Chill Lofi
==================================================

1. Library Rain — Paper Lanterns
   Final score: 3.90
   Reasons:
     • genre match (+2.0)
     • mood match (+1.0)
     • energy similarity (+0.90)

2. Midnight Coding — LoRoom
   Final score: 3.76
   Reasons:
     • genre match (+2.0)
     • mood match (+1.0)
     • energy similarity (+0.76)

3. Focus Flow — LoRoom
   Final score: 2.80
   Reasons:
     • genre match (+2.0)
     • energy similarity (+0.80)

4. Spacewalk Thoughts — Orbit Bloom
   Final score: 1.96
   Reasons:
     • mood match (+1.0)
     • energy similarity (+0.96)

5. Baby Powder — Jenevieve
   Final score: 1.70
   Reasons:
     • mood match (+1.0)
     • energy similarity (+0.70)
```

### Profile Comparisons — Weight-Shift Run

- **Ballad / Romantic and Deep Intense Rock:** These top-five lists share no
  songs. Ballad / Romantic targets energy `0.45` and rewards ballad or romantic
  matches, while Deep Intense Rock targets `0.9` and rewards rock or intense
  matches, so their different results fit their different preferences.
- **Ballad / Romantic and High-Energy Pop:** These lists also share no songs.
  The pop profile targets happy, high-energy pop, while the ballad profile
  targets romantic songs at lower energy; the different genre, mood, and energy
  preferences lead to different recommendations.
- **Ballad / Romantic and Chill Lofi:** Both lists include Midnight Coding and
  Baby Powder. Their target energies (`0.45` and `0.3`) are relatively close, so
  those songs earn useful energy points for both profiles; the other results
  reflect their different genre and mood preferences.
- **Deep Intense Rock and High-Energy Pop:** These lists share four songs:
  Storm Runner, The Emptiness Machine, Gym Hero, and Sunrise City. Both profiles
  target energy `0.9`, so high-energy songs can score well for both, even when
  the genre or mood is not an exact match. Gym Hero is pop and intense, so it
  earns genre and energy points for Pop and mood and energy points for Rock.
- **Deep Intense Rock and Chill Lofi:** These lists share no songs. The rock
  profile favors intense songs near energy `0.9`, while Chill Lofi favors lofi
  or chill songs near `0.3`; the opposite energy targets and different moods
  explain the split.
- **High-Energy Pop and Chill Lofi:** These lists share no songs. Happy pop near
  energy `0.9` fits the Pop profile, while chill lofi near `0.3` fits Chill
  Lofi, so their recommendations are expected to differ.

### Profile Observation

The Ballad / Romantic profile returned only two ballads, which is the full
number of ballad songs in the catalog. Only one song, All of My Life, matches
both the ballad genre and romantic mood; Goodbye is labeled melancholic. Slow
earned points for romantic mood and energy, while Baby Powder and Midnight
Coding earned points for energy. The small catalog cannot fill five spots with
ballads, and genre is a bonus rather than a filter, so songs from other genres
can still rank in the top five.

### Weight-Shift Experiment

I tested the effect of halving the genre weight from 2 points to 1 point and
doubling the energy-similarity weight from at most 1 point to at most 2 points.
The mood weight stayed at 1 point. The energy formula still gives a similarity
between 0 and 1 before its weight is applied, so the total score remains between
0 and 4 points.

The changes were modest overall: the top-five order stayed the same for Ballad /
Romantic and High-Energy Pop. For Deep Intense Rock, Gym Hero moved from fourth
to third, BIBI Vengeance moved from fifth to fourth, and Sunrise City entered the
top five as The Drug In Me Is Reimagined dropped out. For Chill Lofi,
Spacewalk Thoughts moved ahead of Focus Flow into third place.

The experiment made a few rankings different, but I cannot conclude the results
are more accurate without user feedback or a labeled set of preferred songs.

---

## 7. Intended Use and Non-Intended Use

This model is for educational exploration of content-based recommendations and
weighted scoring. It can show how different preferences change a small song
catalog's rankings. It is not designed for production use or to predict a real
listener's taste because it has no listening history, user feedback, or
validated ground-truth preferences.

## 8. Ideas for Improvement

- Add more songs, especially for underrepresented genres and moods.
- Add a genre filter or genre similarity option for listeners who want strict
  genre matches.
- Independently verify the song metadata and add explicit preference handling
  for numeric features such as valence or acousticness.
- Compare ranking strategies and diversity settings with listener feedback or
  a labeled set of preferred songs.

## 9. Personal Reflection

My biggest learning moment was seeing how much the weights shape the results.
Changing the balance between genre and energy moved songs in and out of the top
five, even though the catalog and profiles stayed the same. That helped me see
that a recommender's rankings reflect design choices, not an objective answer
about what someone will like. AI tools helped me think through the scoring rule,
organize the evaluation, and put the behavior into plain language. I also used
AI to explore edge-case profiles and explain why songs ranked where they did. I
needed to double-check the code, score arithmetic, and terminal output because
an AI explanation can be inconsistent with what the program actually returns.

I was surprised that such a simple score can still feel like a recommendation:
matching a few familiar signals and explaining them makes the list seem tailored
to a listener. But exact labels and a tiny catalog can make that feeling stronger
than the evidence supports. If I extended the project, I would expand the song
catalog, support related genres and moods, and ask listeners to rate or save
songs. That feedback could help me check whether the rankings actually match
their taste instead of relying only on the profile I designed.

## 10. Current RAG music discovery service

The sections above preserve the earlier CSV system and its historical experiment.
The current CLI/Streamlit service adds Tavily retrieval, Gemini generation, custom
documents and MusicBrainz metadata. See the [README](README.md) for the configured
model, provider eligibility, architecture and setup.

### Evidence and responsible use

The enhanced path selectively retrieves a project-authored guide and 56
source-backed song cards, then looks up candidate recording metadata in
MusicBrainz. Cards span several genres and languages but remain a tiny,
editorially selected collection, primarily derived from English Wikipedia.
Well-documented and popular recordings are overrepresented. Source summaries can
be wrong or incomplete, and duplicates of a source are not independent evidence.
Genre, emotion and activity suitability are not objective labels of enjoyment.

A work, a recording, a cover and a remix are different entities. Cards retain
version information; MusicBrainz preserves distinct recording IDs, artist credits,
release information and disambiguation. Even one exact title/artist match does
not prove the intended version. Missing metadata means unknown coverage, not
nonexistence. MusicBrainz is not used to infer vocal emotion, BPM or language.

The first generation call separates explicit requirements and exclusions from
weighted preferences, then produces at most two focused search queries. Tavily
uses the first for a general-web search and the second (or reuses the first) for a
search restricted to 12 music-focused domains. Both searches retain the same safe-URL,
content and request-relevance filters. Results from allowlisted hosts and their real
subdomains are ordered first, but the hostname does not count as evidence and does
not bypass the normal citation checks. If only the focused call fails, the general
results remain available. A second
call extracts song identities while excluding albums, playlists and other
non-song entities. The final call evaluates hard criteria and soft preferences;
Python rejects unestablished requirements and ranks supported or partial matches
with deterministic weighted scoring. Mechanical guardrails check source
membership, identity co-occurrence, candidate-specific citations, exact excerpts,
criterion coverage, safe URLs and duplicates. Each song is checked on its own, so
one that fails is dropped and logged while the rest are kept. A short quote can still be
misinterpreted: these checks do not prove that every paraphrase follows from
evidence or that a listener will enjoy the result. Common English and Portuguese
vocal exclusions receive extra omission checks, not a complete multilingual parser.

### Actual before/after retrieval experiment

Date: November 11, 2026. Six varied requests used short editorial paraphrases of
consulted source pages as fixed web fixtures. Each baseline contained one source;
the same source was retained while the actual lexical retriever selected guide
sections and cards. The fixtures are not captured Tavily API responses. No
Gemini calls or MusicBrainz calls were successful in this environment. Therefore
there are no generated before/after recommendations or measured MusicBrainz
improvements to report.

| Request | Baseline evidence | Added evidence | Evidence review |
| --- | --- | --- | --- |
| Slow expressive vocals | Teardrop/Fraser identity | Four vocal cards and two guide sections | No improvement: the lexical card selection misses Teardrop and includes unrelated styles; neither slow pace nor expressive delivery is established. |
| Electronic running music | Daft Punk identity/electronic description | Daft Punk, Underworld, Eno and two guide sections | Mixed: adds genre context and version cautions; ambient Eno is a weak activity match. Fast/upbeat suitability remains unverified. |
| Instrumental piano, no vocals | River Flows in You/First Love identity | Four piano cards and two guide sections | Adds instrumental evidence for the specified Sakamoto album version and Enya's Miss Clare Remembers. Study suitability and calmness remain personal judgments. |
| Portuguese/percussion | Jorge Ben language/samba identity | Two Portuguese cards and two guide sections | Improves language-specific selection, but adds no established percussion or energy evidence. |
| Heavy guitar, clean singing/no screaming | Metallica power-ballad identity | Guitar/vocal cards and two guide sections | Fails to establish the vocal exclusion; unrelated bossa nova/house cards remain. No qualifying match can be asserted. |
| Reference + modal jazz/piano/bass | So What/modal-jazz identity | Take Five, So What, two other cards and two guide sections | Adds the piano/bass introduction and distinction between moderate studio and faster live renditions. Reference and weak matches remain in context. |

The concrete instrumental-piano example improves the available evidence: the
baseline says only that Yiruma's track appears on a pianist's album, which does
not prove no vocals. Retrieved cards add a recording on Sakamoto's explicitly
instrumental solo-piano album and Enya's track whose vocals were discarded.
These are potential supported candidates for the generator, not reported model
selections. Two cards with limited evidence (Yiruma/Einaudi) retain unknown vocal
status rather than assuming that a pianist always records instrumentals.

An initial index also matched the word “fast” in *Fast Car* and generic language
wording in a Korean card. The revised index scores supported tags/language and
uses full identity phrases, with explicit-language weighting. It now avoids that
running-title match and ranks Portuguese cards first. The failed expressive and
clean-vocal cases show the remaining limits of lexical matching. More retrieved
text is not itself a quality score.

**Follow-up, November 17, 2026: filtering weak matches.** The two failed cases
shared a cause. The word "vocal" appears in 15 of the 32 documents, so any request
mentioning vocals scored every vocal card the same and the tie was broken by card
id, which is why Águas de Março and Despacito were returned for an expressive-vocals
request. Because the extraction step treats every retrieved card as a possible
recommendation, those cards were candidates, not just noise. Four changes followed:
a document must match at least one distinctive word (not one found in over 35% of
documents); negations such as "no" are not search words (it had pulled the piano
guide into a metal request); song cards scoring under 35% of the best card are cut,
except songs the listener named and cards in a requested language; and song and
artist names match whole words only (a request for "geeky" no longer matched the
card for *Gee*).

I measured this offline on 12 requests: the six above plus six I wrote afterward,
with the cards I judged relevant to each, taken from each card's tags and
description. The old retriever returned 37 cards of which 18 were relevant
(precision 0.49, recall 0.95); the new one returns 19, with the same 18 relevant
(precision 0.95, recall 0.95). These are my own labels, and I chose the thresholds
after seeing these requests, so the numbers show the failure patterns are gone, not
how it will do on new requests. The comparison also showed the first version of the
cut was too aggressive: it dropped two of the four piano cards, which is why the cut
ignores the language boost and sits at 35%, between the 0.30 share of a noise match
and the 0.39 share of a useful one. That margin is narrow and was tuned to a 26-card
collection, so it needed to be rechecked when the collection grew. The Teardrop miss
is not fixed offline: lexical matching cannot link "expressive" to "mournful", and
only the synonyms from the live analysis step bridge that (covered by
`test_expanded_query_still_reaches_cards_the_original_words_miss`). The table above
is the dated record of the original run and is left as recorded; the live
Gemini answers were not re-run, so the effect on generated recommendations is
unmeasured.

**Reproducible regression, November 17, 2026.** The six documented requests are
now stored as an offline fixture with fixed query expansions and human-labeled
relevant card IDs. After the collection expanded to 56 cards, generic terms stopped
contributing rank weight, broad mood/energy terms became supporting signals, and
explicit language, instrumental and metal requests gained bounded filters. The
current retriever returns all 15 labeled-relevant cards among 15 cards total
(micro precision 1.00, recall 1.00). The test requires both
metrics to remain at or above 0.90 as the card collection grows. This is a local
retrieval measurement, not a generated-answer or listener-satisfaction score.

The offline run's JSON report recorded actual document IDs, card titles, guide
sections and context sizes. A real generated-answer comparison would use the same
Tavily evidence, model, temperature and output limits across web-only,
document-enhanced and full MusicBrainz conditions. Human review of
identity/version, each explanation, source support, exclusions, diversity and
helpfulness remains pending, and the offline report does not invent a human
rating.

### Reliability evidence and limits

The [README testing summary](README.md#testing-summary) records operational
limits, automated checks and what live verification is still outstanding. Provider
and UI tests use controlled responses; they do not establish live model quality.
The offline retrieval experiment above likewise does not measure generated-answer
quality. A successful live comparison and human review remain pending.

## 11. Limitations and biases

**Intended use.** Exploring music by describing a sound, with sources a listener
can inspect. **Not intended for** judging a listener's taste, replacing a
streaming service's personalization, or establishing facts about a recording
without checking the cited source.

- **Coverage and popularity bias.** The 56 song cards are hand-selected and all
  summarize English Wikipedia pages, so well-documented, popular and
  English-language recordings are overrepresented. Web discovery through Tavily
  inherits that search engine's ranking and language skew. Its focused call favors
  an editorial list of 12 established music domains; this can improve music-specific
  context, but can also reinforce mainstream, English-language and publication-access
  bias. The parallel general search remains available, and specialist status is only
  an ordering tie-breaker, not a truth signal. A missing record means
  unknown coverage, not that a song does not exist.
- **Unverified catalog labels.** The 1,000-song CSV carries genre, mood, energy and
  other labels that have not been independently verified, and some entries are
  illustrative. Offline rankings reflect those labels, not audio analysis.
- **Subjective fit.** Mood, pace, energy and activity suitability are listener
  judgments. The system checks whether a source supports a trait (supported,
  partial or unknown). Its fit score is the weighted share of supported
  preferences, not a probability that a listener will enjoy the song.
- **Hybrid retrieval remains bounded.** The recorded comparison predates semantic
  retrieval: its lexical index missed the Teardrop card for an expressive-vocals
  request and could not establish a clean-singing exclusion. The current enhanced
  path keeps those lexical filters and can add at most two semantic-only cards above
  a fixed similarity threshold when fewer than four lexical cards qualify. This can
  recover paraphrases, but semantic similarity is not evidence that a requested
  trait is true; the normal citation and hard-constraint validators still decide
  what may be recommended. Language handling and the live quality gain remain
  unmeasured.
- **Guardrails check form, not truth.** Exact-excerpt checks show that a quote
  appears in the cited source and that the source names the song. They cannot show
  that a paraphrase is fair. Duplicated or syndicated text is not independent
  evidence, and AI-prepared card summaries are derived evidence.
- **Model interpretation.** Gemini interprets the request, so an unfamiliar genre
  or a request in a less common language may be understood less accurately. I
  have not measured the size of that effect.
- **Recording identity.** A work, a recording, a cover and a remix are different
  things, and even one exact MusicBrainz title and artist match does not prove the
  intended version.
- **Unscored fallback listings.** When the model's fit checks fail validation twice,
  or a broad request with no hard requirements leaves fewer than nine checked songs,
  the app lists songs that a retrieved page names, marked "NOT SCORED". Each shows a
  window of that page's own text near the song's title and notes which requested
  traits the text mentions. Detection is word matching with crude stemming and a few
  synonyms, so a mention of "slow" in the excerpt does not show that the song is slow,
  and the list order only reflects how many requested words appear. A page that lists
  many songs, such as a playlist, can yield several songs whose only link to the
  request is that they appear on the same page.
- **No personalization.** The system has no accounts, listening history or
  feedback loop.
- **Unmeasured generation quality.** No saved side-by-side comparison of
  generated answers exists yet, so claims about answer quality are limited to what
  the tests and the retrieval comparison show.

## 12. Potential misuse and how it is limited

| Risk | Mitigation | Remaining gap |
| --- | --- | --- |
| Prompt injection through a web page or document | Retrieved text is treated as data; the model must cite source IDs; validators reject unknown IDs, quotes missing from the source and cross-song citations; Python, not retrieved text, controls ranking | A carefully worded page could still shape a paraphrase that passes the exact-quote checks |
| Quota or cost abuse of a shared deployment | Server-held keys; at most 2 Tavily searches and 4 Gemini calls per request; 15-second session cooldown; 1,200-character input cap; timeouts and a 1 MB response cap; no automatic move to paid models or advanced search | The cooldown is per session, not per person, so many sessions could still add up |
| Credential exposure | Keys come from ignored files or environment variables, the repository holds placeholders only, and a test confirms a fixture secret never appears in errors or logs | The host must protect its own secrets |
| Presenting generated claims as fact | Results show supporting excerpts, a list of what could not be confirmed and tentative wording; Spotify buttons are search links, and the YouTube button opens a video only after a strict title-and-artist match, otherwise a search | A confident-sounding explanation can still be over-trusted |
| Unsafe links | Source URLs are rejected if they use other schemes, embed credentials, or point to localhost, local-network or non-global addresses | The content of a legitimate linked site is outside the system's control |
| Copyright | No lyrics are stored; supporting quotes are limited to 25 words; cards are paraphrases with a source link | Linked sites carry their own terms |
| Profiling listeners | No accounts and nothing written to disk; logs omit prompts and keys | In-memory caches hold results keyed by request-derived search queries for 15 minutes to an hour until restart, and providers apply their own retention terms |

## 13. Reliability testing: results and surprises

| Check | Result |
| --- | --- |
| Automated tests (controlled provider responses, no network) | 638 of 638 pass |
| Deterministic six-request local retrieval fixture | 1.00 micro precision, 1.00 recall (15 relevant cards retrieved out of 15 returned) |
| Hybrid fusion and fallback tests | Semantic agreement reranks qualified cards; at most two high-similarity cards fill unused slots; failures reproduce lexical output |
| Music-source search tests | One general and one domain-focused call stay within the two-call limit; safe/relevance filters still apply; lookalike hosts remain general; focused failure keeps general results |
| Retrieval comparison, six requests | 2 clear gains, 2 partial gains, 2 no improvement |
| Live end-to-end CLI runs, November 17, 2026 (8 attempts, listed in the [README](README.md#5-live-end-to-end-runs-tavily-musicbrainz-gemini)) | 1 passed every validator, 2 listed source-named songs with fit unconfirmed, 2 returned nothing rather than relax a required trait, 3 hit a Gemini error |
| Blank input | No provider calls; helpful message |
| Missing API keys | Safe error; no secret or traceback |
| Citation of an unknown source, a fabricated quote or another song's source | Rejected |
| Hard requirement without supporting evidence | Candidate dropped; requirement not relaxed |
| Live generated-answer comparison | Not recorded; human review pending |

**What surprised me**

1. **A fully green test suite did not mean the live app worked.** Entering
   `happy low beat pop` produced the error "The response could not be supported by
   the retrieved sources." Real Gemini output parsed correctly but violated one of
   the strict semantic validators, and every such failure surfaced as the same
   generic message. The mocked tests could not reveal that. Validation is now per
   song, so one bad song no longer discards the valid ones. The pipeline allows one
   retry per request, and the recommendation retry tells the model which checks
   failed. If that still fails it degrades with a warning instead of aborting:
   retrieval falls back to the original wording, and only source-identified songs
   are shown as "not scored", each with the source's own words about it.
2. **More retrieved text was not better evidence.** The expressive-vocals request
   pulled in four unrelated cards, and the word "fast" matched the song *Fast Car*
   for running music until I changed the index.
3. **The guardrails were weaker than they looked.** Exact-quote checks stop
   invented sources, but they say nothing about whether a paraphrase follows from
   the quote, so a human still has to read the excerpts.
4. **A saved session object broke after an update.** A discovery result stored in
   a Streamlit session before a code change lacked fields the new renderer
   expected, and the page crashed on rerun. The renderer now tolerates older
   results, and a regression test covers it.

## 14. Collaboration with AI

I used an AI coding assistant, Claude Code, to plan the extension, write tests
before the implementation, draft the provider adapters and validators, diagnose
failures and draft documentation. I treated its output as a draft to verify: an AI
explanation of a ranking or a behavior can disagree with what the program prints,
so I checked claims against test results, command output and provider
documentation, and I did not count mocked tests as live evidence.

**A suggestion I accepted.** The assistant proposed a leaner structure: fold the
separate service module into `src/rag.py` so `find_music` is the single entry point
that both the command line and Streamlit call, and consolidate duplicated
evaluation write-ups into this model card while the README keeps a short summary
and a link. I accepted it because the removed module only forwarded calls, and one
entry point and one detailed write-up leave less to keep consistent across the code,
tests, diagrams and docs. The full test suite still passed afterward (50 tests at
that point) and the modules still import independently.

That single module later grew to roughly 470 lines and was split by responsibility:
shared types in `src/models.py`, text and URL helpers in `src/utils.py`, evidence
checks in `src/validation.py`, the staged workflow in `src/discovery.py`, and provider
wiring plus `find_music` in `src/composition.py`. `find_music` is still the one entry
point for the command line and Streamlit, `src/rag.py` no longer exists, and the
suite now has 638 passing tests. `tests/test_rag.py` keeps its name and now covers
the whole discovery pipeline.

**A suggestion I did not accept as written.** The assistant's first retrieval index
scored documents by overlapping words. Testing showed it was too loose: the word
"fast" in a running-music request matched the song *Fast Car*, and a generic word
like "language" pulled in an unrelated card. I had the index changed to score
supported tags and languages, match full identity phrases and weight an explicit
language above generic wording, and I added regression tests for both cases
(`test_fast_activity_does_not_match_fast_car_title` and
`test_explicit_language_outweighs_generic_language_word`). The revised index still
fails on paraphrases such as the expressive-vocals request, so I documented that
limit rather than adding a vector database, which would have added a dependency
and an extra API for a collection of 56 cards.
