# Music Discovery

<!-- Behavior: explains how to set up, run and test the music discovery app and how it works. -->

A retrieval-augmented music discovery assistant. Describe what you want to hear:
a mood, genre, language, instrument, activity or vocal preference, optionally with
a reference song. The system retrieves evidence from the web, from a prepared music
guide and song cards, and from MusicBrainz. Gemini selects up to five songs from
that evidence, and Python validates every citation before anything is shown. Each
result comes with an explanation, the traits that could not be confirmed, and
links to its sources.

**Why it matters.** A language model answering from memory can invent songs or
claim traits it cannot know. Here a song is shown only when a retrieved source
names it, each claimed trait has to be backed by an exact excerpt, and a request
the evidence cannot support returns fewer matches or a clear "not established"
message rather than a guess.

## Original project

This builds on **Music Recommender Simulation** (model name VibeFinder 1.0), a
content-based recommender over a CSV catalog that started with 28 songs and now
holds 1,000. It scored genre, mood and energy preferences plus optional subgenre,
mood-tag, lyrical-theme, instrumentation and language matches. It offered four
ranking strategies (Balanced, Genre-First, Mood-First, Energy-Focused), applied
artist and genre diversity penalties while picking the top results, and printed
each score's reasons in a terminal table. It could only recommend what was in the
CSV, so this version keeps it as an offline mode and adds retrieval so discovery
reaches beyond the catalog.

## Architecture overview

```mermaid
flowchart TD
    Strategy["User selects --strategy"] --> Scoring["Score preferences and record reasons"]
    Profiles["Four hardcoded preference profiles"] --> Scoring
    Catalog["data/songs.csv"] --> Loader["load_songs: parse catalog metadata"]
    Loader --> Scoring
    Scoring --> Selection["recommend_songs: select top five with diversity penalties"]
    Selection --> Output["Terminal tables: songs, scores and rule-based reasons"]

    %% Existing development test, run separately from the application.
    Tests["pytest: repeated-artist penalty test"] -.->|"Checks ranking and penalty reason"| Selection

    UI["CLI query or Streamlit form"] -->|"CSV mode"| Scoring
    UI -->|"Description and optional reference"| RAG["discovery.py: shared discovery workflow"]
    RAG --> Intent["Gemini: analyze criteria and focused queries"]
    Intent --> Web["Tavily: one general + one music-source-focused search"]
    RAG -->|"Enhanced mode"| Docs["Hybrid lexical + semantic retrieval: up to four cards and two guides"]
    Web --> Extract["Gemini: extract supported candidates"]
    Docs --> Extract
    Extract -->|"Enhanced mode"| Metadata["MusicBrainz: recording identities and versions"]
    Extract --> Generate["Gemini: evidence-grounded selection"]
    Metadata --> Generate
    Generate --> Validate["Validate evidence and score weighted preferences"]
    Validate --> Results["Up to five exact or partial matches with uncertainties"]
    Results --> Human["Listener reviews excerpts and original sources"]
    Checks["pytest: controlled pipeline, provider and UI checks"] -.-> Validate
    Review["Recorded six-request retrieval comparison"] -.-> Docs

    Validate -->|"Invalid model output"| Degrade["Retry once, then warn and keep only source-identified songs"]
    Degrade --> Results
    Results -->|"No supported match"| Fallback["Up to five CSV catalog songs, else unverified songs, else a related video, else a plain message"]
    Fallback --> Human
    Guard["Input checks, call limits, timeouts, safe errors"] -.-> RAG
```

The diagram source is [diagrams/architecture.mmd](diagrams/architecture.mmd).

**Data flow.** A CLI query or Streamlit form builds one `MusicRequest` and calls
`find_music` in `src/composition.py`, so both interfaces share a single pipeline. Gemini
turns the request into criteria and one or two focused search queries. Tavily uses
at most two calls: one general-web search and one search restricted to a small,
explicit set of music-focused domains. It applies the same URL, content and
request-relevance checks to both, deduplicates them, and places qualifying specialist
sources first without treating their domain as proof. In enhanced mode (always on in the Streamlit app, opt-in
with `--enhanced` on the CLI) a hybrid lexical/semantic index adds relevant guide
sections and song cards, and MusicBrainz adds recording identities. Gemini extracts and then
evaluates candidates against that evidence. Python validates the result, computes
the fit score and returns the top five.

**Where people and tests check the AI.** The listener is the human in the loop:
every result shows its supporting excerpts, stated uncertainties and original
source links, and platform buttons are search links; the YouTube button opens a
video only when a strict match on title and artist confirms it. The automated tests exercise validation, guardrails, providers and the UI
with controlled responses, and a recorded six-request comparison measures what
the extra retrieval adds (see [Testing summary](#testing-summary)).

## Setup

**Requirements.** Python 3.13 (tested with 3.13.9) and Git, on macOS, Linux or
Windows. Nothing else is needed to run the tests, the offline command line or the
Music Collection panel. Live discovery ("Find music") also needs a Tavily API key and a
Gemini API key from your own accounts (see Credentials below).

**Install.** Run every command from the repository root, inside the virtual
environment:

```bash
git clone https://github.com/KarinaS0uza/applied-ai-music-recommender.git
cd applied-ai-music-recommender
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

On Windows PowerShell, create the environment with `py -3.13 -m venv .venv` and
activate it with `.\.venv\Scripts\Activate.ps1`. The tests make no network calls and
need no keys; every test passing confirms the install.

Pinned versions: google-genai 2.18.1, Streamlit 1.67.0, requests 2.34.2,
pytest 9.2.0. The tests rely on the pinned Streamlit version, so always run the
project from this virtual environment, not from a global or Anaconda Python.

**What works without keys.**

| Feature | Keys needed |
|---|---|
| `python -m pytest -q` | None |
| Streamlit **Music Collection** panel ("Browse collection") | None |
| `python -m src.main` (offline catalog ranking) | None |
| Streamlit **Find music** and `python -m src.main --query ...` | Tavily and Gemini |
| Exact YouTube links instead of search links | Optional YouTube key |

**Credentials.** Live discovery needs a Tavily key and a Gemini key. Create them in
the [Tavily dashboard](https://app.tavily.com) and in
[Google AI Studio](https://aistudio.google.com/apikey), then create
`.streamlit/secrets.toml` with your keys; the file is ignored by Git and never leaves
your machine:

```toml
TAVILY_API_KEY = "your-tavily-key"
GEMINI_API_KEY = "your-gemini-key"
# GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
# SEMANTIC_RETRIEVAL_ENABLED = false     # optional lexical-only kill switch
# MUSIC_SOURCE_FOCUS_ENABLED = false     # optional general-web-only kill switch
# YOUTUBE_API_KEY = "your-youtube-key"   # optional, see below
# SPOTIFY_CLIENT_ID = "your-spotify-client-id"          # optional, both or neither
# SPOTIFY_CLIENT_SECRET = "your-spotify-client-secret"
```

Streamlit looks for `.streamlit/secrets.toml` in the folder you start the app from, so
launch it from the repository root:

```bash
python -m streamlit run src/app.py
```

Streamlit prints the local address (usually `http://localhost:8501`); stop it with
`Ctrl+C`. You can also set `TAVILY_API_KEY` and `GEMINI_API_KEY` as environment variables
instead. Never commit them. The Streamlit app reads
server-side configuration, so viewers never enter keys. `GEMINI_MODEL` defaults to
`gemini-3.5-flash-lite` and `GEMINI_EMBEDDING_MODEL` to `gemini-embedding-001`.
Set `SEMANTIC_RETRIEVAL_ENABLED=false` to use the original lexical local retrieval
without disabling the rest of enhanced mode. The web search also makes one bounded
music-focused request by default, using AllMusic, Bandcamp, Billboard, Discogs,
Last.fm, MusicBrainz, MusicRadar, NPR, Official Charts, Pitchfork, Rolling Stone and
Stereogum. Set `MUSIC_SOURCE_FOCUS_ENABLED=false` to restore two unrestricted
queries (or one when Gemini produced only one query). This changes source targeting,
not validation: every result still needs safe, relevant evidence. `MUSICBRAINZ_CONTACT` defaults to this project's GitHub
URL; set it to your own contact URL or email. MusicBrainz needs no key.

**Optional: exact YouTube links.** Add a `YOUTUBE_API_KEY` (a YouTube Data API v3 key
from the Google Cloud console; no OAuth needed) and each recommendation links to its
exact video instead of a YouTube search. The match is deliberately strict: the title and
artist must both appear, and covers, live takes, remixes, karaoke and mixes are skipped,
so an unsure song keeps its search link. Google's default allowance is about 100
searches a day, so the app looks up at most five songs per request, caches answers for
24 hours, and stops at 90 searches a day. If the key is missing, rejected or out of
quota, results still appear with search links.

**Optional: exact Spotify links.** Spotify's Web API has no single key: create an app in
the [Spotify developer dashboard](https://developer.spotify.com/dashboard) and add its
`SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` (both, or neither). The app uses the
client-credentials flow, so nobody has to log in to Spotify and the secret stays on the
server. Each recommendation then links to its exact track instead of a Spotify search,
under the same strict rule as YouTube: the title and an artist must both match, and live
takes, remixes, karaoke and covers are skipped (reissue labels such as "Remastered 2013"
are fine). An unsure song keeps its search link. Lookups cover at most five songs per
request, are cached for 24 hours, and pause after a rejected secret or a rate limit.

The command-line interface reads environment variables:

```bash
export TAVILY_API_KEY="..."
export GEMINI_API_KEY="..."
python -m src.main --query "instrumental piano for studying, no vocals" --enhanced
python -m src.main --query "Japanese math rock with guitar" --reference "a song title — artist"
python -m src.main                       # offline catalog ranking, no keys
python -m src.main --strategy mood-first
```

The command line reads only environment variables, not `.streamlit/secrets.toml`. In
Windows PowerShell, set them with `$env:TAVILY_API_KEY = "..."`.

**Troubleshooting.**

- **"Configure TAVILY_API_KEY and GEMINI_API_KEY privately on the server…" in the app.**
  The keys were not found. Check that `.streamlit/secrets.toml` exists in the folder
  you launched from, then restart the app (Streamlit reads secrets at startup).
- **Tests fail with errors such as `[] == ['song_card_1']`.** The wrong Streamlit
  version is installed. Run `python -c "import streamlit; print(streamlit.__version__)"`;
  it must print `1.67.0`. If it does not, activate `.venv` (`which python` should point
  inside it) and run `python -m pip install -r requirements.txt` again.
- **`python3.13: command not found`.** Install Python 3.13 (python.org or your package
  manager). On Windows, use `py -3.13`.
- **`ModuleNotFoundError: No module named 'src'`.** Run the commands from the
  repository root, not from inside `src/`.

**Provider availability.** The project uses Google's supported `google-genai` SDK.
Google's [pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite)
and [model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
listed free-tier text input and output for `gemini-3.5-flash-lite` on
November 11, 2026. Account eligibility, region and quota still apply. The app
never switches to a paid model or advanced Tavily search on its own.

## How retrieval changes the answer

1. Gemini first converts the request into source-linked taste criteria and one or
   two focused search queries. Mood, energy and activity fit stay weighted
   preferences unless the listener explicitly makes them mandatory. Explicit vocal
   phrases such as "clean singing", "instrumental" and "no screaming" are always hard
   criteria: if Gemini grades one as a soft vocals preference it is promoted, and if
   Gemini drops one entirely the analysis is rejected.
2. Tavily runs at most two basic searches and keeps excerpts that overlap the
   interpreted intent. It returns source excerpts, not a generated answer.
3. With enhanced retrieval on, the local index combines the existing lexical rank
   with Gemini text embeddings and still selects at most four song cards and two
   guide sections. Lexical specificity and autocut rules remain the primary safety
   gates. Semantic agreement can reorder lexically qualified cards, and up to two
   cards above cosine similarity 0.65 can fill unused slots; semantic-only cards do
   not displace a full four-card lexical result. Named references and requested
   languages retain priority, guides remain lexical, and any embedding failure
   returns the exact lexical result.
4. A second Gemini call extracts up to eight song candidates with source IDs,
   excluding albums, playlists and other non-song entities. A candidate must be
   named in its cited web or song-card evidence.
5. Enhanced retrieval looks up at most five candidates in MusicBrainz. Exact
   title and artist matches keep recording IDs, release details and version
   ambiguity. Metadata does not establish mood, vocal quality or musical fit.
6. A third Gemini call evaluates hard criteria and soft preferences for up to eight
   candidates. Python computes a weighted fit score (the share of weighted
   preferences that are supported, with partial matches counting half) and returns
   the five strongest supported or partial matches. The model is asked for its best
   candidates and may judge only some. If it judged some but none survived
   validation, one follow-up call asks about the candidates it skipped, using the
   request's single shared retry.
7. Validation rejects unknown sources, unsupported identity references, missing
   or paraphrased supporting excerpts, cross-song citations, unsafe URLs and
   unestablished hard-constraint checks. Duplicates are removed, and soft
   preferences never relax an explicit requirement or exclusion.

The full description and reference reach all three model calls, and retrieved text
is treated as untrusted data. Because the generator sees different evidence for
different requests, changing the evidence changes the recommendations; an empty
retrieval produces no recommendation rather than a fabricated one.

**Graceful degradation.** Recommendations are validated song by song: a song whose
quotes or checks fail is dropped and the rest are kept, and the rejected check is
logged. If every song fails, or a model stage otherwise returns invalid output, the
pipeline retries once (one retry is shared across the whole request). The retry of
the recommendation stage tells the model which checks failed so it can correct them.
If it still fails, the pipeline degrades instead of aborting:
retrieval falls back to the original wording, unverifiable candidates are dropped,
and recommendations fall back to songs that a source identifies (never when the
request has hard requirements). Those songs carry no fit score and the Streamlit
app labels them "NOT SCORED". Each one quotes the passage of its source that
names it, with links and markdown stripped, and says which requested traits that
passage or the page title mentions or does not mention. Songs whose source text
mentions more of the requested traits are listed first. This is word matching with
simple stemming and a few synonyms ("happy" also finds "upbeat"), so a mention is
not proof that the song fits. A warning explains what happened. When no supported
match remains, or discovery fails outright, the Streamlit app falls back in order to:
up to five songs ranked from the CSV catalog (labeled "CATALOG FALLBACK", each with a
match score, a "Why it fits" line and platform links); songs a source names but no
check could verify (only when the request had hard requirements, labeled "NOT
VERIFIED"); a related YouTube video from the retrieved sources; and finally a plain
"no evidence-backed matches" message. The later steps appear only if the catalog
cannot be read or returns nothing.

### Retrieval enhancements

| Addition | What it contributes |
| --- | --- |
| General music guide | Vocabulary for pace, energy, instrumentation, vocals, languages, references and exclusions; not evidence about particular songs. |
| 56 song cards | Source-backed descriptions spanning jazz, folk, country, rock, pop, K-pop, R&B, hip hop, house, techno, reggaeton, Afrobeats, amapiano, ambient, piano, samba, bossa nova, raï and several vocal languages. Thirty newer cards add sourced tempo, mood, dance, running and sports context where the linked article establishes it. |
| MusicBrainz | Structured recording identities, releases and disambiguation; missing records and ambiguous versions remain uncertain. |

Cards in `data/knowledge/song_cards.json` carry source URLs, retrieval dates,
recording versions and explicit unknowns. They are paraphrased summaries of
English Wikipedia pages (text available under CC BY-SA 4.0; attribution and the
source list are in [data/knowledge/SOURCES.md](data/knowledge/SOURCES.md)), not independent
corroboration. The retriever selects relevant
documents rather than sending the whole collection. A small, deterministic alias
list normalizes close vocabulary such as dreamy/ethereal, melancholic/mournful,
synth/synthesizer and EDM/electronic, plus a few Portuguese and English variants.
The optional semantic layer embeds card descriptions, versions, languages and tags,
but excludes their unknown-trait lists so missing evidence does not attract a card.
Every card the model sees can become a recommended candidate, so the
retriever prefers returning nothing to returning a loosely related card (the
thresholds are `GENERIC_TERM_SHARE` and `AUTOCUT_SHARE` in `src/knowledge.py`).
Tavily keeps discovering beyond the prepared cards in both modes.

No database is used. Live discoveries are not saved. Tavily responses and semantic
query embeddings use bounded 15-minute in-memory caches, document embeddings use a
24-hour content-addressed cache, and MusicBrainz lookups a one-hour cache; all
disappear when the process restarts, and Streamlit keeps displayed results only in
the current session.

## Sample interactions

These are real command outputs from this repository. Samples 1–4 cover the
offline catalog mode, the retrieval step that feeds the model and the guardrails
(no keys needed). Sample 5 shows live end-to-end runs with Tavily, MusicBrainz and
Gemini, including the runs that failed or degraded. Run the `--query` commands
above with your own keys to reproduce them; generated text varies from run to run.

### 1. Offline catalog ranking (original recommender)

Input: the "Ballad / Romantic" profile in Mood-First mode.

```bash
python -m src.main --strategy mood-first
```

```text
Loaded songs: 1000

==================================================
TOP MUSIC RECOMMENDATIONS — My Preference — Ballad / Romantic
Ranking mode: Mood-First
==================================================
+-----+--------------------------+--------------------+---------+--------------------------------------------------+
| #   | Song                     | Artist             | Score   | Reasons                                          |
+-----+--------------------------+--------------------+---------+--------------------------------------------------+
| 1   | All of My Life           | Park Won           | 4.40    | genre match (+1.0); mood match (+3.0); energy    |
|     |                          |                    |         | similarity (+0.40)                               |
| 2   | Scott and Zelda          | BIBI               | 3.50    | mood match (+3.0); energy similarity (+0.50)     |
| 3   | Just the Two of Us       | Grover Washington  | 3.50    | mood match (+3.0); energy similarity (+0.50)     |
|     |                          | Jr. & Bill Withers |         |                                                  |
| 4   | Tum Hi Ho                | Arijit Singh       | 3.50    | mood match (+3.0); energy similarity (+0.50)     |
| 5   | Satin Dawn               | After Hours        | 3.49    | mood match (+3.0); energy similarity (+0.49)     |
|     |                          | Collective         |         |                                                  |
+-----+--------------------------+--------------------+---------+--------------------------------------------------+
```

The command prints four profiles; the first is shown. Every score is explained by
rule, with no model involved.

### 2. Retrieval adds the evidence a request needs (improvement)

Input: `Calm instrumental piano for studying with no vocals`, enhanced mode.

The offline retrieval check makes no API calls: it runs the actual local retriever
on a saved web source and produces a JSON report. Below is its `piano` case,
reformatted, with the `reference_song`, `metadata_status` and `generation_status`
fields omitted:

```json
{
  "id": "piano",
  "description": "Calm instrumental piano for studying with no vocals",
  "baseline_source_ids": ["web_1"],
  "custom_document_source_ids": [
    "web_1", "card_merry-christmas", "card_miss-clare", "card_nuvole",
    "card_river-flows", "guide_27", "guide_30"
  ],
  "retrieved_cards": [
    "Merry Christmas Mr. Lawrence — Ryuichi Sakamoto",
    "Miss Clare Remembers — Enya",
    "Nuvole bianche — Ludovico Einaudi",
    "River Flows in You — Yiruma"
  ],
  "retrieved_guides": [
    "Instrumental piano and concentration",
    "Instrumentation and texture"
  ],
  "added_context_characters": 2193,
  "evidence_review": "Cards add explicit instrumental evidence for Sakamoto's Playing the Piano version and Enya's Miss Clare Remembers. Calmness and study benefit remain subjective."
}
```

The web-only baseline says only that Yiruma's track is on a pianist's album, which
does not prove there are no vocals. The retrieved cards add explicit instrumental
evidence for two recordings that the generator can then cite.

### 3. Retrieval that does not help (honest failure)

Input: `Heavy guitar music with clean singing and no screaming`, enhanced mode.
This is the offline `clean-metal` case, with the same reformatting and with the
baseline source fields also omitted.

```json
{
  "id": "clean-metal",
  "description": "Heavy guitar music with clean singing and no screaming",
  "retrieved_cards": [
    "Du hast — Rammstein",
    "Smells Like Teen Spirit — Nirvana"
  ],
  "retrieved_guides": [
    "Vocal delivery and emotion",
    "Instrumentation and texture"
  ],
  "added_context_characters": 1558,
  "evidence_review": "The guide warns that genre labels do not establish absence of screaming. Guitar-related Rammstein and Nirvana appear; the unrelated bossa nova/house cards that generic vocal overlap used to add are no longer retrieved. No selected source establishes the requested clean-singing exclusion."
}
```

No retrieved source establishes the exclusion, so the validators would reject any
recommendation that claimed it. The system returns no qualifying match rather than
relaxing the requirement. This case used to retrieve two more cards, Águas de Março
and Around the World, matched only by the word "vocal", plus the piano guide, matched
by "no". The retrieval filter described above removed them; the missing evidence about
clean singing is a gap retrieval cannot close.

### 4. Input and configuration guardrails

```bash
python -m src.main --query "   "
```

```text
Describe a genre, mood, activity, instrument, or sound to search for.
```

```bash
python -m src.main --query "instrumental piano for studying, no vocals" --enhanced
```

```text
Configure TAVILY_API_KEY and GEMINI_API_KEY privately on the server before using discovery.
```

Both commands exit with status 1 and make no provider calls; the message never
includes a key or a traceback.

### 5. Live end-to-end runs (Tavily, MusicBrainz, Gemini)

Recorded on November 17, 2026 with `gemini-3.5-flash-lite` and the default
operating limits. Eight CLI attempts were made and all eight are listed, including
the failures. The first six were run once each; attempts 7 and 8 retried the two
that ended in a Gemini error.

| # | Request | Mode | Outcome |
| --- | --- | --- | --- |
| 1 | instrumental piano for studying, no vocals | enhanced | 5 songs named in one retrieved page; fit not confirmed (intent step degraded) |
| 2 | Japanese math rock with guitar, reference song | standard | 0 results: no candidate had evidence for every required trait |
| 3 | upbeat Korean pop for a morning run | enhanced | 0 results: one candidate rejected, a required trait was not established |
| 4 | mellow lo-fi hip hop for relaxing | enhanced | Gemini unavailable, no results |
| 5 | classic bossa nova for a dinner party | enhanced | 1 fully validated result |
| 6 | high-energy rock for a workout | standard | Gemini unavailable, no results |
| 7 | retry of 4 | enhanced | Gemini unavailable after one retry |
| 8 | retry of 6 | standard | 5 songs named in one retrieved page; fit not confirmed |

These runs predate the current fallback listing: attempts 1 and 8 were recorded
before each fallback entry started quoting its source and the app began labeling
such songs "NOT SCORED", so the wording in their output below is as recorded.
Only attempt 5 passed every validator. Two attempts (1, 8) returned songs whose
existence the source supports but whose musical fit the validator could not confirm,
and the output says so. Two (2, 3) returned nothing rather than relax a required
trait. Three (4, 6, 7) hit a Gemini error that the app reports as "Gemini is
unavailable." Those logs record only the safe code `provider_unavailable`, so I
cannot say what caused them. A later live run, after the log began recording the
HTTP status, showed the same message for a Google 503 `UNAVAILABLE` response, so a
temporary overload is the likeliest cause, though not proven for these three. A
quota limit would have been reported as a rate limit. Attempt 2 used
`Goodbye — toe` as the reference song; I did not verify that it exists.

**Attempt 5: a validated result that used the local song cards**

```bash
python -m src.main --query "classic bossa nova for a dinner party" --enhanced
```

```text
1. Águas de Março — Elis Regina and Antonio Carlos Jobim
Aguas de Março is a Portuguese bossa nova song by Jobim, recorded as a duet with Elis Regina for Elis & Tom.
Uncertainty: exact BPM
Uncertainty: measured energy
Uncertainty: listener-specific mood or activity fit
Uncertainty: absence of screaming or other unmentioned vocal sounds
Source [card_aguas]: https://en.wikipedia.org/wiki/Waters_of_March
YouTube search: https://www.youtube.com/results?search_query=%C3%81guas%20de%20Mar%C3%A7o%20Elis%20Regina%20and%20Antonio%20Carlos%20Jobim
Spotify search: https://open.spotify.com/search/%C3%81guas%20de%20Mar%C3%A7o%20Elis%20Regina%20and%20Antonio%20Carlos%20Jobim
Notice: MusicBrainz did not match every checked candidate; missing metadata does not mean a song is nonexistent.
```

```text
INFO:src.providers:generation provider=Gemini call=1 latency_ms=1706
INFO:src.providers:retrieval provider=Tavily cache_hit=False sources=6 latency_ms=1538
INFO:src.providers:retrieval provider=Tavily cache_hit=False sources=12 latency_ms=1849
INFO:src.providers:generation provider=Gemini call=2 latency_ms=6519
INFO:src.providers:retrieval provider=MusicBrainz cache_hit=False matches=0 latency_ms=729
INFO:src.providers:generation provider=Gemini call=3 latency_ms=7770
INFO:src.discovery:discovery_complete enhanced=True criteria=3 queries=2 evidence=15 candidates=1 recommendations=1 partial=False latency_ms=20123
```

The cited source `card_aguas` was one of the 26 local song cards at the time of
that recorded run (the collection now has 56), so enhanced
retrieval changed what the model could cite. Gemini extracted one candidate and it
passed. The four uncertainties are the traits the evidence does not establish,
such as exact BPM.

**Attempt 8: songs found, fit unconfirmed** (first two of five entries shown; the
other three have the same format)

```bash
python -m src.main --query "high-energy rock for a workout"
```

```text
1. Sleep Now In the Fire — Rage Against The Machine
This song and artist are identified in the retrieved source; its fit with your request still needs verification.
Uncertainty: The generated musical-fit details could not be validated against the retrieved source.
Source [web_3]: https://open.spotify.com/playlist/3RMYoItrRWT0Qa9y6TdbfY
YouTube search: https://www.youtube.com/results?search_query=Sleep%20Now%20In%20the%20Fire%20Rage%20Against%20The%20Machine
Spotify search: https://open.spotify.com/search/Sleep%20Now%20In%20the%20Fire%20Rage%20Against%20The%20Machine

2. Wait and Bleed — Slipknot
This song and artist are identified in the retrieved source; its fit with your request still needs verification.
Uncertainty: The generated musical-fit details could not be validated against the retrieved source.
Source [web_3]: https://open.spotify.com/playlist/3RMYoItrRWT0Qa9y6TdbfY
YouTube search: https://www.youtube.com/results?search_query=Wait%20and%20Bleed%20Slipknot
Spotify search: https://open.spotify.com/search/Wait%20and%20Bleed%20Slipknot

Notice: The generated musical-fit details could not be fully validated; source-identified songs are shown with their fit marked as uncertain.
```

```text
INFO:src.discovery:generation_retry stage=recommendations reason=invalid_response
WARNING:src.discovery:generation_degraded stage=recommendations reason=invalid_response
INFO:src.discovery:discovery_complete enhanced=False criteria=3 queries=2 evidence=7 candidates=8 recommendations=5 partial=False latency_ms=15756
```

Gemini's fit explanations failed validation twice, so the pipeline fell back to
songs that the retrieved page names, with no claim about how well they fit. These
five entries are five songs from one Spotify playlist page; nothing here confirms
that they suit a workout. The fallback has since changed: each entry now quotes the
page's own words about the song and says which requested traits they mention, its
uncertainty reads "We couldn't check this song's fit against the source.", and the
notice reads "We couldn't verify how well these songs fit your request, so they are
listed without a match score." This request was not re-run live after that change;
the new wording is covered by `tests/test_rag.py`.

**Attempt 3: nothing shown rather than an unsupported answer**

```bash
python -m src.main --query "upbeat Korean pop for a morning run" --enhanced
```

```text
Notice: MusicBrainz is unavailable; recording versions remain uncertain.
Notice: No candidate had evidence supporting every required or excluded trait.
```

```text
WARNING:src.providers:provider_failure provider=MusicBrainz code=provider_unavailable
INFO:src.validation:recommendation_rejected reason=hard_criterion_unestablished
INFO:src.discovery:discovery_complete enhanced=True criteria=4 queries=2 evidence=17 candidates=8 recommendations=0 partial=False latency_ms=20144
```

Eight candidates were extracted and none became a recommendation; the log records
one rejection because a required trait was not established. That is the designed
behavior, but for a mainstream request it also shows that the validators are strict
enough to return nothing. The log also shows only one rejection for eight candidates,
which suggests Gemini returned a verdict for one song and the other seven were never
judged. The pipeline now spends the request's shared retry on a follow-up call about
the skipped candidates when none survived. This was not re-run live, so whether it
turns this request into results is unmeasured; the behavior is covered by
`tests/test_rag.py`.

**What these runs show.** The pipeline runs end to end against live services, and
the guardrails fire as designed: no unsupported song is presented as a confirmed fit.
They also show a limit: on one run in eight the model's output passed every check, and
the system leans on "source names the song" when it cannot confirm fit. Treat the
live answer quality as unproven; the hand-reviewed comparison in the Testing summary
below is the stronger evidence for the retrieval step. The log blocks for attempts 8
and 3 are excerpts of the lines that matter; attempt 5's is complete.

## Screenshots

These are the Streamlit app after live runs on November 17, 2026 (Tavily,
MusicBrainz and Gemini). They show output that validated; the failure cases are
the pasted text in sample 5, which is the evidence that gets graded. Live results
vary from run to run, so a rerun can return different songs.

![The Find music tab after "classic bossa nova for a dinner party": five recommendation cards with match scores, a note on how many details could not be confirmed, and YouTube and Spotify links](assets/01-find-music-bossa-nova-results.png)

*Find music: "classic bossa nova for a dinner party" returned five songs. Each card
shows a match score and says how many details the sources could not confirm.*

![The "Why these songs?" panel, listing for each song the reason it may fit, what could not be confirmed, a quoted excerpt from the source and a link to that source](assets/02-why-these-songs-evidence.png)

*The "Why these songs?" panel is the human check: for each song it shows what the
evidence supports, what it does not establish, the quoted excerpt and the source.*

![The Find music tab after "upbeat Korean pop for a morning run": a "must-haves" line reading Korean pop and five K-pop recommendation cards](assets/03-find-music-korean-pop-must-haves.png)

*A request with a hard requirement ("Korean pop") is shown under "Your must-haves"
and every returned song was checked against it.*

![The Music Collection panel ranking the offline catalog for jazz and relaxed, with five catalog cards showing match scores and why each fits](assets/04-music-collection-jazz-relaxed.png)

*Music Collection: ranking the offline catalog for jazz and relaxed. This path needs
no API keys.*

## Testing summary

```bash
python -m pytest -q
```

```text
........................................................................ [ 12%]
........................................................................ [ 24%]
........................................................................ [ 36%]
........................................................................ [ 48%]
........................................................................ [ 60%]
........................................................................ [ 72%]
........................................................................ [ 84%]
........................................................................ [ 97%]
.................                                                        [100%]
593 passed in 6.63s
```

| Test file | Tests | What it covers |
| --- | --- | --- |
| `tests/test_rag.py` | 92 | The full discovery pipeline: evidence rules, retries, fallbacks, provider wiring |
| `tests/test_validation.py` | 89 | Validators: request and intent limits, citation and quote checks, scoring, model feedback |
| `tests/test_utils.py` | 36 | Text normalization and the safe-URL rules |
| `tests/test_models.py` | 5 | Shared types and their defaults |
| `tests/test_providers.py` | 147 | Tavily, focused-source search, Gemini generation/embeddings and MusicBrainz: limits, caching, safe errors, malformed output |
| `tests/test_knowledge.py` | 54 | Hybrid local retrieval: lexical safety, semantic fusion/fallback, aliases, relevance benchmark, document loading |
| `tests/test_song_cards.py` | 4 | Expanded card count, source metadata, coverage groups and visible mood/tempo/activity evidence |
| `tests/test_app.py` | 90 | Streamlit pages, configuration, fallbacks, escaping, throttling, catalog mode |
| `tests/test_main.py` | 15 | Command-line output and failure exits |
| `tests/test_recommender.py` | 29 | CSV ranking, strategies, diversity penalties, genre resolution |
| `tests/test_youtube.py` | 32 | Strict video matching, quota limits, safe fallback |

The guardrail checks can be run on their own:

```bash
python -m pytest -v tests/test_rag.py -k "uses_retrieved_evidence or empty_input or no_evidence or unknown_source or unknown_hard_constraint or cross_song or unsafe_source_urls or identity_hallucination or fabricated_quote"
```

```text
tests/test_rag.py::test_discovery_uses_retrieved_evidence PASSED
tests/test_rag.py::test_empty_input_skips_providers PASSED
tests/test_rag.py::test_no_evidence_skips_generation PASSED
tests/test_rag.py::test_unknown_source_is_rejected PASSED
tests/test_rag.py::test_identity_hallucination_is_not_recommended PASSED
tests/test_rag.py::test_unrelated_citation_and_fabricated_quote_rejected PASSED
tests/test_rag.py::test_unknown_hard_constraint_rejects_candidate_without_relaxing_it PASSED
tests/test_rag.py::test_cross_song_citation_cannot_support_traits PASSED
tests/test_rag.py::test_unsafe_source_urls_are_rejected[javascript:alert(1)] PASSED
tests/test_rag.py::test_unsafe_source_urls_are_rejected[https://user:password@example.org] PASSED
tests/test_rag.py::test_unsafe_source_urls_are_rejected[http://127.0.0.1/a] PASSED
tests/test_rag.py::test_unsafe_source_urls_are_rejected[http://localhost] PASSED
tests/test_rag.py::test_unsafe_source_urls_are_rejected[https://example.org/ bad] PASSED
13 passed, 71 deselected
```

### Evaluation results

Retrieval comparison on six varied requests, each with a written evidence review
(full table and provenance in the
[model card](model_card.md#actual-beforeafter-retrieval-experiment)):

| Test input | Evaluation criteria | Result |
| --- | --- | --- |
| Slow-paced music with emotionally expressive vocals | Retrieves relevant vocal cards | Fail: the Teardrop card is still missed (unrelated cards are no longer returned) |
| Fast upbeat electronic music for running | Adds relevant genre context | Partial: adds electronic context, but ambient Eno is a weak activity match |
| Calm instrumental piano for studying with no vocals | Adds explicit instrumental evidence | Pass: Sakamoto and Enya cards state it |
| Energetic Portuguese-language music with percussion | Ranks Portuguese-language cards first | Partial: language improves, percussion is not established |
| Heavy guitar music with clean singing and no screaming | Establishes the vocal exclusion | Fail: no source establishes it (unrelated cards are no longer returned) |
| Reference track plus modal jazz, piano and bass | Adds the requested instrumentation evidence | Pass: adds the piano/bass introduction and pace distinction |

The same six requests now form a deterministic offline regression fixture with
fixed query expansions and human-labeled relevant card IDs. It retrieves all 15
relevant cards among 15 returned cards: micro precision **1.00** and recall
**1.00**. This measures local card retrieval only; it does not call providers or
claim that the generated recommendations are good.

Guardrail behavior, from the automated suite:

| Test input | Evaluation criteria | Result |
| --- | --- | --- |
| Blank description | No provider calls, helpful message | Pass |
| Missing API keys | Safe error, no secret or traceback | Pass |
| Citation of an unknown source ID | Rejected as invalid | Pass |
| Quote that is not in the cited source | Rejected | Pass |
| Trait cited from a different song's source | Rejected | Pass |
| Hard requirement with no supporting evidence | Candidate dropped, requirement not relaxed | Pass |
| `javascript:`, credential or localhost source URLs | Rejected | Pass |
| Malformed model output | Retried once, then bounded | Pass |
| One invalid song among valid ones | Only that song is dropped; the rest are kept | Pass |
| Failed validation after the retry | Source-identified songs listed as unscored; none when the request has hard requirements | Pass |
| MusicBrainz requests across sessions | At least 1.05 s apart, cached repeats skip the request | Pass (injected clock) |

**Result.** 593 of 593 automated tests pass, and the retrieval comparison shows a
clear gain on 2 of 6 requests, a partial gain on 2 and no gain on 2.

**What worked.** Requiring exact excerpts for every claim made unsupported answers
fail loudly instead of looking plausible. Keeping scoring in Python made fit scores
deterministic and explainable. Injected provider dependencies let the whole pipeline
be tested without network calls.

**What did not.** Lexical retrieval alone misses paraphrases, and until the
specificity filter and autocut were added, generic words such as "vocal" pulled in
unrelated cards. The hybrid layer now has deterministic fusion and fallback tests,
but its live relevance improvement has not been measured. All-mock tests also missed a live problem: real
Gemini output sometimes failed the strict validators and the whole request ended in
a generic "could not be supported" error. The pipeline now allows one retry and
then degrades gracefully, as described above, and the app offers clearer
fallbacks.

**What I learned.** Validation can confirm that a quote exists and that a source
names a song; it cannot confirm that a paraphrase is fair or that a listener will
enjoy the result. The automated tests establish the guardrails, not live answer
quality. A saved side-by-side live comparison of generated answers is not included,
and its semantic review would stay a human task.

### Operating limits

Each discovery makes at most two basic Tavily searches and normally three Gemini
calls (four with one retry, which the follow-up check on skipped candidates also
uses); a Gemini 503 (overloaded) response is retried once after a two-second pause
and uses one of those four calls, and other provider and network failures are not
retried. Enhanced mode also makes one semantic-query embedding request; the first
request after startup or a card change makes one additional batched document
embedding request. Embedding failures are not retried and fall back to lexical
retrieval. HTTP
requests use 5-second connect and 20-second read timeouts and a 1 MB response cap;
Gemini has a 30-second timeout and bounded JSON output. Descriptions are capped at
1,200 characters and references at 200. The Streamlit app adds a 15-second session
cooldown. Logs record counts, cache hits, latency and safe status codes, never
prompts or keys. MusicBrainz starts are at least 1.05 seconds apart within one
application process; multiple workers would need a shared rate limiter.

## Design decisions and trade-offs

| Decision | Why | Trade-off |
| --- | --- | --- |
| Fixed, bounded pipeline instead of an autonomous agent | Predictable cost and behavior, and every stage can be tested with controlled inputs | No second attempt with a different strategy when a search misses |
| Three evidence sources: Tavily, local documents, MusicBrainz | Each does a different job: discovery, domain vocabulary and curated cards, recording identity | More providers to fail, quota and a one-request-per-second MusicBrainz limit |
| In-memory hybrid lexical/semantic index instead of a vector database | Lexical safety gates remain inspectable while embeddings can recover paraphrases; no persistent service is needed for the small collection | Adds one embedding request per query, needs a similarity threshold, and resets its cache on restart |
| Gemini extracts and judges, Python validates and scores | The model cannot award itself a good score, and results are reproducible | Strict exact-quote checks reject some fair paraphrases, which is why retry and degradation exist |
| Mood, energy and activity are weighted preferences; only explicit requirements and exclusions are hard | "Happy, peaceful pop" should return partial matches, not nothing | Partial matches are less precise and are labeled as such |
| Retrieved text is data, never instructions | Web pages can contain prompt injection | Cannot execute anything a source says, by design |
| Server-side keys, no viewer input of credentials | Keys never reach the browser | The host must supply and protect the keys |
| In-memory caches, no database | Nothing about listeners is stored | Caches reset on restart; the rate limiter works in one process only |
| Keep the CSV recommender as an offline mode | A reproducible baseline and a fallback that needs no keys | Its labels are unverified and its coverage is the catalog |

## Responsible use

The [model card](model_card.md) documents limitations and biases, how the system
could be misused and how that is limited, what surprised me while testing
reliability, and how I worked with AI on this project.

## Reflection

The most useful lesson was that a trustworthy AI system is mostly the code around
the model. Retrieval, exact-quote checks, bounded calls and honest fallbacks decide
whether an answer can be trusted, and they only help if they are tested against the
cases where they fail. This project shows how I build AI features: scope the workflow, make every claim
traceable to a source, test the guardrails and report what the results do and do
not prove, including the cases where retrieval failed.

## Repository layout

```text
src/            app.py (Streamlit), main.py (CLI), composition.py (provider
                wiring and find_music), discovery.py (shared workflow),
                validation.py (evidence checks), models.py (shared types),
                utils.py (text and URL helpers), providers.py (Tavily, Gemini,
                MusicBrainz, YouTube), knowledge.py (local index), recommender.py (CSV
                ranking)
data/           songs.csv (1,000-song catalog), knowledge/ (music guide, song cards,
                SOURCES.md with attribution)
diagrams/       architecture.mmd (system diagram: components, data flow, checks)
tests/          pytest suite; fixtures/ holds the intent cases
.streamlit/     config.toml (theme); secrets.toml holds your keys and is
                ignored by Git
requirements.txt  pinned dependencies
model_card.md   limitations, misuse, reliability findings, AI collaboration
ai_interactions.md  log of AI-assisted work on the original recommender
```

## Original catalog ranking mode

Core scoring uses genre, mood and energy; optional preferences can match subgenre,
mood tags, lyrical theme, instrumentation and language. Greedy selection applies
artist and repeated non-preferred-genre penalties, and genre is a scoring bonus,
not a hard filter. Lines in `data/songs.csv` that start with `#` are skipped when
it is loaded. The catalog labels are descriptive metadata that has not been
independently verified, so the entries should not be presented as verified
recordings. The Streamlit app exposes this mode as the **Music Collection** panel
with a genre, mood, energy and strategy picker. A typed genre is resolved to the
catalog labels it names: an exact label, a short alias list ("baile funk" finds
`funk carioca`), or a song whose genre, subgenre and language together contain every
word ("brazilian funk" finds `funk carioca` and `funk melody`). It returns nothing
rather than guessing, and the panel says so when a genre or mood matches nothing
(suggesting related genres for a genre). Cards show a match percentage, the share of the best score
attainable for the preferences given, before diversity penalties. The historical weight-shift
experiment and personal reflection are kept in `model_card.md`.
