# Knowledge sources and attribution

This folder is the local evidence set that enhanced retrieval searches in addition
to the web. It holds two files:

- `song_cards.json`: 26 short song cards, each a paraphrase of one English
  Wikipedia article.
- `music_guide.md`: a vocabulary guide written for this project. It explains
  how people describe pace, vocals, instruments and similar traits. It is not derived
  from any outside source, and it is not evidence that any recording has a trait.

The cards are not audio analysis. Each card lists the facts its source does not
establish (exact BPM, measured energy, vocal language and similar) under `unknowns`,
and the system treats those as unconfirmed.

## License and attribution

The card descriptions are adapted from articles written by Wikipedia contributors.
Wikipedia text is available under the
[Creative Commons Attribution-ShareAlike 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0); see Wikipedia's
[reuse guidance](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content).
The cards are shortened paraphrases, so they are changed from the originals. The
article linked in each row is the source and lists its authors in its page history.
Any reuse of the card text should keep this attribution and the same license.

A Wikipedia summary is a single source and is not independent corroboration of a
claim; each card's `provenance` field says so.

## Card sources

All 26 cards were prepared from the pages below and retrieved on the date shown.

| Card id | Title | Artist | Source | Retrieved |
| --- | --- | --- | --- | --- |
| `take-five` | Take Five | Dave Brubeck Quartet | <https://en.wikipedia.org/wiki/Take_Five> | 2026-11-11 |
| `so-what` | So What | Miles Davis | <https://en.wikipedia.org/wiki/So_What_(Miles_Davis_composition)> | 2026-11-11 |
| `around-the-world` | Around the World | Daft Punk | <https://en.wikipedia.org/wiki/Around_the_World_(Daft_Punk_song)> | 2026-11-11 |
| `born-slippy` | Born Slippy .NUXX | Underworld | <https://en.wikipedia.org/wiki/Born_Slippy_.NUXX> | 2026-11-11 |
| `fast-car` | Fast Car | Tracy Chapman | <https://en.wikipedia.org/wiki/Fast_Car> | 2026-11-11 |
| `respect` | Respect | Aretha Franklin | <https://en.wikipedia.org/wiki/Respect_(song)> | 2026-11-11 |
| `jolene` | Jolene | Dolly Parton | <https://en.wikipedia.org/wiki/Jolene_(song)> | 2026-11-11 |
| `nothing-else-matters` | Nothing Else Matters | Metallica | <https://en.wikipedia.org/wiki/Nothing_Else_Matters> | 2026-11-11 |
| `aguas` | Águas de Março | Elis Regina and Antonio Carlos Jobim | <https://en.wikipedia.org/wiki/Waters_of_March> | 2026-11-11 |
| `mas-que-nada` | Mas que nada | Jorge Ben | <https://en.wikipedia.org/wiki/Mas_que_Nada> | 2026-11-11 |
| `despacito` | Despacito | Luis Fonsi featuring Daddy Yankee | <https://en.wikipedia.org/wiki/Despacito> | 2026-11-11 |
| `chan-chan` | Chan Chan | Buena Vista Social Club | <https://en.wikipedia.org/wiki/Chan_Chan_(song)> | 2026-11-11 |
| `sodade` | Sodade | Cesária Évora | <https://en.wikipedia.org/wiki/Sodade> | 2026-11-11 |
| `didi` | Didi | Khaled | <https://en.wikipedia.org/wiki/Didi_(song)> | 2026-11-11 |
| `gee` | Gee | Girls' Generation | <https://en.wikipedia.org/wiki/Gee_(Girls%27_Generation_song)> | 2026-11-11 |
| `sukiyaki` | Sukiyaki | Kyu Sakamoto | <https://en.wikipedia.org/wiki/Sukiyaki_(song)> | 2026-11-11 |
| `river-flows` | River Flows in You | Yiruma | <https://en.wikipedia.org/wiki/River_Flows_in_You> | 2026-11-11 |
| `nuvole` | Nuvole bianche | Ludovico Einaudi | <https://en.wikipedia.org/wiki/Una_Mattina> | 2026-11-11 |
| `ascent` | An Ending (Ascent) | Brian Eno | <https://en.wikipedia.org/wiki/An_Ending_(Ascent)> | 2026-11-11 |
| `paper-planes` | Paper Planes | M.I.A. | <https://en.wikipedia.org/wiki/Paper_Planes_(M.I.A._song)> | 2026-11-11 |
| `teen-spirit` | Smells Like Teen Spirit | Nirvana | <https://en.wikipedia.org/wiki/Smells_Like_Teen_Spirit> | 2026-11-11 |
| `du-hast` | Du hast | Rammstein | <https://en.wikipedia.org/wiki/Du_hast> | 2026-11-11 |
| `superstition` | Superstition | Stevie Wonder | <https://en.wikipedia.org/wiki/Superstition_(song)> | 2026-11-11 |
| `teardrop` | Teardrop | Massive Attack | <https://en.wikipedia.org/wiki/Teardrop_(song)> | 2026-11-11 |
| `merry-christmas` | Merry Christmas Mr. Lawrence | Ryuichi Sakamoto | <https://en.wikipedia.org/wiki/Playing_the_Piano> | 2026-11-11 |
| `miss-clare` | Miss Clare Remembers | Enya | <https://en.wikipedia.org/wiki/Watermark_(Enya_album)> | 2026-11-11 |

## Adding or changing a card

The application never writes these files. When you add a card, give it a source URL,
a retrieval date, a `provenance` note and an `unknowns` list in `song_cards.json`,
and add a matching row to the table above.
