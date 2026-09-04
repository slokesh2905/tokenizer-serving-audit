# Part A1 — Eval corpus: source, construction, and caveats

## Languages

English, Hindi, Kannada, Telugu — 4 languages, including English, Hindi, and two Dravidian
languages as required.

**Deviation from the plan, flagged explicitly:** the original plan (agreed with the user
before building this) was Kannada + Tamil as the two Dravidian languages. Tamil is not
present in the corpus source used here (see below) — checked directly against the full
108-language file listing of `christos-c/bible-corpus`, confirmed absent (would sort
between "Tagalog.xml" and "Telugu.xml"; nothing there). Telugu was available and is
substituted. This is a data-availability decision, not a preference — flagged here and in
`NOTEBOOK.md` (Day 3) rather than made silently. If a Tamil-inclusive parallel source is
found later, the corpus can be rebuilt from a different source with the same pipeline.

## Source and why

FLORES-200's usual hosts (Hugging Face, dl.fbaipublicfiles.com) are blocked by this
environment's network egress policy (see `NOTEBOOK.md`, Day 1). `openlanguagedata/flores`
on GitHub — the actively maintained successor — no longer hosts the data in-repo; it
redirects to a Hugging Face dataset, also blocked. Checked directly, not assumed.

Used instead: **`christos-c/bible-corpus`** (GitHub, github.com/christos-c/bible-corpus),
a multilingual parallel corpus of Bible translations, verse-aligned by book/chapter/verse
ID, encoded as Corpus Encoding Standard XML. Citable source: Christodoulopoulos & Steedman,
*"A massively parallel corpus: the Bible in 100 languages"*, Language Resources and
Evaluation 49(2), 2015. Each language file is licensed for research/NLP use (this is a
long-standing, widely used resource in low-resource-language NLP for exactly this reason:
it's one of the few texts translated into hundreds of languages with verse-level
alignment).

## Construction

1. Downloaded the first ~700KB of each of `English.xml`, `Hindi.xml`, `Kannada.xml`,
   `Telugu.xml` via HTTP range request (full files are 6-12MB each; this environment's
   bandwidth to raw.githubusercontent.com is slow — ~50KB/s measured directly — so a
   partial download was used rather than the full text; see `raw_bible/` for the partial
   XML files actually used).
2. Parsed `<seg id="b.BOOK.CHAPTER.VERSE" type="verse">text</seg>` entries per language
   with a regex over the XML (code: `build_corpus.py`), collapsing internal whitespace.
3. Took the intersection of verse IDs present in **all four** languages within the partial
   downloads, restricted to the Book of Genesis specifically (chosen because it's complete
   — all 50 chapters, all verses — in every language's partial download, giving a clean,
   principled boundary rather than an arbitrary truncation point).
4. Wrote line-aligned `eval_corpus/{eng,hin,kan,tel}.txt` (line N is the same verse in
   every file) plus `eval_corpus/manifest.tsv` mapping each line number to its verse ID,
   for full traceability back to the source.

## Size

**1,532 parallel sentences (verses) per language**, all of Genesis, aligned 1:1 across all
four languages. This is ~150x the size of the toy sample corpus used for A2 (10 lines).

| lang | lines | bytes |
|---|---|---|
| eng | 1532 | 196,729 |
| hin | 1532 | 454,965 |
| kan | 1532 | 502,226 |
| tel | 1532 | 467,298 |

## Domain and preprocessing

Domain: narrative religious text (Genesis — creation, patriarchal narratives, genealogies),
translated by different translation teams/eras per language, so register and verbosity
conventions are not perfectly matched across languages even though content is. No
code-switching, no numbers in modern formats (dates, currency), no URLs, no emoji, no
conversational register — this is nothing like production chat traffic.

Preprocessing applied: internal whitespace collapsed to single spaces (source XML has
line-wrapped text with irregular indentation inside `<seg>` tags); no case-folding, no
Unicode normalization applied at corpus-build time (kept as extracted — any normalization
is applied downstream by the specific experiment, as in A2, so its effect stays visible and
attributable rather than baked in silently here).

## What this corpus cannot tell you

This is 19th/20th-century-style Bible translation, not representative of the register a
production assistant actually serves (casual conversational text, code-switched queries,
short chat messages, product/technical vocabulary) — a routing/cost decision built purely
on this corpus risks generalizing fertility numbers from formal narrative prose to a
completely different register where tokenizer behavior can differ substantially (short
informal sentences, transliterated/Romanized Indic text mixed with English, emoji, numbers,
proper nouns). It is also a single, narrow domain (one book of one genre), translated by
different teams per language with their own conventions for sentence length and
literalness — some of the cross-language length variation measured here is translator
style, not an inherent property of the language. And because it's translation (not
independently authored text in each language), it does not capture how each language is
actually *used* natively (e.g. code-mixed Hinglish, colloquial contractions, SMS-style
abbreviations) which is exactly the traffic a production router would see. Treat the
numbers from this corpus as a controlled, reproducible signal about tokenizer/metric
behavior — not as a direct estimate of production serving cost. A4's recommendation memo
should carry this caveat forward explicitly.
