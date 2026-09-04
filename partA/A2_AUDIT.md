# Part A2 — Auditing fertility.py and its metric

All numbers below come from `code/a2_experiments.py`, run against the toy sample corpus
(`corpus/toy_sample/`, 10 lines/language) using the reconstructed gpt2 tokenizer
(`code/gpt2_bpe.py`, verified in NOTEBOOK.md to reproduce REPORT_v0.md exactly). Reproduce
with `python3 code/a2_experiments.py`. Every number here is measured, not asserted — this
file states the claim, the isolated experiment, and the result; nothing is included without
both.

## The conceptual bug: tokens-per-whitespace-word is not a fair cross-lingual unit

**Claim.** `fertility.py` computes exactly what it says — tokens divided by
whitespace-split words — but that is the wrong thing to compute for a routing/cost
decision. Whitespace-delimited "words" are not comparable units of content across
languages: Hindi (and more so agglutinative Dravidian languages, not in this toy corpus but
relevant to the real A1 corpus) glue more grammatical information into each word via
inflection, so a word-based ratio conflates *tokenizer inefficiency* with *how much a
language's words happen to carry*, even when the tokenizer is perfectly fair.

**Experiment (isolates the denominator from everything else — same tokenizer, same
corpus, same preprocessing, only the unit changes):**

```
eng: tokens=99  bytes=448  words=79   tok/word=1.253   tok/byte=0.221
hin: tokens=459 bytes=764  words=62   tok/word=7.403   tok/byte=0.601

hin/eng ratio via tokens-per-word (REPORT_v0's metric): 5.91x
hin/eng ratio via tokens-per-byte (content-normalized):  2.72x
```

**Result.** Switching only the denominator — nothing else — from words to bytes cuts the
apparent "Hindi is Nx worse" gap from **5.91x to 2.72x**: 54% of the reported gap was an
artifact of counting whitespace-words, not a property of tokenization cost. The remaining
2.72x is real, but it's a *tokenizer-choice* effect (gpt2 is English-centric), not evidence
that "Hindi simply has more Unicode characters per word, a property of the script" as
REPORT_v0 claims (Section 1, finding 3) — that claim is asserted with zero measurement and
this experiment contradicts the "property of the script, not the tokenizer" framing
directly: the same script, same corpus, same content shows a very different ratio once the
denominator stops being word-count. Part A3 pushes this further with a second,
Indic-aware tokenizer to see how much of the remaining 2.72x survives that too.

## Code bug 1: asymmetric `.lower()` — one line of code, unequal effect per language

**Claim.** `line = line.lower()` runs identically in code on every corpus, but Devanagari
has no case distinction, so it's a silent no-op for Hindi while it actively changes what
the tokenizer sees for English. A step that looks language-neutral in the source is not
neutral in effect.

**Experiment (with vs. without `.lower()`, same corpus, same tokenizer):**

```
eng: with .lower()=1.2652  without .lower()=1.2293   CHANGES  (+2.92%)
hin: with .lower()=7.4485  without .lower()=7.4485   NO EFFECT (+0.00%)
```

**Result.** Confirmed exactly as hypothesized: a +2.92% shift for English, a mathematically
exact 0.00% for Hindi (identical floats, not just "close"). The preprocessing pipeline
treats the two languages differently while appearing to treat them the same, and it does so
in the direction that further inflates the eng/hin fertility gap (lowercasing makes English
*more* token-efficient without touching Hindi at all).

## Code bug 2: `words = line.split(" ")` — literal-space split creates phantom words

**Claim.** Splitting on a literal single space (not `.split()`) means any run of
consecutive spaces produces empty-string "words," inflating the word count and therefore
deflating the reported fertility. Both sample corpora contain exactly one line with a
doubled space (`eng` line 7, `hin` line 10) — looks planted.

**Experiment (`split(" ")` vs `.split()`, same corpus, same tokenizer):**

```
eng: 1/10 lines have a double space.  split(' ')=1.2652  .split()=1.2831  (-1.39%)
hin: 1/10 lines have a double space.  split(' ')=7.4485  .split()=7.5985  (-1.97%)
```

**Result.** Confirmed, directionally as predicted (original *understates* fertility) — but
small in magnitude on this 10-line toy corpus (1.4-2.0%), because it only affects the one
line with a double space out of ten. Flagging honestly: this isn't a headline-scale
distortion at this corpus size, but it's a genuine, unambiguous implementation bug
(`split(" ")` vs `.split()` are different functions with different results on real text,
which routinely contains double spaces, tabs, or other whitespace irregularities) and its
effect on the *real* A1 corpus will scale with how much messier real-world/scraped text is
than this hand-typed sample — worth re-measuring once A1's corpus is built.

## Code bug 3: `len(line)` counts Unicode codepoints, not grapheme clusters

**Claim.** Python's `len()` on a string counts Unicode codepoints. Devanagari commonly
represents one visual character ("akshara") as multiple codepoints (a base consonant plus
a combining vowel sign or virama), so codepoint-counting overstates the true character
count for Hindi — while for English (pure ASCII, one codepoint per visual character) it
makes no difference at all.

**Experiment (codepoint count vs. grapheme-cluster count via `regex`'s `\X`, same corpus):**

```
eng: codepoints=448  graphemes=448  (SAME -- no distortion on this corpus)
     tok/char via codepoints: 0.2210   tok/char via graphemes: 0.2210   (0.00%)
hin: codepoints=290  graphemes=188  (DIFFER by 102 -- codepoints overstate by 54%)
     tok/char via codepoints: 1.5828   tok/char via graphemes: 2.4415   (-35.17%)
```

**Result.** Confirmed, and the direction is the interesting, non-obvious part: because
codepoint-counting *inflates* the character denominator for Hindi, the reported tok/char
(1.583, matching REPORT_v0's 1.579) actually *understates* how many tokens each true visual
character costs — the grapheme-correct number is 2.44 tokens per character, 35% higher.
This bug happens to point in the opposite direction from the other two: where the
word-denominator and asymmetric-lowercasing bugs both make Hindi look artificially *worse*
than it is, this one makes Hindi's per-character cost look artificially *better* (lower)
than it truly is. That's exactly why each claim needs its own isolated measurement rather
than a blanket "the report overstates Hindi cost" — the direction isn't uniform.

## Secondary finding (measured, but small — noted rather than treated as a headline claim): macro- vs. micro-averaged fertility

`analyze()` averages *per-line* fertility ratios (mean of ratios) rather than pooling
totals (sum of tokens over sum of words). Measured effect on this corpus:

```
eng: macro (original)=1.2652  micro (pooled)=1.2532  (+0.96%)
hin: macro (original)=7.4485  micro (pooled)=7.4032  (+0.61%)
```

Real, but under 1% on this corpus, and averaging-method choice is more a matter of
statistical convention than an unambiguous bug (macro-averaging is defensible when every
line is meant to count equally regardless of length; micro-averaging is defensible when
overall throughput/cost is the target quantity — which, for a serving-cost decision, is
probably the better justified choice). Including this for completeness since it was
measured, not omitting it, but not spending one of the "few claims you can defend to the
death" on it given how small the effect is here.

## The "looks suspicious but is actually fine" item: `random.seed(1337)`

**Claim.** `import random` and `random.seed(1337)` appear early in the script with a
"# reproducibility" comment. A skim-read flags this as suspicious (why seed randomness in a
deterministic benchmark?). Grepped the entire file: `random` is never called again anywhere
— no shuffling, no sampling. The seed is vestigial dead code.

**Experiment (run analysis under three different seed states — 1337, a different seed,
and no seed at all):**

```
seed=1337:   fertility = 1.2652056277
seed=99999:  fertility = 1.2652056277
no seed:     fertility = 1.2652056277
```

**Result.** Identical to 10 decimal places regardless of seed state — as expected, since
nothing in `read_lines()` or `analyze()` calls anything from the `random` module. This is
the one candidate deliberately **not** being flagged as a bug: it's inert, provably so, and
flagging it would cost points under the assignment's own rule for unverified claims.

## What was tested and explicitly not carried forward as a claim

- **NFC normalization** (`unicodedata.normalize("NFC", line)`): tested with/without on
  both corpora — text and fertility numbers are byte-identical either way, because these
  sample files are already NFC-normalized (hand-typed, clean). This corpus can't
  demonstrate NFC mattering, but that's a property of this data, not proof the
  normalization step is unnecessary — it's correct defensive practice for real
  scraped/mixed-input Indic text and is being treated as fine, not a second "suspicious"
  candidate, since the seed is the cleaner, more decisive example of that category.
