# Part A3 — Fertility on the real corpus, with a second tokenizer

## Setup

Corpus: `partA/corpus/eval_corpus/` — 1,532 line-aligned Genesis verses, English, Hindi,
Kannada, Telugu (see `partA/SOURCE.md` for construction and the Tamil→Telugu substitution).

Tokenizers:
1. **gpt2-bpe** — the same hand-reconstructed GPT-2 byte-level BPE from A1/A2
   (`partA/code/gpt2_bpe.py`), vocab 50,257, English-centric (trained on WebText).
2. **indic4-spm** — a SentencePiece BPE tokenizer trained locally on this project's own
   corpus (`partA/code/spm/train_spm.py`): vocab_size 16,000, `character_coverage=1.0`,
   `byte_fallback=True`, trained on the combined eng+hin+kan+tel text. Used because every
   pretrained multilingual/Indic tokenizer (mBERT, XLM-R, IndicBERT, etc.) lives behind a
   blocked host in this environment (see `NOTEBOOK.md` Day 1) and the `tokenizers` pip
   wheel would not download in time over this connection either. Round-trip correctness
   verified on English/Hindi/Kannada test strings before use.

Metric: macro-averaged (mean of per-line ratios) tokens/word and tokens/byte, computed with
`partA/code/a3_analysis.py`.

## Results

| tokenizer | lang | tok/word | tok/byte |
|---|---|---|---|
| gpt2-bpe | eng | 1.296 | 0.252 |
| gpt2-bpe | hin | 6.813 | 0.588 |
| gpt2-bpe | kan | 21.285 | 0.984 |
| gpt2-bpe | tel | 21.151 | 1.000 |
| indic4-spm | eng | 1.249 | 0.243 |
| indic4-spm | hin | 1.199 | 0.104 |
| indic4-spm | kan | 1.565 | 0.073 |
| indic4-spm | tel | 1.531 | 0.073 |

Ratio vs. English (the report's own framing — "language X is N× worse"):

| tokenizer | lang | word-ratio | byte-ratio |
|---|---|---|---|
| gpt2-bpe | hin | 5.26× | 2.34× |
| gpt2-bpe | kan | 16.42× | 3.91× |
| gpt2-bpe | tel | 16.32× | 3.98× |
| indic4-spm | hin | 0.96× | **0.43×** |
| indic4-spm | kan | 1.25× | **0.30×** |
| indic4-spm | tel | 1.23× | **0.30×** |

## The headline finding

**Which tokenizer you use changes the conclusion more than which metric (tok/word vs.
tok/byte) you use — and A2 already showed the metric alone matters a lot.**

With gpt2-bpe, Kannada and Telugu need roughly **4× the tokens per byte** of English even
after fixing the tok/word-vs-tok/byte denominator bug from A2. That's because gpt2-bpe's
50,257-token vocabulary was trained on English WebText and has essentially no
Kannada/Telugu subword coverage — most Kannada/Telugu text falls back to encoding
individual UTF-8 bytes of each 3-byte Devanagari/Kannada/Telugu codepoint almost one at a
time (visible directly in the tok/word numbers: 21.3 and 21.2 tokens per word, vs. 1.3 for
English).

With indic4-spm — a tokenizer that has actually seen these scripts during training —
Kannada and Telugu need **roughly a third** the tokens per byte of English, not four times
more. Hindi comes in under English too (0.43×). This flips the entire conclusion of the
original report: it isn't that "Indic languages are inherently more expensive to tokenize."
It's that gpt2-bpe specifically is a bad tokenizer for these languages. A production system
that already uses (or could reasonably switch to) an Indic-aware tokenizer would see the
*opposite* of the report's claimed cost problem for Kannada and Telugu.

(Why would Indic languages end up *cheaper* per byte once the tokenizer knows them?
Devanagari/Kannada/Telugu are abugida scripts — each character already encodes a
consonant+vowel or a fuller syllable than a single Latin letter does, and each such
character costs 3 bytes in UTF-8 vs. 1 byte per Latin letter. So an efficient subword
tokenizer merging common Indic morphemes into single tokens can end up needing fewer
tokens per *byte* than English even while needing a comparable or higher number of tokens
per *word* — the per-byte numbers above show this isn't happening for Hindi/Kannada/Telugu
tokens-per-word either, once the tokenizer is fit to the script; that ratio moved from
21×/6.8× down to ~1.2–1.6×.)

## Evidence: this isn't a leaderboard artifact of testing on the training corpus

indic4-spm was trained on the exact 1,532-verse corpus it's then measured on — a real risk
that its low tok/byte numbers are just memorized whole words (proper nouns like "Abraham"
repeat constantly in Genesis) rather than genuine subword efficiency for these scripts.

Checked with `partA/code/a3_heldout.py`: retrained indic4-spm on the first 80% of lines
only (1,225 lines), then measured fertility on the held-out last 20% (307 lines) never
seen during training.

| tokenizer | lang | byte-ratio vs eng, full corpus | byte-ratio vs eng, held-out only |
|---|---|---|---|
| indic4-spm | hin | 0.428 | 0.427 |
| indic4-spm | kan | 0.300 | 0.329 |
| indic4-spm | tel | 0.299 | 0.331 |
| gpt2-bpe | hin | 2.337 | 2.400 |
| gpt2-bpe | kan | 3.909 | 4.006 |
| gpt2-bpe | tel | 3.975 | 4.068 |

Numbers barely move (all within ~10%, in the direction of the effect being *slightly*
weaker on held-out data, as expected, not disappearing). This is real subword-segmentation
behavior driven by tokenizer design, not memorization of this specific corpus.

## Limitations of this A3 result (carried into A4)

- indic4-spm was trained on in-domain text (this exact corpus's languages/register), not
  a general-purpose multilingual tokenizer built for production traffic. A held-out split
  within the same corpus and register isn't the same test as a genuinely different domain
  (chat messages, code-switched text). The conclusion "a tokenizer aware of these scripts
  changes the outcome" is solid; the specific magnitude (0.3×–0.4×) should not be quoted as
  a production number without testing on production-like text (see `partA/SOURCE.md`'s
  "what this corpus cannot tell you" section — it applies here too).
- vocab_size=16,000 was a size choice made for this corpus's scale, not tuned against any
  target; a production Indic tokenizer would likely use a larger, more carefully-built
  vocabulary (e.g. IndicBERT/IndicBART-scale), which this experiment cannot substitute for
  — it demonstrates the *direction and existence* of the tokenizer effect, not a
  production-ready number.
- Only 2 tokenizers were compared. The finding is "tokenizer choice matters enormously,"
  not "indic4-spm specifically is the tokenizer to deploy."
