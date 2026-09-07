# The Audit — tokenizer fertility + serving capacity review

An audit of a previous intern's tokenizer benchmark report (`REPORT_v0.md`) and serving
load-test log before either is used for routing or capacity decisions. Every claim below
is backed by a script in this repo that reproduces it — see **Reproducing the numbers**.

## Start here

- **`NOTEBOOK.md`** — chronological log of the actual work: hypotheses, dead ends,
  measurements, revisions. This is the real record of how each finding below was reached.
- **`AI_USAGE.md`** — honest disclosure of where AI helped and where it needed correcting.

## Structure and headline findings

**`partA/`** — the tokenizer audit.
- `SOURCE.md` — how the 1,532-verse, 4-language eval corpus (English/Hindi/Kannada/Telugu)
  was built, and why Telugu stands in for Tamil (not present in the source corpus used;
  substitution confirmed with the assignment owner — see `NOTEBOOK.md`).
- `A2_AUDIT.md` — audits `fertility.py`. Headline: measuring tokens-per-word instead of
  tokens-per-byte inflates the reported "Hindi is 5.91x worse" gap — switching only the
  denominator (same tokenizer, same corpus) drops it to 2.72x. Plus three code bugs, each
  with before/after evidence, and one thing that looks like a bug but measurably isn't
  (`random.seed(1337)`, confirmed inert).
- `A3_AUDIT.md` — recomputes fertility on the real corpus with two tokenizers. Headline:
  which tokenizer you use matters far more than which metric you use. An English-centric
  tokenizer (gpt2-bpe) makes Kannada/Telugu look ~4x more expensive than English; a
  tokenizer trained on these scripts makes them look *cheaper* per byte, not more
  expensive. Verified this isn't just memorization of the training corpus via a held-out
  split.
- `A4_RECOMMENDATION.md` — tokens/byte over tokens/word, and never quote a fertility
  number without naming the tokenizer it was measured with.

**`partB/`** — serving capacity reconciliation.
- `ANSWERS.md` — KV-cache arithmetic from the model spec, an explanation of the
  throughput anomaly at high batch sizes (preemption thrashing past the KV-cache
  ceiling), a reverse-engineered formula for the log's `reported_tok_s` column showing
  it's a throughput metric that includes prefill (not a decode-speed/goodput metric), and
  the resulting honest goodput number vs. the report's overstated one.

**`partC/memo.md`** — a resource-constrained decision memo on shipping casual-tone
replies across six languages, given a reviewer who only covers two of them.

## Reproducing the numbers

Every number in `A2_AUDIT.md`, `A3_AUDIT.md`, and `partB/ANSWERS.md` comes from a script
in this repo, not a hand-typed claim:

```
python3 partA/code/repro_baseline.py     # reproduces REPORT_v0's original numbers exactly
python3 partA/code/a2_experiments.py     # all A2 bugs/findings, isolated one at a time
python3 partA/code/a3_analysis.py        # A3's two-tokenizer, two-denominator comparison
python3 partA/code/a3_heldout.py         # A3's held-out-split robustness check (retrains a model)
python3 partB/capacity_calc.py           # all of B1-B4
```

Requires Python 3 with `sentencepiece` and `regex` installed
(`pip install sentencepiece regex`). No network access is needed to reproduce any of
these — the gpt2 vocabulary files and the SentencePiece tokenizer are already in
`partA/code/gpt2_vocab/` and `partA/code/spm/`.
