# NOTEBOOK

Chronological lab notebook. Hypothesis -> experiment -> result -> revision, dead ends included.
Dates in IST (Asia/Calcutta), assignment work started 2026-09-03.

## 2026-09-03 -- Day 1: orientation + environment + Part B

### Orientation
Read AI ASSIGNMENT 2026.pdf and the starter_kit in full (fertility.py, REPORT_v0.md,
corpus_sample/{eng,hin}_sample.txt, bench/model_spec.md, bench/bench_log.csv).

First read of fertility.py turned up several candidate issues before running anything:
whitespace-word denominator as a cross-lingual unit, `line.split(" ")` vs `line.split()`,
`.lower()` applied identically in code but asymmetrically in effect (Devanagari has no case),
`len(line)` counting Unicode codepoints not grapheme clusters, macro- vs micro-averaging
across lines, and an unused `random.seed(1337)` that looks suspicious but does nothing.
None of these count yet -- they're hypotheses, not claims, until measured (evidence rule).

### Dead end #1: tiktoken can't load gpt2 out of the box here
Tried `tiktoken.get_encoding("gpt2")` to establish the exact baseline the original report
used. Failed:
  `ProxyError(... host='openaipublic.blob.core.windows.net' ... 403 Forbidden ...)`
This host is blocked by the network egress policy in both the local machine and the cloud
sandbox (tested both directly with curl -- same allowlist). Also blocked: huggingface.co
(so no `AutoTokenizer.from_pretrained` either), dl.fbaipublicfiles.com (FLORES-200's usual
host), storage.googleapis.com, opus.nlpl.eu, tatoeba.org.
Reachable: pypi.org, github.com, raw.githubusercontent.com.

### Revision
Can't use the networked loaders. Plan:
- gpt2 baseline: pull `encoder.json` + `vocab.bpe` from a GitHub mirror
  (raw.githubusercontent.com/rasbt/LLMs-from-scratch/.../gpt2_model/) and build the BPE
  encoder locally instead of via tiktoken's own downloader.
- Second, "Indic-aware" tokenizer for A3: rather than a blocked pretrained HF model, train
  a small SentencePiece BPE model locally on the assembled multilingual corpus. Fully
  offline, and I can actually explain every part of it in the defense.
- A1 corpus: FLORES-200's usual hosts are blocked. Found `christos-c/bible-corpus` on
  GitHub as a reachable, verse-aligned parallel corpus across 100+ languages. Domain is
  religious/archaic register -- flagging this as an explicit caveat, not hiding it. Will
  make one real attempt at an in-repo FLORES mirror first (Day 3) before committing to it.


### gpt2 tokenizer reconstructed and verified
Pulled encoder.json + vocab.bpe from the GitHub mirror, hand-wrote a byte-level BPE
encoder (your-submission/partA/code/gpt2_bpe.py) implementing the same algorithm as
OpenAI's original encoder.py / tiktoken's "gpt2". Self-test: vocab size 50257 (matches
known gpt2 vocab size exactly), round-trip encode/decode correct on 6 test strings
including edge cases (empty string, repeated internal spaces), and spot-checked token ids
against well-known examples ("hello world" -> [31373, 995], "Hello, world!" ->
[15496, 11, 995, 0]) which match published tiktoken gpt2 output. Confident this is
byte-identical to the real thing, not an approximation.

Ran fertility_original.py's own read_lines()/analyze() functions (untouched) against this
reconstructed tokenizer on the sample corpora (partA/code/repro_baseline.py):
  eng   1.27 / 0.226
  hin   7.45 / 1.579
Matches REPORT_v0.md exactly. Good -- confirms the substitute tokenizer isn't introducing
its own distortion before we even start auditing the script itself. Everything from here
audits fertility.py's logic, not our tokenizer substitution.

### Part B done and verified by script (see partB/capacity_calc.py + partB/ANSWERS.md)
Script output matches every number worked out by hand earlier:
- B1: 114,688 bytes/token KV cache; ~25.7 -> ~25 max concurrent 4096-token sequences
  (decimal-GB reading; ~29 if GiB -- flagged as an open caveat on the exact ceiling).
  Confirmed against the log: batch 24 clean, batch 32 already preempting.
- B2: reported_tok_s peaks at batch 24 (1607.4) then drops at 32 (1384.0) and 48 (1298.5),
  exactly where kv_cache_util hits 0.97 and preempted_seqs goes 0->7->23. Preemption
  thrashing past the ~25-seq ceiling.
- B3: confirmed reported_tok_s = (prompt_len+gen_len)*num_requests/wall_clock_s holds
  exactly for all 13 rows in bench_log.csv (script checks every row, not just the two
  REPORT_v0 cites). Honest goodput at batch24/prompt3584: 200.9 tok/s (gen-tokens method)
  and 249.8 tok/s (inter-token-latency method) -- both ~6-8x below the reported 1607.4.
  The two methods disagree with each other by ~25% (prefill time in the denominator vs
  not) -- noted as its own caveat rather than picking whichever is more convenient.

Day 1 done. Next: Day 2, run the A2 evidence experiments on the toy corpus.

## 2026-09-03 -- Day 2: A2 evidence experiments

Ran all 7 candidate-flaw experiments from Day 1's hypothesis list against the toy corpus
(partA/code/a2_experiments.py, reproducible with `python3 a2_experiments.py`). Full
write-up in partA/A2_AUDIT.md. Headline results:

1. **Conceptual bug, quantified**: switching only the denominator (tokens/word ->
   tokens/byte, same tokenizer, same corpus) cuts the "Hindi is Nx worse" ratio from
   5.91x to 2.72x -- 54% of the reported gap was the word-denominator itself, not
   tokenization cost. Directly contradicts REPORT_v0's "property of the script, not the
   tokenizer" claim.
2. **Asymmetric .lower()**: +2.92% for English, exactly 0.00% for Hindi. Clean, decisive.
3. **split(" ") double-space bug**: confirmed, right direction, but small on this 10-line
   corpus (-1.4% to -2.0%, only 1/10 lines affected per language). Real bug, modest here --
   flagged honestly rather than oversold.
4. **len() codepoints vs grapheme clusters**: the interesting one. Hindi: 290 codepoints
   vs 188 graphemes (54% overstatement). This makes tok/char (1.58, matches REPORT_v0)
   UNDERSTATE true per-character cost -- true grapheme-based number is 2.44, +35%. Direction
   is OPPOSITE the other two bugs (which inflate the Hindi-looks-worse story; this one
   deflates it). Important nuance for A4: not all these bugs point the same way, so "the
   report overstates Hindi cost" isn't a clean blanket claim.
5. **Dead end confirmed as dead end**: random.seed(1337) genuinely inert -- identical
   fertility to 10 decimal places across three different seed states. This is the
   "looks-suspicious-but-fine" item. NOT flagging it as a bug.
6. **Tested, not carried forward**: NFC normalization -- byte-identical with/without on
   this corpus (sample files are already clean/normalized). Can't demonstrate on this
   data; treating as fine, not flagging as suspicious (seed is the cleaner example).
7. **Tested, kept as minor note only**: macro- vs micro-averaging, real but <1% effect on
   this corpus, statistical convention as much as bug. Not spending one of the "few claims
   you can defend to the death" on it, but reported since it was actually measured.

Decision: carrying forward 4 headline claims (conceptual + 3 code bugs) plus the one
"fine" item, per the assignment's own steer toward fewer, defensible claims over many
shaky ones. Split bug and macro/micro averaging are the two weakest by magnitude on this
toy corpus -- worth re-measuring once the real A1 corpus (larger, messier) is built in
case the picture changes at scale.

Day 2 done. Next: Day 3, build the real eval corpus (eng/hin/kan/tam).

## 2026-09-03 -- Day 3 (in progress): A1 eval corpus

Confirmed openlanguagedata/flores (FLORES-200's maintained successor) no longer hosts data
in-repo -- redirects to a blocked HF dataset. Committed to christos-c/bible-corpus on
GitHub instead (per Day 1 plan).

Dead end: Tamil is not in this corpus. Checked the complete 108-language file listing
directly (via GitHub's embedded page JSON, since api.github.com is unusable here) --
confirmed absent, would sort between Tagalog.xml and Telugu.xml, nothing there. Swapped
Tamil -> Telugu (also Dravidian, confirmed present). Flagged explicitly to the user and in
partA/SOURCE.md rather than silently substituted.

Bandwidth to raw.githubusercontent.com measured directly at ~50KB/s (a full Telugu.xml,
11.4MB, would take ~4 minutes -- too slow for 4 full files in the per-call time budget).
Used HTTP range requests (first 700KB per language) instead of full downloads. This
happened to capture the complete Book of Genesis (all 50 chapters) in every language's
partial download -- used Genesis as the corpus boundary since it's a complete, principled
unit rather than an arbitrary truncation point.

Built: 1,532 parallel verses x 4 languages (eng/hin/kan/tel), line-aligned, with a
manifest.tsv for traceability back to verse IDs. Full source/construction/caveats
documented in partA/SOURCE.md, including the "what this corpus can't tell you" paragraph
(religious/narrative register, not representative of production chat traffic; translator-
style variance is not the same as language-inherent variance).

Status: corpus built and documented. Paused here at the user's request (mid Day 3) --
resuming will pick up with sanity-checking the corpus (spot-checked lines 1 and 500
already look correct) and then moving into A3's corrected analysis.

Resumed. Finished sanity-checking the corpus before moving on:
- Line counts: 1,532 lines in all four files, no mismatch.
- Zero empty lines in any of the four files.
- Per-line length distribution sane (min 14-25 chars, max 296-403 chars, avg ~111-127
  chars depending on language) -- no truncated/garbage lines.
- Line 1532 (the last line) is Genesis 50:26 in all four languages -- the actual final
  verse of the Book of Genesis ("So Joseph died, being an hundred and ten years old...").
  This confirms the corpus boundary is clean: it captured the complete book, not a
  truncation mid-book that happened to stop at byte 700000.

Day 3 done. Corpus is verified and ready for A3.

## 2026-09-04 -- Day 4: A3, second tokenizer + corrected fertility analysis

Plan: A3 needs a second tokenizer to compare against gpt2-BPE, since the assignment
wants the audit to show whether the original report's conclusions survive a different
tokenizer, not just a fixed metric. HF pretrained tokenizers (mBERT, XLM-R, etc.) are
blocked (same network restriction as Day 1). `tokenizers` (Rust/HF) pip wheel timed out
downloading during Day 1 already (too slow a connection for the compiled wheel). Decision:
train a local SentencePiece BPE tokenizer on the actual A1 corpus itself (already have
`sentencepiece` installed from Day 1). This is also more defensible than reaching for
another pretrained English-centric tokenizer -- it's trained on the same 4 languages we're
measuring, which is a more realistic proxy for what an Indic-aware production tokenizer
would look like, and the training corpus is fully in-repo and reproducible.

Trained indic4-spm (SentencePiece BPE, vocab_size=16000, char_coverage=1.0, byte_fallback,
trained on the combined eng+hin+kan+tel A1 corpus) -- see partA/code/spm/train_spm.py.
Round-trip verified on 4 test strings across all 4 languages before using it for anything.

Ran partA/code/a3_analysis.py: 2 tokenizers (gpt2-bpe, indic4-spm) x 4 languages x 2
denominators (tok/word, tok/byte) on the real 1,532-verse corpus. Headline result:

  gpt2-bpe byte-ratio vs eng: hin=2.34x, kan=3.91x, tel=3.98x  (word-ratio: 5.26x/16.4x/16.3x)
  indic4-spm byte-ratio vs eng: hin=0.43x, kan=0.30x, tel=0.30x (word-ratio: 0.96x/1.25x/1.23x)

This is the big A3 finding: with gpt2-bpe (an English-centric tokenizer, near-zero Kannada/
Telugu representation in its 50k vocab) Kannada/Telugu need ~4x the tokens/byte of English.
With a tokenizer actually trained on these scripts, they need *fewer* tokens/byte than
English, not more. The entire "Indic languages are N times more expensive to serve"
conclusion is largely a property of *which tokenizer got used*, not an inherent property
of the languages -- an even bigger effect than the tok/word-vs-tok/byte denominator bug
found in A2.

Dead-end-shaped worry, checked and ruled out: indic4-spm was trained on the exact same
corpus it's being measured on, so the low ratios could just be memorization of frequent
words/names (Abraham, Isaac, etc.) rather than genuine subword efficiency. Ran a held-out
check (partA/code/a3_heldout.py): retrained indic4-spm on lines 1-1225 only (80%), measured
fertility on lines 1226-1532 (20%, never seen in training). Result barely moved (hin 0.427x,
kan 0.329x, tel 0.331x vs the full-corpus 0.428x/0.300x/0.299x) -- confirms this is real
subword-segmentation behavior, not memorization. gpt2-bpe on the same held-out lines stayed
at 2.40x/4.01x/4.07x, consistent with the full-corpus numbers. Effect is robust.

Day 4 (A3) done. Writing up partA/A3_AUDIT.md, then A4 recommendation.

Wrote partA/A3_AUDIT.md (full write-up: setup, results table, headline finding, held-out
robustness check, limitations) and partA/A4_RECOMMENDATION.md (<=1 page: tokens/byte over
tokens/word, and "which tokenizer" matters more than "which metric" as the real
operational takeaway, with the A1/A3 caveats carried forward explicitly).

Day 4 done: A3 (12 pts) and A4 (8 pts) both complete with reproducible scripts and
before/after evidence tables. Next: Day 5 -- Part C decision memo, then finalize
NOTEBOOK.md/AI_USAGE.md, re-run everything clean, prep defense.

## 2026-09-04 -- Day 5: Part C decision memo

Re-read the assignment PDF directly (not from memory) before writing this, specifically to
get Part C's exact constraints right: 1x A100-80GB for 2 weeks, 1 native-speaker reviewer
(Hindi + Kannada ONLY) at 10h/week, 3-week launch review, no external API budget, 6 target
languages (Hindi, Kannada, Tamil, Telugu, Bengali, Marathi).

Key realization while planning the memo: the binding constraint isn't compute, it's
review coverage -- only 2 of 6 languages have a native reviewer at all. This shaped the
recommendation more than anything else: SFT (path a) touches the main model's weights and
needs broad eval coverage to catch regressions, which we don't have for 4/6 languages,
so it's the wrong choice under this specific constraint set even though it might otherwise
be the "best" long-term fix. Recommended prompt-engineering as an immediate day-1 ship
(path c) plus a small bolt-on rewriter model in parallel (path b) -- decoupled from the
main model, cheaper to validate, easy to disable per-language if something's wrong.

First draft came out at 942 words -- over the "<=1 page" limit. Cut to 516 words while
keeping every required labeled section (assumptions, arithmetic, success metric, kill
criterion, day-1 experiment) intact. Wrote partC/memo.md.

Day 5 (Part C) done. Remaining: finalize NOTEBOOK.md/AI_USAGE.md, re-run everything from
clean to confirm reproducibility, review the whole submission end-to-end, prep for defense
counterfactuals.

## 2026-09-04 -- Reproducibility pass

Ran every executable script fresh, end to end, to confirm the whole submission holds
together as delivered (not just "worked when I wrote it"):
- partA/code/repro_baseline.py -- reproduces REPORT_v0's exact numbers (eng 1.27/0.226,
  hin 7.45/1.579). OK.
- partA/code/a2_experiments.py -- all 7 experiments run clean, numbers match A2_AUDIT.md.
  OK.
- partB/capacity_calc.py -- B1-B4 numbers match ANSWERS.md exactly (KV bytes/token
  114,688; ~25.7 max concurrent seqs; B2 anomaly at batch 24 vs 32/48; B3 formula matches
  all rows). OK.
- partA/code/a3_analysis.py and a3_heldout.py already run during Day 4 (results captured
  in A3_AUDIT.md); not re-run here since a3_heldout.py retrains a model and is slow, but
  the two-tokenizer/two-denominator numbers were captured live in NOTEBOOK.md Day 4 at
  the time they were produced.

Structure check against the assignment's required layout (NOTEBOOK.md, AI_USAGE.md,
partA/, partB/, partC/memo.md) -- matches. All A1-A4, B1-B4, and C deliverables present.

This closes out active work. Remaining before actual submission: the user should confirm
the Tamil->Telugu substitution is acceptable (flagged repeatedly throughout, not a call
AI can make final), and do a last read-through of the four write-ups
(A2_AUDIT/A3_AUDIT/A4_RECOMMENDATION/partC memo) before the live defense.

## 2026-09-04 -- Tamil->Telugu substitution confirmed by user

Discussed the Tamil->Telugu substitution (see Day 3 entry and partA/SOURCE.md) with the
user directly, including the option of searching for an alternate Tamil-inclusive parallel
corpus. User confirmed keeping Telugu as-is -- the substitution is accepted as a documented
data-availability constraint, not a gap to fix before submission. No further corpus rework
planned.
