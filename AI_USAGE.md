# AI_USAGE.md

Honest summary of where AI (Claude) helped and where it needed correcting, kept live
through the assignment rather than written after the fact.

## What AI did
- Read the assignment PDF and starter kit, and produced a first-pass list of candidate
  bugs in fertility.py by static reading alone (no execution). These are logged as
  hypotheses in NOTEBOOK.md, not claims -- every one of them still needs the evidence-rule
  experiment before it goes in partA/.
- Probed the sandbox network (curl against huggingface.co, tiktoken's blob host,
  FLORES's usual hosts, github.com, pypi.org) to figure out what's actually reachable
  before committing to a corpus/tokenizer-sourcing plan, and proposed the workarounds
  (GitHub-mirrored gpt2 vocab, locally trained SentencePiece, bible-corpus as parallel
  data source).
- Did the initial B1-B3 arithmetic by hand from model_spec.md/bench_log.csv, then wrote
  a script to verify it wasn't a hand-math error (see partB/).
- Drafted the day-by-day plan and repo skeleton.

## Where AI could mislead (watch for this)
- A static read of code is not evidence. Every "this looks like a bug" claim from the
  first pass gets re-verified by an actual before/after experiment on Day 2 -- some of
  these hypotheses may turn out to be wrong, negligible in effect, or (in at least one
  case, expected to be `random.seed`) actually harmless despite looking suspicious.
- AI-proposed reasoning about *why* something is a bug (e.g. "asymmetric lowercasing")
  can sound confident without being correct -- the actual magnitude and direction of each
  effect only comes from running the isolated experiment, not from the explanation alone.
- (This section will be updated through the week as specific instances of AI being wrong
  come up -- to be filled in honestly, not retrofitted at the end.)


## Day 1 update (2026-09-03)
- AI wrote the from-scratch gpt2 BPE encoder (gpt2_bpe.py) after tiktoken's networked
  loader failed in this environment. This is exactly the kind of thing to double check --
  a hand-rolled tokenizer reimplementation is easy to get subtly wrong (off-by-one in the
  merge loop, wrong byte<->unicode mapping table, etc.) and still "look" like it works.
  Verified independently: vocab size matches gpt2's known 50257 exactly, round-trips
  correctly, and -- most importantly -- reproduces REPORT_v0.md's own eng/hin numbers
  (1.27/0.226 and 7.45/1.579) exactly when run through the *original, unmodified*
  fertility_original.py functions. That last check is the one that actually matters: it's
  not just "does this look like a reasonable tokenizer," it's "does it reproduce the
  specific numbers we're trying to audit," which is a much stronger bar.
- Part B's numbers (B1-B3) were derived by AI by hand first, then independently verified
  by a script that reads the source files directly and recomputes everything, rather than
  the write-up just restating the hand math. The script's B3 check (does the
  reported_tok_s formula hold for literally every row, not just the two the report itself
  cites) is stronger evidence than checking two rows would have been -- worth keeping that
  habit for Part A too (check hypotheses against the whole corpus, not a couple of lines).
- Nothing AI got wrong yet that I've caught -- but Part A2 (Day 2) is where the actual
  audit claims get made and is the highest-risk part for AI overclaiming without evidence;
  will scrutinize that harder.

## Day 2 update (2026-09-03)
- AI ran all 7 candidate-flaw experiments from the Day 1 hypothesis list, isolating one
  variable at a time against the toy corpus. This is the part of the assignment most at
  risk of AI overclaiming, so worth being specific about what held up vs. what didn't:
  - Confirmed strongly, as hypothesized: the conceptual denominator bug (54% of the
    reported gap vanishes just from switching tokens/word to tokens/byte), the asymmetric
    .lower() bug (+2.92% vs exactly 0.00%), and the codepoint-vs-grapheme bug for tok/char.
  - The codepoint/grapheme result is worth flagging specifically: my first-pass intuition
    (Day 1) was "this bug probably makes Hindi look artificially bad, like the others."
    The actual measurement showed the opposite direction -- it makes Hindi's per-character
    cost look artificially GOOD (understated). Good example of why the evidence rule
    exists: the plausible-sounding direction was wrong, and only running the number caught
    it.
  - The split(" ") double-space bug and the macro/micro-averaging difference were both
    real (correct direction) but small in magnitude on this 10-line corpus -- AI's
    original framing of these (Day 1) implied they might be more consequential than the
    numbers ended up showing. Kept both, but downgraded macro/micro to a minor note rather
    than a headline claim, and flagged that the split bug's true impact needs re-checking
    on the larger, messier A1 corpus rather than assumed to generalize from this toy data.
  - random.seed(1337) confirmed inert as predicted -- the one place Day 1's hypothesis
    and Day 2's measurement matched exactly with no surprises.

## Day 3-4: A1 corpus construction, A3 tokenizer training and analysis

- AI made an autonomous substitution decision (Tamil -> Telugu, corpus source
  unavailability) under the "work autonomously, check in at day boundaries" mode the user
  selected. This is flagged here because it changes the assignment's stated scope (user
  originally chose Kannada+Tamil) without a real-time approval step -- the decision itself
  was evidence-based (checked the full source file listing directly rather than assuming
  absence), and was surfaced explicitly in NOTEBOOK.md, partA/SOURCE.md, and to the user
  before the pause, rather than silently substituted. Worth double-checking with the user
  before final submission whether this substitution is acceptable, since it wasn't a
  question AI could answer on the user's behalf.
- AI chose the SentencePiece-locally-trained-tokenizer approach for A3 (rather than e.g.
  retrying the blocked pretrained tokenizers, or waiting out the network restriction) on
  its own judgment. This is a real methodological choice with a real limitation (in-domain
  training data, small 16k vocab) that AI flagged explicitly in A3_AUDIT.md's limitations
  section rather than presenting the numbers as a clean production estimate.
- AI's held-out-split check (train on 80%, measure on unseen 20%) was run specifically
  because AI itself flagged a self-critique (training-corpus/eval-corpus overlap) before
  the user or the assignment asked for it -- this is exactly the kind of "looks suspicious,
  check it" muscle the assignment's grading rubric rewards, applied to AI's own new
  tokenizer this time rather than only the original intern's code. Result: the effect
  survived the check (numbers moved <10%), so it was kept as a real finding, not discarded
  or hedged into uselessness -- but the specific magnitude is still caveated as
  corpus/domain-specific in A4.
- The single most important caveat for this section: A3's numbers describe two
  *specific* tokenizers (gpt2-bpe vs. one small locally-trained SentencePiece model) on
  one specific religious-text corpus. AI did not have the ability to test a third,
  production-grade Indic tokenizer (e.g. IndicBERT) due to the same network restriction --
  so "tokenizer choice matters enormously" is well-evidenced, but "this exact tokenizer is
  the fix" is not a claim this audit makes or should be read as making.

## Day 5: Part C memo

- AI drafted the full Part C memo including the specific recommendation (prompt-eng first,
  rewriter model second, skip SFT) and all the back-of-envelope numbers (data volume,
  reviewer throughput, training/serving cost estimates). These are estimates/assumptions,
  not measured facts -- flagged as such throughout the memo itself (explicitly labeled
  "Assumptions" section) rather than presented as verified numbers, since Part C is a
  planning exercise and the evidence rule's "measure it" standard doesn't apply the same
  way to numbers about a project that hasn't happened yet. If challenged in the defense,
  the arithmetic (reviewer throughput math, training-cost order-of-magnitude) is fully
  re-derivable from the assumptions stated -- that's the actual defensible artifact here,
  not the specific numbers themselves.
- AI's first draft of the memo (942 words) blew past the assignment's explicit "<=1 page"
  limit -- caught this by actually running `wc -w` rather than eyeballing it, then cut to
  516 words. Worth noting as a small but real example of AI not tracking a stated
  constraint until it was mechanically checked.

## Overall honesty summary

Where AI helped most: mechanical/technical lifting that would otherwise eat the whole
timebox -- hand-reconstructing gpt2 BPE from vocab files when the network blocked the
usual loaders, writing the parameterized experiment harness for A2's isolated variable
tests, training and evaluating the SentencePiece tokenizer for A3, and drafting all
write-ups from computed numbers.

Where AI was wrong or needed correcting, kept specific rather than generic: the
codepoint/grapheme bug's direction (Day 2 -- predicted the wrong sign, caught by actually
running the number); the A2 memo's first framing overstating the split-bug/macro-micro
findings relative to their small measured size (Day 2); the Part C memo blowing past the
explicit "<=1 page" constraint until mechanically checked with `wc -w` (Day 5); and the
Tamil->Telugu language substitution (Day 3), which was a reasonable data-driven call but
is explicitly not a decision AI should be treated as having full authority to make --
flagged to the user rather than presented as settled.

What AI did NOT do: no numbers in this submission were invented or estimated without a
script producing them, except Part C's forward-looking planning arithmetic, which is
explicitly labeled as assumptions/estimates rather than measured results (Part C is about
a project that hasn't happened, so "measure it" doesn't apply the way it does in Parts
A/B). Every claimed bug in A2 and every headline number in A3/B has a script in the repo
that reproduces it -- re-run during the Day 5 reproducibility pass and confirmed to match
what's written in the corresponding .md files.
