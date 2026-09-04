# Part C — Localization tone memo: casual replies for Hindi, Kannada, Tamil, Telugu, Bengali, Marathi

## Recommendation

**Prompt-engineering first (ship day 1-2); in parallel build a small (≤1B) inference-time
rewriter — paths (c) then (b).** Skip a full SFT pass (a): the binding constraint here is
review capacity, not compute. One reviewer covers 2 of 6 languages, 10h/week, in a
3-week window. SFT touches the main model's weights and needs broad eval coverage to
catch regressions — coverage we don't have for 4 of 6 languages. A rewriter is a bolt-on,
disable-able per language, and needs a much lower "does this look right" bar to ship
safely.

## Assumptions

Casual tone is a register problem, not a correctness one (replies are already accurate).
No API budget → synthetic pairs must come from an open model run locally on the A100.
The A100 is for training/data-gen only, not added serving infra. "Reviewer: Hindi+Kannada
only" means the other four languages get **zero native QA** for this launch — a hard
constraint to design around. Assume ~150-250 tokens/reply.

## Back-of-envelope arithmetic

**Data:** ~800 synthetic formal→casual pairs/language × 6 = ~4,800, generated via an open
instruction-tuned model on the A100, seeded with reviewer-written exemplars for
Hindi/Kannada.

**Reviewer throughput:** ~2 min/pair (rate casualness + faithfulness) → 30/hr → 300/week
→ ~600 pairs over 2 weeks, split HI/KN (~300 each) — enough to spot-check ~35-40% of each
language's 800 pairs and calibrate an automated faithfulness filter, not enough for
full review, and none for the other four languages.

**Training cost:** a ≤1B rewriter on ~4,800 short pairs, a few epochs, is a few A100-hours
— comfortably inside the 2-week window alongside data generation and 2-3 iteration
rounds.

**Serving cost:** one extra ≤1B forward pass per reply, feature-flagged to the 6
languages only — a modest fraction of the main model's own per-request cost, not a global
rollout.

## Success metric

Hindi/Kannada (the only reviewed languages): on 50 held-out prompts/language, ≥70% rated
casual/conversational (4-5/5) **and** ≤10% rated meaning-lost — both required. Tamil/
Telugu/Bengali/Marathi: gated on an automated faithfulness-similarity score, threshold
calibrated to best match the HI/KN reviewer labels — an explicit proxy, not real review.

## Biggest caveat

Four of six languages ship with no native validation at all; the automated proxy is
calibrated on Hindi/Kannada and assumed (not verified) to transfer. Flag to product
before launch and recommend a smaller initial rollout percentage for the unreviewed four,
plus fast-following native review.

## Kill criterion

By **end of week 1** of the compute window: if HI/KN reviewer-rated pairs are <50%
casual+faithful (well below the 70%/≤10% bar), abandon the rewriter path — not enough of
the 3-week window remains to fix data generation, retrain, and re-review. Fall back
entirely to prompt-engineering-only, using whatever the day-1 experiment already
validated.

## Day-1 experiment

Write a casual-tone system prompt + 3-5 few-shot exemplars per language (HI/KN with the
reviewer; other four drafted and explicitly labeled "unreviewed"). Run against ~20
existing formal replies/language; get the reviewer's same-day HI/KN ratings on the same
5-point scale. This gives an immediate baseline (prompting alone may suffice given the
timeline) and the first calibration data the automated proxy for the other four languages
will depend on.
