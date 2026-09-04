# Part A4 — Which number should drive a routing/cost decision?

**Recommendation: tokens-per-byte (UTF-8), measured with the actual tokenizer the serving
stack will use — not tokens-per-word with any general-purpose tokenizer.**

## Why not tokens/word (the original report's metric)

"Words" aren't a comparable unit across scripts. Whitespace-delimited "words" undercount
morphologically richer languages and over/under-count depending on script conventions —
A2 showed this alone cuts the reported Hindi penalty in this corpus from 5.91× to 2.72×
using nothing but a denominator swap on the *same* tokenizer and corpus. Tokens/word
answers "how many tokens per word" — a question no capacity-planning system asks.
Capacity, cost, and latency all scale with **token count**, and what a service actually
receives and bills on is **bytes/characters of input**, not word counts. Tokens/byte
directly answers the question that matters: given N bytes of a request, how many tokens
will the model see (and pay for, and need KV-cache for)?

## Why "which tokenizer" matters more than which metric

A3 is the stronger finding here: switching the *tokenizer* (gpt2-bpe → indic4-spm, an
SentencePiece BPE actually trained on these scripts) moves Kannada's byte-ratio from
3.91× to 0.30× — a 13x swing, dwarfing the ~2x swing from fixing the metric alone. **The
single biggest driver of a language's serving cost is whether the deployed tokenizer was
built with that language's script in mind — not whether you measure in words or bytes.**
A routing/cost decision based on gpt2-bpe fertility numbers is really measuring "how bad
is gpt2-bpe at Kannada," not "how expensive is Kannada to serve."

## What this means operationally

1. **Never quote a fertility/cost number without naming the tokenizer it was measured
   with.** "Hindi costs 2.3x English" is meaningless without "...under gpt2-bpe." The
   same corpus under indic4-spm gives 0.43x.
2. Before setting per-language routing multipliers or capacity budgets, confirm which
   tokenizer the production serving stack actually uses. If it's an English-centric
   tokenizer (gpt2-bpe-like), the inflated Indic-language costs in the original report are
   real *for that tokenizer* — but the fix on the table isn't necessarily "charge Hindi
   users more," it's "the tokenizer choice is costing you 3-4x on Indic traffic before any
   routing logic runs."
3. Use tokens/byte (not tokens/word) as the standard unit whenever comparing fertility
   across languages, on whatever tokenizer is actually deployed.

## Caveat carried from A1/A3

Both A1's corpus (formal Bible narrative, not production chat traffic) and A3's
indic4-spm (trained in-domain, at a small 16k vocab) are proxies, not a final production
number — see `partA/SOURCE.md` and `partA/A3_AUDIT.md` limitations sections. The
*direction* of both findings (byte over word; tokenizer choice dominates) is solid and
should drive the decision process; the exact multipliers should be re-measured on
production-representative traffic with the actual deployed tokenizer before being used in
a real capacity/pricing formula.
