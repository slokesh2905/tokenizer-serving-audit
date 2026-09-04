"""
Part A2 -- evidence-rule experiments against fertility_original.py.

Every candidate flaw from the first-pass reading (see NOTEBOOK.md) is tested here in
isolation: one change at a time, before/after numbers on the SAME corpus, so the delta
can only be attributed to that one change. Run on the toy sample corpus (10 lines/lang) --
small enough to hand-verify a couple of lines by eye, which is the point of using it here
rather than jumping straight to the real A1 corpus.

Usage: python3 a2_experiments.py
"""
import os
import sys
import unicodedata
import regex as re

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from gpt2_bpe import GPT2BPE  # noqa: E402

enc = GPT2BPE(
    os.path.join(HERE, "gpt2_vocab", "encoder.json"),
    os.path.join(HERE, "gpt2_vocab", "vocab.bpe"),
)
encode = enc.encode

CORPUS_DIR = os.path.join(HERE, "..", "corpus", "toy_sample")
FILES = {"eng": "eng_sample.txt", "hin": "hin_sample.txt"}


def read_lines_raw(path):
    """Same as fertility_original.read_lines but keeps the exact stripped line,
    no NFC applied -- used as the base for the NFC on/off experiment."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def load(lang, nfc=True):
    lines = read_lines_raw(os.path.join(CORPUS_DIR, FILES[lang]))
    if nfc:
        lines = [unicodedata.normalize("NFC", l) for l in lines]
    return lines


def fertility_variant(lines, encode, lowercase=True, word_split="literal_space", avg="macro"):
    """Reimplements analyze() with each design choice as a parameter, so a single flag
    flip isolates a single change against the original's behavior (lowercase=True,
    word_split='literal_space', avg='macro' reproduces the original exactly)."""
    per_line_fert = []
    tok_total, word_total = 0, 0
    for line in lines:
        L = line.lower() if lowercase else line
        tokens = encode(L)
        if word_split == "literal_space":
            words = L.split(" ")
        else:  # "whitespace" -- collapses runs of whitespace, no empty tokens
            words = L.split()
        n_tok, n_word = len(tokens), len(words)
        per_line_fert.append(n_tok / n_word)
        tok_total += n_tok
        word_total += n_word
    macro = sum(per_line_fert) / len(per_line_fert)
    micro = tok_total / word_total
    return macro if avg == "macro" else micro


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
section("EXPERIMENT 1: word_split literal-space bug (split(' ') vs split())")
# ---------------------------------------------------------------------------
for lang in ["eng", "hin"]:
    lines = load(lang)
    n_double_space_lines = sum(1 for l in lines if "  " in l)
    before = fertility_variant(lines, encode, word_split="literal_space")  # original
    after = fertility_variant(lines, encode, word_split="whitespace")      # fixed
    pct = (before - after) / after * 100
    print(f"{lang}: {n_double_space_lines} line(s) with a double space out of {len(lines)}")
    print(f"  original split(' '):  fertility = {before:.4f}")
    print(f"  fixed .split():       fertility = {after:.4f}")
    print(f"  delta: original overstates word count -> understates fertility by {pct:+.2f}% "
          f"on this corpus (magnitude scales with how many double-spaces are in a corpus,"
          f" not a fixed correction factor)")

# ---------------------------------------------------------------------------
section("EXPERIMENT 2: asymmetric .lower() (English changes, Hindi doesn't)")
# ---------------------------------------------------------------------------
for lang in ["eng", "hin"]:
    lines = load(lang)
    before = fertility_variant(lines, encode, lowercase=True)   # original
    after = fertility_variant(lines, encode, lowercase=False)   # no lowercasing
    changed = before != after
    pct = (before - after) / after * 100 if after else float("nan")
    print(f"{lang}: with .lower()={before:.4f}  without .lower()={after:.4f}  "
          f" {'CHANGES' if changed else 'NO EFFECT'} ({pct:+.2f}%)")

# ---------------------------------------------------------------------------
section("EXPERIMENT 3: macro- vs micro-averaged fertility")
# ---------------------------------------------------------------------------
for lang in ["eng", "hin"]:
    lines = load(lang)
    macro = fertility_variant(lines, encode, avg="macro")  # original
    micro = fertility_variant(lines, encode, avg="micro")
    pct = (macro - micro) / micro * 100
    print(f"{lang}: macro (original) = {macro:.4f}   micro (pooled) = {micro:.4f}   "
          f"delta = {pct:+.2f}%")

# ---------------------------------------------------------------------------
section("EXPERIMENT 4: len(line) codepoints vs grapheme clusters, for tok/char")
# ---------------------------------------------------------------------------
for lang in ["eng", "hin"]:
    lines = load(lang)
    cp_total, gc_total, tok_total = 0, 0, 0
    for line in lines:
        L = line.lower()
        tokens = encode(L)
        tok_total += len(tokens)
        cp_total += len(L)                          # original: codepoints
        gc_total += len(re.findall(r"\X", L))        # fixed: grapheme clusters
    tpc_codepoint = tok_total / cp_total
    tpc_grapheme = tok_total / gc_total
    pct = (tpc_codepoint - tpc_grapheme) / tpc_grapheme * 100
    print(f"{lang}: codepoints={cp_total} graphemes={gc_total} "
          f"({'SAME -- no distortion on this corpus' if cp_total == gc_total else f'DIFFER by {cp_total - gc_total}'})")
    print(f"  tok/char via codepoints (original): {tpc_codepoint:.4f}")
    print(f"  tok/char via grapheme clusters:      {tpc_grapheme:.4f}   delta = {pct:+.2f}%")

# ---------------------------------------------------------------------------
section("EXPERIMENT 5: random.seed(1337) -- is it actually inert?")
# ---------------------------------------------------------------------------
import random  # noqa: E402
lines_eng = load("eng")

random.seed(1337)
out_a = fertility_variant(lines_eng, encode)
random.seed(99999)  # different seed
out_b = fertility_variant(lines_eng, encode)
random.seed()  # no seed at all (system entropy)
out_c = fertility_variant(lines_eng, encode)
print(f"seed=1337:   fertility = {out_a:.10f}")
print(f"seed=99999:  fertility = {out_b:.10f}")
print(f"no seed:     fertility = {out_c:.10f}")
print("IDENTICAL REGARDLESS OF SEED (as expected -- random is never called in analyze/read_lines)"
      if out_a == out_b == out_c else "DIFFERS BY SEED -- retract the 'inert' claim, something uses randomness")

# ---------------------------------------------------------------------------
section("EXPERIMENT 6: NFC normalization -- does it change anything on THIS corpus?")
# ---------------------------------------------------------------------------
for lang in ["eng", "hin"]:
    with_nfc = load(lang, nfc=True)
    without_nfc = load(lang, nfc=False)
    identical_text = with_nfc == without_nfc
    fert_with = fertility_variant(with_nfc, encode)
    fert_without = fertility_variant(without_nfc, encode)
    print(f"{lang}: raw text identical after NFC = {identical_text}  "
          f"fertility_with_nfc={fert_with:.6f}  fertility_without_nfc={fert_without:.6f}")
print("Note: sample files are apparently already NFC-normalized text (typed cleanly), so this")
print("particular corpus can't show NFC mattering. That's a property of THIS corpus, not proof")
print("normalization is unnecessary in general -- it's still correct defensive practice for")
print("real-world/scraped Indic text with mixed composed/decomposed input. Treating it as 'fine'")
print("(not a bug) here.")

# ---------------------------------------------------------------------------
section("EXPERIMENT 7: how much of the '5.9x worse' gap is the word-denominator itself?")
# ---------------------------------------------------------------------------
# Same tokenizer (gpt2), same corpus, same lowercasing -- ONLY the denominator changes.
# This isolates the conceptual bug (A2's required conceptual flaw) from tokenizer choice.
totals = {}
for lang in ["eng", "hin"]:
    lines = load(lang)
    tok_total = byte_total = word_total = 0
    for line in lines:
        L = line.lower()
        tokens = encode(L)
        tok_total += len(tokens)
        byte_total += len(L.encode("utf-8"))
        word_total += len(L.split(" "))
    totals[lang] = dict(tok=tok_total, byte=byte_total, word=word_total,
                         tok_per_word=tok_total / word_total, tok_per_byte=tok_total / byte_total)
    print(f"{lang}: {totals[lang]}")

word_ratio = totals["hin"]["tok_per_word"] / totals["eng"]["tok_per_word"]
byte_ratio = totals["hin"]["tok_per_byte"] / totals["eng"]["tok_per_byte"]
print(f"\nhin/eng ratio via tokens-per-word (REPORT_v0's metric): {word_ratio:.2f}x")
print(f"hin/eng ratio via tokens-per-byte  (content-normalized): {byte_ratio:.2f}x")
print(f"Same tokenizer, same corpus, only the denominator changed -- yet "
      f"{(1 - byte_ratio / word_ratio) * 100:.1f}% of the reported 'Hindi is Nx worse' gap "
      f"evaporates. The remaining {byte_ratio:.2f}x is attributable to gpt2 itself being "
      f"English-centric (a tokenizer-choice question, which A3 tests directly with a second, "
      f"Indic-aware tokenizer) -- not to Hindi 'having more Unicode characters per word' as "
      f"REPORT_v0 claims.")
