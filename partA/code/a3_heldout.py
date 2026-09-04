"""
Part A3 follow-up experiment -- the indic4-spm tokenizer was trained on the exact same
1,532-verse corpus it's then measured on. That's a legitimate overfitting concern: BPE
merges could just be memorizing frequent whole words/names from this specific text
(e.g. "Abraham", "Isaac") rather than learning genuinely useful subword segmentation for
these scripts, artificially deflating its own tok/byte numbers. This experiment checks
whether the effect survives a held-out split: train on lines 1-1225 (80%), measure
fertility on lines 1226-1532 (the held-out 20%) never seen in training, and compare to
gpt2-bpe on the same held-out lines.
"""
import os
import sys
import random

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from gpt2_bpe import GPT2BPE  # noqa: E402
import sentencepiece as spm  # noqa: E402

CORPUS_DIR = os.path.join(HERE, "..", "corpus", "eval_corpus")
LANGS = ["eng", "hin", "kan", "tel"]


def load_lines(lang):
    path = os.path.join(CORPUS_DIR, f"{lang}.txt")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def n_words(line):
    return len(line.split())


def macro_ratio(lines, encode_fn, denom_fn):
    ratios = []
    for line in lines:
        n_tok = len(encode_fn(line))
        d = denom_fn(line)
        if d > 0:
            ratios.append(n_tok / d)
    return sum(ratios) / len(ratios)


corpus = {lang: load_lines(lang) for lang in LANGS}
n = len(corpus["eng"])
split = int(n * 0.8)
print(f"Total lines: {n}, train: {split}, held-out: {n - split}")

# Write train-only combined file for retraining
train_dir = os.path.join(HERE, "spm_heldout")
os.makedirs(train_dir, exist_ok=True)
train_path = os.path.join(train_dir, "train_80pct.txt")
with open(train_path, "w", encoding="utf-8") as f:
    for lang in LANGS:
        for line in corpus[lang][:split]:
            f.write(line + "\n")

model_prefix = os.path.join(train_dir, "indic4_bpe_80pct")
spm.SentencePieceTrainer.train(
    input=train_path,
    model_prefix=model_prefix,
    vocab_size=16000,
    model_type="bpe",
    character_coverage=1.0,
    input_sentence_size=0,
    shuffle_input_sentence=True,
    byte_fallback=True,
)
print("Retrained on 80% split.")

sp_heldout = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
gpt2 = GPT2BPE(
    os.path.join(HERE, "gpt2_vocab", "encoder.json"),
    os.path.join(HERE, "gpt2_vocab", "vocab.bpe"),
)

print(f"\n{'tokenizer':<20}{'lang':<6}{'tok/byte (held-out 20%)':<26}")
results = {}
for tok_name, enc in [
    ("indic4-spm(80pct)", lambda s: sp_heldout.encode(s, out_type=int)),
    ("gpt2-bpe", lambda s: gpt2.encode(s)),
]:
    results[tok_name] = {}
    for lang in LANGS:
        heldout_lines = corpus[lang][split:]
        tpb = macro_ratio(heldout_lines, enc, lambda l: len(l.encode("utf-8")))
        results[tok_name][lang] = tpb
        print(f"{tok_name:<20}{lang:<6}{tpb:<26.4f}")

print("\n=== Ratio vs English, held-out 20% only ===")
for tok_name in results:
    eng = results[tok_name]["eng"]
    for lang in ["hin", "kan", "tel"]:
        print(f"{tok_name:<20}{lang:<6} byte-ratio={results[tok_name][lang]/eng:.3f}")
