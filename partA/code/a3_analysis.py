"""
Part A3 -- recompute fertility on the REAL eval corpus (partA/corpus/eval_corpus/,
1,532 parallel Genesis verses x eng/hin/kan/tel) with:
  - 2 tokenizers: gpt2 byte-level BPE (reconstructed in Day 1, partA/code/gpt2_bpe.py)
                  and a locally-trained SentencePiece BPE (Day 4, partA/code/spm/,
                  vocab_size=16000, trained on this same corpus -- see spm/train_spm.py
                  for why: HF/tiktoken pretrained tokenizers are network-blocked here).
  - 2 denominators: tokens/word (the original report's metric) and tokens/byte (the
                  corrected metric from A2 -- word-count is not comparable across
                  languages/scripts, byte count is).
  - macro-averaging (mean of per-line ratios), the direction A2's Experiment 5 found
    more defensible for this line-count regime, and consistent with what the toy-corpus
    baseline reproduction used.

Purpose: A2 already showed, on the 10-line TOY corpus, that switching the denominator
from words to bytes cuts the headline "Hindi is Nx worse" ratio roughly in half (5.91x ->
2.72x). A3 asks: does that finding hold on a much larger, real, 4-language corpus, and
does it hold across a SECOND, independently-built tokenizer -- or was it an artifact of
either the toy 10-line sample or of gpt2-BPE specifically (which was never designed for
Indic scripts, so a tokenizer that WAS built with Indic text in mind might behave
differently).
"""
import os
import sys
import re

HERE = os.path.dirname(__file__)
CODE_DIR = HERE
CORPUS_DIR = os.path.join(HERE, "..", "corpus", "eval_corpus")

sys.path.insert(0, CODE_DIR)
from gpt2_bpe import GPT2BPE  # noqa: E402
import sentencepiece as spm  # noqa: E402

LANGS = ["eng", "hin", "kan", "tel"]


def load_lines(lang):
    path = os.path.join(CORPUS_DIR, f"{lang}.txt")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def n_words(line):
    # correct whitespace split (A2 code-bug-2 fix: split() not split(" "))
    return len(line.split())


def macro_ratio(lines, encode_fn, denom_fn):
    ratios = []
    for line in lines:
        n_tok = len(encode_fn(line))
        d = denom_fn(line)
        if d > 0:
            ratios.append(n_tok / d)
    return sum(ratios) / len(ratios)


def main():
    gpt2 = GPT2BPE(
        os.path.join(CODE_DIR, "gpt2_vocab", "encoder.json"),
        os.path.join(CODE_DIR, "gpt2_vocab", "vocab.bpe"),
    )
    sp = spm.SentencePieceProcessor(model_file=os.path.join(CODE_DIR, "spm", "indic4_bpe.model"))

    tokenizers = {
        "gpt2-bpe": lambda s: gpt2.encode(s),
        "indic4-spm": lambda s: sp.encode(s, out_type=int),
    }

    results = {}  # tok_name -> lang -> {tok_per_word, tok_per_byte}
    corpus = {lang: load_lines(lang) for lang in LANGS}

    for tok_name, enc in tokenizers.items():
        results[tok_name] = {}
        for lang in LANGS:
            lines = corpus[lang]
            tpw = macro_ratio(lines, enc, n_words)
            tpb = macro_ratio(lines, enc, lambda l: len(l.encode("utf-8")))
            results[tok_name][lang] = {"tok_per_word": tpw, "tok_per_byte": tpb}

    # Print table
    print(f"{'tokenizer':<12}{'lang':<6}{'tok/word':<12}{'tok/byte':<12}")
    for tok_name in tokenizers:
        for lang in LANGS:
            r = results[tok_name][lang]
            print(f"{tok_name:<12}{lang:<6}{r['tok_per_word']:<12.4f}{r['tok_per_byte']:<12.4f}")
        print()

    # Headline ratio: hin / eng, kan / eng, tel / eng, for both metrics, both tokenizers
    print("=== Ratio vs English (the report's headline framing: 'X is Ny worse') ===")
    print(f"{'tokenizer':<12}{'lang':<6}{'word-ratio':<12}{'byte-ratio':<12}")
    for tok_name in tokenizers:
        eng_tpw = results[tok_name]["eng"]["tok_per_word"]
        eng_tpb = results[tok_name]["eng"]["tok_per_byte"]
        for lang in ["hin", "kan", "tel"]:
            r = results[tok_name][lang]
            wr = r["tok_per_word"] / eng_tpw
            br = r["tok_per_byte"] / eng_tpb
            print(f"{tok_name:<12}{lang:<6}{wr:<12.3f}{br:<12.3f}")
        print()

    return results


if __name__ == "__main__":
    main()
