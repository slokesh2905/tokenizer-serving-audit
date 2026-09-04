"""
Reproduce REPORT_v0.md's Section 1 numbers using fertility_original.py's own
read_lines()/analyze() functions, unmodified -- only the tokenizer *loading*
is swapped (tiktoken's gpt2 loader needs network we don't have here; gpt2_bpe.py
is a from-scratch reconstruction of the identical algorithm from the same
released vocab files, verified round-trip-correct and vocab-size-correct
against the real gpt2 tokenizer in gpt2_bpe.py's own self-test).

This is a sanity check, not an audit: if we can't reproduce 1.27 / 7.45 (eng)
and whatever REPORT_v0 got for hindi, then something about our substitute
tokenizer is wrong and nothing downstream can be trusted.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gpt2_bpe import GPT2BPE  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "fertility_original", os.path.join(os.path.dirname(__file__), "fertility_original.py")
)
fo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fo)

HERE = os.path.dirname(__file__)
enc = GPT2BPE(
    os.path.join(HERE, "gpt2_vocab", "encoder.json"),
    os.path.join(HERE, "gpt2_vocab", "vocab.bpe"),
)

corpus_dir = os.path.join(HERE, "..", "corpus", "toy_sample")
results = {}
for lang, fname in [("eng", "eng_sample.txt"), ("hin", "hin_sample.txt")]:
    lines = fo.read_lines(os.path.join(corpus_dir, fname))
    fert, tpc = fo.analyze(lines, enc.encode)
    results[lang] = (fert, tpc)
    print(f"{lang:<8}{fert:>22.2f}{tpc:>12.3f}")

print()
print(f"REPORT_v0 claimed: eng 1.27 / 0.226   hin 7.45 / 1.579")
print(
    f"reproduced here:   eng {results['eng'][0]:.2f} / {results['eng'][1]:.3f}"
    f"   hin {results['hin'][0]:.2f} / {results['hin'][1]:.3f}"
)
eng_match = abs(results["eng"][0] - 1.27) < 0.01 and abs(results["eng"][1] - 0.226) < 0.001
hin_match = abs(results["hin"][0] - 7.45) < 0.01 and abs(results["hin"][1] - 1.579) < 0.001
print("BASELINE REPRODUCED EXACTLY" if (eng_match and hin_match) else "MISMATCH -- investigate before trusting anything downstream")
