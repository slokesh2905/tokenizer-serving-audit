"""
Part A1 -- build the real multilingual eval corpus.

Source: christos-c/bible-corpus (GitHub, CC-BY-SA-style multilingual parallel Bible
corpus, see corpus/raw_bible/SOURCE.md for why this was used instead of FLORES-200).
Selection: the complete Book of Genesis (50 chapters, all verses present in all four
languages) -- a clean, principled, reproducible boundary rather than an arbitrary cut.

Languages: English, Hindi, Kannada, Telugu. (Tamil was the original plan -- not present
in this corpus; swapped for Telugu, see NOTEBOOK.md Day 3 and corpus/raw_bible/SOURCE.md
for the full explanation.)

Output: one line per verse per language file, all four files line-aligned (same verse
order), plus manifest.tsv recording the verse id for each line number for traceability.
"""
import re
import os

HERE = os.path.dirname(__file__)
RAW_DIR = os.path.join(HERE, "raw_bible")
OUT_DIR = os.path.join(HERE, "eval_corpus")
os.makedirs(OUT_DIR, exist_ok=True)

LANGS = {"eng": "eng_partial.xml", "hin": "hin_partial.xml", "kan": "kan_partial.xml", "tel": "tel_partial.xml"}


def parse_segs(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    last_close = content.rfind("</seg>")
    content = content[: last_close + 6]
    pattern = re.compile(r'<seg id="([^"]+)" type="verse">\s*(.*?)\s*</seg>', re.DOTALL)
    segs = {}
    for m in pattern.finditer(content):
        vid, text = m.group(1), m.group(2).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            segs[vid] = text
    return segs


all_segs = {lang: parse_segs(os.path.join(RAW_DIR, fname)) for lang, fname in LANGS.items()}

common_ids = set(all_segs["eng"].keys())
for lang in ["hin", "kan", "tel"]:
    common_ids &= set(all_segs[lang].keys())
gen_ids = [i for i in common_ids if i.startswith("b.GEN.")]


def sort_key(vid):
    parts = vid.split(".")
    return (int(parts[2]), int(parts[3]))


gen_ids_sorted = sorted(gen_ids, key=sort_key)

print(f"Common Genesis verse IDs across all 4 languages: {len(gen_ids_sorted)}")
print(f"Range: {gen_ids_sorted[0]} .. {gen_ids_sorted[-1]}")

files = {lang: open(os.path.join(OUT_DIR, f"{lang}.txt"), "w", encoding="utf-8") for lang in LANGS}
manifest = open(os.path.join(OUT_DIR, "manifest.tsv"), "w", encoding="utf-8")
manifest.write("line_number\tverse_id\n")

for i, vid in enumerate(gen_ids_sorted, start=1):
    for lang in LANGS:
        files[lang].write(all_segs[lang][vid] + "\n")
    manifest.write(f"{i}\t{vid}\n")

for f in files.values():
    f.close()
manifest.close()

for lang in LANGS:
    n_lines = sum(1 for _ in open(os.path.join(OUT_DIR, f"{lang}.txt"), encoding="utf-8"))
    n_bytes = os.path.getsize(os.path.join(OUT_DIR, f"{lang}.txt"))
    print(f"{lang}.txt: {n_lines} lines, {n_bytes} bytes")

print("\nDone. Files are line-aligned 1:1 across eng/hin/kan/tel.txt -- line N in each")
print("file is the same Bible verse (see manifest.tsv for the verse id of each line).")
