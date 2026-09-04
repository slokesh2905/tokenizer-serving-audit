"""
Minimal, self-contained GPT-2 byte-level BPE encoder, built directly from the
original released vocab files (encoder.json + vocab.bpe), reconstructing the
same algorithm OpenAI's own encoder.py / tiktoken's "gpt2" encoding use.

Built this by hand because tiktoken's own get_encoding("gpt2") tries to
download its BPE ranks from openaipublic.blob.core.windows.net at runtime,
which is blocked by the network egress policy in this environment (see
NOTEBOOK.md, Day 1). encoder.json / vocab.bpe here were pulled from a public
GitHub mirror of the same files instead.
"""
import json
import functools
import regex as re

_GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@functools.lru_cache()
def _bytes_to_unicode():
    """Reversible byte <-> printable-unicode-char mapping (OpenAI's original trick
    to make every byte value representable as a visible character before BPE)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class GPT2BPE:
    def __init__(self, encoder_json_path, vocab_bpe_path):
        with open(encoder_json_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(vocab_bpe_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        merges = [tuple(line.split()) for line in bpe_data.split("\n")[1:-1] if line]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        ids = []
        for tok in re.findall(_GPT2_PATTERN, text):
            tok_bytes = tok.encode("utf-8")
            tok_translated = "".join(self.byte_encoder[b] for b in tok_bytes)
            for bpe_tok in self._bpe(tok_translated).split(" "):
                ids.append(self.encoder[bpe_tok])
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[i] for i in ids)
        b = bytearray(self.byte_decoder[c] for c in text)
        return b.decode("utf-8", errors="replace")


if __name__ == "__main__":
    import sys

    enc = GPT2BPE(sys.argv[1], sys.argv[2])
    print("vocab size:", len(enc.encoder))

    tests = [
        "hello world",
        "Hello, world!",
        "tiktoken is great!",
        "",
        "  double  space  test  ",
        "The quick brown fox jumps over the lazy dog.",
    ]
    all_ok = True
    for t in tests:
        ids = enc.encode(t)
        back = enc.decode(ids)
        ok = back == t
        all_ok = all_ok and ok
        print(f"{'OK ' if ok else 'FAIL'} n_tok={len(ids):3d}  {t!r} -> {ids[:12]}{'...' if len(ids)>12 else ''}")
    print("ALL ROUND-TRIP OK" if all_ok else "ROUND-TRIP FAILURE")
