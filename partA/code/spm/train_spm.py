"""
Part A3 -- train a local SentencePiece BPE tokenizer on the A1 eval corpus itself
(eng+hin+kan+tel combined), since pretrained HF/tiktoken tokenizers are network-blocked
in this environment (see NOTEBOOK.md Day 1 and Day 4).

Trained on the same 4 languages being measured -- deliberately not held out from training,
because the point of A3 is "does a tokenizer actually built with Indic scripts in mind
change the fertility gap", not "how does an unseen-language tokenizer generalize". This
is documented as a limitation in A3_AUDIT.md (in-domain vocab, not a general-purpose
production tokenizer).

vocab_size=16000 chosen as a reasonable size for a ~1.6MB / 6128-line training corpus
across 4 scripts (Latin + Devanagari + Kannada + Telugu) -- large multilingual vocabs
(32k-250k, as in mBERT/XLM-R) assume much larger training corpora; 16k keeps merges
meaningful without over/under-fitting this corpus size. character_coverage=1.0 to avoid
dropping rare Indic characters (default 0.9995 can silently drop conjuncts).
"""
import sentencepiece as spm
import os

HERE = os.path.dirname(__file__)

spm.SentencePieceTrainer.train(
    input=os.path.join(HERE, "train_all.txt"),
    model_prefix=os.path.join(HERE, "indic4_bpe"),
    vocab_size=16000,
    model_type="bpe",
    character_coverage=1.0,
    input_sentence_size=0,
    shuffle_input_sentence=True,
    byte_fallback=True,  # so any char not in vocab still round-trips, no <unk> data loss
)

print("Trained. Files:")
for f in os.listdir(HERE):
    if f.startswith("indic4_bpe"):
        print(" ", f)
