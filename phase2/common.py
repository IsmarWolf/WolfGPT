"""Shared helpers for Phase 2. Streams FineWeb-Edu 'sample-10BT' document by document,
so we never store the raw text (only the small tokenized result)."""
import itertools, os

DATASET, CONFIG = "HuggingFaceFW/fineweb-edu", "sample-10BT"
EOT = "<|endoftext|>"          # separator token placed between documents
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_fw")
os.makedirs(OUT, exist_ok=True)

# Document split of the stream (same order every time):
#   [0, TOK_DOCS)                       -> train the tokenizer
#   [TOK_DOCS, TOK_DOCS + VAL_DOCS)     -> validation set (never trained on)
#   [TOK_DOCS + VAL_DOCS, ...)          -> training set
TOK_DOCS, VAL_DOCS = 100_000, 10_000

def doc_stream(skip=0, take=None):
    """Yield document texts [skip, skip+take) from FineWeb-Edu."""
    if os.environ.get("WOLFGPT_FAKE"):                      # offline self-test only
        import random; random.seed(0)
        words = "the of and a to in is that it was for on are with as his they at be this from".split()
        def fake():
            while True:
                yield " ".join(random.choice(words) for _ in range(random.randint(50, 400)))
        return itertools.islice(fake(), skip, None if take is None else skip + take)
    from datasets import load_dataset
    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    it = (d["text"] for d in ds)
    return itertools.islice(it, skip, None if take is None else skip + take)
