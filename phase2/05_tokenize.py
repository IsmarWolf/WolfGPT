"""WolfGPT Phase 2 - step 2: turn FineWeb-Edu into token ids on disk.

Writes data_fw/val.bin (never trained on) and data_fw/train.bin as uint16 arrays
(2 bytes per token; fine because vocab < 65,536). Documents are separated by <|endoftext|>.
Usage: python 05_tokenize.py [train_tokens]     default 3e9 (use 2e7 for a quick test)
"""
import sys, time
import numpy as np
from tokenizers import Tokenizer
from common import *

target = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000_000
tok = Tokenizer.from_file(f"{OUT}/tokenizer.json")
eot = tok.token_to_id(EOT)
assert tok.get_vocab_size() < 65536

def encode_to(path, docs, max_tokens=None, batch=2000):
    """Tokenize docs in batches (Rust, multi-threaded) and append to a binary file."""
    n, t0, buf = 0, time.time(), []
    with open(path, "wb") as f:
        def flush():
            nonlocal n, buf
            for enc in tok.encode_batch(buf):
                arr = np.array(enc.ids + [eot], dtype=np.uint16)
                arr.tofile(f); n += len(arr)
            buf = []
            print(f"  {path.split('/')[-1]}: {n/1e6:8.1f}M tokens | {n/(time.time()-t0)/1e6:.2f}M tok/s", flush=True)
        for d in docs:
            buf.append(d)
            if len(buf) == batch:
                flush()
                if max_tokens and n >= max_tokens: break
        if buf and not (max_tokens and n >= max_tokens): flush()
    return n

print("validation set...")
nv = encode_to(f"{OUT}/val.bin", doc_stream(TOK_DOCS, VAL_DOCS))
print("training set...")
nt = encode_to(f"{OUT}/train.bin", doc_stream(TOK_DOCS + VAL_DOCS), max_tokens=target)
print(f"\nDONE: train {nt:,} tokens ({nt*2/1e9:.1f} GB) | val {nv:,} tokens")
