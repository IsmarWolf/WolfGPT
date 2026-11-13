"""WolfGPT Phase 1 - step 1: download Shakespeare + build a character-level tokenizer.

Source: the "tiny Shakespeare" file (~1.1 MB, public-domain plays), the standard small baseline.
Outputs (in ./data): input.txt, train.bin, val.bin, meta.json
Safe to re-run: skips the download if input.txt already exists.
"""
import json, os, urllib.request
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
path = os.path.join(DATA, "input.txt")

if not os.path.exists(path):
    print("downloading", URL)
    urllib.request.urlretrieve(URL, path)
text = open(path, encoding="utf-8").read()
print(f"characters: {len(text):,}")

# --- tokenizer: every distinct character gets an integer id -------------------
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}   # string -> int
itos = {i: c for c, i in stoi.items()}       # int -> string
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print(f"vocab size: {len(chars)}  ->  {''.join(chars)!r}")

# round-trip check: encoding then decoding must give the exact same text
sample = text[:100]
assert decode(encode(sample)) == sample
print("sample:", repr(sample[:40]), "->", encode(sample)[:12], "...")

# --- split 90% train / 10% validation (validation = text the model never trains on)
ids = np.array(encode(text), dtype=np.uint16)
n = int(0.9 * len(ids))
ids[:n].tofile(os.path.join(DATA, "train.bin"))
ids[n:].tofile(os.path.join(DATA, "val.bin"))
json.dump({"vocab_size": len(chars), "chars": chars}, open(os.path.join(DATA, "meta.json"), "w"))
print(f"train tokens: {n:,} | val tokens: {len(ids)-n:,}")
