"""WolfGPT Phase 1 - step 1 (Portuguese): Machado de Assis + character-level tokenizer.

Books (Project Gutenberg, public domain in the USA / author died 1908):
  - Dom Casmurro                      (ebook #55752)
  - Memorias Posthumas de Braz Cubas  (ebook #54829)
NOTE: these are the ORIGINAL 1800s editions, so spelling is pre-reform ("Braz", "Posthumas"),
not modern Brazilian orthography. The model will learn that older spelling.
Outputs (in ./data_pt): input.txt, train.bin, val.bin, meta.json. Safe to re-run.
"""
import json, os, urllib.request
import numpy as np

BOOKS = {55752: "Dom Casmurro", 54829: "Memorias Posthumas de Braz Cubas"}
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_pt")
os.makedirs(DATA, exist_ok=True)
path = os.path.join(DATA, "input.txt")

def strip_gutenberg(raw: str) -> str:
    """Keep only the book itself: drop the license header/footer Gutenberg adds."""
    start = raw.find("*** START OF")
    if start != -1:
        raw = raw[raw.find("\n", start) + 1:]
    end = raw.find("*** END OF")
    if end != -1:
        raw = raw[:end]
    return raw.strip()

if __name__ == "__main__":
    if not os.path.exists(path):
        parts = []
        for num, title in BOOKS.items():
            url = f"https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt"
            print("downloading", title, url)
            raw = urllib.request.urlopen(url).read().decode("utf-8-sig")
            parts.append(strip_gutenberg(raw.replace("\r\n", "\n")))
        open(path, "w", encoding="utf-8").write("\n\n".join(parts))
    text = open(path, encoding="utf-8").read()
    print(f"characters: {len(text):,}")

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    print(f"vocab size: {len(chars)}  ->  {''.join(chars)!r}")
    accented = [c for c in chars if ord(c) > 127]
    print("accented/special characters:", ''.join(accented))

    sample = text[:100]
    assert decode(encode(sample)) == sample
    print("sample:", repr(sample[:40]))

    ids = np.array(encode(text), dtype=np.uint16)
    n = int(0.9 * len(ids))
    ids[:n].tofile(os.path.join(DATA, "train.bin"))
    ids[n:].tofile(os.path.join(DATA, "val.bin"))
    json.dump({"vocab_size": len(chars), "chars": chars}, open(os.path.join(DATA, "meta.json"), "w"))
    print(f"train tokens: {n:,} | val tokens: {len(ids)-n:,}")
