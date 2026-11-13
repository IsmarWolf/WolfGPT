"""Phase 5 step 1b: build the identity mini fine-tune data, in the same on-disk format as prep_sft.py.

It reads the dataset prep_sft.py already wrote (--src_dir) and writes a new folder (--out) that sft.py
reads unchanged:
  train: --mix_n random training conversations from src (keeps general ability)
         + the identity conversations, each repeated --id_rep times (source name "identity")
  val:   --val_orig conversations sampled from src's validation set
         + the last --val_id identity conversations, held out (removed from train)
Files: {train,val}_{tokens,mask,off,src}.npy + sources.json + sources.txt (same dtypes as prep_sft.py).

Usage:  python prep_identity.py
        python prep_identity.py --src_dir ~/wolfgpt-data/sft --out ~/wolfgpt-data/sft_id
"""
import argparse, collections, json, os, random
import numpy as np
from tokenizers import Tokenizer
from chatlib import encode_prefix, encode_response, DEFAULT_TOK
from identity_data import CONVS

p = argparse.ArgumentParser()
p.add_argument("--src_dir", default=os.path.expanduser("~/wolfgpt-data/sft"))
p.add_argument("--out", default=os.path.expanduser("~/wolfgpt-data/sft_id"))
p.add_argument("--tokenizer", default=DEFAULT_TOK)
p.add_argument("--mix_n", type=int, default=10000)        # training conversations sampled from src (keeps general ability)
p.add_argument("--id_rep", type=int, default=3)           # copies of each identity conversation in TRAIN
p.add_argument("--val_orig", type=int, default=900)       # validation conversations sampled from src's validation set
p.add_argument("--val_id", type=int, default=20)          # identity conversations held out: the LAST ones in CONVS (after dedup)
p.add_argument("--block_size", type=int, default=1024)    # same as prep_sft.py; must not exceed the model's context
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
tok = Tokenizer.from_file(args.tokenizer)
eot = tok.token_to_id("<|endoftext|>")


def encode_conv(pairs):
    """Same as prep_sft.py: tokens + mask for one conversation. Turns that do not fit in block_size are cut off whole."""
    ids, mask = [], []
    for u, a in pairs:
        pre, resp = encode_prefix(tok, u), encode_response(tok, a, eot)
        if len(ids) + len(pre) + len(resp) > args.block_size:
            break
        ids += pre + resp
        mask += [0] * len(pre) + [1] * len(resp)
    return (ids, mask) if ids else None


# ---- 1. identity conversations: check, remove repeated first prompts (same key as prep_sft.py), encode
seen, id_convs, n_dup = set(), [], 0
for c in CONVS:
    assert c and all(isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip() for u, a in c), f"bad conversation: {c!r}"
    k = " ".join(c[0][0].lower().split())
    if k in seen:
        n_dup += 1; continue
    seen.add(k); id_convs.append((k, c))
id_enc = []
for k, c in id_convs:
    e = encode_conv(c)
    if e:
        id_enc.append((e[0], e[1], k))
n_drop = len(id_convs) - len(id_enc)
cut = len(id_enc) - args.val_id
assert cut > 0, f"only {len(id_enc)} identity conversations: not enough for {args.val_id} held out"
id_train, id_val = id_enc[:cut], id_enc[cut:]
assert not {k for _, _, k in id_train} & {k for _, _, k in id_val}, "held-out identity conversation also in train"
print(f"identity: {len(CONVS)} written | {n_dup} repeated first prompts removed | {n_drop} dropped (first turn longer than {args.block_size} tokens)")
print(f"identity: {len(id_train)} for train (x{args.id_rep} = {len(id_train)*args.id_rep}) | {len(id_val)} held out for validation (the last ones)")

# ---- 2. the existing prepared data (written by prep_sft.py)
src_names = json.load(open(f"{args.src_dir}/sources.json"))


def load_split(name):
    d = dict(tokens=np.load(f"{args.src_dir}/{name}_tokens.npy", mmap_mode="r"),
             mask=np.load(f"{args.src_dir}/{name}_mask.npy", mmap_mode="r"),
             off=np.load(f"{args.src_dir}/{name}_off.npy"),
             src=np.load(f"{args.src_dir}/{name}_src.npy"))
    d["n"] = len(d["src"])
    assert len(d["off"]) == d["n"] + 1, f"{name}: off and src sizes do not match"
    d["lens"] = np.diff(d["off"])
    return d


def src_rec(d, i):
    a, b = int(d["off"][i]), int(d["off"][i + 1])
    return (np.array(d["tokens"][a:b], dtype=np.uint16), np.array(d["mask"][a:b], dtype=np.uint8), src_names[int(d["src"][i])])


tr, va = load_split("train"), load_split("val")
tr_pool = [i for i in range(tr["n"]) if tr["lens"][i] <= args.block_size]     # drop rule, as in sft.py's assert
va_pool = [i for i in range(va["n"]) if va["lens"][i] <= args.block_size]
mix_idx = random.Random(args.seed).sample(tr_pool, min(args.mix_n, len(tr_pool)))
val_idx = random.Random(args.seed + 2).sample(va_pool, min(args.val_orig, len(va_pool)))
print(f"src: {tr['n']:,} train / {va['n']:,} val conversations | sampled {len(mix_idx):,} train and {len(val_idx):,} validation")

# ---- 3. assemble train and validation (identity is added AFTER sampling; no held-out identity reaches train)
tr_recs = [src_rec(tr, i) for i in mix_idx]
for ids, mask, _ in id_train:
    for _ in range(args.id_rep):
        tr_recs.append((np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8), "identity"))
random.Random(args.seed + 3).shuffle(tr_recs)
va_recs = [src_rec(va, i) for i in val_idx]
for ids, mask, _ in id_val:
    va_recs.append((np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8), "identity"))
random.Random(args.seed + 4).shuffle(va_recs)

names = sorted(set(r[2] for r in tr_recs + va_recs) | {"identity"})
assert len(names) < 256
top = max(int(r[0].max()) for r in tr_recs + va_recs if len(r[0]))
assert top < tok.get_vocab_size(), f"token id {top} is outside the tokenizer vocabulary: wrong --tokenizer?"


def save(name, recs):
    """Same files and dtypes as prep_sft.py.save()."""
    off = np.zeros(len(recs) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(r[0]) for r in recs])
    tokens = np.concatenate([r[0] for r in recs]).astype(np.uint16)
    mask = np.concatenate([r[1] for r in recs]).astype(np.uint8)
    np.save(f"{args.out}/{name}_tokens.npy", tokens)
    np.save(f"{args.out}/{name}_mask.npy", mask)
    np.save(f"{args.out}/{name}_off.npy", off)
    np.save(f"{args.out}/{name}_src.npy", np.array([names.index(r[2]) for r in recs], dtype=np.uint8))
    tot, sup = int(off[-1]), int(mask.sum())
    print(f"  {name}: {len(recs):,} conversations | {tot/1e6:.2f}M tokens | {100*sup/tot:.0f}% are answer tokens (loss) | mean length {tot/len(recs):.0f}")


save("train", tr_recs)
save("val", va_recs)
json.dump(names, open(f"{args.out}/sources.json", "w"))

print("train conversations per source:", dict(collections.Counter(r[2] for r in tr_recs).most_common()))
print("validation conversations per source:", dict(collections.Counter(r[2] for r in va_recs).most_common()))

# ---- 4. one example (answer tokens in [[double brackets]]), as in prep_sft.py
ids, mask, _ = id_train[0]
print("\nExample (answer tokens in [[double brackets]]):")
out, cur, on = "", [], 0
for t, m in zip(ids, mask):
    if m != on:
        out += tok.decode(cur) + ("[[" if m else "]]"); cur, on = [], m
    cur.append(t)
print((out + tok.decode(cur) + ("]]" if on else ""))[:900])

src_txt = open(f"{args.src_dir}/sources.txt").read() if os.path.exists(f"{args.src_dir}/sources.txt") else ""
with open(f"{args.out}/sources.txt", "w") as f:
    f.write("Sources: " + str(dict(collections.Counter(r[2] for r in tr_recs + va_recs))) + "\n"
            + "identity: hand-written for the WolfGPT identity fine-tune (identity_data.py). No external data.\n"
            + "Other sources: copied from " + args.src_dir + "/sources.txt\n" + src_txt)
