"""Phase 5 step 1: build the instruction-tuning data (tokenized, with a mask saying which tokens count).

Sources (all open licenses):
  - HuggingFaceTB/smol-smoltalk   Apache-2.0, synthetic chats made for models under 1B parameters (all ~460k by default)
  - HuggingFaceTB/smoltalk, config everyday-conversations   Apache-2.0, ~2.4k greetings / small talk
  - databricks/databricks-dolly-15k   CC BY-SA 3.0, 15k questions/answers written by humans

Output in --out (default ~/wolfgpt-data/sft/): {train,val}_{tokens,mask,off,src}.npy + sources.json
  tokens: all conversations concatenated (uint16) | mask: 1 = the model must learn to write this token,
  0 = prompt (given, no loss) | off: start offset of each conversation | src: which source it came from.
Usage:  python prep_sft.py                     (downloads ~1 GB once)
        python prep_sft.py --smol_n 50000      (smaller)
"""
import argparse, collections, json, os, random
import numpy as np
from tokenizers import Tokenizer
from chatlib import encode_prefix, encode_response, DEFAULT_TOK

p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.expanduser("~/wolfgpt-data/sft"))
p.add_argument("--tokenizer", default=DEFAULT_TOK)
p.add_argument("--smol_n", type=int, default=10**9)       # conversations taken from smol-smoltalk (default: all)
p.add_argument("--no_everyday", action="store_true")
p.add_argument("--everyday_rep", type=int, default=3)     # copies of each everyday chat in TRAIN only (it is tiny)
p.add_argument("--no_dolly", action="store_true")
p.add_argument("--val_n", type=int, default=1000)
p.add_argument("--block_size", type=int, default=1024)    # must not exceed the model's context
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--fake", type=int, default=0)             # offline self-test: N random conversations
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
tok = Tokenizer.from_file(args.tokenizer)
eot = tok.token_to_id("<|endoftext|>")


def pairs_from_messages(msgs, use_system):
    """Keep user->assistant pairs in order. If use_system, the system message is put in front of the
    FIRST user turn (in the rewrite/summarize subsets the task itself lives in the system message)."""
    pairs, user, system, used = [], None, "", False
    for m in msgs:
        r, c = m.get("role"), (m.get("content") or "")
        if r == "system":
            system = c.strip()
        elif r == "user":
            user = c.strip()
            if use_system and system and not pairs:
                user, used = f"{system}\n\n{user}", True
        elif r == "assistant" and user and c.strip():
            pairs.append((user, c)); user = None            # answers keep their indentation (see encode_response)
    return pairs, used


convs, srcs, n_sys = [], [], 0          # srcs[i] = source name of convs[i]
if args.fake:
    rnd = random.Random(0)
    w = "the of and a to in is that it was for on are with as his they at be this from water heart book".split()
    sent = lambda n: " ".join(rnd.choice(w) for _ in range(n)) + "."
    for k in range(args.fake):
        convs.append([(sent(rnd.randint(3, 15)) + f" #{k}", sent(rnd.randint(5, 40))) for _ in range(rnd.randint(1, 3))])
        srcs.append("fake-a" if k % 2 else "fake-b")
else:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train").shuffle(seed=args.seed)
    ds = ds.select(range(min(args.smol_n, len(ds))))
    has_src = "source" in ds.column_names
    for ex in ds:
        src = (ex.get("source") if has_src else None) or "smol"
        # rewrite/summarize conversations: the instruction is in the system message, so keep it.
        # If the dataset has no source column we cannot tell, so keep all system messages.
        use_sys = (not has_src) or any(k in src.lower() for k in ("rewrite", "summar"))
        pr, used = pairs_from_messages(ex["messages"], use_sys)
        if pr:
            convs.append(pr); srcs.append(src); n_sys += used
    print("smol-smoltalk sources found:", dict(collections.Counter(srcs).most_common()))
    print(f"system message put in front of the first user turn in {n_sys:,} conversations")
    if has_src and n_sys == 0:
        print("WARNING: no source name contains 'rewrite' or 'summar'; check the names above and adjust use_sys")
    if not args.no_everyday:              # greetings / small talk: almost absent from smol-smoltalk
        for ex in load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train"):
            pr, _ = pairs_from_messages(ex["messages"], False)
            if pr:
                convs.append(pr); srcs.append("everyday-conversations")
    if not args.no_dolly:
        for ex in load_dataset("databricks/databricks-dolly-15k", split="train"):
            q = ex["instruction"].strip() + (("\n\n" + ex["context"].strip()) if ex["context"].strip() else "")
            if q and ex["response"].strip():
                convs.append([(q, ex["response"])]); srcs.append("dolly")

# Remove duplicate prompts BEFORE the train/validation split (a repeated prompt on both sides would
# make the validation loss look better than it is). Key = the first user turn.
seen, keep = set(), []
for i, c in enumerate(convs):
    k = " ".join(c[0][0].lower().split())
    if srcs[i] == "everyday-conversations":          # many start with "Hi"; keep the different answers
        k = "everyday|" + k + "|" + " ".join(c[0][1].lower().split())
    if k not in seen:
        seen.add(k); keep.append(i)
print(f"dedup: {len(convs):,} -> {len(keep):,} conversations ({len(convs)-len(keep):,} repeated first prompts removed)")
convs, srcs = [convs[i] for i in keep], [srcs[i] for i in keep]

order = list(range(len(convs)))
random.Random(args.seed).shuffle(order)
convs, srcs = [convs[i] for i in order], [srcs[i] for i in order]
names = sorted(set(srcs))
assert len(names) < 256


def encode_conv(pairs):
    """Tokens + mask for one conversation. Turns that do not fit in block_size are cut off whole."""
    ids, mask = [], []
    for u, a in pairs:
        pre, resp = encode_prefix(tok, u), encode_response(tok, a, eot)
        if len(ids) + len(pre) + len(resp) > args.block_size:
            break
        ids += pre + resp
        mask += [0] * len(pre) + [1] * len(resp)
    return (ids, mask) if ids else None


enc = [(e, names.index(s)) for e, s in ((encode_conv(c), s) for c, s in zip(convs, srcs)) if e]
print(f"{len(convs):,} conversations -> {len(enc):,} kept ({len(convs)-len(enc):,} dropped: first turn longer than {args.block_size} tokens)")
val, train = enc[:args.val_n], enc[args.val_n:]
if "everyday-conversations" in names and args.everyday_rep > 1:   # repeat AFTER the split: no copy reaches val
    ev = [e for e in train if names[e[1]] == "everyday-conversations"]
    train = train + ev * (args.everyday_rep - 1)
    random.Random(args.seed + 1).shuffle(train)
    print(f"everyday-conversations: {len(ev):,} in train, repeated x{args.everyday_rep} -> {len(ev)*args.everyday_rep:,}")


def save(name, items):
    off = np.zeros(len(items) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(i[0][0]) for i in items])
    np.save(f"{args.out}/{name}_tokens.npy", np.fromiter((t for i in items for t in i[0][0]), dtype=np.uint16, count=int(off[-1])))
    np.save(f"{args.out}/{name}_mask.npy", np.fromiter((m for i in items for m in i[0][1]), dtype=np.uint8, count=int(off[-1])))
    np.save(f"{args.out}/{name}_off.npy", off)
    np.save(f"{args.out}/{name}_src.npy", np.array([i[1] for i in items], dtype=np.uint8))
    tot, sup = int(off[-1]), sum(sum(i[0][1]) for i in items)
    print(f"  {name}: {len(items):,} conversations | {tot/1e6:.1f}M tokens | {100*sup/tot:.0f}% are answer tokens (loss) | mean length {tot/len(items):.0f}")


save("train", train)
save("val", val)
json.dump(names, open(f"{args.out}/sources.json", "w"))
print("  validation conversations per source:", dict(collections.Counter(names[i[1]] for i in val).most_common()))
(ids, mask), _ = enc[0]
print("\nExample (answer tokens in [[double brackets]]):")
out, cur, on = "", [], 0
for t, m in zip(ids, mask):
    if m != on:
        out += tok.decode(cur) + ("[[" if m else "]]"); cur, on = [], m
    cur.append(t)
print((out + tok.decode(cur) + ("]]" if on else ""))[:900])
with open(f"{args.out}/sources.txt", "w") as f:
    f.write("Sources: " + str(dict(collections.Counter(srcs))) + "\nsmol-smoltalk and smoltalk/everyday-conversations: Apache-2.0 (HuggingFaceTB). databricks-dolly-15k: CC BY-SA 3.0 (Databricks, Inc.).\n")
