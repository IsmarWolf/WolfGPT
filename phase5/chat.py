"""Chat with the fine-tuned WolfGPT.   python chat.py     (empty line to quit, /reset starts a new chat)

  python chat.py --ckpt ~/wolfgpt-data/ckpt/sft_best.pt --temperature 0.7 --rep_penalty 1.1
"""
import argparse, os
import torch
from tokenizers import Tokenizer
from chatlib import load_model, encode_prefix, encode_response, sample, repeat_score, cut_at_marker, DEFAULT_TOK

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.expanduser("~/wolfgpt-data/ckpt/sft_best.pt"))
p.add_argument("--tokenizer", default=DEFAULT_TOK)
p.add_argument("--temperature", type=float, default=0.7)   # 0 = always the most likely word
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--rep_penalty", type=float, default=0.0)   # 0 = off. If answers loop, try 0.3 (subtracted from used words)
p.add_argument("--no_repeat", type=int, default=0)         # 0 = off. 4: never repeat a 4-token phrase from this or earlier replies
p.add_argument("--retries", type=int, default=0)           # 0 = off. If a reply repeats an earlier one, flag it and try again (up to this many times)
p.add_argument("--show_flags", action="store_true")        # print a [flag: ...] line each time a reply is rejected and retried
p.add_argument("--repeat_thresh", type=float, default=0.3) # share of repeated 4-token phrases above which a reply counts as a repeat
p.add_argument("--max_new", type=int, default=256)
p.add_argument("--cpu", action="store_true")
args = p.parse_args()

dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
tok = Tokenizer.from_file(args.tokenizer)
eot = tok.token_to_id("<|endoftext|>")
model, ck = load_model(args.ckpt, dev)
print(f"loaded {args.ckpt} on {dev} | context {model.c.block_size} tokens (older turns fall out of memory)\n")
replies = []                         # the assistant's earlier replies (tokens), used by --no_repeat
turns = []                           # one token list per finished turn, in exactly the training format
budget = model.c.block_size - args.max_new          # room left for the prompt: the answer needs its own space
while True:
    q = input("You> ")
    q = q.encode("utf-8", "ignore").decode("utf-8", "ignore").strip()   # Windows console can send invalid bytes: drop them
    if not q:
        break
    if q == "/reset":
        turns = []; replies = []; print("(new chat)\n"); continue
    pre = encode_prefix(tok, q)
    while turns and sum(map(len, turns)) + len(pre) > budget:
        turns.pop(0)                 # forget the oldest WHOLE turn, never half of one
    def attempt(k):
        """Attempt 0 = your settings. 1: warmer. 2: warmer and only the last exchange as memory. 3+: warmer and no memory."""
        temp = args.temperature if k == 0 else args.temperature + (0.2 if k < 3 else 0.3)
        hist = turns if k < 2 else (turns[-1:] if k == 2 else [])
        h = list(hist)
        while h and sum(map(len, h)) + len(pre) > budget:
            h.pop(0)
        ids = [t for tr in h for t in tr] + pre[-budget:]
        o = sample(model, ids, eot, dev, args.max_new, temp, args.top_k, args.rep_penalty, args.no_repeat, replies)
        return cut_at_marker(tok, o)

    best, best_score = None, 2.0
    for k in range(args.retries + 1):
        out = attempt(k)
        score = repeat_score(out, replies)
        if score < best_score:
            best, best_score = out, score
        if score < args.repeat_thresh:
            break
        if k < args.retries:
            how = ["warmer sampling", "warmer sampling + short memory", "no memory of the chat"][min(k, 2)]
            if args.show_flags:
                print(f"   [flag: reply repeated itself ({100*score:.0f}% seen before) -> retry {k+1}/{args.retries}: {how}]")
    out = best
    print(f"WolfGPT> {tok.decode(out) if out else '(no answer)'}\n")
    turns.append(pre + out + [eot]); replies.append(out)
