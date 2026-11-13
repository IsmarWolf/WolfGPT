"""Chat with Llama-WolfGPT-110M (a 110M-parameter GPT trained from scratch).

  pip install torch tokenizers safetensors
  python chat.py                      (empty line quits, /reset starts a new chat)
  python chat.py --temperature 0.7 --retries 0 --show_flags

The files model.py, config.json, model.safetensors and tokenizer.json must be in the same folder as this script.
"""
import argparse, json, os, sys
import torch
from tokenizers import Tokenizer
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import GPT, Config                      # noqa: E402


def load(folder, dev):
    """Load the model from config.json + model.safetensors. The output layer shares the embedding table, so
    'head.weight' is not stored in the file (it is tied when the model is built)."""
    cfg = json.load(open(os.path.join(folder, "config.json")))
    model = GPT(Config(**{k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}))
    missing, unexpected = model.load_state_dict(load_file(os.path.join(folder, "model.safetensors")), strict=False)
    assert not unexpected and set(missing) == {"head.weight"}, f"unexpected weights: {unexpected} / missing: {missing}"
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), "weight tying lost"
    return model.to(dev).eval(), cfg


def encode_prefix(tok, text):
    return tok.encode(f"### User:\n{text.replace('<|endoftext|>', '<endoftext>').strip()}\n\n### Assistant:\n").ids


@torch.no_grad()
def sample(model, ids, eot, dev, max_new=256, temperature=0.7, top_k=40, rep_penalty=0.0, no_repeat=0, avoid=()):
    """Continue token list `ids`; returns new tokens (without the final end token)."""
    T = model.c.block_size
    x = torch.tensor([ids], device=dev)
    out, seen = [], {}
    if no_repeat:
        for seq in avoid:
            for k in range(len(seq) - no_repeat + 1):
                seen.setdefault(tuple(seq[k:k + no_repeat - 1]), set()).add(seq[k + no_repeat - 1])
    for _ in range(max_new):
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
            logits = model(x[:, -T:])[0][:, -1, :].float()
        if rep_penalty and out:
            logits[0, torch.tensor(sorted(set(out)), device=dev)] -= rep_penalty
        if no_repeat and len(out) >= no_repeat - 1:
            banned = [b for b in seen.get(tuple(out[len(out) - no_repeat + 1:]) if no_repeat > 1 else (), ()) if b != eot]
            if banned and len(banned) < logits.size(-1):
                logits[0, torch.tensor(banned, device=dev)] = float("-inf")
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        t = nxt.item()
        if t == eot:
            break
        if no_repeat and len(out) >= no_repeat - 1:
            seen.setdefault(tuple(out[len(out) - no_repeat + 1:]) if no_repeat > 1 else (), set()).add(t)
        out.append(t)
        x = torch.cat([x, nxt], dim=1)
    return out


def repeat_score(out, replies, n=4):
    """Share of this answer's n-token phrases already seen in earlier replies or earlier in this answer (1 = identical)."""
    if not out:
        return 1.0
    if any(out == r for r in replies):
        return 1.0
    seen = set()
    for r in replies:
        seen.update(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    grams = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
    if not grams:
        return 0.0
    hit = 0
    for g in grams:
        hit += g in seen
        seen.add(g)
    return hit / len(grams)


def cut_at_marker(tok, out, marker="###"):
    """Keep only what comes before a '###' (the model starting a new '### User:' block instead of stopping)."""
    text = tok.decode(out)
    if marker not in text:
        return out
    head = text.split(marker)[0]
    return tok.encode(head.rstrip()).ids if head.strip() else []


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=HERE)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--rep_penalty", type=float, default=0.3)
    p.add_argument("--no_repeat", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--repeat_thresh", type=float, default=0.3)
    p.add_argument("--show_flags", action="store_true")
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    tok = Tokenizer.from_file(os.path.join(args.dir, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    model, cfg = load(args.dir, dev)
    print(f"Llama-WolfGPT-110M on {dev} | context {model.c.block_size} tokens | empty line quits, /reset = new chat\n")
    replies, turns = [], []
    budget = model.c.block_size - args.max_new
    while True:
        q = input("You> ").encode("utf-8", "ignore").decode("utf-8", "ignore").strip()
        if not q:
            break
        if q == "/reset":
            turns, replies = [], []; print("(new chat)\n"); continue
        pre = encode_prefix(tok, q)

        def attempt(k):
            temp = args.temperature if k == 0 else args.temperature + (0.2 if k < 3 else 0.3)
            h = list(turns if k < 2 else (turns[-1:] if k == 2 else []))
            while h and sum(map(len, h)) + len(pre) > budget:
                h.pop(0)
            ids = [t for tr in h for t in tr] + pre[-budget:]
            return cut_at_marker(tok, sample(model, ids, eot, dev, args.max_new, temp, args.top_k,
                                             args.rep_penalty, args.no_repeat, replies))

        best, best_score = None, 2.0
        for k in range(args.retries + 1):
            out = attempt(k)
            score = repeat_score(out, replies)
            if score < best_score:
                best, best_score = out, score
            if score < args.repeat_thresh:
                break
            if args.show_flags and k < args.retries:
                print(f"   [flag: reply repeated itself ({100*score:.0f}% seen before) -> retry {k+1}/{args.retries}]")
        out = best
        print(f"WolfGPT> {tok.decode(out) if out else '(no answer)'}\n")
        turns.append(pre + out + [eot]); replies.append(out)
