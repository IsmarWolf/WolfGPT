"""Shared helpers for Phase 5: the chat format, model loading and sampling.

Chat format (plain text, no new tokens, so the model/vocab stay unchanged):

    ### User:
    <question>

    ### Assistant:
    <answer><|endoftext|>

A multi-turn chat is these blocks one after another. The model learns to end its answer with
<|endoftext|>, which is how generation knows when to stop.
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))
from model import GPT, Config                      # noqa: E402

DEFAULT_TOK = os.path.expanduser("~/wolfgpt-data/data_fw/tokenizer.json")


def _safe(text):
    """The literal text <|endoftext|> inside data would be turned into the stop token: defuse it."""
    return text.replace("<|endoftext|>", "<endoftext>")


def encode_prefix(tok, user_text):
    """Everything the model is GIVEN for one turn (no loss on these tokens)."""
    return tok.encode(f"### User:\n{_safe(user_text).strip()}\n\n### Assistant:\n").ids


def encode_response(tok, text, eot):
    """What the model must WRITE for one turn (loss on these tokens), ending with <|endoftext|>.
    Only leading blank lines and trailing space are removed, so an answer that starts with
    indented code keeps its indentation."""
    return tok.encode(_safe(text).lstrip("\n").rstrip()).ids + [eot]


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), ck


@torch.no_grad()
def sample(model, ids, eot, dev, max_new=256, temperature=0.7, top_k=40, rep_penalty=0.0,
           no_repeat=0, avoid=()):
    """Continue token list `ids`; returns the new tokens (without the final <|endoftext|>).
    rep_penalty: fixed amount subtracted from the score of every token already used in this answer
    (0 = off; try 0.3 only if you see loops). Callers must keep len(ids) + max_new within the context.
    no_repeat: if n > 0, a word-piece sequence of n tokens that already appeared in this answer or in one of the
    token lists in `avoid` (the assistant's earlier replies) may not appear again: the last n-1 tokens are
    looked up and any token that would complete a seen n-gram is forbidden (0 = off; 4 is a good start)."""
    T = model.c.block_size
    x = torch.tensor([ids], device=dev)
    out = []
    seen = {}                                   # (n-1 tokens) -> set of tokens that followed them before
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
            banned = seen.get(tuple(out[len(out) - no_repeat + 1:]) if no_repeat > 1 else (), ())
            banned = [b for b in banned if b != eot]
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
    """How much of this answer is a repeat: the share of its n-token phrases that already appeared in an earlier
    reply or earlier in this same answer (0 = all new, 1 = all repeated). An answer identical to an earlier one is 1."""
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
    """The model sometimes starts a new '### User:' block instead of stopping (and then writes junk): keep only
    what comes before the marker. Returns the (possibly shortened) token list."""
    text = tok.decode(out)
    if marker not in text:
        return out
    return tok.encode(text.split(marker)[0].rstrip()).ids if text.split(marker)[0].strip() else []
