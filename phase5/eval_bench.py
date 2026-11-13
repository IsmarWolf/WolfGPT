"""Phase 6 step 1: zero-shot benchmark scores for WolfGPT checkpoints.

Each question is a multiple choice. The model is NOT given the choices: we compute how likely it finds each
answer text after the question (sum of log-probabilities of the answer tokens) and pick the likeliest. Two scores:
  acc       = pick the answer with the highest total log-probability
  acc_norm  = same, but divided by the answer's length in characters (so longer answers are not unfairly penalised)
Which score is reported per task follows the usual convention (acc_norm for HellaSwag/ARC/PIQA, acc for Winogrande).
Plain text is used (no chat template), the standard way these tests are run, for base and chat models alike.

Usage (WSL, venv active):
  python eval_bench.py --ckpt ~/wolfgpt-data/ckpt/main_best.pt ~/wolfgpt-data/ckpt/wolf2_last.pt
  python eval_bench.py --ckpt <file> --limit 200        # quick check on 200 questions per task
Random guessing scores: HellaSwag 25%, ARC 25%, PIQA 50%, Winogrande 50%.
"""
import argparse, json, math, os, re, sys, time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from chatlib import load_model, DEFAULT_TOK

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]
CHANCE = {"hellaswag": 25, "arc_easy": 25, "arc_challenge": 25, "piqa": 50, "winogrande": 50}
REPORT = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc_norm", "winogrande": "acc"}


# ---------------------------------------------------------------- task loaders -> list of (context, [continuations], label)
def _hs_prep(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def load_hellaswag(limit):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = []
    for ex in ds:
        ctx = _hs_prep(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
        items.append((ctx, [" " + _hs_prep(e) for e in ex["endings"]], int(ex["label"])))
    return items


def _arc(name, limit):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", name, split="test")
    items = []
    for ex in ds:
        labels = list(ex["choices"]["label"])
        if ex["answerKey"] not in labels:
            continue
        items.append((f"Question: {ex['question']}\nAnswer:", [" " + t for t in ex["choices"]["text"]], labels.index(ex["answerKey"])))
    return items


def load_arc_easy(limit): return _arc("ARC-Easy", limit)
def load_arc_challenge(limit): return _arc("ARC-Challenge", limit)


def load_piqa(limit):
    from datasets import load_dataset
    last = None
    for repo in ("ybisk/piqa", "baber/piqa", "nthngdy/piqa"):
        try:
            ds = load_dataset(repo, split="validation")
            return [(f"Question: {ex['goal']}\nAnswer:", [" " + ex["sol1"], " " + ex["sol2"]], int(ex["label"])) for ex in ds]
        except Exception as e:           # a dataset repo may use a loading script the installed datasets version refuses
            last = e
    raise RuntimeError(f"no PIQA copy could be loaded ({last})")


def load_winogrande(limit):
    """Partial scoring: the blank is filled with each option and we score the words AFTER the blank."""
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    items = []
    for ex in ds:
        s = ex["sentence"]; i = s.index("_")
        cont = " " + s[i + 1:].strip()
        # one request per option: (context, [continuation], 0); regrouped below via the 'pair' marker
        items.append(([s[:i] + ex["option1"], s[:i] + ex["option2"]], cont, int(ex["answer"]) - 1))
    return items


LOADERS = dict(hellaswag=load_hellaswag, arc_easy=load_arc_easy, arc_challenge=load_arc_challenge,
               piqa=load_piqa, winogrande=load_winogrande)


# ---------------------------------------------------------------- scoring
def build_requests(items):
    """Flatten items into (item_index, choice_index, context, continuation) requests."""
    reqs = []
    for qi, (ctx, conts, label) in enumerate(items):
        if isinstance(ctx, list):                       # winogrande: different contexts, same continuation
            for ci, c in enumerate(ctx):
                reqs.append((qi, ci, c, conts))
        else:
            for ci, c in enumerate(conts):
                reqs.append((qi, ci, ctx, c))
    return reqs


@torch.no_grad()
def score_requests(model, tok, reqs, dev, batch_tokens=16384):
    """Return {(qi, ci): (sum_logprob, n_chars)}. Sequences are sorted by length and batched by token budget."""
    T = model.c.block_size
    enc = []
    for qi, ci, ctx, cont in reqs:
        c_ids = tok.encode(ctx).ids
        t_ids = tok.encode(cont).ids
        ids = (c_ids + t_ids)[-T:]
        enc.append((qi, ci, ids, min(len(t_ids), len(ids) - 1), len(cont)))
    order = sorted(range(len(enc)), key=lambda k: len(enc[k][2]))
    out, i = {}, 0
    while i < len(order):
        L = len(enc[order[i]][2]); j = i
        while j < len(order) and (j - i + 1) * len(enc[order[j]][2]) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        group = [enc[k] for k in order[i:j]]
        Lmax = max(len(g[2]) for g in group)
        x = torch.zeros(len(group), Lmax, dtype=torch.long)
        for b, g in enumerate(group):
            x[b, :len(g[2])] = torch.tensor(g[2])
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
            logits = model(x.to(dev))[0]
        for b, (qi, ci, ids, n_cont, n_chars) in enumerate(group):
            n = len(ids)
            lp = F.log_softmax(logits[b, n - n_cont - 1:n - 1].float(), dim=-1)
            tgt = torch.tensor(ids[n - n_cont:], device=lp.device)
            out[(qi, ci)] = (lp.gather(1, tgt[:, None]).sum().item(), max(n_chars, 1))
        i = j
    return out


def evaluate_items(model, tok, items, dev):
    scores = score_requests(model, tok, build_requests(items), dev)
    acc = acc_norm = 0
    for qi, (ctx, conts, label) in enumerate(items):
        n = len(ctx) if isinstance(ctx, list) else len(conts)
        lps = [scores[(qi, ci)] for ci in range(n)]
        acc += int(max(range(n), key=lambda c: lps[c][0]) == label)
        acc_norm += int(max(range(n), key=lambda c: lps[c][0] / lps[c][1]) == label)
    return dict(n=len(items), acc=100 * acc / len(items), acc_norm=100 * acc_norm / len(items))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--tokenizer", default=DEFAULT_TOK)
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--limit", type=int, default=0, help="questions per task (0 = all)")
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    tok = Tokenizer.from_file(args.tokenizer)

    data = {}
    for t in args.tasks:
        try:
            items = LOADERS[t](args.limit)
            if args.limit:
                items = items[:args.limit]
            data[t] = items
            print(f"loaded {t}: {len(items)} questions", flush=True)
        except Exception as e:
            print(f"!! could not load {t}: {type(e).__name__}: {e}", flush=True)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for path in args.ckpt:
        name = os.path.basename(path).replace(".pt", "")
        model, _ = load_model(os.path.expanduser(path), dev)
        results[name] = {}
        for t, items in data.items():
            t0 = time.time()
            r = evaluate_items(model, tok, items, dev)
            results[name][t] = r
            print(f"  {name:>14} | {t:<14} | acc {r['acc']:5.1f} | acc_norm {r['acc_norm']:5.1f} | n={r['n']} | {time.time()-t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
        json.dump(results, open(args.out, "w"), indent=1)

    names = [os.path.basename(c).replace(".pt", "") for c in args.ckpt]
    print("\n=== summary (the usual reported score per task; random guessing in the last column) ===")
    print(f"{'task':<15}" + "".join(f"{n:>16}" for n in names) + f"{'random':>9}")
    for t in data:
        print(f"{t:<15}" + "".join(f"{results[n][t][REPORT[t]]:>15.1f}%" for n in names) + f"{CHANCE[t]:>8}%")
    print(f"\nsaved {args.out}")
