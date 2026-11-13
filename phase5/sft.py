"""Phase 5 step 2: supervised fine-tuning (SFT) of the base WolfGPT on chat data.

  python sft.py                                  # defaults: 2 epochs, lr 1e-4
  python sft.py --lr 5e-5 --tag sft_lr5e-5       # try another learning rate (each run is short)
  python sft.py --epochs 1 --max_steps 20 --eval_every 10 --tag smoke   # quick check

Differences from pretraining: starts from main_best.pt, loss only on answer tokens (prompt tokens are
masked), AdamW at a small learning rate, sequences padded per batch (no torch.compile: shapes vary).
Saves the best-validation model to <ckpt_dir>/<tag>_best.pt.
"""
import argparse, copy, json, math, os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less memory fragmentation with varying shapes
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from chatlib import load_model, encode_prefix, sample, DEFAULT_TOK, HERE

p = argparse.ArgumentParser()
p.add_argument("--base", default=os.path.expanduser("~/wolfgpt-data/ckpt/main_best.pt"))
p.add_argument("--data_dir", default=os.path.expanduser("~/wolfgpt-data/sft"))
p.add_argument("--tokenizer", default=DEFAULT_TOK)
p.add_argument("--ckpt_dir", default=os.path.expanduser("~/wolfgpt-data/ckpt"))
p.add_argument("--tag", default="sft")
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--max_steps", type=int, default=0)         # >0: stop early (for smoke tests)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup_frac", type=float, default=0.03)
p.add_argument("--batch_seqs", type=int, default=32)       # conversations per optimizer step
p.add_argument("--micro_tokens", type=int, default=4096)   # max tokens (incl. padding) per GPU pass: this is what sets memory use. Lower it if out of memory
p.add_argument("--micro", type=int, default=16)            # also at most this many conversations per GPU pass
p.add_argument("--eval_every", type=int, default=200)
p.add_argument("--max_train", type=int, default=0)         # >0: train on only the first N conversations (a random subset): quick learning-rate comparisons
p.add_argument("--ckpt_every", type=int, default=1000)     # save a resumable snapshot every N steps
p.add_argument("--resume", action="store_true")            # continue an interrupted run with the same --tag and settings
p.add_argument("--device", default="cuda")
args = p.parse_args()
dev = args.device
dev_type = dev.split(":")[0]
torch.manual_seed(1337)
rng = np.random.default_rng(1337)
os.makedirs(args.ckpt_dir, exist_ok=True); os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)

tok = Tokenizer.from_file(args.tokenizer)
eot = tok.token_to_id("<|endoftext|>")
model, ck = load_model(args.base, dev)
model.train()
V, BLOCK = model.c.vocab_size, model.c.block_size


def load(name):
    d = {k: np.load(f"{args.data_dir}/{name}_{k}.npy", mmap_mode="r") for k in ("tokens", "mask", "off")}
    d["lens"] = np.diff(d["off"]); d["n"] = len(d["lens"])
    if name == "train" and args.max_train:
        d["n"] = min(d["n"], args.max_train); d["lens"] = d["lens"][:d["n"]]
    d["src"] = np.load(f"{args.data_dir}/{name}_src.npy")
    assert d["lens"].max() <= BLOCK, f"{name}: a conversation is longer than the model context ({BLOCK}); rerun prep_sft.py with --block_size {BLOCK}"
    return d


train, val = load("train"), load("val")
src_names = json.load(open(f"{args.data_dir}/sources.json"))


def pad_len(n):
    """Round up to a multiple of 64 (far fewer distinct shapes, friendlier to the GPU), never above the context."""
    return min(-(-n // 64) * 64, BLOCK)


def chunks(d, idx, budget, cap):
    """Split conversations into GPU passes whose padded size (conversations x longest) stays within `budget`
    tokens, so memory use is predictable whether the conversations are short or long."""
    out, cur, mx = [], [], 0
    for i in idx:
        L = pad_len(int(d["lens"][i]) - 1)
        if cur and (max(mx, L) * (len(cur) + 1) > budget or len(cur) >= cap):
            out.append(cur); cur, mx = [], 0
        cur.append(i); mx = max(mx, L)
    if cur:
        out.append(cur)
    return out


def make_micro(d, idx):
    """Pad a few conversations into one (x, y) pair. y = next token, or -100 where there is no loss."""
    seqs = [(np.asarray(d["tokens"][d["off"][i]:d["off"][i + 1]], dtype=np.int64),
             np.asarray(d["mask"][d["off"][i]:d["off"][i + 1]])) for i in idx]
    L = pad_len(max(len(t) for t, _ in seqs) - 1)
    x = np.zeros((len(seqs), L), np.int64); y = np.full((len(seqs), L), -100, np.int64)
    for k, (t, m) in enumerate(seqs):
        n = len(t) - 1
        x[k, :n] = t[:-1]
        yy = t[1:].copy(); yy[m[1:] == 0] = -100       # the first token of a conversation is never predicted
        y[k, :n] = yy
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


def ce_sum(x, y):
    """Summed loss over answer tokens. Same math as model(x)[0] + cross-entropy, but the big output layer
    (vocab 32,768) is only applied at positions that have a target (prompt positions are skipped)."""
    sel = y != -100
    with torch.autocast(dev_type, dtype=torch.bfloat16):
        h = model.tok_emb(x)
        for blk in model.blocks:
            h = blk(h, model.cos, model.sin)
        logits = model.head(model.norm(h)[sel])
    return F.cross_entropy(logits.float(), y[sel], reduction="sum")


@torch.no_grad()
def evaluate(per_source=False):
    """Token-weighted validation loss over answer tokens (long chats count more). With per_source=True
    also returns the loss of each data source separately."""
    model.eval()
    tot, cnt, by = 0.0, 0, {}
    for s in np.unique(val["src"]):
        idx = np.where(val["src"] == s)[0]; idx = idx[np.argsort(val["lens"][idx])]; st, sc = 0.0, 0
        for c in chunks(val, idx, 8192, 32):                 # no backward pass in evaluation: a bigger budget is fine
            x, y = make_micro(val, c)
            st += ce_sum(x, y).item(); sc += (y != -100).sum().item()
        tot += st; cnt += sc; by[src_names[s]] = (st / max(sc, 1), len(idx))
    model.train()
    return (tot / cnt, by) if per_source else tot / cnt


def show_sources(by):
    print("   per source: " + " | ".join(f"{k} {v[0]:.3f} (n={v[1]})" for k, v in sorted(by.items())))


def epoch_batches():
    """Shuffle, then group conversations of similar length (less padding), then shuffle the batches."""
    perm = rng.permutation(train["n"]); chunk = args.batch_seqs * 50; out = []
    for c in range(0, len(perm), chunk):
        part = perm[c:c + chunk]; part = part[np.argsort(train["lens"][part])]
        out += [part[b:b + args.batch_seqs] for b in range(0, len(part) - args.batch_seqs + 1, args.batch_seqs)]
    return [out[i] for i in rng.permutation(len(out))]


steps_per_epoch = train["n"] // args.batch_seqs
total = min(args.max_steps or 10**9, steps_per_epoch * args.epochs)
warm = max(1, int(args.warmup_frac * total))
lr_at = lambda s: args.lr * ((s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=(dev_type == "cuda"))
if dev_type == "cuda":
    free, tot_mem = torch.cuda.mem_get_info()
    print(f"GPU memory free before training: {free/2**30:.1f} of {tot_mem/2**30:.1f} GB (the rest is used by other programs: close other programs)")
print(f"[{args.tag}] {train['n']:,} train / {val['n']:,} val conversations | {args.epochs} epochs = {total} steps of {args.batch_seqs} conversations | lr {args.lr}"
      + (f" | {args.max_train:,} training conversations used" if args.max_train else ""))

history, best, step = [], float("inf"), 0


def save(name, v):
    path = os.path.join(args.ckpt_dir, f"{args.tag}_{name}.pt")
    torch.save(dict(model=model.state_dict(), cfg=ck["cfg"], step=step, val=v, base=args.base), path + ".tmp")
    os.replace(path + ".tmp", path)


def save_resume(ep, next_batch, epoch_rng):
    """Full snapshot (weights, optimizer, position in the data) so an interrupted run can continue exactly."""
    path = os.path.join(args.ckpt_dir, f"{args.tag}_resume.pt")
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, ep=ep, next_batch=next_batch,
                    epoch_rng=epoch_rng, history=history, best=best, run_loss=run_loss,
                    train_time=train_time, train_tokens=train_tokens), path + ".tmp")
    os.replace(path + ".tmp", path)


run_loss, start_ep, skip_batches = [], 0, 0
t0, done = time.time(), False
train_time, train_tokens = 0.0, 0          # training-only time (evaluation excluded) for speed and ETA
if args.resume:
    r = torch.load(os.path.join(args.ckpt_dir, f"{args.tag}_resume.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"])
    step, start_ep, skip_batches = r["step"], r["ep"], r["next_batch"]
    rng.bit_generator.state = r["epoch_rng"]                 # the data order of the interrupted epoch is rebuilt exactly
    history, best, run_loss = r["history"], r["best"], r["run_loss"]
    train_time, train_tokens = r["train_time"], r["train_tokens"]
    v0 = history[0]["val"]
    print(f"[{args.tag}] RESUMED at step {step} of {total} (best val so far {best:.3f})")
else:
    v0, by0 = evaluate(per_source=True); history.append(dict(step=0, val=v0, by_source={k: v[0] for k, v in by0.items()}))
    print(f"step    0 | val {v0:.3f} (base model, before fine-tuning)"); show_sources(by0)
for ep in range(start_ep, args.epochs):
    epoch_rng = copy.deepcopy(rng.bit_generator.state)       # state before this epoch's shuffles
    for bi, batch in enumerate(epoch_batches()):
        if ep == start_ep and bi < skip_batches:
            continue                                         # already trained before the interruption
        if step >= total:
            done = True; break
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        mbs = [make_micro(train, c) for c in chunks(train, sorted(batch, key=lambda i: train["lens"][i]), args.micro_tokens, args.micro)]
        n_tgt = sum((y != -100).sum().item() for _, y in mbs)           # average over answer tokens of the whole step
        loss_acc = 0.0
        for x, y in mbs:
            l = ce_sum(x, y) / n_tgt
            l.backward(); loss_acc += l.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not (math.isfinite(loss_acc) and torch.isfinite(gn)):
            print(f"step {step}: loss/grad is NaN/inf - stopping"); done = True; break
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if dev_type == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - ts; train_tokens += int(train["lens"][batch].sum())
        if step <= 5:
            print(f"   step {step}: {time.time()-ts:.1f}s for {int(train['lens'][batch].sum()):,} tokens in {len(mbs)} GPU passes")
        run_loss.append(loss_acc)
        if step % args.eval_every == 0 or step == total:
            v = evaluate(); train_avg = sum(run_loss) / len(run_loss); run_loss = []     # train loss averaged since last eval
            history.append(dict(step=step, train=train_avg, val=v))
            tag_best = ""
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev_type == "cuda" else 0.0
            res = torch.cuda.max_memory_reserved() / 2**30 if dev_type == "cuda" else 0.0
            left = int((total - step) * train_time / step); eta = f"{left // 3600}h{left % 3600 // 60:02d}m"   # training time left, evaluation not included
            if v < best:
                best = v; save("best", v); tag_best = " *best*"
            print(f"step {step:5d} | epoch {ep+1} | train {train_avg:.3f} | val {v:.3f} | gnorm {float(gn):.2f} | {train_tokens/train_time/1e3:.1f}k tok/s | mem {mem:.1f} GB used / {res:.1f} GB reserved | ETA {eta}{tag_best}")
        if step % args.ckpt_every == 0 and step < total:
            save_resume(ep, bi + 1, epoch_rng)
    if done:
        break
save("last", history[-1]["val"])
json.dump(dict(args=vars(args), history=history, best_val=best, base_val=v0),
          open(os.path.join(HERE, "runs", f"{args.tag}.json"), "w"), indent=1)
print(f"\nbest val {best:.3f} (base model {v0:.3f}) | saved {args.ckpt_dir}/{args.tag}_best.pt")
if os.path.exists(os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")):
    model, _ = load_model(os.path.join(args.ckpt_dir, f"{args.tag}_best.pt"), dev)
    vb, byb = evaluate(per_source=True); print(f"best model re-checked: val {vb:.3f}"); show_sources(byb)

# ---- what did it learn? Sample the best model on fixed prompts ----------------------------
if os.path.exists(os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")):
    for q in ["hello", "What is the capital of France?", "Explain what photosynthesis is in two sentences.",
              "Give me three tips for studying.", "What is 12 + 15?", "Write a short poem about the sea."]:
        a = sample(model, encode_prefix(tok, q), eot, dev, max_new=120, temperature=0.7)
        print(f"\nUser: {q}\nWolfGPT: {tok.decode(a)}")
