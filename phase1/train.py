"""WolfGPT Phase 1 - training loop.   python train.py --data data      (Shakespeare)
                                      python train.py --data data_pt   (Machado de Assis)
Smoke test first:                     python train.py --data data --max_iters 30
"""
import argparse, json, math, os, time
import numpy as np
import torch
from gpt import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")            # folder made by 03_prepare_data*.py
p.add_argument("--out", default=None)               # where to save (default: out_<data>)
p.add_argument("--max_iters", type=int, default=5000)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--patience", type=int, default=4)   # stop after this many evals in a row with no new best val loss
args = p.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(here, args.data)
out_dir = os.path.join(here, args.out or f"out_{args.data}")
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
dev = "cuda"
meta = json.load(open(os.path.join(data_dir, "meta.json")))
chars = meta["chars"]
cfg = GPTConfig(vocab_size=meta["vocab_size"])

train = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)

def get_batch(split):
    """Random windows of block_size chars. y is x shifted by one: at every position the
    target is simply 'the next character'. One window = block_size training examples."""
    d = train if split == "train" else val
    ix = np.random.randint(0, len(d) - cfg.block_size - 1, args.batch_size)
    x = np.stack([d[i:i + cfg.block_size] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

model = GPT(cfg).to(dev)
print(f"parameters: {model.num_params()/1e6:.2f}M | vocab {cfg.vocab_size} | "
      f"untrained loss should be about ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f}")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)

def lr_at(it, warmup=100):
    """Warm up linearly, then decay along a cosine to 10% of the peak."""
    if it < warmup:
        return args.lr * (it + 1) / warmup
    t = (it - warmup) / max(1, args.max_iters - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

@torch.no_grad()
def estimate_loss(n=40):
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = float(np.mean([model(*get_batch(split))[1].item() for _ in range(n)]))
    model.train()
    return out

def sample(prompt="\n", n=400, temperature=0.8):
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([[stoi[c] for c in prompt]], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(ids, n, temperature=temperature, top_k=40)
    model.train()
    return "".join(chars[i] for i in out[0].tolist())

history, t0, best_val, bad_evals = [], time.time(), float("inf"), 0
ckpt = lambda: {"model": model.state_dict(), "cfg": cfg.__dict__, "chars": chars}
for it in range(args.max_iters + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    if it % args.eval_every == 0 or it == args.max_iters:
        l = estimate_loss()
        history.append({"iter": it, **l, "sec": round(time.time() - t0, 1)})
        mark = ""
        if l["val"] < best_val:      # keep the checkpoint with the best validation loss
            best_val, mark, bad_evals = l["val"], "  <- best so far", 0
            torch.save(ckpt(), os.path.join(out_dir, "ckpt_best.pt"))
        else:
            bad_evals += 1
        print(f"iter {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time()-t0:5.0f}s{mark}")
        if bad_evals >= args.patience:
            print(f"early stop: no new best validation loss in {args.patience} evals (best {best_val:.3f})")
            break
    if it == args.max_iters:
        break
    x, y = get_batch("train")
    with torch.autocast("cuda", dtype=torch.bfloat16):    # matmuls in bf16 (fast), the rest stays fp32
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()                                        # backprop: how should each weight change?
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stop one bad batch from wrecking the weights
    opt.step()                                             # AdamW update

dt = time.time() - t0
tps = it * args.batch_size * cfg.block_size / dt
torch.save(ckpt(), os.path.join(out_dir, "ckpt_last.pt"))
model.load_state_dict(torch.load(os.path.join(out_dir, "ckpt_best.pt"))["model"])   # sample from the BEST model
print(f"best validation loss: {best_val:.3f}")
text = sample()
print(f"\ntrained {it} iters in {dt:.0f}s ({tps:,.0f} tokens/s)\n--- sample ---\n{text}\n--------------")
json.dump({"history": history, "tokens_per_sec": tps, "best_val": best_val, "params": model.num_params()},
          open(os.path.join(out_dir, "log.json"), "w"), indent=1)
open(os.path.join(out_dir, "sample.txt"), "w", encoding="utf-8").write(text)
