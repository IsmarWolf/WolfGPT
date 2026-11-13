"""WolfGPT Phase 3 trainer: pilot runs and (later) the main run.

  python train.py --size pilot --optimizer adamw --lr 1e-3 --tag adamw_1e-3
  python train.py --size pilot --optimizer muon  --muon_lr 0.02 --tag muon_0.02
Quick check that everything works:  python train.py --size pilot --steps 10 --eval_every 5 --tag smoke
Main run (3B tokens, steps computed from --tokens; resumable after a crash with --resume):
  python train.py --compile --size main --tokens 3e9 --eval_every 250 --tag main [--resume]
"""
import argparse, json, math, os, signal, sys, time
import numpy as np
import torch
from model import GPT, Config, SIZES
from muon import Muon

p = argparse.ArgumentParser()
p.add_argument("--size", default="pilot", choices=list(SIZES))
p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
p.add_argument("--lr", type=float, default=1e-3)          # AdamW lr (for muon: used for embeddings/norms)
p.add_argument("--muon_lr", type=float, default=0.02)     # Muon lr for the hidden weight matrices
p.add_argument("--muon_wd", type=float, default=0.0)     # Muon weight decay (AdamW matrices use 0.1)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--tokens", type=float, default=None)      # if set, steps = tokens // batch_tokens (use for the main run)
p.add_argument("--warmup", type=int, default=40)          # minimum warm-up steps...
p.add_argument("--warmup_frac", type=float, default=0.01) # ...or this share of all steps, whichever is larger
p.add_argument("--batch_tokens", type=int, default=131072)
p.add_argument("--micro_batch", type=int, default=16)     # sequences per GPU pass; lower it if you run out of VRAM
p.add_argument("--block_size", type=int, default=1024)
p.add_argument("--eval_every", type=int, default=50)
p.add_argument("--val_seqs", type=int, default=320)       # fixed validation windows (independent of micro_batch)
p.add_argument("--data_dir", default=os.path.expanduser("~/wolfgpt-data/data_fw"))
p.add_argument("--ckpt_dir", default=os.path.expanduser("~/wolfgpt-data/ckpt"))   # Linux disk: fast writes
p.add_argument("--ckpt_every", type=int, default=1000)    # full checkpoint (model + optimizer + data position)
p.add_argument("--resume", action="store_true")           # continue from <ckpt_dir>/<tag>_last.pt
p.add_argument("--tag", default=None)
p.add_argument("--compile", action="store_true")          # torch.compile: often faster, optional
p.add_argument("--save", action="store_true")             # save a checkpoint at the end
args = p.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
tag = args.tag or f"{args.size}_{args.optimizer}"
os.makedirs(os.path.join(here, "runs"), exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
if args.tokens:
    args.steps = int(args.tokens // args.batch_tokens)
warmup = max(args.warmup, int(args.warmup_frac * args.steps))
torch.manual_seed(1337); np.random.seed(1337)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

train = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
T = args.block_size
seqs_per_step = args.batch_tokens // T
assert seqs_per_step % args.micro_batch == 0, "batch_tokens/block_size must be divisible by micro_batch"
accum = seqs_per_step // args.micro_batch

def batch_from(data, starts):
    # int32 on the CPU (half the bytes of int64), pinned so the copy really runs in the background
    x = np.stack([data[s:s + T] for s in starts]).astype(np.int32)
    y = np.stack([data[s + 1:s + 1 + T] for s in starts]).astype(np.int32)
    to_gpu = lambda a: torch.from_numpy(a).pin_memory().to(dev, non_blocking=True).long()
    return to_gpu(x), to_gpu(y)

# Training data: every non-overlapping window once per epoch, in shuffled order (sampling with
# replacement would leave ~37% of the tokens unseen in a one-epoch run and repeat others).
data_rng = np.random.default_rng(1337)
train_starts = np.arange(0, len(train) - T - 1, T)
data_rng.shuffle(train_starts)
data_pos, data_epoch = 0, 0

def train_batch():
    global data_pos, data_epoch
    if data_pos + args.micro_batch > len(train_starts):        # finished an epoch: reshuffle
        data_rng.shuffle(train_starts); data_pos, data_epoch = 0, data_epoch + 1
    s = train_starts[data_pos:data_pos + args.micro_batch]
    data_pos += args.micro_batch
    return batch_from(train, s)

# Fixed validation windows: non-overlapping, chosen by seed only, so they are identical at every eval
# and in every run (pilot and main), whatever micro_batch is.
val_windows = np.arange(0, len(val) - T - 1, T)
val_sel = np.sort(np.random.RandomState(0).choice(val_windows, min(args.val_seqs, len(val_windows)), replace=False))
val_starts = [val_sel[i:i + args.micro_batch] for i in range(0, len(val_sel), args.micro_batch)]

cfg = Config(block_size=T, **SIZES[args.size])
model = GPT(cfg).to(dev)
n_params = model.num_params()
fpt = model.flops_per_token(T)
print(f"[{tag}] {args.size}: {n_params/1e6:.1f}M params | batch {args.batch_tokens:,} tokens "
      f"({args.micro_batch} seq x {accum} accum) | {args.steps} steps = {args.steps*args.batch_tokens/1e6:.0f}M tokens "
      f"| warmup {warmup} | train windows {len(train_starts):,} | val windows {len(val_sel)}")
fwd = torch.compile(model) if args.compile else model

# ---- optimizer ---------------------------------------------------------------------------
matrices = [q for n, q in model.blocks.named_parameters() if q.ndim == 2]
others = [q for n, q in model.blocks.named_parameters() if q.ndim != 2] + list(model.norm.parameters())
embed = [model.tok_emb.weight]   # tied with the output head
if args.optimizer == "adamw":
    groups = [dict(params=matrices, lr=args.lr, weight_decay=0.1),
              dict(params=embed, lr=args.lr, weight_decay=0.0),
              dict(params=others, lr=args.lr, weight_decay=0.0)]
    opts = [torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=True)]
else:
    opts = [Muon(matrices, lr=args.muon_lr, weight_decay=args.muon_wd),
            torch.optim.AdamW([dict(params=embed, weight_decay=0.0), dict(params=others, weight_decay=0.0)],
                              lr=args.lr, betas=(0.9, 0.95), fused=True)]
for o in opts:
    for g in o.param_groups:
        g["base_lr"] = g["lr"]

def lr_factor(it):
    """Linear warm-up, then cosine decay to 10% of the peak."""
    if it < warmup:
        return (it + 1) / warmup
    t = (it - warmup) / max(1, args.steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for s in val_starts:
        x, y = batch_from(val, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(fwd(x, y)[1].item())
    model.train()
    return float(np.mean(losses))

# ---- checkpoints -------------------------------------------------------------------------
def save_ckpt(name, step, full=True):
    """Write to a temp file, then rename: a crash mid-write never destroys the previous checkpoint."""
    path = os.path.join(args.ckpt_dir, f"{tag}_{name}.pt")
    state = dict(model=model.state_dict(), cfg=cfg.__dict__, step=step, best=best)
    if full:
        state.update(opts=[o.state_dict() for o in opts], data_pos=data_pos, data_epoch=data_epoch,
                     history=history, t_train=t_train, tokens_done=tokens_done)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)

def write_summary():
    tps = tokens_done / max(t_train, 1e-9)
    summary = dict(tag=tag, args=vars(args), params=n_params, final_val=history[-1]["val"], best_val=best,
                   tok_s=tps, tflops=tps * fpt / 1e12, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                   history=history)
    json.dump(summary, open(os.path.join(here, "runs", f"{tag}.json"), "w"), indent=1)
    return summary

# ---- training loop -------------------------------------------------------------------------
history, best = [], float("inf")
t_train, tokens_done, start_step = 0.0, 0, 1
if args.resume:
    ck = torch.load(os.path.join(args.ckpt_dir, f"{tag}_last.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    for o, s in zip(opts, ck["opts"]):
        o.load_state_dict(s)
    for _ in range(ck["data_epoch"]):            # replay the reshuffles so the data order matches
        data_rng.shuffle(train_starts)
    data_pos, data_epoch = ck["data_pos"], ck["data_epoch"]
    history, best, t_train, tokens_done = ck["history"], ck["best"], ck["t_train"], ck["tokens_done"]
    start_step = ck["step"] + 1
    print(f"resumed from step {ck['step']} (best val {best:.3f})")
else:
    v = evaluate(); history.append(dict(step=0, val=v)); print(f"step    0 | val {v:.3f}")
step, stopped = start_step - 1, False
interrupted = False
def _on_sigint(signum, frame):
    """Ctrl+C: finish the current step, save a full checkpoint, exit cleanly. A second Ctrl+C aborts at once."""
    global interrupted
    if interrupted:
        raise KeyboardInterrupt
    interrupted = True
    print("\nCtrl+C received: finishing this step, then saving a checkpoint...", flush=True)
signal.signal(signal.SIGINT, _on_sigint)
for step in range(start_step, args.steps + 1):
    torch.cuda.synchronize(); t0 = time.time()
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lr_factor(step - 1)
    loss_acc = torch.zeros((), device=dev)       # summed on the GPU: no CPU-GPU sync per micro-batch
    for _ in range(accum):
        x, y = train_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(x, y)
        (loss / accum).backward()
        loss_acc += loss.detach() / accum
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    loss_sum = loss_acc.item()                   # one sync per step
    if not (math.isfinite(loss_sum) and torch.isfinite(gnorm)):   # check BEFORE stepping: weights stay clean
        print(f"step {step}: loss/grad is NaN/inf - stopping (last good checkpoint is kept)")
        stopped = True; break
    for o in opts:
        o.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    if step > start_step + 2:         # skip the first steps: they include GPU/compile warm-up
        t_train += dt; tokens_done += args.batch_tokens
    if step % args.eval_every == 0 or step == args.steps:
        v = evaluate()
        tps = tokens_done / max(t_train, 1e-9)
        tflops = tps * fpt / 1e12
        history.append(dict(step=step, train=loss_sum, val=v, gnorm=float(gnorm), tok_s=tps, tflops=tflops))
        print(f"step {step:4d} | train {loss_sum:.3f} | val {v:.3f} | gnorm {float(gnorm):.2f} | "
              f"{tps/1e3:5.1f}k tok/s | {tflops:4.1f} TFLOP/s | mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
        if v < best:
            best = v
            if args.save or args.ckpt_every < args.steps:   # best-val weights (model only)
                save_ckpt("best", step, full=False)
        write_summary()                         # partial history survives a crash
    if step % args.ckpt_every == 0:
        save_ckpt("last", step)
    if interrupted:
        save_ckpt("last", step)
        write_summary()
        print(f"PAUSED at step {step}/{args.steps}. Checkpoint saved. Resume with the same command plus --resume.", flush=True)
        sys.exit(0)

if args.ckpt_every < args.steps and not stopped:
    save_ckpt("last", step)
summary = write_summary()
tps = summary["tok_s"]
print(f"\n[{tag}] final val {summary['final_val']:.3f} | {tps:,.0f} tokens/s | {summary['tflops']:.1f} TFLOP/s "
      f"(benchmark peak ~50) | -> runs/{tag}.json")
if args.save:
    torch.save(dict(model=model.state_dict(), cfg=cfg.__dict__), os.path.join(here, "runs", f"{tag}.pt"))
