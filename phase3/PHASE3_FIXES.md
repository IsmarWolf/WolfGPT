# WolfGPT Phase 3: fixes applied to model.py, muon.py, train.py (2026-11-09)

Originals are kept next to them as `model.py.bak`, `muon.py.bak`, `train.py.bak`.
`run_pilot.sh` is unchanged and still works.

## What was already correct (checked, not changed)
RoPE, causal attention, weight tying (one tensor, on AdamW), every parameter in exactly one
optimizer group, the Muon math, gradient accumulation and clipping, and no overlap between the
train/validation/tokenizer documents.

## What changed and why

1. **Checkpoints and resume** (train.py). Before, a crash at hour 20 lost the whole run.
   - Every `--ckpt_every` steps (default 1000, about 1 h) it saves model, optimizer and data position
     to `~/wolfgpt-data/ckpt/<tag>_last.pt`, on the Linux disk because it's fast. It writes a temp
     file and then renames it, so a crash during the save can't corrupt the previous checkpoint.
   - It also keeps the best-validation weights in `<tag>_best.pt`.
   - `--resume` continues from `<tag>_last.pt`. Tested: crash at step 20 plus resume gives
     bit-identical weights to an uninterrupted run, with both AdamW and Muon, and also across an
     epoch boundary.
   - The NaN/inf check now runs BEFORE the optimizer step, so a bad step never damages the weights
     or the checkpoint.
   - `runs/<tag>.json` is rewritten at every eval, so the history survives a crash.
   - Pilot runs (400 steps) write no checkpoints, because 400 < 1000.

2. **Data is read once, shuffled** (train.py). Before, random start positions with repeats meant
   ~37% of the 3B tokens were never seen and ~26% were seen 2+ times. Now every non-overlapping
   1024-token window is used once per epoch, in a shuffled order (seed 1337). The main run's
   22,888 steps use 2,929,664 of 2,931,656 windows: almost exactly one pass.

3. **Muon orthogonalizes fused weights per block** (muon.py, model.py). `qkv` is 3 matrices and
   SwiGLU `up` is 2 (gate + value). Muon used to treat each as one big matrix, so the largest block
   (e.g. q) took most of the update. In a test where q's gradient was 10x larger, q got an update
   3.5x bigger than k and v; now all three get the same. model.py marks the weights with
   `muon_split = 3` / `= 2`, and Newton-Schulz now handles a batch of matrices.
   This changes the effective step size (qkv used to get a sqrt(3) boost), so **the Muon
   learning-rate pilot must be rerun**.

4. **Muon weight decay flag** `--muon_wd` (default 0, so behavior is the same as before).
   AdamW matrices use wd 0.1, so the two optimizers weren't being compared equally. A 400-step
   pilot is too short to judge weight decay. Untested suggestion for the main run:
   `--muon_wd 0.005` at `--muon_lr 0.02`, which shrinks weights by the same amount per step as
   AdamW (1e-3 x 0.1).

5. **Run length and warmup** (train.py). `--tokens 3e9` sets steps = tokens // batch_tokens
   (22,888). Warmup = max(`--warmup` 40, 1% of steps), so 228 steps for the main run and still 40
   for the pilots. Before, the main run would have had a 40-step warmup, and forgetting `--steps`
   meant a 52M-token run.

6. **Validation set is fixed** (train.py). `--val_seqs 320` non-overlapping windows, chosen only by
   seed 0. Before, the set depended on `--micro_batch`, so lowering it for VRAM changed the
   validation data. `--eval_batches` was removed. Validation numbers will differ slightly from the
   earlier smoke runs; compare only runs made with the new code.

7. **Less GPU idle time** (train.py). The loss is summed on the GPU, with one sync per step instead
   of one per micro-batch. Batches are copied as int32 from pinned memory, so `non_blocking`
   actually works.

8. **Honest TFLOP/s** (model.py). The attention term is now 6*L*C*T, not 12*L*C*T: causal
   attention skips half the work. Speeds read about 8-10% lower; the real speed hasn't changed.

## Commands
Pilot (unchanged): `bash run_pilot.sh`, then pick the best optimizer and learning rate.

Main run (about 24 h; turn off Windows sleep, close other programs):
```
python train.py --compile --size main --tokens 3e9 --eval_every 250 \
    --optimizer <best> --muon_lr <best> --lr <best> --tag main
```
After a crash, run the same command with `--resume` added. Final weights: `--save` writes
`runs/main.pt`; best-validation weights are always in `~/wolfgpt-data/ckpt/main_best.pt`.

## Not changed (optional)
- The learning rate still decays to 10% of peak. Decaying to ~0 usually helps the final loss a
  little.
- No full-validation eval at the end yet. Running all ~9.8k windows once would give a more
  precise final number.
