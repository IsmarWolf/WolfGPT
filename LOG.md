# WolfGPT build log

## 2026-11-08 - Phase 0: Environment
- Checked first, installed only what was missing.
- WSL: was NOT installed -> installed WSL 3.0.3 + VirtualMachinePlatform (reboot needed).
- NVIDIA driver (Windows): 621.14, CUDA UMD 13.5 -> RTX 5060 Ti (Blackwell) needs >= 570: OK, no change.
- Firmware virtualization (SVM): confirmed enabled by user.
- Windows apps held 3.4 GB of 16 GB VRAM (other programs) -> close before benchmarking.
- Rebooted; Ubuntu (WSL2) installed via 01_install_ubuntu.ps1.
- 02_setup_env.sh OK: torch 2.11.0+cu128, Python 3.14, capability 12.0, sm_120 kernels present, sanity matmul OK. Installed numpy 2.5.4.
- Benchmark (2 runs, matched within ~2%), median TFLOP/s:
  - bf16: 47.4-47.7 @4096, 50.1 @8192
  - fp16: 49.0-49.9 @4096, 48.7 @8192
  - tf32: 24.2-24.3 @4096, 25.1 @8192
- Realistic bf16 ceiling: ~50 TFLOP/s. Training MFU of 30-50% => ~15-25 TFLOP/s effective.
- Rough rule: 125M params => ~7.5e8 FLOPs/token (6*N) => ~20-33k tokens/s => ~1.5-2.4B tokens in 21 h.
- Phase 0 gate: PASSED. Next: Phase 1 (tiny GPT by hand).

## 2026-11-08 - Phase 1: tiny GPT by hand (PASSED)
- Model: 10.75M params, 6 layers x 6 heads, width 384, context 256, char-level, bf16, AdamW lr 1e-3, batch 64.
- Speed: ~240k chars/s (~30% of the 50 TFLOP/s bf16 benchmark for a model this small).
- First run (5000 iters, no early stopping): val loss bottomed at 1.469 (~step 1500-2000), then rose to 1.679 while train fell to 0.625 = overfitting. Lesson: judge by VALIDATION loss.
- Added best-checkpoint saving + early stopping (patience 4 evals).
- English (tiny Shakespeare, vocab 65): best val 1.462 @ step 1750, stopped at 2750, 187 s.
- Portuguese (Machado de Assis: Dom Casmurro + Braz Cubas, vocab 107, ORIGINAL 1800s spelling): best val 1.379 @ step 1750, stopped at 2750, 191 s. Learned old spellings (cousa, vêr, vae) and dialogue format; copied character names (José Dias, Escobar) from training text.
- Note: val losses of different datasets/vocabs are not directly comparable.
- Gotcha: device_commit_files silently left an old train.py on disk twice; always verify with `grep` on the real file.

## 2026-11-08/09 - Phase 2: data + tokenizer (DONE)
- Dataset: FineWeb-Edu `sample-10BT` (HuggingFaceFW/fineweb-edu, ODC-By 1.0; web text from Common Crawl). Streamed, raw text never stored.
- Split by document order: first 100k docs -> tokenizer, next 10k -> validation, rest -> training.
- Tokenizer: byte-level BPE, vocab 32,768, special token <|endoftext|>. Trained in 57 s. 4.61 characters/token on unseen text (char-level model = 1.00).
- Tokenized: train 3,002,016,685 tokens (6.0 GB, uint16), val 10,032,689 tokens. Speed ~1.1M tokens/s on the Ryzen 5 5600X, ~45 min total.
- Used a READ-only Hugging Face token (HF_TOKEN env var, never stored in the repo). Added .gitignore for data/model files.
- Decision: English-only main model. Portuguese deferred (needs modern-orthography data; Gutenberg is pre-reform spelling).
- Next: Phase 3 pilot (50M, AdamW vs Muon); move data to the Linux filesystem (~/) for faster reads.

## 2026-11-09 - Phase 3 prep: speed, review, fixes
- Smoke: 10.75M->47.9M pilot model: 56k tok/s eager, 77.7k tok/s with --compile (+38%), memory 11.6 -> 7.2 GB, identical loss. compile needed `sudo apt install python3-dev`.
- Estimated main run: 110M params, ~35k tok/s => ~2.6B tokens in 21 h, full 3B in ~24 h.
- Code review by Opus (fixes in claude/phase3-code-review-fixes.md): checkpoint/resume, shuffled one-pass data, Muon orthogonalizes q/k/v and gate/value separately, Muon weight-decay flag, fixed validation set, honest TFLOP/s (attention term 6LCT).
- Independently verified (CPU): resume after a simulated crash gives bit-identical weights for AdamW and Muon; per-block Muon gives equal q/k/v update norms (old code: q got 3.5x more).
- Numbers from before the fixes (smoke runs) are not comparable with the new validation set.
- Next: run_pilot.sh (4 runs, ~50 min) -> pick optimizer/lr -> main run (use --micro_batch 8 for the 110M model).

## 2026-11-09 - Phase 3 pilot result
- 47.9M model, 52M tokens each. Final val loss: muon_lr 0.02 = 4.418 | muon_lr 0.05 = 4.537 | adamw 6e-4 = 5.280 | adamw 2e-3 = 5.352.
- Muon wins by 0.86 nats (perplexity ~83 vs ~196) for ~4% less speed. Caveat: short run, AdamW grid did not bracket its optimum.
- Speed 90-95k tok/s (29-30 TFLOP/s, with the corrected FLOPs formula). Main run estimate: ~40k tok/s => 3B tokens in ~21-24 h.
- Decision: main run = Muon, muon_lr 0.02, AdamW 2e-3 for embeddings/norms, muon_wd 0, micro_batch 8 (run_main.sh asks for "yes" before starting).
