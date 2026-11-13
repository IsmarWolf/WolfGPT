#!/usr/bin/env bash
# Phase 3 pilot: 4 fair runs of the 49M model. Each prints its result and saves runs/<tag>.json.
# Close other programs first. Run from the phase3 folder with the venv active.
set -e
python train.py --compile --size pilot --optimizer adamw --lr 6e-4  --tag adamw_6e-4
python train.py --compile --size pilot --optimizer adamw --lr 2e-3  --tag adamw_2e-3
python train.py --compile --size pilot --optimizer muon  --muon_lr 0.02 --lr 2e-3 --tag muon_0.02
python train.py --compile --size pilot --optimizer muon  --muon_lr 0.05 --lr 2e-3 --tag muon_0.05
python - <<'PY'
import json, glob
rows = sorted((json.load(open(f)) for f in glob.glob("runs/*.json")), key=lambda r: r["final_val"])
print("\n=== pilot ranking (lower final val loss is better) ===")
for r in rows:
    print(f"{r['tag']:14s} val {r['final_val']:.3f} | {r['tok_s']/1e3:5.1f}k tok/s | {r['tflops']:4.1f} TFLOP/s")
PY
printf '\a'
echo
echo "=============================================================="
echo " PILOT FINISHED. The ~24-hour main run has NOT started."
echo " Paste the ranking table above into the chat to choose settings."
echo "=============================================================="
