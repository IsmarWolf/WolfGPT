#!/usr/bin/env bash
# Phase 5 overnight run, unattended:
#   1. compare a few learning rates, each on a random part of the data (about 50 min each)
#   2. full fine-tune on ALL the data with the best learning rate (about 2.5 h per pass over the data)
# If it is interrupted (reboot, crash, Ctrl+C), just run it again: finished stages are skipped and an
# interrupted stage continues from its last snapshot (saved every 1000 steps).
# Run with the venv active:   bash run_overnight.sh        Output is also saved in phase5/logs/
# Turn off Windows sleep first, and close other programs.
set -euo pipefail
cd "$(dirname "$0")"

LRS="${LRS:-5e-5 1e-4 3e-4}"        # learning rates to compare
SWEEP_N="${SWEEP_N:-150000}"        # training conversations used in each comparison run
FINAL_EPOCHS="${FINAL_EPOCHS:-2}"   # passes over all the data in the final run
EXTRA="${EXTRA:-}"                  # extra sft.py options for every run, e.g. EXTRA="--micro_tokens 3072"
CKPT="${CKPT_DIR:-$HOME/wolfgpt-data/ckpt}"
mkdir -p logs runs

run_stage() {   # run_stage <tag> <sft.py options...>
  local tag="$1"; shift
  if [ -f "runs/$tag.json" ]; then echo "[skip] $tag is already finished"; return; fi
  local resume=""
  if [ -f "$CKPT/${tag}_resume.pt" ]; then resume="--resume"; echo "[resume] $tag continues from its last snapshot"; fi
  echo "=== $(date '+%F %T') start $tag ==="
  python sft.py --tag "$tag" --ckpt_dir "$CKPT" $resume $EXTRA "$@" 2>&1 | tee -a "logs/$tag.log"
}

for lr in $LRS; do
  run_stage "sweep_lr$lr" --lr "$lr" --epochs 1 --max_train "$SWEEP_N"
done

BEST=$(python - <<'PY'
import glob, json, sys
rows = sorted((json.load(open(f)) for f in glob.glob("runs/sweep_lr*.json")), key=lambda r: r["best_val"])
print("\n=== learning-rate comparison (lower validation loss is better) ===", file=sys.stderr)
for r in rows:
    print(f"  lr {r['args']['lr']:g}: best val {r['best_val']:.4f}", file=sys.stderr)
print(rows[0]["args"]["lr"])
PY
)
echo "best learning rate: $BEST" | tee logs/best_lr.txt

run_stage final --lr "$BEST" --epochs "$FINAL_EPOCHS"

printf '\a'
echo
echo "=============================================================="
echo " OVERNIGHT RUN FINISHED."
echo " Best model: $CKPT/final_best.pt   (log: phase5/logs/final.log)"
echo " Chat with it: python chat.py --ckpt $CKPT/final_best.pt"
echo "=============================================================="
