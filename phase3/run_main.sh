#!/usr/bin/env bash
# WolfGPT main run: 110M model, 3B tokens, Muon (best of the pilot). Resumable.
#   bash run_main.sh            start from scratch
#   bash run_main.sh --resume   continue after a crash / reboot
set -e
cd "$(dirname "$0")"
mkdir -p runs
echo "=============================================================="
echo " MAIN RUN: 110M model, 3B tokens. Expected time: about 21-24 hours."
echo " Before starting: Windows sleep OFF, close other programs,"
echo " keep the PC cool and plugged in. Checkpoint about every hour."
echo " After a crash:  bash run_main.sh --resume"
echo "=============================================================="
read -p "Type yes to start: " ans
[ "$ans" = "yes" ] || { echo "cancelled"; exit 1; }
trap '' INT   # Ctrl+C is handled by train.py: it saves a checkpoint, then exits
python -u train.py --compile --size main --tokens 3e9 --micro_batch 8 --eval_every 250 \
  --optimizer muon --muon_lr 0.02 --lr 2e-3 --save --tag main "$@" 2>&1 | tee -a runs/main.log
printf '\a'
echo
echo "Script ended. Read the last line above:"
echo "  'PAUSED at step ...'  -> not finished. Continue with: bash run_main.sh --resume"
echo "  'final val ...'       -> the run is complete."
