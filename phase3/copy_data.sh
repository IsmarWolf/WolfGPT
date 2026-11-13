#!/usr/bin/env bash
# Copy the tokenized data from the Windows drive (slow for Linux) to the Linux disk (fast).
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/phase2/data_fw"
DST="$HOME/wolfgpt-data/data_fw"
mkdir -p "$DST"
echo "copying $SRC -> $DST (6 GB, about a minute)"
cp -v "$SRC"/train.bin "$SRC"/val.bin "$SRC"/tokenizer.json "$DST"/
ls -la "$DST"
