#!/usr/bin/env bash
# WolfGPT - Phase 0 (inside Ubuntu/WSL2). Run:  bash 02_setup_env.sh
set -euo pipefail

echo "== [1/5] GPU visible from WSL? (driver comes from Windows) =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== [2/5] System packages (only what is missing) =="
need=()
for p in python3-venv python3-pip build-essential git; do
  dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if [ ${#need[@]} -gt 0 ]; then sudo apt-get update -y && sudo apt-get install -y "${need[@]}"; else echo "all present"; fi

echo "== [3/5] Virtual env on the Linux filesystem (fast I/O, not /mnt/c) =="
VENV="$HOME/wolfgpt-env"
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

echo "== [4/5] PyTorch built for CUDA 12.8 (cu128) - required for Blackwell sm_120 =="
if python -c "import torch,sys; sys.exit(0 if torch.version.cuda and tuple(map(int,torch.version.cuda.split('.')[:2]))>=(12,8) else 1)" 2>/dev/null; then
  echo "suitable torch already installed"
else
  pip install torch --index-url https://download.pytorch.org/whl/cu128
fi

echo "== [5/5] Verify =="
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
assert any("sm_120" in a for a in torch.cuda.get_arch_list()), "No sm_120 kernels in this build!"
x = torch.randn(1024,1024,device="cuda"); print("sanity matmul OK:", float((x@x).sum().isfinite()))
PY
echo "Environment OK. Next: python benchmark.py   (after: source ~/wolfgpt-env/bin/activate)"
