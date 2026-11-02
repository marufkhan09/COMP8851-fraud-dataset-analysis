#!/usr/bin/env bash
# One-time setup on a Vast.ai instance.
# Recommended image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel (torch 2.4.0 preinstalled).
# Usage: bash scripts/vastai_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "== torch"
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.4.0') else 1)" 2>/dev/null; then
  echo "installing torch 2.4.0 (cu121)"
  "$PY" -m pip install --no-cache-dir "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121
fi
CUDA_TAG=$("$PY" -c "import torch; v = torch.version.cuda or ''; print({'12.1': 'cu121', '12.4': 'cu124', '11.8': 'cu118'}.get(v, 'cu121'))")
echo "torch CUDA tag: ${CUDA_TAG}"

echo "== dgl 2.4.0 (${CUDA_TAG})"
"$PY" -m pip install --no-cache-dir "dgl==2.4.0" -f "https://data.dgl.ai/wheels/torch-2.4/${CUDA_TAG}/repo.html"

echo "== common requirements"
"$PY" -m pip install --no-cache-dir -r env/requirements-common.txt

echo "== pinned official implementations"
"$PY" scripts/fetch_official.py

echo "== datasets (hash-verified against the previous benchmark)"
"$PY" scripts/prepare_data.py

echo "== environment report"
mkdir -p results
"$PY" scripts/verify_env.py --require-gpu
"$PY" -m pip freeze > results/pip_freeze.txt
echo "setup complete"
