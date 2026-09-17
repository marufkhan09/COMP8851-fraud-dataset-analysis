#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export LD_LIBRARY_PATH="/workspace/pmp_vast/envs/pmp-author/lib/python3.10/site-packages/nvidia/cublas/lib:/workspace/pmp_vast/envs/pmp-author/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/workspace/pmp_vast/envs/pmp-author/lib/python3.10/site-packages/nvidia/cusparse/lib:/workspace/pmp_vast/envs/pmp-author/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
exec "/workspace/pmp_vast/envs/pmp-author/bin/python" "$@"
