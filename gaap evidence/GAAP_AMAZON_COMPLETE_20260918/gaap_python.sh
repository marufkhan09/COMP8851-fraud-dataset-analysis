#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH="/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cublas/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cudnn/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cufft/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/curand/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cusolver/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/cusparse/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/nccl/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/nvidia/nvtx/lib:/workspace/gaap_vast/envs/gaap-author/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

exec "/workspace/gaap_vast/envs/gaap-author/bin/python" "$@"
