#!/usr/bin/env bash
export LD_LIBRARY_PATH="/workspace/comp8851/env/eogfd-legacy/lib/python3.10/site-packages/nvidia/cusparse/lib:/workspace/comp8851/env/eogfd-legacy/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/cuda-11.3/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export DGLBACKEND=pytorch
export PYTHONDONTWRITEBYTECODE=1
