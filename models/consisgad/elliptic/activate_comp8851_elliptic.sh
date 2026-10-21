#!/usr/bin/env bash
source /workspace/comp8851_elliptic_env/bin/activate
SITE="$VIRTUAL_ENV/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/cusparse/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=0
