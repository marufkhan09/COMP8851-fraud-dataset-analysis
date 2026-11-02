export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/workspace/amnet_vast/envs/amnet/lib/python3.9/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}
