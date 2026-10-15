# Dependency/environment installation — safe, manual stage

**Never pip-install all models into a common Python interpreter. Never repair a dead Python 3.10 virtualenv by linking its python to Python 3.12.** This destroyed portability during earlier GAAP recovery. The base Vast image is for provisioning; specific model runtimes come after GPU driver checks.

## Basic

Verify `nvidia-smi` and `df -h /workspace` first. The host driver is separate from the CUDA runtime in torch wheels. `torch.cuda.is_available()` plus a small DGL CUDA graph operation must both pass; a successful import alone is not sufficient. Record `pip freeze`, Python executable, torch/dgl versions and driver after installing each environment.

## RGTAN

Upstream requirements are legacy: Python 3.7-era package versions, `torch==1.12.1+cu113`, `dgl-cu113==0.8.1`, numpy 1.21.6, scikit-learn 1.0.2, pandas 1.3.5. Test against CUDA driver and Python compatibility **before** install. Prefer an isolated Python 3.9 environment or dedicated image; isolate from modern SAWGAD and preserve authoritative upstream versions unless a necessary compatibility patch is documented. Some older DGL/CUDA distributions are unavailable on current indexes; do not blindly substitute newer DGL. Record any change. The old NVIDIA driver has to be new enough; installed `nvidia-smi` CUDA version is maximum driver support, not necessarily the exact runtime wheel.

## SAWGAD

Upstream lock is based on `pyproject.toml` (Python 3.12, torch 2.4.0, DGL 2.4.0, CUDA 11.8 wheels). In its own checkout, install `uv` and run `uv sync --frozen` *if `uv.lock` exists*, otherwise `uv sync` creates a lock to capture. Verify `uv run python -c 'import torch,dgl; print(torch.__version__, dgl.__version__, torch.cuda.is_available())'`. Then only native smoke run, **not canonical** until split/schedule rule is approved.

## EOGFD

No trusted upstream requirements file or dependency lock verified. Derive versions from imports and test `torch`, `dgl`, `numpy`, `scipy`, `sklearn`, not from the SAWGAD environment. Treat import alias mismatch and test-in-loop evaluation as two hard gates. Canonical run requires a separate audited runner; cannot just call `run_eogfd.py` and label result canonical.

## Disk

Under 145 GB disk: create one model environment at a time when possible; after exporting `pip freeze`, wheels/config/hash and valid evidence, remove disposable cache only if no active run depends on it. Keep canonical assets and all evidence external-backed. Under 60 GB disk, do not expect to hold three modern environments plus full T-Social graph formats and checkpoints; either rent larger storage or use externally mounted volumes if supported and verified.
