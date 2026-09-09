# COMP8851 BWGNN on T-Finance

This folder contains a reproducible, instrumented BWGNN workflow for the
T-Finance dataset. It is based on the official implementation supplied for:

> Tang, J., Li, J., Gao, Z., & Li, J. (2022). Rethinking graph neural networks
> for anomaly detection. *Proceedings of the 39th International Conference on
> Machine Learning*, 21076–21089.

Paper: https://proceedings.mlr.press/v162/tang22b.html

Official code: https://github.com/squareRoot3/Rethinking-Anomaly-Detection

## What the workflow does

`run_all.py` performs these stages in order:

1. Loads the actual T-Finance DGL graph.
2. Creates a persistent, stratified 40/20/40 split using split seed 2.
3. Creates nested TR30, TR20 and TR10 training subsets while keeping the
   validation and test node IDs fixed.
4. Runs a two-epoch GPU smoke test.
5. Runs one 100-epoch author-setting reference check, selecting the checkpoint
   by validation Macro-F1 as in the supplied author training code.
6. Tunes hidden dimension 32 versus 64 on TR40, seed 72, using validation
   AUROC only. Test evaluation is disabled during tuning.
7. Runs TR40/TR30/TR20/TR10 using seeds 2, 42 and 72.
8. Records every epoch's synchronized GPU training time and exports all
   configurations, metrics, split evidence, logs, aggregate means and sample
   standard deviations into one ZIP.

## Controlled settings

- Model: homogeneous BWGNN
- Beta-wavelet order: 2
- Optimizer: Adam
- Learning rate: 0.01
- Weight decay: 0
- Split seed: 2
- Training seeds: 2, 42, 72
- Maximum final epochs: 100
- Validation interval: 5 epochs
- Final-run early-stopping patience: 20 epochs
- Hyperparameter selection: validation AUROC only
- Hardware: one NVIDIA Tesla T4, GPU 0

The core BWGNN architecture and fraud-specific mechanism are unchanged. The
only compatibility edit in `BWGNN.py` places temporary tensors on the same
device as the input tensor so the official model can run on a GPU. The
instrumented runner adds experiment control and evidence collection without
redesigning the model.

## Required input

The raw T-Finance dataset is intentionally not included. Attach the DGL file
named `tfinance` (or `tfinance.bin`/`tfinance.dgl`) to the Kaggle notebook as a
private dataset. The verified execution in this folder used SHA-256
`b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700` and
the graph has 39,357 nodes, 42,445,086 stored directed edge entries, 10
features, 1,804 fraud/anomaly nodes and 37,553 normal nodes. See
`../docs/DATASET_VERSION_DECISION_REQUIRED.md` before cross-model comparison.

## Direct command

After installing the requirements and a CUDA-compatible DGL build:

```bash
python -u run_all.py \
  --data-path /path/to/tfinance \
  --output-zip /kaggle/working/bwgnn_tfinance_complete_results.zip
```

The workflow is idempotent within the same writable session: a stage whose
result folder already contains `summary.json` is skipped.

## Output

The final ZIP contains source files, persistent split files, run
configurations, per-epoch timing CSVs, validation and test metrics, terminal
logs, the tuning decision, a complete run registry, aggregate mean/SD tables,
environment information and SHA-256 file hashes. Model checkpoints and the raw
dataset are excluded to keep the package suitable for GitHub.
