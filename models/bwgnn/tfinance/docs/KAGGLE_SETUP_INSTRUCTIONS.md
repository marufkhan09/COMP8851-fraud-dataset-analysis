# Run instructions: BWGNN on T-Finance

## Include these two inputs in Kaggle

1. **Code/model input:** `BWGNN_TFinance_COMP8851_Bundle.zip`
2. **Dataset input:** the raw T-Finance DGL file named `tfinance`

Do not put the 652 MB raw dataset in GitHub. Upload it as a private Kaggle
dataset, then attach that dataset to the notebook.

## Kaggle settings

- Accelerator: **GPU T4 x2** if available, otherwise a single **GPU T4**.
- The benchmark uses **GPU 0 only**, even when two GPUs are visible.
- Internet: **ON** while the environment is installed.
- Start a fresh Kaggle notebook by importing
  `BWGNN_TFinance_COMP8851_Run_All.ipynb`.

## Run

Use **Run All**. The notebook has four main cells:

1. Locate and copy the code and dataset.
2. Create a Python 3.11 environment and verify CUDA-enabled DGL.
3. Run the complete experiment workflow.
4. Verify and download the final evidence ZIP.

Progress is printed throughout. If a cell stops but the Kaggle session is
still active, run the workflow cell again; completed runs will be skipped.

## Send back

When the notebook reports `COMPLETE WORKFLOW: TRUE`, download and send:

`bwgnn_tfinance_complete_results.zip`

That single file contains all configurations, split files, logs, metrics,
per-epoch timings and aggregate results needed for verification and the master
experiment record.
