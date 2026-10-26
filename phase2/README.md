# COMP8851 phase 2: SAGAD, SpaceGNN and CRoC under the frozen fraud-GNN benchmark protocol

This bundle extends the 10-model benchmark (CARE-GNN … SplitGNN; report Table 6) with three newer models:

- **SAGAD** (WWW 2026)
- **SpaceGNN** (ICLR 2025)
- **CRoC** (ECAI 2025)

They run on the same six datasets (YelpChi, Amazon, T-Finance, Elliptic, FDCompCN, T-Social) under the same protocol, `vast-v4.4`:

- frozen split registry: TR40 = 40/20/40 with nested TR30–TR10;
- Elliptic split chronologically;
- seeds 2, 42 and 72;
- at most 12 tuning trials, on validation AUPRC only;
- validation-AUPRC checkpointing, at most 100 epochs, patience 20;
- the shared evaluator, which reports AUPRC plus 7 other metrics.

The models' **official code** is used unchanged. It is fetched at pinned commits and verified file by file. Each model has a thin, documented adapter.

## Contents

| Path | What |
|---|---|
| `shared/comp8851/` | The previous benchmark's shared library, **copied verbatim** (hashes in `audit/copied_assets_manifest.json`) |
| `shared/splits/` | Frozen split registry. YelpChi, Amazon, T-Finance and T-Social are byte copies. Elliptic and FDCompCN are regenerated and verified against the previous runs. |
| `data/canonical/` | The six canonical datasets. Each is **byte-identical** to the data used by CARE-GNN/GHRN, verified by view hash (`data/canonical/data_status.json`). |
| `phase2/` | New shared machinery: data and label isolation, run harness (checkpointing, resume, predictions, reload check), configs, tuning |
| `models/{sagad,spacegnn,croc}/` | Adapters around the official code, plus `run_one.py` |
| `configs/*.yaml` | Official per-dataset hyperparameters with their sources, budgets, and the papers' search grids; `matrix.yaml` holds execution caps |
| `scripts/` | Setup, data, orchestration (`run_matrix.py`), aggregation, integrity verification, packaging, Kaggle push |
| `tests/` | Automated tests on real data: exactness of the SAGAD MRQ sampler vs official code, leakage, resume, failure paths |
| `kaggle/` | Kaggle GPU integration kernel and secure-authentication guide |
| `audit/` | `AUDIT_REPORT.md`, `PROVENANCE.md`, `DEVIATIONS.md`, `COMPATIBILITY_MATRIX.md` |
| `evidence/local_cpu/` | Pre-deployment verification evidence: test logs, CPU smoke matrix, and end-to-end pipeline check |
| `results/` | Complete production GPU results (`raw/` 54 multi-seed runs, `aggregated/final/` summary tables, `ledger.csv`) |
| `third_party/manifest.json` | Pinned commits and per-file SHA256 of the official code. The code itself is fetched and verified via `scripts/fetch_official.py`. |

## Production GPU Execution: One Command

1. **Environment.** Dedicated **NVIDIA RTX A6000 48 GB** GPU workstation with at least 64 GB RAM and 100 GB disk (`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`).
2. **Unpack.** `unzip comp8851_phase2_vastai.zip && cd comp8851_phase2`.
3. **Run full pipeline** inside `tmux new -s comp8851`:

   ```bash
   CONCURRENCY=2 bash scripts/run_all.sh
   ```

`run_all.sh` runs these stages in order:

1. setup (torch 2.4.0 + DGL 2.4.0, pinned) and environment verification;
2. fetching the official code;
3. data verification;
4. the test suite;
5. the 18-pair GPU smoke test;
6. a production pilot (T-Social and FDCompCN);
7. tuning: 12 trials × 18 pairs;
8. final runs: 3 seeds × 18 pairs (54 runs);
9. aggregation, independent integrity verification, and packaging.

It is **resumable**: after any interruption, re-run the same command and completed cells are skipped while interrupted ones resume from `checkpoints/last.pt`.

### Individual commands

```bash
python scripts/verify_env.py --require-gpu
python scripts/fetch_official.py
python scripts/prepare_data.py                       # verify (or rebuild) all six datasets
python -m pytest tests -q
python scripts/run_matrix.py smoke --require-cuda
python scripts/run_matrix.py pilot --datasets tsocial fdcompcn --require-cuda
python scripts/run_matrix.py tune  --require-cuda --concurrency 2
python scripts/run_matrix.py final --require-cuda --concurrency 2
python scripts/run_matrix.py status
python scripts/aggregate_results.py --phase final
python scripts/verify_results.py --phase final --deep
bash scripts/package_results.sh
# a single cell:
python models/sagad/run_one.py --dataset amazon --ratio TR40 --seed 42 --hp lr=0.01
# resume one cell:
python models/croc/run_one.py --dataset tsocial --ratio TR40 --seed 2 --resume
# label-scarcity ladder (optional, as in the report): add  --ratios TR40 TR30 TR20 TR10  to `final`
# authors' epoch budgets as a sensitivity track:  --extra --budget official
```

## Outputs

- **One directory per cell:** `results/raw/<model>/<dataset>/unified/<ratio>/seed_<seed>/`. It holds the previous benchmark's artefact set (`summary.json`, `run_config.yml`, `protocol_identity.json`, environment and hardware files, dataset and view manifests, `split_indices.npz`, `epoch_times.csv`, `validation_metrics.csv`, `test_metrics.json`, `terminal.log`, `checkpoints/`). It adds three things:
  - `predictions.npz`: validation and test probabilities with node IDs and true labels;
  - `checkpoints/last.pt`, used for resuming;
  - a checkpoint-reload reproduction check.
- **Tables** in `results/aggregated/final/`:
  - `per_seed.csv/json`: every seed, every status, failure reason, runtime, memory, artefact paths;
  - `mean_sd.csv/json`: mean ± sample SD;
  - `status_matrix.md`: all 18 pairs;
  - `comparison_table6.md`: the new models next to the previous ten, with comparability markers;
  - `integrity_report.json`: every metric recomputed from the saved predictions.
- **Ledger:** `results/ledger.csv` records every job's status (PENDING, RUNNING, COMPLETED, INCOMPLETE, FAILED, BLOCKED_DATA, SKIPPED_BUDGET).

## Retrieving results

`scripts/package_results.sh` writes two archives to `results/`:

- `results_full_<stamp>.tar.gz`, which includes checkpoints;
- `results_light_<stamp>.tar.gz`, which does not.

Copy them off the instance with `scp` **before destroying it**: a destroyed Vast.ai instance loses its disk, while a stopped one keeps it.

## Integrity rules

- Model code never receives validation or test labels; non-training labels are masked to -1.
- Tuning trials run with `--skip-test`, so the test set is never scored during tuning.
- Thresholds are chosen on validation only.
- A COMPLETE run is never overwritten; `--force` archives the old run to `results/_attempts/`.
- Every reported number can be recomputed from `predictions.npz` with `scripts/verify_results.py`.
- Smoke and pilot outputs are feasibility evidence only and never enter the result tables.
