# PC-GNN COMP8851 run guide

This package keeps the original PC-GNN Pick–Choose–Aggregate architecture and
adds experiment recording required for the COMP8851 benchmark.

## What was changed

- Records the UTC start and end of every training epoch.
- Uses `time.perf_counter()` and synchronises CUDA before and after GPU timing.
- Saves every epoch to `epoch_times.csv`, including validation time separately.
- Saves data-loading time, total training time, mean/median/SD epoch time,
  test inference time, and peak GPU memory when CUDA is used.
- Saves AUROC, AUPRC, Macro-F1, fraud-class F1, G-Mean, fraud precision,
  fraud recall, accuracy, and the confusion-matrix counts.
- Saves the exact configuration, environment details, checkpoint, split IDs,
  split counts, and final summary for each run.
- Accepts persistent `.npz` split files for fair dataset-specific comparison.
- Provides a generator for nested TR40/TR30/TR20/TR10 training subsets with
  fixed validation and test nodes.

No message-passing layer, neighbour-selection rule, aggregation mechanism,
sampling mechanism, model loss, or relation meaning was redesigned.

## Required YelpChi data files

Keep these locally under `data/`; they are deliberately ignored by Git:

```text
YelpChi.mat
yelp_homo_adjlists.pickle
yelp_rur_adjlists.pickle
yelp_rtr_adjlists.pickle
yelp_rsr_adjlists.pickle
```

## A. Instrumented author-code reproduction

Activate the environment that already ran PC-GNN successfully, then execute:

```bash
python main.py --config ./config/pcgnn_yelpchi_smoke.yml 2>&1 | tee pcgnn_smoke_test.log
```

The smoke test runs only two epochs. Confirm that it creates `epoch_times.csv`
and `summary.json` before starting the complete instrumented run:

```bash
python main.py --config ./config/pcgnn_yelpchi_author_reproduction.yml 2>&1 | tee pcgnn_author_instrumented.log
```

The supplied reproduction configuration preserves the successful local run:
51 epochs, seed 72, Adam, learning rate 0.01, batch size 1024, 40/20/40-style
random stratified splitting, and CPU execution.

Do not present this as an exact paper reproduction. It uses 32-dimensional
YelpChi features, 51 epochs, one seed, and a modern ARM64 CPU environment. The
paper used 100-dimensional features, 100 epochs, ten runs, Python 3.7,
PyTorch 1.6.0, and different hardware.

## B. Create the persistent unified splits

Run once from the PC-GNN project directory:

```bash
python scripts/create_nested_splits.py \
  --data-file ./data/YelpChi.mat \
  --dataset-name yelp \
  --output-dir ./splits \
  --seed 2
```

This produces:

```text
splits/yelp_TR40_seed2.npz
splits/yelp_TR30_seed2.npz
splits/yelp_TR20_seed2.npz
splits/yelp_TR10_seed2.npz
splits/yelp_nested_split_manifest_seed2.json
```

The required nesting is enforced:

```text
TR10 subset of TR20 subset of TR30 subset of TR40
```

Validation and test IDs remain identical in every file. Nodes removed from a
smaller training subset become unused; they are not moved into validation or
test.

## C. Unified benchmark runs

Use one configuration at a time:

```bash
python main.py --config ./config/unified/pcgnn_yelpchi_TR40.yml 2>&1 | tee pcgnn_TR40.log
python main.py --config ./config/unified/pcgnn_yelpchi_TR30.yml 2>&1 | tee pcgnn_TR30.log
python main.py --config ./config/unified/pcgnn_yelpchi_TR20.yml 2>&1 | tee pcgnn_TR20.log
python main.py --config ./config/unified/pcgnn_yelpchi_TR10.yml 2>&1 | tee pcgnn_TR10.log
```

For the final comparison, use the team-approved common training seeds while
keeping the persistent split files unchanged. Run on the same benchmark GPU as
the other models. The supplied seed 2 is the initial controlled-run setting,
not a claim that Venus mandated a particular seed value.

## Saved output from every instrumented run

Each run receives a unique directory under the configured `results_dir`:

```text
epoch_times.csv
run_config.json
split_indices.npz
split_summary.json
summary.json
test_metrics.csv
checkpoints/yelp_PCGNN.pkl
```

Use `summary.json` as the main machine-readable record. Use `epoch_times.csv`
for the table or graph of per-epoch training time.

## Timing convention

`train_seconds` begins before PC-GNN's label-balanced sampling and ends after
the last optimiser step. Validation is excluded from that value and recorded
separately in `validation_seconds`. GPU operations are synchronised at the
timing boundaries. This convention must be applied consistently to the other
models before comparing their epoch times.

## Important fairness rule

Only compare or rank models that used the same dataset version, label
definitions, persistent split IDs, experimental condition, metric code,
hardware, and agreed seed schedule. The local ARM64 result is reproduction
evidence, not a timing result to compare against Kaggle GPU runs.
