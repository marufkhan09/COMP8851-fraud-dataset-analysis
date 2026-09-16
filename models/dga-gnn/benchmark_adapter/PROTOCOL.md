# DGA-GNN protocol-safe default-run adapter

Upstream source commit: `0907392f6060e18230339ca30eaca0c917414820`.

## Purpose

The upstream validation loader predicts every graph node because DGA-GNN uses
all-node predictions for its dynamic grouping feedback. That transductive,
label-free prediction pass is retained. The benchmark adapter changes how
labels and model selection are used:

- validation loss, ROC-AUC and AP use validation nodes only;
- checkpoint selection and early stopping monitor validation AP;
- test AP, ROC-AUC, Macro-F1 and fraud-class F1 are computed once, after the
  best validation checkpoint is loaded;
- the classification threshold used on test nodes is selected on validation
  nodes;
- structured JSON records the configuration, validation history, split sizes,
  final metrics, parameter count and timing.

## Parallel-run interpretation

The four default seed-2 runs are launched concurrently to obtain accuracy and
compatibility evidence quickly. Their wall time, per-epoch time, inference time
and GPU-memory trace are resource-contended and are therefore marked as not
reportable for the efficiency/scalability research question.

After configuration selection and multi-seed accuracy runs, each selected
configuration must be rerun alone on GPU 0 to obtain reportable efficiency
measurements.

## Scope

These are upstream-default, single-seed runs. They are not a replacement for
the predefined equal-budget hyperparameter search or final multi-seed runs.
