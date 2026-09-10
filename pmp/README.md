# PMP Benchmark

This directory contains the COMP8851 implementation and benchmarking work for **PMP (Partitioning Message Passing for Graph Fraud Detection)**.

## Contents

- `pmp-benchmark.ipynb` — reproducibility, compatibility auditing, dataset validation, controlled split integration, training, evaluation, and benchmarking workflow for PMP.

## Benchmark Protocol

The notebook follows the COMP8851 controlled graph-fraud evaluation protocol, including:

- reproducible dataset and split validation;
- separation of author-code reproduction from unified benchmarking;
- validation-only model selection and hyperparameter tuning;
- controlled random seeds and hardware;
- predictive-performance, runtime, and GPU-memory measurements;
- preservation of PMP's original message-partitioning mechanism.

The workflow begins with YelpChi and can be extended to other datasets that pass the PMP compatibility gate.
