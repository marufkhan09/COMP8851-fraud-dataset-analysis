# YelpChi pause checkpoint

Recorded UTC: 2026-10-17T15:33:47.161336+00:00

## Completed

- Canonical YelpChi intake and frozen-split validation
- LEX-GNN two-epoch compatibility test
- ConsisGAD two-epoch compatibility test
- LEX-GNN five-trial validation-only tuning
- ConsisGAD five-trial validation-only tuning
- Test data was not evaluated during either tuning batch

## Selected configuration

LEX-GNN:

- Learning rate: 0.005
- Beta: 0.5
- Validation AUROC: 0.9093330502274346
- Validation AUPRC: 0.6726585169950963

## Pending next session

1. Verify and rank all five ConsisGAD trials.
2. Select ConsisGAD hyperparameters using validation AUROC and AUPRC.
3. Package the combined YelpChi tuning evidence.
4. Create the 24 final YelpChi configurations.
5. Run final multi-ratio, multi-seed experiments.

## Full pause archive

The complete local archive, including checkpoints and predictions, is:

`/workspace/comp8851_two_models/exports/lexgnn_consisgad_yelpchi_pause_20261017T153344Z.tar.gz`

Large checkpoints, predictions, datasets, environments and logs are not
stored in Git.
