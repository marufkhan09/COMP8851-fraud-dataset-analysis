# YelpChi tuning selection summary

## Selection protocol

- Tuning partition: frozen TR40
- Split seed: 2
- Training seed: 2
- Primary metric: validation AUROC
- Tie-breaker: validation AUPRC
- Test used during tuning: no

## LEX-GNN

- Selected trial: trial_01_author_baseline
- Learning rate: 0.005
- Beta: 0.5
- Validation AUROC: 0.909333
- Validation AUPRC: 0.672659
- Validation Macro-F1: 0.777853
- Validation Fraud-F1: 0.624017
- Validation G-Mean: 0.780488

## ConsisGAD

- Selected trial: trial_05_weight_2p0
- Learning rate: 0.001
- Consistency weight: 2.0
- Validation AUROC: 0.871477
- Validation AUPRC: 0.545380
- Validation Macro-F1: 0.727473
- Validation Fraud-F1: 0.540218
- Validation G-Mean: 0.724671

These settings are frozen before final multi-seed and
training-ratio experiments.
