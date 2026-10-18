# FDCompCN pre-final benchmark status

## Protocol

- Models: LEX-GNN and ConsisGAD
- Canonical nodes: 5,317
- Features: 57
- Graph view: three semantic relations
- Derived homogeneous relation: excluded
- Split seed: 2
- Training ratios: TR40, TR30, TR20 and TR10
- Training seeds: 2, 42 and 72
- Final configurations: 24
- Test evaluation: once after training
- Final training started: no

## Selected LEX-GNN configuration

- Trial: trial_01_author_baseline
- Learning rate: 0.005
- Beta: 0.5
- Validation AUROC: 0.5838929321728691
- Validation AUPRC: 0.14628319878009288

## Selected ConsisGAD configuration

- Trial: trial_03_lr_0p005
- Learning rate: 0.005
- Consistency weight: 1.5
- Validation AUROC: 0.6076680672268908
- Validation AUPRC: 0.17500110492159052
