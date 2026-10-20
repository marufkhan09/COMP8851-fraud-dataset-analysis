# Elliptic — LEX-GNN and ConsisGAD

Protocol: elliptic_reconstructed_v1

Models: LEX-GNN, ConsisGAD
Training ratios: 10%, 20%, 30%, 40%
Seeds: 2, 42, 72

Training experiments: 24/24 completed
Checkpoint-based test evaluations: 24/24 completed
Test nodes per run: 17,626

## Protocol limitation

The original team-frozen Elliptic split ID files were
unavailable. The training subsets were reconstructed.

Therefore, these results are not verified as directly
comparable with experiments using the original frozen
split IDs.

Hyperparameters were inherited from previously established
configurations, without a documented Elliptic-specific search.

## Contents

tables/: Individual and aggregated test results
splits/: Reconstructed split files and manifest
final_evaluation/: Test inference scripts and evidence
elliptic_full_evidence.tar.gz: Training outputs, checkpoints,
logs, configurations and adapter evidence

Raw Elliptic dataset files are not included.
