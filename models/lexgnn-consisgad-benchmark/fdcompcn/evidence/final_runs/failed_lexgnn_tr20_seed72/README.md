# FDCompCN — LEX-GNN TR20 Seed 72: Failed Final Run

## Experiment identity

- Dataset: FDCompCN
- Model: LEX-GNN
- Training ratio: 20%
- Training seed: 72
- Split seed: 2
- Status: FAILED
- Process exit code: 1
- Successful test evaluation: No

## Observed failure

The final run ended because LEX-GNN exceeded its configured
initialization restart limit.

The original execution made six initialization attempts.
Each attempt reached three training epochs before the
restart logic either restarted training or terminated the run.

The restart policy used a validation AUROC threshold of 0.510.

On the sixth attempt, the recorded validation AUROC at
epoch 3 was 0.500136. Training then terminated with:

RuntimeError: LEX-GNN exceeded the configured initialization restart limit

The final-batch log recorded an elapsed runtime of
approximately 10 seconds for this failed experiment.

## Interpretation

The failure occurred during initialization, before a
completed final training run or test evaluation.

Several initialization attempts showed near-chance
validation AUROC and fraud-class F1 of zero.

This is evidence of initialization instability under the
frozen experimental configuration. It does not establish
that the dataset is corrupted or that LEX-GNN can never
learn on this dataset.

## Treatment in benchmark results

- The original failed experiment is retained.
- No replacement training run is performed.
- No test metrics are fabricated or imputed.
- The experiment counts as one unsuccessful run.
- LEX-GNN TR20 summaries use only the two successful
  seeds, 2 and 42, and explicitly report n=2.
- All other FDCompCN conditions retain three seeds.

Final FDCompCN completion:
- LEX-GNN: 11 successful / 12 attempted
- ConsisGAD: 12 successful / 12 attempted
- Combined: 23 successful / 24 attempted

## Original evidence

- lexgnn_fdcompcn_tr20_seed72.log
- Original partial-run restart_attempts.json
  (preserved in the external experiment backup)

No model checkpoint or valid final test result was
produced for this failed run.
