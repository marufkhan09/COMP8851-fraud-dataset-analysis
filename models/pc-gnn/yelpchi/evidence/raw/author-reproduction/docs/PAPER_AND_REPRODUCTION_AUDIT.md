# PC-GNN paper and reproduction audit

## Verified paper

Yang Liu, Xiang Ao, Zidi Qin, Jianfeng Chi, Jinghua Feng, Hao Yang, and Qing He.
“Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud
Detection.” The Web Conference 2021, pages 3168–3177.

- DOI: <https://doi.org/10.1145/3442381.3449989>
- Author code: <https://github.com/PonderLY/PC-GNN>
- Recorded source commit: `9d7d7fae491081178b6e12e193fb89cfc330f2be`

## Paper settings versus the supplied successful run

| Item | Paper | Supplied successful run |
|---|---:|---:|
| Dataset | YelpChi | YelpChi |
| Feature dimension | 100 | 32 |
| Train/validation/test | 40%/20%/40% | 18,381/9,099/18,474 nodes |
| Epochs | 100 | 51 |
| Runs | 10 | 1 |
| Optimiser | Adam | Adam |
| Learning rate | 0.01 | 0.01 |
| Batch size | 1,024 | 1,024 |
| Hidden dimension | 64 | 64 |
| Alpha | 2 | 2 |
| Paper software | Python 3.7, PyTorch 1.6.0 | Python 3.14.3, PyTorch 2.13.0 |
| Hardware | Ubuntu server, 40 cores, 128 GB memory | macOS ARM64 CPU |

The repository `requirements.txt` requests PyTorch 1.4.0 even though the paper
states PyTorch 1.6.0. Both must be recorded rather than silently treated as the
same environment.

## Result comparison

| Metric | Paper mean ± SD | Supplied run | Difference |
|---|---:|---:|---:|
| Macro-F1 | 0.6300 ± 0.0230 | 0.6530 | +0.0230 |
| AUROC | 0.7987 ± 0.0014 | 0.8178 | +0.0191 |
| G-Mean | 0.7160 ± 0.0130 | 0.7408 | +0.0248 |

The supplied run is reasonably consistent with the reported PC-GNN result but
is not an exact replication because its data representation, epoch count,
number of runs, software, and hardware differ.

## Existing timing evidence

The original console log contains all 51 epoch durations:

- Total logged minibatch time: 89.4990 seconds.
- Mean: 1.7549 seconds per epoch.
- Median: 1.7358 seconds.
- Population SD: 0.0470 seconds.
- Minimum: 1.6853 seconds.
- Maximum: 1.8509 seconds.

Those legacy times sum only the individual minibatch blocks and exclude graph
sampling, validation, and final inference. They are preserved as evidence but
must not be compared directly with the new full training-epoch timing.

## Split detail that must be reported

The configuration seed is 72, but the supplied source hard-codes
`random_state=2` for both stratified split operations. Therefore:

- seed 72 controls model/random sampling behaviour;
- random state 2 controls the author-style train/validation/test node split.

The instrumented package records the actual node IDs used by every run and can
instead load shared persistent split files for the controlled benchmark.
