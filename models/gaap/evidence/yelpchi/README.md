# GAAP × YelpChi Controlled Evidence

Controlled GAAP benchmark evidence for YelpChi.

## Final status

- Final runs: 12/12
- Ratios: TR40, TR30, TR20, TR10
- Seeds: 2, 42, 72
- Hardware: NVIDIA RTX A6000
- Model/checkpoint selection: validation only
- Test isolation: PASS
- GAAP author repository tracked source: clean

## Final AUPRC

| Ratio | Mean ± SD |
| --- | --- |
| TR40 | 0.898626 ± 0.007925 |
| TR30 | 0.870239 ± 0.004646 |
| TR20 | 0.828499 ± 0.008484 |
| TR10 | 0.712022 ± 0.018234 |

The evidence directory preserves final summaries, ratio summaries, scripts, logs, split/configuration records, environment information, repository provenance and checksum manifests.

The full archival ZIP and raw dataset bytes are intentionally retained outside Git.

A packaging-only DGL import probe produced a GraphBolt C++ library loading error after all 12 final runs had already completed. The event is preserved as environment evidence and did not affect training, model selection or final evaluation.
