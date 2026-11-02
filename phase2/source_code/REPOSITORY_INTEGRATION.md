# Integration with the COMP8851 repository

This directory is a self-contained, reproducible phase-2 execution bundle from Akib's branch. Its `models/`, `phase2/`, `scripts/`, `configs/`, `shared/`, and `tests/` paths deliberately remain together, because code and scripts use paths relative to this directory. Moving just the Python model files into the top-level `models/` directory would break that structure.

| Component | Repository location |
| --- | --- |
| Model index/owner | `models/sagad/`, `models/spacegnn/`, `models/croc/` |
| Source adapters | `phase2/source_code/models/{sagad,spacegnn,croc}/` |
| Runnable scripts and environment | `phase2/source_code/scripts/`, `env/`, `configs/` |
| Pinned official source manifests | `phase2/source_code/third_party/manifest.json` |
| Existing raw result records | `phase2/results/raw/{sagad,spacegnn,croc}/` |
| Existing aggregated results | `phase2/results/aggregated/final/` |
| Audits and deviations | `phase2/source_code/audit/` |

**Run commands from `phase2/source_code/`** following its `README.md`. This merge archives code and provenance but does not claim that the complete GPU experiment was independently rerun.
