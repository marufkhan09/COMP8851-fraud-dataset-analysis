# LEX-GNN - COMP8851 fraud-GNN benchmark

One folder per dataset, same layout as the other models. Shared adapter code,
environment and provenance files are in `source/`.

Datasets: amazon, yelpchi, elliptic, fdcompcn, tfinance, tsocial

Evidence excluded from git by design: raw datasets, checkpoints, prediction arrays, split arrays.

## Verified results and limitations

- **FDCompCN:** 11/12 successful final runs. TR20 seed72 failed the
  configured initialization-restart gate; no successful test result
  exists for this condition. See
  `fdcompcn/evidence/final_runs/failed_lexgnn_tr20_seed72/README.md`.
- **Elliptic:** 12/12 final runs completed using reconstructed
  splits (`elliptic_reconstructed_v1`).
- **T-Social:** 12/12 final runs completed using provisional splits
  reconstructed from the documented convention.

For Elliptic and T-Social, do not describe these experiments as
using the original team-frozen splits.
