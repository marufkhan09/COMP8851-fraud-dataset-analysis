# ConsisGAD - COMP8851 fraud-GNN benchmark

One folder per dataset, same layout as the other models. Shared adapter code,
environment and provenance files are in `source/`.

Datasets: amazon, yelpchi, elliptic, fdcompcn, tfinance, tsocial

Evidence excluded from git by design: raw datasets, checkpoints, prediction arrays, split arrays.

## Verified results and limitations

- **FDCompCN:** 12/12 final runs completed.
- **Elliptic:** 12/12 final runs completed using reconstructed
  splits (`elliptic_reconstructed_v1`).
- **T-Social:** 12/12 final runs completed using provisional splits
  reconstructed from the documented convention.

For Elliptic and T-Social, do not describe these experiments as
using the original team-frozen splits.
