# PMP × T-Finance Evidence

Controlled COMP8851 PMP benchmark evidence for T-Finance.

Protocol:
- Frozen benchmark v4.4
- Static nested split
- Split seed: 2
- TR40 / TR30 / TR20 / TR10
- Validation-only model selection
- Test isolation during tuning and checkpoint selection
- Final seeds: 2, 42, 72
- RTX A6000 execution

Dataset:
- T-Finance
- 39,357 nodes
- 1,804 fraud
- 37,553 normal

The `archive/` directory preserves the original evidence package.
The `extracted/` directory contains the unpacked scripts, metrics,
configuration, provenance, summaries and execution evidence.

See `SHA256SUMS.txt` for integrity hashes.
