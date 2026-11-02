# Dataset Registry Notebook

`dataset-registry.ipynb` provides the shared dataset foundation for the COMP8851 graph fraud benchmark.

It currently performs:

- Kaggle NVIDIA T4 / GPU 0 runtime verification
- Detection of the six canonical datasets: YelpChi, Amazon, T-Finance, T-Social, Elliptic and FDCompCN
- Exact mounted-path and file-size recording
- SHA-256 checksum generation for dataset source files
- Structural verification of YelpChi, Amazon and Elliptic
- T-Social source-hash verification
- Creation of the canonical dataset registry and checksum metadata

Current status:

- YelpChi — structural verification PASS
- Amazon — structural verification PASS
- Elliptic — structural verification PASS
- T-Social — canonical hash verified; DGL load verification pending
- T-Finance — DGL structural verification pending
- FDCompCN — DGL structural verification pending

Raw datasets are not stored in this Git repository. Only reproducibility metadata, checksums and notebook code are committed.
