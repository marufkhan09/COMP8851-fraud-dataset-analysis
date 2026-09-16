# COMP8851 Fraud Dataset Analysis

Repository for the COMP8851 graph fraud detection benchmark project. This repository contains benchmark scripts, dataset documentation, model-specific experiment files, and reproducibility material.

## Start Here

| Resource | Location | Purpose |
| --- | --- | --- |
| Final BWGNN findings | `docs/pathik/BWGNN_Findings_Final-2.docx` | Final documentation of the completed BWGNN benchmark and findings. |
| Six-dataset characteristics | `docs/pathik/Six_Dataset_Characteristics_Final.docx` | Reference document describing the six graph-fraud datasets used in the benchmark. |
| BWGNN Python scripts | `bwgnn notebook and docx/BWGNN_ALL_PY_EXPORT/` | Final dataset-specific BWGNN benchmark scripts, reporting utilities, adapter, and checksum manifest. |
| PMP Python scripts | `pmp notebooks and docx/` | PMP recovery, diagnostic, tuning, and final benchmark scripts. |

The primary documentation location is `docs/pathik/`.

## Important Documents

### BWGNN Findings

**File:** `docs/pathik/BWGNN_Findings_Final-2.docx`

Contains the final BWGNN benchmark findings, dataset-level experimental results, benchmark observations, and the main conclusions from the completed BWGNN work.

A convenience copy is also stored at:

`bwgnn notebook and docx/BWGNN_Findings_Final-2.docx`

### Six Dataset Characteristics

**File:** `docs/pathik/Six_Dataset_Characteristics_Final.docx`

Contains the benchmark reference information for YelpChi, Amazon, T-Finance, FDCompCN, Elliptic, and T-Social, including dataset identity, size, labels, graph structure, and other benchmark-relevant characteristics.

A convenience copy is also stored at:

`dataset characteristics/Six_Dataset_Characteristics_Final.docx`

## BWGNN Python Files

BWGNN benchmark scripts are located in:

`bwgnn notebook and docx/BWGNN_ALL_PY_EXPORT/`

Important files:

- `02_bwgnn_yelp_final.py` - YelpChi final benchmark workflow.
- `03_bwgnn_amazon_final.py` - Amazon final benchmark workflow.
- `04_bwgnn_tfinance_final.py` - T-Finance final benchmark workflow.
- `05_bwgnn_fdcompcn_final.py` - FDCompCN final benchmark workflow.
- `06_bwgnn_elliptic166_final.py` - Elliptic final benchmark workflow.
- `07_bwgnn_tsocial_a6000_smoke.py` - T-Social A6000 compatibility smoke test.
- `08_bwgnn_tsocial_final.py` - T-Social final benchmark workflow.
- `90_bwgnn_progress_report.py` - BWGNN benchmark progress reporting utility.
- `91_bwgnn_final_report.py` - Consolidated BWGNN final reporting utility.
- `adapters/BWGNN_gpu.py` - GPU-compatible BWGNN benchmark adapter.
- `SHA256SUMS.txt` - SHA-256 manifest for the exported BWGNN files.

## PMP Python Files

PMP benchmark scripts are located directly in:

`pmp notebooks and docx/`

Current committed PMP material contains the completed YelpChi workflow. Additional PMP datasets can be added to this same directory structure as they are completed.

### Current PMP structure

```text
pmp notebooks and docx/
├── final/
│   ├── 05_pmp_yelp_final_12_runs.py
│   └── PMP_Yelp_Final_12_Runs.py
├── recovery/
│   ├── 02_recover_pmp_yelp_split.py
│   └── PMP_Find_Exact_Yelp_Split.py
├── smoke/
│   └── 03_pmp_yelp_2epoch_smoke.py
├── tuning/
│   └── 04_pmp_yelp_tr40_tune.py
└── SHA256SUMS.txt
```

### PMP file roles

- `recovery/` contains utilities used to recover and verify the exact benchmark split.
- `smoke/03_pmp_yelp_2epoch_smoke.py` is an early diagnostic compatibility run and is not the final benchmark.
- `tuning/04_pmp_yelp_tr40_tune.py` contains the TR40 validation-based tuning workflow.
- `final/05_pmp_yelp_final_12_runs.py` contains the final PMP Yelp benchmark workflow.
- `final/PMP_Yelp_Final_12_Runs.py` is the preserved final runner artifact.
- `SHA256SUMS.txt` records file checksums for the exported PMP material.

## Important Repository Structure

```text
COMP8851-fraud-dataset-analysis/
│
├── docs/
│   └── pathik/
│       ├── BWGNN_Findings_Final-2.docx
│       └── Six_Dataset_Characteristics_Final.docx
│
├── bwgnn notebook and docx/
│   ├── BWGNN_Findings_Final-2.docx
│   └── BWGNN_ALL_PY_EXPORT/
│       ├── dataset-specific final Python scripts
│       ├── reporting utilities
│       ├── adapters/
│       └── SHA256SUMS.txt
│
├── pmp notebooks and docx/
│   ├── final/
│   ├── recovery/
│   ├── smoke/
│   ├── tuning/
│   └── SHA256SUMS.txt
│
└── dataset characteristics/
    └── Six_Dataset_Characteristics_Final.docx
```

## Benchmark Datasets

The current benchmark covers six graph-fraud datasets:

1. YelpChi
2. Amazon
3. T-Finance
4. FDCompCN
5. Elliptic
6. T-Social

For detailed dataset definitions and characteristics, see `docs/pathik/Six_Dataset_Characteristics_Final.docx`.

## Reproducibility

Benchmark artifacts are organised so that model code, benchmark orchestration, dataset documentation, adapters, tuning scripts, final runs, and reporting utilities remain clearly separated.

Where present, `SHA256SUMS.txt` files provide file-level integrity records for exported benchmark artifacts.
