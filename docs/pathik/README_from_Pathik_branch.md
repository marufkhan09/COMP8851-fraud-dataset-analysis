# COMP8851 Fraud Dataset Analysis

Repository for the COMP8851 graph fraud detection benchmark project. This repository contains benchmark scripts, dataset documentation, model-specific experiment files, controlled evidence, findings documents, and reproducibility material.

## Start Here

| Resource | Location | Purpose |
| --- | --- | --- |
| Final BWGNN findings | `docs/pathik/BWGNN_Findings_Final-2.docx` | Final documentation of the completed BWGNN benchmark and findings. |
| Final PMP findings | `docs/pathik/PMP_Findings_Final.docx` | Consolidated findings from the current PMP benchmark work. |
| Six-dataset characteristics | `docs/pathik/Six_Dataset_Characteristics_Final.docx` | Reference document describing the six graph-fraud datasets used in the benchmark. |
| BWGNN Python scripts | `bwgnn notebook and docx/BWGNN_ALL_PY_EXPORT/` | Final dataset-specific BWGNN scripts, reporting utilities, adapter, and checksum manifest. |
| PMP YelpChi workflow | `pmp notebooks and docx/` | PMP recovery, smoke, tuning, and final-run scripts for YelpChi. |
| PMP controlled evidence | `models/pmp/evidence/` | Controlled PMP evidence packages for T-Finance, Amazon, FDCompCN, and Elliptic. |

The primary documentation location is `docs/pathik/`.

## Benchmark Datasets

The project benchmark uses six graph-fraud datasets:

1. YelpChi
2. Amazon
3. T-Finance
4. FDCompCN
5. Elliptic
6. T-Social

For detailed dataset definitions and characteristics, see `docs/pathik/Six_Dataset_Characteristics_Final.docx`.

## BWGNN

### Final Findings

**File:** `docs/pathik/BWGNN_Findings_Final-2.docx`

Contains the final BWGNN benchmark findings, dataset-level experimental results, benchmark observations, and main conclusions.

A convenience copy is also stored at:

`bwgnn notebook and docx/BWGNN_Findings_Final-2.docx`

### Python Files

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

## PMP

### Final Findings

**File:** `docs/pathik/PMP_Findings_Final.docx`

Contains the consolidated findings from the current PMP benchmark work.

### YelpChi Workflow

The earlier PMP YelpChi workflow is retained in:

`pmp notebooks and docx/`

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

File roles:

- `recovery/` - split recovery and benchmark-identity verification.
- `smoke/` - short compatibility and diagnostic execution.
- `tuning/` - TR40 validation-based tuning workflow.
- `final/` - final YelpChi execution scripts.
- `SHA256SUMS.txt` - integrity manifest for the exported PMP YelpChi material.

### Controlled PMP Evidence

Controlled PMP evidence is stored under:

`models/pmp/evidence/`

```text
models/pmp/evidence/
├── tfinance/
├── amazon/
├── fdcompcn/
└── elliptic/
```

The evidence packages retain experiment material needed for auditing and reproducibility, including where available:

- source and repository identity;
- environment and GPU information;
- fixed or chronology-preserving split evidence;
- compatibility or smoke-test records;
- frozen tuning configuration;
- final controlled run summaries;
- per-run epoch histories;
- validation threshold grids;
- execution logs;
- checkpoints;
- SHA-256 manifests and source-archive hashes.

### PMP Dataset Status

| Dataset | Current repository status |
| --- | --- |
| YelpChi | Recovery, smoke, tuning, and final-run workflow committed under `pmp notebooks and docx/`. |
| T-Finance | Controlled evidence committed under `models/pmp/evidence/tfinance/`. |
| Amazon | Controlled evidence committed under `models/pmp/evidence/amazon/`. |
| FDCompCN | Controlled evidence committed under `models/pmp/evidence/fdcompcn/`. |
| Elliptic | Chronology-preserving controlled evidence committed under `models/pmp/evidence/elliptic/`. |
| T-Social | No PMP controlled evidence package is currently committed on this branch. |

The PMP layout deliberately distinguishes the earlier YelpChi workflow export from the newer controlled evidence packages.

## Six Dataset Characteristics

**File:** `docs/pathik/Six_Dataset_Characteristics_Final.docx`

Contains benchmark reference information for YelpChi, Amazon, T-Finance, FDCompCN, Elliptic, and T-Social, including dataset identity, size, labels, graph structure, provenance, and other benchmark-relevant characteristics.

A convenience copy is also stored at:

`dataset characteristics/Six_Dataset_Characteristics_Final.docx`

## Important Repository Structure

```text
COMP8851-fraud-dataset-analysis/
│
├── docs/
│   └── pathik/
│       ├── BWGNN_Findings_Final-2.docx
│       ├── PMP_Findings_Final.docx
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
├── models/
│   ├── bwgnn/
│   │   ├── README.md
│   │   └── references/
│   └── pmp/
│       ├── README.md
│       ├── references/
│       └── evidence/
│           ├── tfinance/
│           ├── amazon/
│           ├── fdcompcn/
│           └── elliptic/
│
├── pmp notebooks and docx/
│   ├── final/
│   ├── recovery/
│   ├── smoke/
│   ├── tuning/
│   └── SHA256SUMS.txt
│
├── pmp/
│   ├── README.md
│   └── pmp-benchmark.ipynb
│
└── dataset characteristics/
    └── Six_Dataset_Characteristics_Final.docx
```

## Controlled Benchmark Protocol

The controlled benchmark uses the following labelled-training regimes:

- **TR40** - 40% labelled training allocation.
- **TR30** - 30% labelled training allocation.
- **TR20** - 20% labelled training allocation.
- **TR10** - 10% labelled training allocation.

Validation and test identities remain fixed within each controlled dataset protocol. Elliptic uses chronology-preserving split handling.

Model selection, early stopping, and threshold selection must use validation evidence only. The test set is reserved for final evaluation after configuration freeze.

## Reproducibility

Benchmark artifacts are organised so that model code, dataset documentation, compatibility checks, split adapters, tuning, final runs, findings, and environment/provenance evidence remain clearly separated.

Where present, `SHA256SUMS.txt` files provide file-level integrity records.

Large raw datasets are not duplicated unnecessarily inside Git evidence directories. When source evidence archives contain files too large for normal Git storage, the repository retains Git-safe experimental evidence together with source-archive hashes and provenance records.

## Notes

- Do not interpret compatibility or smoke runs as final benchmark results.
- Keep author-reproduction evidence separate from controlled benchmark evidence.
- Do not use test labels for tuning, checkpoint selection, early stopping, or threshold selection.
- Preserve dataset-specific representation rules, including chronological handling for Elliptic.
- Prefer checksum-backed evidence and recorded source/repository identity when reproducing results.
