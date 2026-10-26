# Audit report: existing COMP8851 benchmark, and readiness for SAGAD, SpaceGNN and CRoC

Prepared 2026-10-11. Every finding cites a file or a command that was executed. Paths are relative to `C:\Users\Admin\Downloads\Abdullah\`.

## 1. What exists (inventory)

- **Workspace.** About 6.4 GB, 2,656 files. It is **not a git repository**, and no git binary is installed, so provenance relies on SHA256 hashes rather than commits.
- **Five overlapping copies of the project:**
  - `project/CLIENT_DELIVERABLE`, `_v2`, `_FINAL`
  - `project/COMP8851_FINAL_SUBMISSION`
  - `project/COMP8851-fraud-dataset-analysis-UPDATED`

  Their `shared/comp8851` libraries agree except for `datasets.py`, `dgl_env.py`, `fetch_datasets.py` and `protocol.py`. The `_FINAL` and `FINAL_SUBMISSION` copies of those four are the newest, so the phase-2 bundle copies `COMP8851_FINAL_SUBMISSION/source_code/shared` verbatim (hashes in `audit/copied_assets_manifest.json`).
- **Report.** `COMP8851_Assessment2_Report_FINAL-069.pdf`, 16 pages, with Table 6 holding the 10-model TR40 AUPRC.
- **Run evidence.** CARE-GNN and GHRN have per-run evidence (summary, epoch times, validation metrics, test metrics, manifests) for five datasets × three seeds under `project/COMP8851_FINAL_SUBMISSION/evidence/`. PC-GNN and BWGNN evidence is under `project/COMP8851-fraud-dataset-analysis-UPDATED/.../models/`. No evidence for GAAP, PMP, DGA-GNN, GAGA, FRAUDRE or SplitGNN is in this workspace.
- **Data.**
  - Only T-Social was present locally, inside `project/comp8851.zip`, as `tsocial_canonical.npz` (sha256 `a30fb6a0…`) plus its split registry.
  - Frozen registries were found for YelpChi, Amazon, T-Finance (under `models/bwgnn/tfinance/shared/splits/`, with non-canonical file names) and T-Social.
  - **No frozen Elliptic or FDCompCN split arrays exist anywhere in the workspace.** Only their `split_summary.json` counts survive.
- **Phase-2 model choice.**
  - `Model Selection Justification FdGars, GraphConsis and LEX-GNN (1).pdf` (6 Oct 2026) proposed FdGars, GraphConsis and LEX-GNN.
  - The current brief names SAGAD, SpaceGNN and CRoC; the brief was followed.

## 2. The established protocol, verified from code and report

- **Source.** `shared/comp8851/protocol.py`, which mirrors report pages 3–4.
- **Split.** Split seed 2. TR40 is a stratified 40/20/40 split over eligible nodes; TR30, TR20 and TR10 are nested subsets of the TR40 training set, while validation and test stay fixed.
- **Elliptic** is split chronologically. Amazon's first 3,305 nodes and Elliptic's unknown-class nodes stay in the graph but are never supervised.
- **Seeds:** **2, 42, 72**, not 1, 2, 3.
- **Tuning:** at most 12 trials on TR40 with seed 2, selected on validation AUPRC.
- **Training:** at most 100 epochs, early-stopping patience 20, checkpoint selection on validation AUPRC.
- **Threshold:** swept on validation from 0.01 to 0.99, maximising Macro-F1, then fraud recall, then closeness to 0.5.
- **Metrics:** AUPRC (sklearn `average_precision_score`, positive class 1) is primary. Also recorded: AUROC, Macro-F1, fraud F1, precision, recall, G-Mean and accuracy.
- **Reporting:** mean ± sample SD (ddof = 1) over seeds.
- **Hardware:** RTX A6000 48 GB, with A100 as the fallback.

## 3. Findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| F1 | **High** (comparability) | PC-GNN's YelpChi rows in Table 6 were not computed on the registry split. PC-GNN used validation/test partitions of 9,099/18,474 nodes; the registry and the CARE-GNN/GHRN runs use 9,191/18,382. The training sets are both 18,381 nodes. Table 6 shows no † marker for this row. | `models/pc-gnn/yelpchi/evidence/raw/final-seed-{2,42,72}/splits/yelpchi_tr40_split_seed2.npz` versus `shared/splits/yelpchi/yelpchi_TR40_seed2.npz` (sha `e9fe555f…`). Content comparison by script. |
| F2 | High (reproducibility) | The shipped GHRN runner crashes: `args.index_dtype` is used but never defined. | `project/COMP8851_FINAL_SUBMISSION/source_code/models/ghrn/scripts/run_one.py:322-323` |
| F3 | Medium | No model in the previous benchmark saved per-node predictions, so no reported score can be recomputed independently, only re-read. | `shared/comp8851/artifacts.py` (no prediction artefact) |
| F4 | Medium | The canonical FDCompCN "relation view" has 4 relations. One of them, `homo`, is exactly the union of the three semantic relations (verified), so relation-aware models such as CARE-GNN received a redundant fourth relation. | `evidence/caregnn_fdcompcn_seed2/dataset_view_manifest.json` (homo 7,268 directed edges); `phase2/data.py::semantic_relations` |
| F5 | Medium | Report Table 3 gives 4,398,392 Amazon edges, the count of the `homo` key in Amazon.mat. The graph every run actually used is the union of the three relations: 4,417,576 undirected edges. | `evidence/caregnn_amazon_seed2/dataset_manifest.json` |
| F6 | Low | The Elliptic split is a count-based 40/20/40 after a stable time sort. Time step 20 is shared by train and validation, and step 32 by validation and test. The report states "1–20 / 21–31 / 32–49". The partitions are still in time order, so no future node enters training. | `tests/test_data_and_leakage.py::test_elliptic_is_chronological`; train steps 1–20, validation 20–32, test 32–49 |
| F7 | Low | `protocol.result_dir` has no attempt suffix, so re-running a cell overwrites its evidence. | `shared/comp8851/protocol.py:216` (phase 2 archives instead: `phase2/harness.py::archive_existing`) |
| F8 | Info | Elliptic and FDCompCN registries were regenerated with the unchanged builder. The TR40 sizes and fraud counts equal the previous runs' records exactly. | `phase2/data.py::PREVIOUS_TR40_COUNTS`; `data/canonical/data_status.json` |
| F9 | Info | Duplicate feature rows among eligible nodes: FDCompCN 76, Amazon 293, Elliptic 28, T-Finance 23, T-Social 1,879,517 (10 features), YelpChi 0. This is not label leakage; it is reported for transparency. | `data/canonical/data_status.json` |
| F10 | Security | A Kaggle API token was pasted into the chat on 2026-10-11. It was not used and must be revoked. | `kaggle/README_KAGGLE.md` |

## 4. Datasets: provenance and integrity (all six available)

Every canonical cache was rebuilt from the original public release, or copied from the owner's frozen cache. Its *view hash*, which covers labels, features and edges, equals the hash recorded by the previous CARE-GNN/GHRN runs. **The new models therefore see byte-identical data.**

| Dataset | Source used here | Source file sha256 | Identity check vs previous runs | TR40 partition check |
|---|---|---|---|---|
| YelpChi | data.dgl.ai FraudYelp.zip | YelpChi.mat `fedb35a8…` | homogeneous and relation views identical | 18,381 / 9,191 / 18,382 ✓ |
| Amazon | data.dgl.ai FraudAmazon.zip | Amazon.mat `4b7e3f9c…` | identical | 3,455 / 1,728 / 3,456 ✓ |
| T-Finance | BWGNN authors' Google Drive `tfinance.zip` | graph `b7d853ec…` (= split manifest's source) | identical | 15,742 / 7,871 / 15,744 ✓ |
| Elliptic | PyG mirror of the Kaggle CSVs | features/classes/edges `fd7f8357…` / `93e2e7b2…` / `a35053ba…` (= previous Kaggle files) | identical | 18,626 / 9,313 / 18,625 ✓ |
| FDCompCN | SplitGNN repository @ `32fbdb8`, `data/FDCompCN.zip` | comp.dgl `e252b9a6…` | identical | 2,126 / 1,063 / 2,128 ✓ |
| T-Social | `comp8851.zip` canonical cache `a30fb6a0…` | — | relation view identical | 2,312,426 / 1,156,213 / 2,312,426 ✓ |

## 5. Readiness verdict

There is no unresolved critical correctness or leakage issue in the phase-2 pipeline. Findings F1 to F5 concern the existing 10-model benchmark and are recommended for the team's attention. The comparison table marks F1 with `#`.
