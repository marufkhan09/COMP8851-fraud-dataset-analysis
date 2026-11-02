"""Assemble the final submission folder from the verified deliverable.

Copies the documents, report, tables, figures, evidence, source and scripts into
``COMP8851_FINAL_SUBMISSION/`` with a submission README and a requirements-coverage
map. Trained checkpoints are left behind: they are 427 MB and nothing a reader can
act on. Their location is recorded in the README.

Run:  python build_submission.py
"""

from __future__ import annotations

import csv
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = os.path.dirname(HERE)
PROJECT = os.path.dirname(DELIVERABLE)
OUT = os.path.join(PROJECT, "COMP8851_FINAL_SUBMISSION")


def read_matrix():
    path = os.path.join(DELIVERABLE, "tables", "experiment_matrix.csv")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def copy_tree(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    # Clear the contents rather than the directory itself: on Windows the folder
    # can be held open by a shell or an indexer, and removing the root then fails
    # with WinError 32 even though everything inside it is deletable.
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(OUT):
        path = os.path.join(OUT, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    # 1. documents
    docs = os.path.join(OUT, "documents")
    os.makedirs(docs)
    for name in sorted(os.listdir(DELIVERABLE)):
        if name.endswith(".md"):
            shutil.copy2(os.path.join(DELIVERABLE, name), os.path.join(docs, name))

    # 2. report
    report = os.path.join(OUT, "report")
    os.makedirs(report)
    for name in os.listdir(DELIVERABLE):
        if name.startswith("COMP8851_") and name.endswith((".docx", ".pdf")):
            shutil.copy2(os.path.join(DELIVERABLE, name), os.path.join(report, name))
    screenshots = os.path.join(DELIVERABLE, "screenshots.pdf")
    if os.path.isfile(screenshots):
        shutil.copy2(screenshots, os.path.join(report, "execution_screenshots.pdf"))

    # 3. data, evidence and code
    for folder in ("tables", "figures", "evidence", "source_code", "scripts", "notebooks"):
        copy_tree(os.path.join(DELIVERABLE, folder), os.path.join(OUT, folder))

    # 3a. execution notebooks that live outside the deliverable. The T-Social
    # escalation notebook is the only record of GHRN's five failed rungs, so it
    # ships as evidence rather than as a convenience.
    notebooks = os.path.join(OUT, "notebooks")
    os.makedirs(notebooks, exist_ok=True)
    extra_notebooks = {
        "notebookdb72ab9871.ipynb": "TSocial_GHRN_escalation_ladder.ipynb",
        "COMP8851_TSocial_ONLY.ipynb": "TSocial_execution.ipynb",
        "COMP8851_TFinance_TSocial_Kaggle.ipynb": "TFinance_TSocial_kaggle.ipynb",
        "COMP8851_TFinance_TSocial_ONE_FILE.ipynb": "TFinance_TSocial_one_file.ipynb",
        "COMP8851_care-gnn.ipynb": "CARE-GNN_analysis.ipynb",
        "COMP8851_ghrn.ipynb": "GHRN_analysis.ipynb",
        "COMP8851_Compare.ipynb": "Model_comparison.ipynb",
    }
    for source, target in extra_notebooks.items():
        path = os.path.join(PROJECT, source)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(notebooks, target))

    # 3b. supporting Word documents produced during the project
    supporting = os.path.join(OUT, "supporting_documents")
    os.makedirs(supporting)
    for source in ("CAREGNN_GHRN_Unified_Report.docx",
                   "CAREGNN_GHRN_Findings_Final.docx",
                   "Individual_Contribution_FINAL.docx",
                   "COMP8851_VastAI_Master_Plan_FINAL_v4.4.docx",
                   "COMP8851_Master_Execution_Plan_Updated.docx",
                   "COMP8851_Verified_Paper_and_Code_Links_Updated.docx",
                   "2020-26_Relevant_Papers_Combined_Reviewed.pdf"):
        path = os.path.join(PROJECT, source)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(supporting, source))

    # 3c. the shared package root, which the runtime bundle carries but the
    # deliverable snapshot omitted
    shared_init = os.path.join(OUT, "source_code", "shared", "__init__.py")
    if not os.path.isfile(shared_init):
        os.makedirs(os.path.dirname(shared_init), exist_ok=True)
        with open(shared_init, "w", encoding="utf-8") as handle:
            handle.write('"""COMP8851 shared protocol package."""\n')

    matrix = read_matrix()
    completed = [r for r in matrix if r["status"] == "COMPLETED"]
    failed = [r for r in matrix if r["status"] == "FAILED"]

    # 4. submission README
    readme = f"""# COMP8851 — final submission

**Graph Neural Networks for Fraud Detection: A Controlled Benchmark of
CARE-GNN and GHRN**

Submitted 17 September 2026.

## Start here

1. `report/COMP8851_CARE-GNN_GHRN_Benchmark_Report.pdf` — the report.
2. `documents/00_START_HERE.md` — guide to the full documentation set.
3. `REQUIREMENTS_COVERAGE.md` — every requirement and where it is answered.

## What the benchmark found

**1. Reliability separates the two models more than accuracy does.** GHRN leads
on every headline metric, but three of its four per-dataset wins are worth under
0.015 AUPRC. What does not dissolve under inspection is the failure rate: **3 of
CARE-GNN's 15 runs collapsed during training, against 0 of GHRN's 15**. The one
large margin, on Amazon, is largely that collapse showing up in the mean — the
two healthy CARE-GNN seeds average 0.8490 against GHRN's 0.8756.

**2. GHRN's advantage does not come from the mechanism its paper emphasises.**
Using GHRN's own two stages as an ablation, the heterophily refinement improved
validation AUPRC in only **9 of 15 runs**, and on Amazon — the dataset carrying
its entire average lead — the *unrefined* stage validated better. Refinement pays
off precisely where it measurably makes the graph more homophilic, and costs
where it does not.

**3. FDCompCN defeats both models.** AUROC 0.5527 and 0.5525 against 0.5 for
random ranking. The structural reason is in the graph: 7,268 labelled edges and a
median fraud node with **no labelled fraud neighbour at all**, so there is no
fraud-to-fraud signal to propagate.

**4. A protocol deviation worth acting on.** GHRN shipped its refined stage in
all 15 runs, including the six where the unrefined stage validated higher. The
defect is identified, the fix is implemented in `source_code/`, and the re-run is
outstanding.

## What was done

| | |
|---|---|
| Models | CARE-GNN (Dou et al., CIKM 2020), GHRN (Gao et al., WWW 2023) |
| Datasets | YelpChi, Amazon, T-Finance, T-Social, Elliptic, FDCompCN |
| Required combinations | {len(matrix)} (2 models × 6 datasets) |
| Completed with a test evaluation | {len(completed)} |
| Attempted but failed | {len(failed)} (both T-Social) |
| Skipped without attempting | 0 |
| Final runs | 30 |
| Protocol | vast-v4.4 — TR40, split seed 2, training seeds 2/42/72 |

**Five of the six datasets are complete for both models.** T-Social was
attempted eight times across the two models and never reached a test evaluation.
It was not skipped, and the attempts failed in three distinct ways, each
recorded in `tables/tsocial_attempts.md` with its timings, configuration and
error:

| Failure | Attempts | What it means |
|---|---|---|
| Session cap reached | 1 | CARE-GNN trained 16 epochs over 1 h 55 m and was **still improving** (validation AUPRC 0.1006 at epoch 14) when time ran out |
| CUDA out of memory | 1 | GHRN at published width needs more than a 16 GB T4 — a real scalability finding |
| DGL index-dtype error | 4 | A code defect exposed by scale, hit once the model was small enough to fit. Fixed in `source_code/`, not yet re-run |

Because shrinking GHRN moved the failure from the memory wall straight into the
bug, its true cost on T-Social is still unmeasured. The loss matters: T-Social is
the **most heterophilic graph in the suite** (global heterophily 0.3761 against
YelpChi's 0.2270), so it is exactly the point where a heterophily-targeted method
would be expected to show most.

## Integrity of the numbers

Every figure in this submission comes from a run that actually executed. No
value is estimated, interpolated, extrapolated from another dataset, or taken
from a published paper.

The tables are generated, not typed: `scripts/consolidate_results.py` and its
companions rebuild every table directly from the per-run records in
`evidence/`, and `scripts/consistency_audit.py` verifies that every figure
quoted in the prose matches those tables. It runs 14 checks and all pass — see
`documents/CONSISTENCY_AUDIT.md`.

## Folder map

| Folder | Contents |
|---|---|
| `report/` | The report as Word and PDF, plus execution screenshots |
| `documents/` | The full documentation set, 00–10, and the consistency audit |
| `tables/` | Every table as CSV and Markdown, including the per-seed master record |
| `figures/` | Comparison charts, heatmap, seed spread, and the CSV behind each |
| `evidence/` | Per-run bundles: config, environment, hardware, splits, timings, metrics, logs — including every T-Social attempt and its dataset records |
| `source_code/` | CARE-GNN and GHRN implementations and the shared protocol layer |
| `scripts/` | The scripts that regenerate every table and the report |
| `notebooks/` | Execution and analysis notebooks, including `TSocial_GHRN_escalation_ladder.ipynb` — the record of all five failed GHRN rungs |
| `supporting_documents/` | Project plan, verified paper and code links, the literature review, findings write-ups and the individual contribution statement |

## Not included

Trained model checkpoints (427 MB) and the raw datasets (4.4 GB) are omitted to
keep this package portable. Checkpoints are in
`CLIENT_DELIVERABLE_FINAL/models/`; the datasets are public releases and their
SHA256 hashes are recorded in `evidence/` so any copy can be verified against
what was actually used.
"""
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(readme)

    # 5. requirements coverage
    coverage = """# Requirements coverage

Every requirement of the benchmark specification, where it is answered, and its
honest status. Prepared 17 September 2026.

## Scope

| Requirement | Status | Where |
|---|---|---|
| Two assigned detectors: CARE-GNN and GHRN | **Met** | Throughout |
| All six datasets attempted: YelpChi, Amazon, T-Finance, T-Social, Elliptic, FDCompCN | **Met** — all six attempted, none skipped | `documents/02_EXPERIMENT_MATRIX.md` |
| Test evaluation on every model-dataset pair | **Partial** — 10 of 12; both T-Social cells failed after repeated attempts | `tables/tsocial_attempts.md` |

## Protocol

| Requirement | Status | Where |
|---|---|---|
| Persistent dataset-specific node IDs | **Met** — split seed 2, identical IDs across both models | `documents/07_REPRODUCIBILITY.md` |
| Fixed validation and test sets | **Met** | `evidence/*/split_summary.json` |
| Common metric implementation across models | **Met** — one shared evaluator | `source_code/shared/comp8851/evaluator.py` |
| Common training seeds | **Met** — 2, 42, 72 on every completed cell | `tables/master_per_seed.csv` |
| One controlled optimiser rule | **Met** — Adam, betas 0.9/0.999, eps 1e-8 | `documents/07_REPRODUCIBILITY.md` |
| Equal tuning budget per pair | **Met** — same budget, protocol maximum 12 trials | `evidence/tuning/` |
| Test set never used for tuning, threshold selection or early stopping | **Met** — enforced by the command, not by convention | `documents/06_QA_NOTES.md` |
| Selection on validation AUPRC | **Met, with one documented deviation** — GHRN shipped its refined stage in all 15 runs, including six where the unrefined stage validated higher | `documents/04_GRAPH_CHARACTERISTICS.md` |
| Same hardware across the comparison | **Partial** — T-Finance ran on a Tesla T4, the other four on an RTX A6000. Predictive metrics unaffected; runtime and memory reported per GPU block and never pooled | `documents/05_EFFICIENCY.md` |

## Training-ratio ladder

| Requirement | Status | Where |
|---|---|---|
| TR40 = 40/20/40 | **Met** — every run in this package | `tables/master_mean_sd.csv` |
| TR30, TR20, TR10 with nested pools | **Not met** — not executed for these two models. The nested split pools are generated and verified; closing this needs roughly 90 further runs | `documents/08_LIMITATIONS_AND_GAPS.md` §2 |
| Training sets strictly nested | **Met in the split generator**, unexercised above TR40 | `source_code/shared/comp8851/splits.py` |
| Elliptic chronology preserved | **Met** — validation and test remain later in time | `source_code/shared/comp8851/splits.py` |

## Research questions

| RQ | Focus | Status | Where |
|---|---|---|---|
| RQ1 | Comparative performance under one protocol | **Met** — 8 metrics, 5 datasets, both models, mean ± SD over 3 seeds | `documents/03_RESULTS.md` |
| RQ2 | Imbalance and label scarcity | **Partial** — class imbalance handled and reported throughout; the controlled label-scarcity ladder was not run | `documents/08_LIMITATIONS_AND_GAPS.md` §2 |
| RQ3 | Efficiency and scalability | **Met** — per-epoch time, peak GPU memory, parameter count and epoch counts per GPU block, plus two measured scaling limits on T-Social: CARE-GNN's validation cost (336 s per pass against 62 s of training) and GHRN's memory ceiling (1.38 GiB short on a 14.56 GiB T4) | `documents/05_EFFICIENCY.md` |
| RQ4 | Fraud-graph characteristics | **Met, with a narrowed range** — global and fraud-local heterophily with mean, median and SD for all six datasets; related to both dataset difficulty and model ranking; plus a within-model refinement ablation. T-Social is profiled but has no result, and it is the most heterophilic graph in the suite, so the axis with results attached spans 0.0292–0.2270 rather than 0.0292–0.3761 | `documents/04_GRAPH_CHARACTERISTICS.md` |

## Reporting

| Requirement | Status | Where |
|---|---|---|
| PR-AUC (AUPRC) as the primary metric | **Met** | `documents/03_RESULTS.md` |
| Mean ± standard deviation across seeds | **Met** — sample SD over seeds 2, 42, 72 | All results tables |
| Runtime, inference time, peak memory, parameter count | **Met** | `tables/efficiency.csv`, `evidence/*/summary.json` |
| Heterophily conventions stated, with mean, median and SD | **Met** | `tables/dataset_profile.md` |
| Relation-specific values for multi-relation datasets | **Partial** — the canonical union view is measured and both models verifiably share it; per-relation breakdowns are not reported | `tables/dataset_profile.md` |
| Every combination given an explicit status | **Met** — 12 of 12, none omitted | `documents/02_EXPERIMENT_MATRIX.md` |
| Failed and unstable runs retained, not dropped | **Met** — 3 collapsed runs included in all means and documented | `documents/06_QA_NOTES.md` |
| Reproducible from the recorded artefacts | **Met** — generated tables, 14 automated consistency checks | `documents/CONSISTENCY_AUDIT.md` |

## Summary

Fully met: the two-model scope, all six datasets attempted, the shared protocol
and its integrity guarantees, TR40 execution, RQ1, RQ3, RQ4, and the reporting
requirements.

Not fully met, and stated as such rather than worked around:

1. **T-Social produced no test evaluation** for either model, after eight
   recorded attempts that failed in three distinct ways. Documented with
   timings, configurations and errors, and its graph is still characterised for
   RQ4 from the frozen dataset manifest.
2. **The TR30/TR20/TR10 label-scarcity ladder was not run**, so RQ2 is only
   partly answered.
3. **Two GPU classes** across the completed cells, so runtime and memory are
   reported per block and never pooled.
4. **GHRN's between-stage selection** did not follow the stated validation-AUPRC
   rule. The defect is identified, the fix is implemented in the source, and the
   re-run is outstanding.

Priority order for closing these is in `documents/08_LIMITATIONS_AND_GAPS.md`,
and the runner is `source_code/scripts/run_remaining_work.sh`.
"""
    with open(os.path.join(OUT, "REQUIREMENTS_COVERAGE.md"), "w", encoding="utf-8") as h:
        h.write(coverage)

    total = files = 0
    for dirpath, _, names in os.walk(OUT):
        for name in names:
            files += 1
            total += os.path.getsize(os.path.join(dirpath, name))
    print(f"submission built: {OUT}")
    print(f"  {files} files, {total / 1e6:.1f} MB")
    print(f"  completed {len(completed)} / failed {len(failed)} / total {len(matrix)}")


if __name__ == "__main__":
    main()
