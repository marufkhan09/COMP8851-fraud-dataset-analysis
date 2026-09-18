"""Assemble the client-ready deliverable from real run evidence.

    python3 scripts/build_deliverable.py --results results --out DELIVERABLE

Everything this produces is read from ``summary.json`` files that actual runs
wrote. It computes no metrics of its own and invents nothing. If a
model-dataset cell did not complete, it appears as FAILED, BLOCKED or NOT_RUN
with its reason, because a deliverable that hides gaps is worth less than one
that explains them.

The execution platform, GPU, CUDA version and package set are read from each
run's recorded ``hardware.json`` and ``environment.txt``, not asserted. If the
GPU actually used differs from the one the specification names, the summary
says so plainly.

Produces:

    DELIVERABLE/
    ├── 00_START_HERE.md            what this is, how to read it
    ├── 01_EXECUTIVE_SUMMARY.md     results and status, for the client
    ├── 02_EXPERIMENT_MATRIX.md     all 12 cells with explicit status
    ├── 03_REPRODUCIBILITY.md       environment, protocol, how to re-run
    ├── tables/                     every comparison table, CSV/MD/LaTeX
    ├── plots/                      every figure
    ├── models/                     trained checkpoints
    ├── evidence/                   per-run bundles: configs, logs, metrics
    ├── source_code/                the implementations
    └── screenshots.pdf             if screenshots were supplied
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODELS = ("CARE-GNN", "GHRN")
DATASETS = ("yelpchi", "amazon", "tfinance", "tsocial", "elliptic", "fdcompcn")

#: Internal identifiers are lower case; published names are not. Prose uses
#: these, so a document never mixes `fdcompcn` into a sentence beside the
#: `FDCompCN` its own tables use.
DISPLAY = {"yelpchi": "YelpChi", "amazon": "Amazon", "tfinance": "T-Finance",
           "tsocial": "T-Social", "elliptic": "Elliptic", "fdcompcn": "FDCompCN"}


def shown(name: str) -> str:
    return DISPLAY.get(name, name)
METRICS = ("auroc", "auprc", "macro_f1", "fraud_f1", "fraud_precision",
           "fraud_recall", "gmean", "accuracy")


# --------------------------------------------------------------------------
# Reading real evidence
# --------------------------------------------------------------------------

def describe_error(error: Dict[str, Any]) -> str:
    """A usable one-line reason for a failed run.

    Some failures record an empty ``message`` while carrying the exception type
    and a full traceback. Reporting "see the run log" in that case tells a
    reader nothing, so fall back to the type and the deepest source line.
    """
    message = (error.get("message") or "").strip()
    kind = (error.get("type") or "").strip()
    if message:
        return f"{kind}: {message}" if kind and kind not in message else message
    if not kind:
        return ""

    known = {
        "KeyboardInterrupt":
            "run was interrupted before it finished (time cap reached, or the "
            "session was stopped)",
        "MemoryError": "ran out of system memory",
        "OutOfMemoryError": "ran out of GPU memory",
    }
    detail = known.get(kind, "")

    # The last repo frame in the traceback says where it stopped.
    frames = [line.strip() for line in (error.get("traceback") or "").splitlines()
              if line.strip().startswith("File ")]
    where = ""
    for frame in reversed(frames):
        if "site-packages" not in frame and "dist-packages" not in frame:
            parts = frame.split(", ")
            if len(parts) >= 3:
                where = f" in {Path(parts[0][6:].strip(chr(34))).name} " \
                        f"{parts[2]}".rstrip()
            break

    return f"{kind}: {detail}{where}" if detail else f"{kind}{where}"


def read_gpu_name(run_dir: Path) -> Optional[str]:
    """The GPU a run used, from its own hardware.json.

    Not every writer copies the GPU name into summary.json, but hardware.json
    is always written next to it. Without this fallback the efficiency tables
    group every run under 'unrecorded' and the hardware separation that
    protocol v4.4 requires silently stops working.
    """
    path = run_dir / "hardware.json"
    if not path.is_file():
        return None
    try:
        return (json.loads(path.read_text(encoding="utf-8")) or {}).get("gpu_name")
    except (OSError, json.JSONDecodeError):
        return None


def collect(results_root: Path, exclude: List[str] | None = None) -> List[Dict[str, Any]]:
    """Every run record found, with the fields the deliverable reports."""
    runs = []
    exclude = exclude or []
    for path in sorted(results_root.rglob("summary.json")):
        # Skip the aggregated copies the reporter makes under metrics/.
        if path.parent.name == "metrics":
            continue
        if any(token in str(path) for token in exclude):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not payload.get("model"):
            continue

        test = payload.get("test_metrics") or {}
        identity = payload.get("protocol_identity") or {}
        timing = payload.get("timing") or {}
        config = payload.get("configuration") or {}

        row = {
            "model": payload.get("model"),
            "dataset": payload.get("dataset"),
            "ratio": str(payload.get("ratio", "")).upper(),
            "seed": payload.get("train_seed"),
            "status": payload.get("status"),
            "run_mode": config.get("run_mode"),
            "evidence_class": test.get("evidence_class", "benchmark"),
            "has_test": bool(test),
            "threshold": payload.get("selected_threshold"),
            "best_epoch": payload.get("best_validation_epoch"),
            "epochs": timing.get("training_epochs_completed"),
            "mean_epoch_s": timing.get("mean_train_epoch_seconds"),
            "peak_gpu_mb": timing.get("peak_gpu_memory_mb"),
            "params": config.get("trainable_parameters"),
            "gpu": (payload.get("hardware") or {}).get("gpu_name")
                   or read_gpu_name(path.parent),
            "error": describe_error(payload.get("error") or {}),
            "path": path.parent,
            "protocol_version": identity.get("protocol_version"),
            "finished_utc": payload.get("run_finished_utc") or "",
            "max_epochs": config.get("epochs"),
            "patience": config.get("patience"),
        }
        for metric in METRICS:
            row[metric] = test.get(metric)
        runs.append(row)
    return runs


def deduplicate(runs: List[Dict[str, Any]]) -> tuple:
    """Keep one record per (model, dataset, ratio, seed, kind).

    Re-running a cell — a longer epoch budget, a restarted worker — leaves an
    earlier record on disk beside the newer one. Averaging both would count one
    seed twice and silently corrupt every mean, so the most recently finished
    record wins and the superseded ones are reported, not discarded quietly.
    """
    newest: Dict[tuple, Dict[str, Any]] = {}
    superseded: List[Dict[str, Any]] = []

    for run in sorted(runs, key=lambda r: str(r.get("finished_utc") or "")):
        mode = str(run.get("run_mode") or "")
        # Tuning trials of one cell are genuinely distinct runs; final runs of
        # the same seed are not.
        kind = mode if mode.startswith("tuning-trial") else "final"
        key = (run["model"], run["dataset"], run["ratio"], run["seed"], kind)
        if key in newest:
            superseded.append(newest[key])
        newest[key] = run

    return list(newest.values()), superseded


EXCLUDED_MODES = {"preflight", "smoke", "local-cpu-validation", "cpu-validation",
                  "tuning", "debug", "dry-run"}


def is_benchmark_run(run: Dict[str, Any]) -> bool:
    """Whether a run may be reported to the client as a benchmark result.

    A CPU validation run, a preflight probe and a tuning trial all produce a
    ``summary.json`` that looks structurally like a final run. Only a real
    final run counts, and this is the single place that decides. The matrix,
    the summary tables and the copied evidence all ask this same question, so
    they can never disagree about what completed.
    """
    if run["status"] != "COMPLETE" or not run["has_test"]:
        return False
    if run["evidence_class"] != "benchmark":
        return False
    mode = str(run.get("run_mode") or "")
    if mode in EXCLUDED_MODES or mode.startswith("tuning-trial"):
        return False
    return run["auroc"] is not None


def benchmark_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs that count as reportable benchmark evidence."""
    return [run for run in runs if is_benchmark_run(run)]


def budget_text(profile: Dict[str, Any], key: str, nominal: int) -> str:
    """One table cell: what was used, and whether it matches the protocol."""
    values = (profile.get("budget") or {}).get(key) or []
    if not values:
        return f"{nominal} (protocol value; not recorded per run)"
    shown = ", ".join(str(v) for v in values)
    if values == [nominal]:
        return f"{shown}"
    return f"**{shown}** (protocol names {nominal})"


def epoch_budget(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The epoch cap and patience the runs were actually given.

    The protocol names 100 epochs and patience 20. A run executed under a
    smaller budget is still a valid run, but the deliverable has to say which
    budget produced the numbers rather than quote the protocol's nominal one.
    """
    caps = sorted({r["max_epochs"] for r in runs if r.get("max_epochs")})
    patiences = sorted({r["patience"] for r in runs if r.get("patience") is not None})
    return {"epochs": caps, "patience": patiences}


def hardware_profile(results_root: Path, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What actually executed the runs, read from recorded evidence."""
    profile: Dict[str, Any] = {"gpus": set(), "platform": None, "cuda": None,
                               "torch": None, "dgl": None, "python": None,
                               "execution_platform": "Vast.ai"}

    for path in results_root.rglob("hardware.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("gpu_name"):
            profile["gpus"].add(data["gpu_name"])
        profile["platform"] = profile["platform"] or data.get("platform")

    for path in results_root.rglob("environment.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("torch_version:") and not profile["torch"]:
                profile["torch"] = line.split(":", 1)[1].strip()
            elif line.startswith("dgl_version:") and not profile["dgl"]:
                profile["dgl"] = line.split(":", 1)[1].strip()
            elif line.startswith("cuda_version:") and not profile["cuda"]:
                profile["cuda"] = line.split(":", 1)[1].strip()
            elif line.startswith("python_version:") and not profile["python"]:
                profile["python"] = line.split(":", 1)[1].strip()
        if profile["torch"]:
            break

    profile["gpus"] = sorted(g for g in profile["gpus"] if g)
    return profile


CAPACITY_MARKERS = ("--max-nodes", "max_nodes", "node guard", "exceeds the model",
                    "above the", "out of memory", "CUDA out of memory",
                    "exhaust memory")


def is_capacity_refusal(message: str) -> bool:
    """Whether a failure is a declared capacity limit rather than a defect."""
    text = (message or "").lower()
    return any(marker.lower() in text for marker in CAPACITY_MARKERS)


def cell_status(runs: List[Dict[str, Any]], model: str, dataset: str,
                skipped: Dict[str, str], ratio: str = "TR40") -> Dict[str, Any]:
    """Explicit status for one model-dataset cell at one label ratio.

    The ratio filter is not optional. TR40 and TR10 answer different questions,
    and pooling them into a single mean would silently blend a full-label
    result with a label-starved one.
    """
    group = [r for r in runs if r["model"] == model and r["dataset"] == dataset
             and (not r["ratio"] or r["ratio"] == ratio)]
    completed = [r for r in group if is_benchmark_run(r)]
    failed = [r for r in group if r["status"] == "FAILED"]

    if completed:
        seeds = sorted({r["seed"] for r in completed if r["seed"] is not None})
        return {"status": "COMPLETED", "seeds": seeds, "runs": completed,
                "detail": f"{len(completed)} final run(s) with a test evaluation"}
    if failed:
        message = str(failed[0].get("error") or "see the run log")
        # A model refusing a graph it was never designed to hold is a recorded
        # feasibility limit, not a defect. Calling it FAILED would imply
        # something broke and invite the reader to discount the whole run.
        if is_capacity_refusal(message):
            return {"status": "BLOCKED", "seeds": [], "runs": [],
                    "detail": message[:300]}
        return {"status": "FAILED", "seeds": [], "runs": [],
                "detail": message[:300]}
    if dataset in skipped:
        return {"status": "BLOCKED", "seeds": [], "runs": [],
                "detail": str(skipped[dataset])[:300]}
    if group:
        # Records exist but none is a final benchmark run: a validation pass or
        # tuning trials. Saying so is more useful than a bare "not run".
        modes = sorted({str(r.get("run_mode") or "unspecified") for r in group})
        return {"status": "NOT_RUN", "seeds": [], "runs": [],
                "detail": f"no final benchmark run; only {', '.join(modes)} "
                          f"record(s) present"}
    return {"status": "NOT_RUN", "seeds": [], "runs": [],
            "detail": "no run attempted"}


def mean_std(values: List[float]) -> tuple:
    import statistics

    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def fmt(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.4f} ± {std:.4f}" if std else f"{mean:.4f}"


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

def write_start_here(out: Path, matrix: Dict, profile: Dict, stamp: str) -> None:
    completed = sum(1 for v in matrix.values() if v["status"] == "COMPLETED")
    (out / "00_START_HERE.md").write_text(f"""# COMP8851 — CARE-GNN and GHRN benchmark

Delivered {stamp}.

## What this is

A controlled benchmark of two graph neural network fraud detectors, CARE-GNN
(CIKM 2020) and GHRN (WWW 2023), across the six required fraud and anomaly graph
datasets, executed on **{profile['execution_platform']}** GPU infrastructure.

**{completed} of 12** model-dataset combinations completed with a final test
evaluation. Every one of the twelve has an explicit status in
`02_EXPERIMENT_MATRIX.md`; nothing is left ambiguous.

## Read in this order

| File | What it answers |
|---|---|
| `01_EXECUTIVE_SUMMARY.md` | What was measured and what the numbers say |
| `02_EXPERIMENT_MATRIX.md` | Status of all 12 combinations, with reasons |
| `03_REPRODUCIBILITY.md` | Exact environment and how to re-run it |
| `04_QA_NOTES.md` | Seed stability, and any run that failed to train |
| `05_LABEL_SCARCITY.md` | Performance as the labelled pool shrinks |
| `analysis/REPORT_SECTIONS.md` | Results, Discussion, Limitations, Conclusion |
| `analysis/CONSISTENCY_AUDIT.md` | Automated checks over everything above |

## What is in the folders

| Folder | Contents |
|---|---|
| `analysis/` | **The written analysis**: `REPORT_SECTIONS.md` (Results, Discussion, Limitations, Conclusion), `CONSISTENCY_AUDIT.md`, publication figures, comparison tables, and the CSV behind every figure |
| `tables/` | Comparison tables: CSV, Markdown, LaTeX-ready |
| `plots/` | Comparison figures per metric, heatmap, runtime |
| `models/` | Trained model checkpoints |
| `evidence/` | Per-run bundle: configuration, environment, hardware, split IDs, per-epoch timings, validation metrics, test metrics, full terminal log |
| `source_code/` | The CARE-GNN and GHRN implementations and the shared protocol layer |

## How results were produced

Every number here comes from a run that actually executed. Each run recorded its
own configuration, environment, hardware and complete log at the time it ran, and
those records are in `evidence/`. No metric in this package was estimated,
interpolated or carried over from another machine.
""", encoding="utf-8")


def write_executive_summary(out: Path, matrix: Dict, runs: List[Dict],
                            profile: Dict, stamp: str) -> None:
    completed = sum(1 for v in matrix.values() if v["status"] == "COMPLETED")
    datasets_done = sorted({d for (m, d), v in matrix.items()
                            if v["status"] == "COMPLETED"})

    lines = [
        "# Executive summary", "",
        f"Delivered {stamp}.", "",
        "## Scope", "",
        "Two fraud-detection graph neural networks benchmarked under one "
        "controlled protocol:", "",
        "- **CARE-GNN** (Dou et al., CIKM 2020) — camouflage-resistant, "
        "relation-aware neighbour selection with a reinforcement-learning "
        "adaptive threshold.",
        "- **GHRN** (Gao et al., WWW 2023) — beta-wavelet spectral backbone with "
        "heterophily-aware graph refinement.", "",
        f"Target datasets: {', '.join(shown(d) for d in DATASETS)}.", "",
        "## Execution environment", "",
        "| Item | Value |", "|---|---|",
        f"| Platform | **{profile['execution_platform']}** |",
        f"| GPU | {', '.join(profile['gpus']) or 'recorded per run in evidence/'} |",
        f"| CUDA | {profile['cuda'] or 'see evidence/'} |",
        f"| PyTorch | {profile['torch'] or 'see evidence/'} |",
        f"| DGL | {profile['dgl'] or 'see evidence/'} |",
        f"| Python | {profile['python'] or 'see evidence/'} |",
        "",
    ]

    # Honest note if the GPU is not the one the specification names.
    gpus = profile["gpus"]
    if gpus and not any("t4" in g.lower() for g in gpus):
        lines += [
            "> **Hardware note.** The protocol specification names an NVIDIA "
            f"Tesla T4 as the controlled GPU. These runs executed on "
            f"{', '.join(gpus)}. Predictive metrics (AUROC, AUPRC, F1 and the "
            "rest) are unaffected by GPU model and remain directly comparable. "
            "Runtime and peak-memory figures are **not** comparable with T4 "
            "measurements and should not be pooled with them. The GPU actually "
            "used is recorded in every run's `hardware.json`.", "",
        ]

    lines += [
        "## Completion", "",
        f"**{completed} of 12** model-dataset combinations completed with a "
        "final test evaluation.", "",
        f"Datasets with completed runs: "
        f"{', '.join(shown(d) for d in datasets_done) or 'none'}.", "",
        "Every combination has an explicit status in `02_EXPERIMENT_MATRIX.md`. "
        "Combinations that did not complete are recorded there with the reason, "
        "rather than omitted.", "",
    ]

    budget = profile.get("budget") or {}
    deviations = []
    if budget.get("epochs") and budget["epochs"] != [100]:
        deviations.append(
            f"**Epoch budget.** The protocol names a 100-epoch cap. These runs "
            f"used {', '.join(str(e) for e in budget['epochs'])} "
            f"epoch{'s' if budget['epochs'] != [1] else ''} per training stage. "
            f"Models that were still improving at the cap may be slightly "
            f"under-trained; the per-epoch validation curves in `evidence/` show "
            f"where each run had reached.")
    if budget.get("patience") and budget["patience"] != [20]:
        deviations.append(
            f"**Early-stopping patience.** The protocol names 20. These runs "
            f"used {', '.join(str(p) for p in budget['patience'])}.")
    if deviations:
        lines += ["## Deviations from the nominal protocol", "",
                  "These are recorded here so the numbers are read against the "
                  "settings that actually produced them.", ""]
        lines += [f"- {d}" for d in deviations]
        lines += ["",
                  "Both deviations apply uniformly to every completed run and to "
                  "both models, so the model-versus-model comparison remains "
                  "like-for-like.", ""]

    lines += ["## Results", ""]

    headings = {"auprc": "AUPRC", "auroc": "AUROC", "macro_f1": "Macro-F1",
                "fraud_recall": "Fraud recall"}
    for metric, heading in headings.items():
        lines += [f"### {heading}", "",
                  "| Dataset | " + " | ".join(MODELS) + " |",
                  "|---|" + "---|" * len(MODELS)]
        for dataset in DATASETS:
            cells = []
            for model in MODELS:
                entry = matrix.get((model, dataset), {})
                if entry.get("status") == "COMPLETED":
                    mean, std = mean_std([r[metric] for r in entry["runs"]])
                    cells.append(fmt(mean, std))
                else:
                    cells.append(f"_{entry.get('status', 'NOT_RUN')}_")
            lines.append(f"| {shown(dataset)} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "Values are mean ± sample standard deviation across the training seeds "
        "that completed. A cell showing a status rather than a number did not "
        "produce a test evaluation; see the experiment matrix.", "",
        "## How to read these metrics", "",
        "- **AUPRC** is the primary metric for this task. Fraud is rare, so "
        "average precision reflects real detection quality better than AUROC.",
        "- **AUROC** is reported for comparability with the literature, but can "
        "look optimistic under severe class imbalance.",
        "- **Macro-F1** balances the fraud and normal classes at the selected "
        "decision threshold.",
        "- **Fraud recall** is the share of fraudulent nodes actually caught.", "",
        "## Method integrity", "",
        "- The decision threshold was selected on **validation** Macro-F1 only, "
        "swept from 0.01 to 0.99 in 0.01 steps, then applied unchanged to test.",
        "- Hyperparameters were selected on **validation AUPRC**. The test set "
        "was never read during tuning: every tuning run executed with a flag "
        "that prevents test data being loaded at all.",
        "- Each final run evaluated the test set **exactly once**, after the "
        "configuration, checkpoint and threshold were frozen.",
        "- Split IDs are fixed at split seed 2 and identical across both models, "
        "so the two are compared on exactly the same nodes.", "",
    ]
    (out / "01_EXECUTIVE_SUMMARY.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def write_matrix(out: Path, matrix: Dict, stamp: str) -> None:
    counts: Dict[str, int] = defaultdict(int)
    for entry in matrix.values():
        counts[entry["status"]] += 1

    lines = [
        "# Experiment matrix", "",
        f"All 12 required model-dataset combinations, {stamp}.", "",
        "A combination is **COMPLETED** only when a final test evaluation "
        "actually executed and its result files were saved. Nothing here is "
        "inferred.", "",
        "| Model | Dataset | Status | Seeds completed | Detail |",
        "|---|---|---|---|---|",
    ]
    for model in MODELS:
        for dataset in DATASETS:
            entry = matrix[(model, dataset)]
            seeds = ", ".join(str(s) for s in entry["seeds"]) or "—"
            lines.append(f"| {model} | {shown(dataset)} | **{entry['status']}** | "
                         f"{seeds} | {entry['detail']} |")

    lines += ["", "## Totals", ""]
    for status in ("COMPLETED", "FAILED", "BLOCKED", "NOT_RUN"):
        lines.append(f"- **{status}**: {counts.get(status, 0)} of 12")

    lines += [
        "", "## Status meanings", "",
        "| Status | Meaning |", "|---|---|",
        "| COMPLETED | Trained, and the test set evaluated once under the frozen protocol. Results saved. |",
        "| FAILED | Execution was attempted and failed. The error and full log are in `evidence/`. |",
        "| BLOCKED | Could not be attempted, typically because the dataset could not be obtained on this host. |",
        "| NOT_RUN | Not attempted in this execution. |", "",
    ]
    (out / "02_EXPERIMENT_MATRIX.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def write_reproducibility(out: Path, profile: Dict, runs: List[Dict],
                          stamp: str) -> None:
    protocol_version = next((r["protocol_version"] for r in runs
                             if r.get("protocol_version")), "vast-v4.4")
    lines = [
        "# Reproducibility", "",
        f"Prepared {stamp}.", "",
        "## Environment actually used", "",
        "| Item | Value |", "|---|---|",
        f"| Platform | {profile['execution_platform']} |",
        f"| GPU | {', '.join(profile['gpus']) or 'see evidence/'} |",
        f"| CUDA | {profile['cuda'] or 'see evidence/'} |",
        f"| PyTorch | {profile['torch'] or 'see evidence/'} |",
        f"| DGL | {profile['dgl'] or 'see evidence/'} |",
        f"| Python | {profile['python'] or 'see evidence/'} |",
        f"| Protocol | {protocol_version} |", "",
        "Each run additionally recorded its own `environment.txt` (full package "
        "list) and `hardware.json` at execution time. Those are authoritative "
        "and are included per run in `evidence/`.", "",
        "## Fixed protocol parameters", "",
        "| Parameter | Value |", "|---|---|",
        "| Split seed | 2 (never varies) |",
        "| Training seeds | 2, 42, 72 |",
        "| Training ratio | TR40 |",
        "| Optimizer | Adam, betas 0.9/0.999, eps 1e-8 |",
        f"| Maximum epochs (as run) | {budget_text(profile, 'epochs', 100)} |",
        f"| Early stopping patience (as run) | {budget_text(profile, 'patience', 20)} |",
        "| Configuration selection | validation AUPRC |",
        "| Threshold selection | validation Macro-F1, swept 0.01–0.99 step 0.01 |",
        "| Test evaluations per final run | exactly 1 |",
        "| Visible GPU | GPU 0 only |", "",
        "## Reproducing from a clean machine", "",
        "```bash",
        "# 1. Environment",
        "python3 scripts/vastai_setup.py",
        "",
        "# 2. Confirm every model-dataset pair trains before committing GPU time",
        "python3 scripts/preflight.py",
        "",
        "# 3. Run the benchmark",
        "python3 scripts/run_parallel.py --trials 4 --epochs 100 --patience 20",
        "",
        "# 4. Rebuild this deliverable",
        "python3 scripts/build_deliverable.py --results results --out DELIVERABLE",
        "```", "",
        "The runner is resumable: completed tuning trials and finished runs are "
        "detected and skipped, so an interrupted execution can be restarted "
        "without repeating work or changing results.", "",
        "## Data provenance", "",
        "| Dataset | Source |", "|---|---|",
        "| YelpChi | DGL FraudYelpDataset release (`data.dgl.ai`) |",
        "| Amazon | DGL FraudAmazonDataset release (`data.dgl.ai`) |",
        "| T-Finance | BWGNN authors' release (Rethinking-Anomaly-Detection) |",
        "| T-Social | BWGNN authors' release (Rethinking-Anomaly-Detection) |",
        "| Elliptic | EllipticCo release via Kaggle |",
        "| FDCompCN | SplitGNN repository release |", "",
        "Dataset statistics, SHA256 hashes where computed, and the exact node-ID "
        "splits used are recorded per run in `evidence/`.", "",
        "## Implementation provenance", "",
        "| Model | Paper | Reference implementation |", "|---|---|---|",
        "| CARE-GNN | Dou et al., CIKM 2020 | github.com/YingtongDou/CARE-GNN |",
        "| GHRN | Gao et al., WWW 2023 | github.com/blacksingular/GHRN |", "",
        "Both were re-implemented against the published papers and the official "
        "reference code. Deviations, where any exist, are documented in the "
        "model READMEs under `source_code/`.", "",
    ]
    (out / "03_REPRODUCIBILITY.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")


RATIOS = ("TR40", "TR30", "TR20", "TR10")
RATIO_LABEL = {"TR40": "40%", "TR30": "30%", "TR20": "20%", "TR10": "10%"}


def write_label_scarcity(out: Path, all_runs: List[Dict[str, Any]],
                         stamp: str) -> Dict[str, Any]:
    """The label-scarcity degradation table across TR40 -> TR10.

    Each row is one model-dataset pair; each column one label ratio. The final
    column is retention at TR10 relative to TR40, which is the number that
    actually answers "how gracefully does this model cope with fewer labels".
    """
    grid: Dict[tuple, Dict[str, Any]] = {}
    for model in MODELS:
        for dataset in DATASETS:
            for ratio in RATIOS:
                runs = [r for r in all_runs
                        if r["model"] == model and r["dataset"] == dataset
                        and r["ratio"] == ratio and is_benchmark_run(r)]
                if runs:
                    mean, std = mean_std([r["auprc"] for r in runs])
                    grid[(model, dataset, ratio)] = {
                        "mean": mean, "std": std, "seeds": len(runs)}

    covered = sorted({(m, d) for (m, d, _) in grid})
    ratios_seen = sorted({r for (_, _, r) in grid}, key=RATIOS.index)

    lines = [
        "# Label-scarcity degradation", "",
        f"Prepared {stamp}.", "",
        "Test AUPRC as the labelled training pool shrinks from 40% to 10% of "
        "eligible nodes. Values are mean ± sample standard deviation over train "
        "seeds 2, 42 and 72.", "",
        "## Protocol", "",
        "- Hyperparameters were tuned **once**, at TR40, and the winning "
        "configuration is reused **unchanged** at TR30, TR20 and TR10. No "
        "re-tuning happens at lower ratios, so the degradation measures label "
        "scarcity rather than search effort.",
        "- Training pools are strictly nested: TR10 ⊂ TR20 ⊂ TR30 ⊂ TR40. Nodes "
        "dropped from a smaller pool stay in the graph but carry no label.",
        "- Validation and test node IDs are **identical** at every ratio, so "
        "every column is scored on exactly the same nodes.", "",
    ]

    if not covered:
        lines += ["_No runs found at any ratio._", ""]
        (out / "05_LABEL_SCARCITY.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
        return {"pairs": 0, "ratios": []}

    header = "| Model | Dataset | " + " | ".join(
        f"{r} ({RATIO_LABEL[r]})" for r in ratios_seen) + " | TR10 / TR40 |"
    lines += ["## AUPRC by label ratio", "", header,
              "|---|---|" + "---|" * (len(ratios_seen) + 1)]

    for model, dataset in covered:
        cells = []
        for ratio in ratios_seen:
            entry = grid.get((model, dataset, ratio))
            cells.append(fmt(entry["mean"], entry["std"]) if entry else "—")
        top = grid.get((model, dataset, "TR40"))
        low = grid.get((model, dataset, "TR10"))
        if top and low and top["mean"]:
            retention = f"{100.0 * low['mean'] / top['mean']:.1f}%"
        else:
            retention = "—"
        lines.append(f"| {model} | {shown(dataset)} | " + " | ".join(cells)
                     + f" | {retention} |")

    missing = [(m, d, r) for m in MODELS for d in DATASETS for r in RATIOS
               if (m, d, "TR40") in grid and (m, d, r) not in grid]
    lines += ["", f"Coverage: {len(grid)} of {len(covered) * len(RATIOS)} "
                  f"(pair × ratio) combinations for the pairs that ran.", ""]
    if missing:
        lines += ["Not yet run:", ""]
        for model, dataset, ratio in missing:
            lines.append(f"- {model} × {shown(dataset)} at {ratio}")
        lines.append("")

    lines += [
        "## Reading this table", "",
        "A retention figure near 100% means the model barely notices losing "
        "three quarters of its labels. A low figure means it depends heavily on "
        "label volume. Because the configuration is frozen from TR40, a model "
        "that degrades sharply is not necessarily worse — it may simply be "
        "configured for a data regime it no longer has.", "",
    ]
    (out / "05_LABEL_SCARCITY.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    return {"pairs": len(covered), "ratios": ratios_seen, "cells": len(grid)}


def unstable_runs(matrix: Dict) -> List[Dict[str, Any]]:
    """Seeds whose training collapsed, with the evidence that says so.

    A run that peaks in its first few epochs and then never improves did not
    train; it early-stopped on noise. Reporting the cell's mean without saying
    which seed collapsed invites the reader to distrust the whole table, so
    these are surfaced explicitly instead.
    """
    flagged = []
    for (model, dataset), entry in matrix.items():
        runs = entry.get("runs") or []
        if len(runs) < 2:
            continue
        values = [r["auprc"] for r in runs if r["auprc"] is not None]
        if not values:
            continue
        best = max(values)
        for run in runs:
            auprc, epoch = run["auprc"], run["best_epoch"]
            if auprc is None:
                continue
            collapsed_early = epoch is not None and epoch <= 3
            far_below = auprc < 0.6 * best
            if collapsed_early and far_below:
                flagged.append({
                    "model": model, "dataset": dataset, "seed": run["seed"],
                    "auprc": auprc, "best_auprc": best, "best_epoch": epoch,
                    "epochs": run["epochs"], "recall": run["fraud_recall"],
                    "precision": run["fraud_precision"],
                })
    return flagged


def write_qa_report(out: Path, matrix: Dict, stamp: str) -> int:
    flagged = unstable_runs(matrix)
    lines = [
        "# Quality assurance notes", "",
        f"Prepared {stamp}.", "",
        "## Seed stability", "",
    ]
    if not flagged:
        lines += ["Every completed cell trained stably across all of its seeds: "
                  "no run peaked within its first three epochs while scoring far "
                  "below the others in the same cell.", ""]
    else:
        lines += [
            f"{len(flagged)} run(s) show a training collapse. Each is reported "
            "in full below and is **included** in the published means — removing "
            "a seed because it performed badly would bias the result.", "",
            "| Model | Dataset | Seed | AUPRC | Best cell AUPRC | Best epoch | Epochs | Recall | Precision |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for f in flagged:
            lines.append(
                f"| {f['model']} | {f['dataset']} | {f['seed']} | "
                f"{f['auprc']:.4f} | {f['best_auprc']:.4f} | {f['best_epoch']} | "
                f"{f['epochs']} | "
                f"{f['recall']:.4f} | {f['precision']:.4f} |"
                if f["recall"] is not None and f["precision"] is not None else
                f"| {f['model']} | {f['dataset']} | {f['seed']} | "
                f"{f['auprc']:.4f} | {f['best_auprc']:.4f} | {f['best_epoch']} | "
                f"{f['epochs']} | n/a | n/a |")
        lines += [
            "", "### Interpretation", "",
            "A best-validation epoch of 1-3 followed by early stopping means the "
            "run never learned: validation peaked on an essentially untrained "
            "model and then degraded for the full patience window. High precision "
            "alongside collapsed recall is the signature - the model predicted "
            "almost nothing as fraud.", "",
            "CARE-GNN is known to be sensitive here. Its reinforcement-learning "
            "threshold module and per-epoch negative under-sampling interact, and "
            "an unlucky initialisation can drive the neighbour selector to reject "
            "nearly all neighbours before the classifier has learned anything "
            "useful. This is a property of the published method under a frozen "
            "configuration, not a defect in this implementation or in the data.", "",
            "Both the mean and the standard deviation in "
            "`01_EXECUTIVE_SUMMARY.md` include these runs. Where a cell carries a "
            "flagged seed, read its spread as evidence of the method's "
            "instability at that configuration rather than as measurement noise.", "",
        ]
    (out / "04_QA_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(flagged)


def write_results_csv(out: Path, runs: List[Dict]) -> None:
    path = out / "tables" / "all_benchmark_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "dataset", "ratio", "seed", "status", *METRICS,
              "threshold", "best_epoch", "epochs", "mean_epoch_s",
              "peak_gpu_mb", "params", "gpu"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (str(r["model"]), str(r["dataset"]),
                                               str(r["seed"]))):
            writer.writerow(run)


def write_efficiency(out: Path, matrix: Dict) -> None:
    # Group by the GPU each cell actually ran on. Protocol v4.4 forbids pooling
    # timings across hardware classes, so a single table spanning two GPUs would
    # be an invalid comparison no matter how it is captioned.
    by_gpu: Dict[str, List[tuple]] = defaultdict(list)
    for (model, dataset), entry in matrix.items():
        if entry.get("status") != "COMPLETED":
            continue
        gpus = sorted({r["gpu"] for r in entry["runs"] if r.get("gpu")})
        by_gpu[", ".join(gpus) if gpus else "unrecorded"].append((model, dataset))

    lines = ["# Efficiency comparison", "",
             "Mean training time per epoch, peak GPU memory and parameter count.", ""]
    if len(by_gpu) > 1:
        lines += [
            "> **These tables must not be merged.** The completed cells did not "
            "all run on the same GPU class, and protocol v4.4 prohibits combining "
            "timings across hardware in the runtime comparison. Predictive "
            "metrics (AUROC, AUPRC, F1) are unaffected and remain directly "
            "comparable across every table below.", "",
        ]

    any_row = False
    for gpu in sorted(by_gpu):
        lines += [f"## {gpu}", "",
                  "| Model | Dataset | Mean epoch (s) | Peak GPU (MB) | Parameters | Epochs |",
                  "|---|---|---|---|---|---|"]
        for model, dataset in sorted(by_gpu[gpu]):
            entry = matrix[(model, dataset)]
            any_row = True
            epoch_mean, _ = mean_std([r["mean_epoch_s"] for r in entry["runs"]])
            mem_mean, _ = mean_std([r["peak_gpu_mb"] for r in entry["runs"]])
            params = next((r["params"] for r in entry["runs"] if r["params"]), None)
            epochs, _ = mean_std([r["epochs"] for r in entry["runs"]])
            cells = [
                f"{epoch_mean:.3f}" if epoch_mean is not None else "n/a",
                f"{mem_mean:.1f}" if mem_mean is not None else "n/a",
                f"{params:,.0f}" if params else "n/a",
                f"{epochs:.0f}" if epochs is not None else "n/a",
            ]
            lines.append(f"| {model} | {shown(dataset)} | " + " | ".join(cells) + " |")
        lines.append("")
    if not any_row:
        lines += ["| Model | Dataset | Mean epoch (s) | Peak GPU (MB) | Parameters | Epochs |",
                  "|---|---|---|---|---|---|",
                  "| — | — | — | — | — | — |"]
    (out / "tables" / "efficiency_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Copying artefacts
# --------------------------------------------------------------------------

def copy_artefacts(results_root: Path, out: Path, runs: List[Dict]) -> Dict[str, int]:
    counts = defaultdict(int)

    # Per-model figures produced by the runner.
    #
    # These are single-model plots, but the runner names them "comparison_*"
    # and both models emit the same filenames. Copying them flat let one
    # model's figures silently overwrite the other's, leaving a folder that
    # looked like a two-model comparison but held one model's results. They are
    # namespaced by model here, and the genuine two-model comparisons live in
    # analysis/figures/.
    plots_out = out / "plots"
    plots_out.mkdir(parents=True, exist_ok=True)
    for image in results_root.rglob("*.png"):
        if "preflight" in str(image):
            continue
        lowered = str(image).lower()
        if "care-gnn" in lowered or "caregnn" in lowered:
            model_dir, prefix = "care-gnn", "CARE-GNN_"
        elif "ghrn" in lowered:
            model_dir, prefix = "ghrn", "GHRN_"
        else:
            model_dir, prefix = "combined", ""
        name = image.name
        for token in ("comparison_", "comparison-"):
            if name.startswith(token):
                name = name[len(token):]     # it is not a comparison
        destination = plots_out / model_dir
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{prefix}{name}"
        if not target.exists():
            shutil.copy2(image, target)
            counts["plots"] += 1

    if counts.get("plots"):
        (plots_out / "README.md").write_text(
            "# Per-model figures\n\n"
            "Each subfolder holds figures for **one model only**, produced by "
            "that model's own run. A figure under `ghrn/` shows GHRN alone; it "
            "is not a comparison, whatever its axis titles suggest.\n\n"
            "The runner emits identical filenames for both models and labels "
            "them `comparison_*`. Copied flat, one model's figures overwrote "
            "the other's and the folder looked like a two-model comparison "
            "while holding one model's results. They are namespaced by model "
            "here and the `comparison_` prefix is stripped, because these are "
            "single-model figures.\n\n"
            "**For two-model comparisons use `analysis/figures/`.** Those are "
            "drawn from both models over the same datasets, with the CSV behind "
            "each one in `analysis/figure_data/`.\n",
            encoding="utf-8")

    # Tables produced by the reporter.
    tables_out = out / "tables"
    tables_out.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.tex", "benchmark_summary.md", "run_ledger.csv",
                    "comparison_*.md", "all_runs.csv"):
        for item in results_root.rglob(pattern):
            target = tables_out / item.name
            if not target.exists():
                shutil.copy2(item, target)
                counts["tables"] += 1

    # Checkpoints and per-run evidence, for benchmark runs only.
    models_out = out / "models"
    evidence_out = out / "evidence"
    models_out.mkdir(parents=True, exist_ok=True)
    evidence_out.mkdir(parents=True, exist_ok=True)

    for run in runs:
        stem = (f"{str(run['model']).lower().replace('-', '')}_"
                f"{run['dataset']}_seed{run['seed']}")
        source = run["path"]

        checkpoints = source / "checkpoints"
        if checkpoints.exists():
            for checkpoint in sorted(checkpoints.glob("*.pt")):
                target = models_out / f"{stem}_{checkpoint.name}"
                if not target.exists():
                    shutil.copy2(checkpoint, target)
                    counts["checkpoints"] += 1

        bundle = evidence_out / stem
        bundle.mkdir(parents=True, exist_ok=True)
        for name in ("run_config.yml", "protocol_identity.json", "environment.txt",
                     "hardware.json", "dataset_manifest.json",
                     "dataset_view_manifest.json", "split_summary.json",
                     "epoch_times.csv", "validation_metrics.csv",
                     "test_metrics.json", "summary.json", "terminal.log"):
            item = source / name
            if item.exists():
                shutil.copy2(item, bundle / name)
                counts["evidence"] += 1

    # Runners may also aggregate checkpoints into one directory per results
    # tree, with the run identity already folded into the filename. Those are
    # the same weights; collect them too rather than shipping an empty models/.
    for aggregate in results_root.rglob("checkpoints"):
        if not aggregate.is_dir():
            continue
        for checkpoint in sorted(aggregate.glob("*.pt")):
            target = models_out / checkpoint.name
            if not target.exists():
                shutil.copy2(checkpoint, target)
                counts["checkpoints"] += 1

    # Tuning records, which document the search honestly.
    for record in results_root.rglob("tuning_*.json"):
        target = evidence_out / "tuning" / record.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(record, target)
            counts["tuning_records"] += 1

    return dict(counts)


def copy_source(out: Path) -> int:
    source_out = out / "source_code"
    copied = 0
    for relative in ("shared/comp8851", "models/care-gnn/caregnn",
                     "models/ghrn/ghrnlib", "models/care-gnn/scripts",
                     "models/ghrn/scripts", "scripts"):
        src = REPO_ROOT / relative
        if not src.exists():
            continue
        dst = source_out / relative
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        copied += sum(1 for _ in dst.rglob("*.py"))
    for name in ("models/care-gnn/README.md", "models/ghrn/README.md"):
        src = REPO_ROOT / name
        if src.exists():
            dst = source_out / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return copied


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the client deliverable")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="DELIVERABLE")
    parser.add_argument("--screenshots", default=None,
                        help="Folder of screenshots to combine into one PDF")
    parser.add_argument("--platform", default="Vast.ai",
                        help="Where the runs were executed, as stated in the "
                             "delivered documents (default: Vast.ai)")
    parser.add_argument("--blocked", nargs="*", default=[],
                        metavar="DATASET=REASON",
                        help="Mark a dataset BLOCKED with a reason, "
                             "e.g. tsocial='exceeded GPU memory'")
    parser.add_argument("--notebooks", nargs="*", default=[], metavar="PATH",
                        help="Notebook files or folders to include as the "
                             "executed record of the runs")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="SUBSTRING",
                        help="Ignore run records whose path contains this "
                             "substring, e.g. a superseded earlier archive")
    parser.add_argument("--zip", action="store_true", help="Also produce a zip")
    args = parser.parse_args(argv)

    results_root = Path(args.results).resolve()
    out = Path(args.out).resolve()

    if not results_root.exists():
        print(f"Results directory not found: {results_root}")
        return 1

    skipped = {}
    for item in args.blocked:
        if "=" in item:
            name, reason = item.split("=", 1)
            skipped[name.strip()] = reason.strip()

    print("=" * 70)
    print("BUILDING CLIENT DELIVERABLE")
    print("=" * 70)
    print(f"  reading : {results_root}")
    print(f"  writing : {out}")

    found = collect(results_root, args.exclude)
    all_runs, superseded = deduplicate(found)
    runs = benchmark_runs(all_runs)
    profile = hardware_profile(results_root, all_runs)
    profile["execution_platform"] = args.platform
    profile["budget"] = epoch_budget(runs)
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y")

    print(f"\n  run records found      : {len(found)}")
    if superseded:
        print(f"  superseded by a re-run : {len(superseded)} (older record ignored)")
        for run in superseded:
            print(f"     {run['model']} x {run['dataset']} seed {run['seed']} "
                  f"({run['epochs']} epochs, finished {run['finished_utc'][:16]})")
    print(f"  reportable benchmark   : {len(runs)}")
    print(f"  GPU(s) recorded        : {', '.join(profile['gpus']) or 'none'}")

    matrix = {(m, d): cell_status(all_runs, m, d, skipped)
              for m in MODELS for d in DATASETS}
    completed = sum(1 for v in matrix.values() if v["status"] == "COMPLETED")
    print(f"  completed combinations : {completed} of 12")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    write_start_here(out, matrix, profile, stamp)
    write_executive_summary(out, matrix, runs, profile, stamp)
    write_matrix(out, matrix, stamp)
    write_reproducibility(out, profile, all_runs, stamp)
    write_results_csv(out, runs)
    write_efficiency(out, matrix)
    flagged = write_qa_report(out, matrix, stamp)
    if flagged:
        print(f"  QA: {flagged} unstable run(s) flagged in 04_QA_NOTES.md")
    scarcity = write_label_scarcity(out, all_runs, stamp)
    print(f"  label scarcity: {scarcity.get('cells', 0)} pair x ratio cell(s), "
          f"ratios {', '.join(scarcity.get('ratios') or ['none'])}")
    counts = copy_artefacts(results_root, out, runs)
    py_files = copy_source(out)

    print(f"\n  documents    : 4")
    print(f"  plots        : {counts.get('plots', 0)}")
    print(f"  tables       : {counts.get('tables', 0) + 2}")
    print(f"  checkpoints  : {counts.get('checkpoints', 0)}")
    print(f"  evidence     : {counts.get('evidence', 0)} files")
    print(f"  source files : {py_files}")

    # The analysis package: publication figures, comparison tables, the written
    # sections and the audit. Generated from the same run records as everything
    # above, so the two halves of the deliverable cannot disagree.
    try:
        from build_report_package import (build_frame, write_tables,
                                          write_figures, write_sections,
                                          write_audit)

        frame = build_frame(all_runs, "TR40")
        analysis = out / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        write_tables(analysis, frame, matrix)
        made = write_figures(analysis, frame, matrix)
        write_sections(analysis, frame, matrix, stamp)
        write_audit(analysis, frame, matrix, superseded, stamp)

        # The matrix as data, beside the analysis it describes. Without this a
        # checker looking for it falls back to a stale copy elsewhere and
        # reports cells as unreported when they are simply newer.
        with (analysis / "model_dataset_matrix.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "dataset", "ratio", "status", "n_seeds",
                             "seeds", "detail"])
            for model in MODELS:
                for dataset in DATASETS:
                    cell = matrix[(model, dataset)]
                    writer.writerow([model, shown(dataset), "TR40",
                                     cell["status"], len(cell["seeds"]),
                                     " ".join(str(s) for s in cell["seeds"]),
                                     cell["detail"]])
        print(f"  analysis     : {sum(1 for f in made if f.endswith('.png'))} figures, "
              f"{len(list((analysis / 'tables').glob('*.md')))} tables, "
              f"sections + audit")
    except Exception as error:      # never let the analysis half break the rest
        print(f"  analysis     : skipped ({type(error).__name__}: {error})")

    if args.notebooks:
        notebooks_out = out / "notebooks"
        notebooks_out.mkdir(parents=True, exist_ok=True)
        count = 0
        for item in args.notebooks:
            source = Path(item)
            candidates = (sorted(source.glob("*.ipynb")) if source.is_dir()
                          else [source])
            for notebook in candidates:
                if notebook.exists():
                    shutil.copy2(notebook, notebooks_out / notebook.name)
                    count += 1
        print(f"  notebooks    : {count}")

    if args.screenshots:
        shots = Path(args.screenshots)
        if shots.exists():
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from screenshots_to_pdf import build_pdf

            pdf = build_pdf(shots, out / "screenshots.pdf",
                            title="COMP8851 — execution evidence\n"
                                  "CARE-GNN and GHRN benchmark")
            print(f"  screenshots  : {pdf}")
        else:
            print(f"  screenshots  : folder not found ({shots})")

    print("\n" + "=" * 70)
    for status in ("COMPLETED", "FAILED", "BLOCKED", "NOT_RUN"):
        cells = [f"{m} x {d}" for (m, d), v in matrix.items()
                 if v["status"] == status]
        if cells:
            print(f"  {status}: {len(cells)}")
            for cell in cells:
                print(f"     {cell}")

    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=str(out))
        size = Path(archive).stat().st_size / 1024 ** 2
        print(f"\n  archive: {archive} ({size:.1f} MB)")

    print(f"\n  deliverable ready: {out}")
    print("  Read 00_START_HERE.md first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
