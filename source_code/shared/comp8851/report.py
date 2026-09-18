"""Assemble the final deliverable: required directory layout, tables, plots, manifest.

The controlled specification fixes the result layout:

    results/raw/          per-run evidence bundles, exactly as written
    results/metrics/      per-run metrics and the tuning trial records
    results/tables/       comparison tables, CSV and Markdown and LaTeX
    results/plots/        comparison figures
    results/logs/         terminal logs, one per run
    results/checkpoints/  the selected checkpoint per run
    results/reports/      benchmark report and reproducibility manifest

This module builds that from the raw run bundles. It computes nothing new: every
number here is read from a ``summary.json`` that a real run wrote.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import aggregate as agg
from . import evaluator, protocol

REQUIRED_DIRECTORIES = ("raw", "metrics", "tables", "plots", "logs",
                        "checkpoints", "reports")


def prepare_layout(results_root: Path) -> Dict[str, Path]:
    """Create the required directory structure and return the paths."""
    results_root = Path(results_root)
    paths = {}
    for name in REQUIRED_DIRECTORIES:
        path = results_root / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


# --------------------------------------------------------------------------
# Collection into the layout
# --------------------------------------------------------------------------

def collect_into_layout(results_root: Path, raw_root: Optional[Path] = None
                        ) -> Dict[str, int]:
    """Copy per-run logs, metrics and checkpoints into the flat directories.

    The raw bundles stay authoritative and untouched; these are indexed copies
    so a reader can find every log or checkpoint without walking the tree.
    """
    results_root = Path(results_root)
    paths = prepare_layout(results_root)
    raw_root = Path(raw_root) if raw_root else paths["raw"]

    counts = {"logs": 0, "metrics": 0, "checkpoints": 0}
    for summary_path in sorted(raw_root.rglob("summary.json")):
        run_dir = summary_path.parent
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        model = str(payload.get("model", "unknown")).lower().replace("-", "")
        dataset = payload.get("dataset", "unknown")
        ratio = str(payload.get("ratio", "")).lower()
        seed = payload.get("train_seed", "na")
        stem = f"{model}_{dataset}_{ratio}_seed{seed}"

        log = run_dir / "terminal.log"
        if log.exists():
            shutil.copy2(log, paths["logs"] / f"{stem}.log")
            counts["logs"] += 1

        for name in ("summary.json", "test_metrics.json"):
            source = run_dir / name
            if source.exists():
                shutil.copy2(source, paths["metrics"] / f"{stem}_{name}")
                counts["metrics"] += 1

        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
                target = paths["checkpoints"] / f"{stem}_{checkpoint.name}"
                if not target.exists():
                    shutil.copy2(checkpoint, target)
                    counts["checkpoints"] += 1

    return counts


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def per_dataset_tables(summary: Dict[tuple, Dict[str, Any]], tables_dir: Path,
                       models: List[str], ratio: str = "TR40") -> List[Path]:
    """One CARE-GNN vs GHRN table per dataset, plus the aggregate comparison."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    metrics = list(evaluator.METRIC_KEYS)

    for dataset in protocol.DATASETS:
        rows = []
        for metric in metrics:
            row: Dict[str, Any] = {"metric": metric}
            for model in models:
                entry = summary.get((model, dataset, ratio))
                stats = entry.get(metric) if entry else None
                row[model] = (f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                              if stats and stats["n"] > 1 else
                              f"{stats['mean']:.4f}" if stats else "n/a")
            rows.append(row)

        path = tables_dir / f"comparison_{dataset}.md"
        lines = [f"# {' vs '.join(models)} on {dataset} ({ratio})", "",
                 "| Metric | " + " | ".join(models) + " |",
                 "|---|" + "---|" * len(models)]
        for row in rows:
            lines.append(f"| {row['metric']} | " +
                         " | ".join(str(row[m]) for m in models) + " |")
        any_value = any(row[m] != "n/a" for row in rows for m in models)
        if not any_value:
            lines += ["", "No completed runs for this dataset. "
                          "See the run ledger for the status of each cell."]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)

    return written


def efficiency_table(runs: List[Dict[str, Any]], tables_dir: Path,
                     ratio: str = "TR40") -> Path:
    """Runtime, memory and parameter-count comparison."""
    tables_dir = Path(tables_dir)
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for run in runs:
        if run.get("status") != "COMPLETE" or run.get("ratio") != ratio:
            continue
        grouped.setdefault((run["model"], run["dataset"]), []).append(run)

    rows = []
    for (model, dataset), group in sorted(grouped.items()):
        def mean_of(key):
            values = [r[key] for r in group
                      if r.get(key) not in (None, "") and np.isfinite(float(r[key]))]
            return float(np.mean(values)) if values else None

        rows.append({
            "model": model,
            "dataset": dataset,
            "runs": len(group),
            "mean_train_epoch_seconds": mean_of("mean_train_epoch_seconds"),
            "total_train_seconds": mean_of("total_train_epoch_seconds"),
            "inference_seconds": mean_of("test_inference_seconds"),
            "peak_gpu_memory_mb": mean_of("peak_gpu_memory_mb"),
            "trainable_parameters": mean_of("trainable_parameters"),
            "epochs_completed": mean_of("epochs_completed"),
        })

    path = tables_dir / "efficiency_comparison.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text("", encoding="utf-8")

    markdown = tables_dir / "efficiency_comparison.md"
    lines = [f"# Efficiency comparison ({ratio})", "",
             "Runtime figures are comparable only within one GPU class. "
             "Check `hardware.json` in each run before comparing across hosts.", "",
             "| Model | Dataset | Mean epoch (s) | Inference (s) | Peak GPU (MB) | Parameters |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        def fmt(value, spec=".3f"):
            return format(value, spec) if isinstance(value, float) else "n/a"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {fmt(row['mean_train_epoch_seconds'])} "
            f"| {fmt(row['inference_seconds'])} | {fmt(row['peak_gpu_memory_mb'], '.1f')} "
            f"| {fmt(row['trainable_parameters'], '.0f')} |")
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def metric_comparison_plots(summary: Dict[tuple, Dict[str, Any]], plots_dir: Path,
                            ratio: str = "TR40") -> List[Path]:
    """One grouped-bar figure per required metric."""
    written: List[Path] = []
    for metric in ("auprc", "auroc", "macro_f1", "fraud_precision", "fraud_recall"):
        path = Path(plots_dir) / f"comparison_{metric}_{ratio}.png"
        produced = agg.plot_model_comparison(summary, path, metric=metric, ratio=ratio)
        if produced:
            written.append(produced)
    return written


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def reproducibility_manifest(results_root: Path, runs: List[Dict[str, Any]],
                             dataset_manifest: Optional[Dict[str, Any]] = None,
                             tuning_records: Optional[List[Dict[str, Any]]] = None,
                             extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the single manifest that makes the benchmark reproducible."""
    results_root = Path(results_root)
    reports = results_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    def package_versions() -> Dict[str, Optional[str]]:
        versions: Dict[str, Optional[str]] = {}
        for module in ("torch", "dgl", "numpy", "scipy", "sklearn", "pandas", "sympy"):
            try:
                versions[module] = __import__(module).__version__
            except Exception:
                versions[module] = None
        return versions

    try:
        import torch

        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        gpu_memory = (int(torch.cuda.get_device_properties(0).total_memory)
                      if torch.cuda.is_available() else None)
    except Exception:
        cuda_version, gpu, gpu_memory = None, None, None

    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=180)
        installed = freeze.stdout.strip().splitlines() if freeze.returncode == 0 else []
    except Exception:
        installed = []

    hosts = sorted({str(r.get("hardware_profile_id")) for r in runs
                    if r.get("hardware_profile_id")})

    manifest: Dict[str, Any] = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "repository_commit": protocol.git_commit(),
        "controlled_specification": {
            "target_gpu": protocol.CONTROLLED_GPU,
            "observed_gpu": gpu,
            "gpu_total_memory_bytes": gpu_memory,
            "hardware_profiles_seen": hosts,
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "split_seed": protocol.SPLIT_SEED,
            "train_seeds": list(protocol.TRAIN_SEEDS),
            "optimizer": "Adam",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "max_epochs": protocol.MAX_EPOCHS,
            "patience": protocol.EARLY_STOPPING_PATIENCE,
            "max_tuning_trials": protocol.MAX_TUNING_TRIALS,
            "tuning_ratio": protocol.TUNING_RATIO,
            "tuning_seed": protocol.TUNING_SEED,
            "configuration_selection_metric": "validation AUPRC",
            "threshold_selection_metric": "validation Macro-F1",
            "threshold_grid": {
                "start": protocol.THRESHOLD_MIN,
                "end": protocol.THRESHOLD_MAX,
                "step": protocol.THRESHOLD_STEP,
            },
            "test_evaluations_per_final_run": 1,
            "cpu_thread_cap": protocol.CPU_THREAD_CAP,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cuda_version": cuda_version,
            "packages": package_versions(),
            "pip_freeze": installed,
        },
        "datasets": dataset_manifest or {},
        "tuning": [
            {
                "model": record.get("model"),
                "dataset": record.get("dataset"),
                "budget": record.get("budget"),
                "trials_attempted": record.get("trials_attempted"),
                "trials_successful": record.get("trials_successful"),
                "selected_trial": record.get("selected_trial"),
                "selected_hyperparameters": record.get("selected_hyperparameters"),
                "selected_validation_auprc": record.get("selected_validation_auprc"),
                "test_used_during_tuning": record.get("test_used_during_tuning", False),
            }
            for record in (tuning_records or [])
        ],
        "runs": [
            {
                "model": r.get("model"), "dataset": r.get("dataset"),
                "ratio": r.get("ratio"), "seed": r.get("seed"),
                "status": r.get("status"), "run_id": r.get("run_id"),
                "config_sha256": r.get("config_sha256"),
                "split_sha256": r.get("split_sha256"),
                "dataset_view_id": r.get("dataset_view_id"),
                "selected_threshold": r.get("selected_threshold"),
                "best_validation_epoch": r.get("best_validation_epoch"),
                "path": r.get("path"),
            }
            for r in runs
        ],
    }
    if extra:
        manifest.update(extra)

    path = reports / "reproducibility_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Status matrix
# --------------------------------------------------------------------------

def status_matrix(runs: List[Dict[str, Any]], models: List[str],
                  reports_dir: Path, skipped: Optional[Dict[str, str]] = None
                  ) -> Path:
    """The 12-cell model x dataset matrix with an explicit status for each.

    Allowed statuses are COMPLETED, FAILED, BLOCKED and NOT_RUN. A cell is
    COMPLETED only when a final test evaluation actually executed and its result
    files were saved.
    """
    index: Dict[tuple, List[Dict[str, Any]]] = {}
    for run in runs:
        index.setdefault((run.get("model"), run.get("dataset")), []).append(run)

    rows = []
    for model in models:
        for dataset in protocol.DATASETS:
            group = index.get((model, dataset), [])
            completed = [r for r in group
                         if r.get("status") == "COMPLETE" and r.get("auroc") is not None]
            failed = [r for r in group if r.get("status") == "FAILED"]

            if completed:
                status = "COMPLETED"
                detail = f"{len(completed)} final run(s) with a test evaluation"
            elif failed:
                status = "FAILED"
                detail = str(failed[0].get("error") or "see the run log")[:200]
            elif skipped and dataset in skipped:
                status = "BLOCKED"
                detail = str(skipped[dataset])[:200]
            else:
                status = "NOT_RUN"
                detail = "no run attempted"

            seeds = sorted({r.get("seed") for r in completed if r.get("seed") is not None})
            rows.append({
                "model": model, "dataset": dataset, "status": status,
                "seeds_completed": ",".join(str(s) for s in seeds),
                "runs_completed": len(completed), "runs_failed": len(failed),
                "detail": detail,
            })

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "experiment_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Experiment matrix", "",
             f"Protocol {protocol.PROTOCOL_VERSION}. "
             "A cell is COMPLETED only when a final test evaluation executed and "
             "its results were saved.", "",
             "| Model | Dataset | Status | Seeds completed | Detail |",
             "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['dataset']} | **{row['status']}** | "
                     f"{row['seeds_completed'] or '-'} | {row['detail']} |")

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines += ["", "## Totals", ""]
    for status in ("COMPLETED", "FAILED", "BLOCKED", "NOT_RUN"):
        lines.append(f"- {status}: {counts.get(status, 0)} of {len(rows)}")

    (reports_dir / "experiment_matrix.md").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
    return csv_path


def build(results_root: Path, models: List[str], ratio: str = "TR40",
          dataset_manifest: Optional[Dict[str, Any]] = None,
          tuning_records: Optional[List[Dict[str, Any]]] = None,
          skipped: Optional[Dict[str, str]] = None,
          raw_root: Optional[Path] = None) -> Dict[str, Any]:
    """Build the whole deliverable and return a summary of what was written."""
    results_root = Path(results_root)
    paths = prepare_layout(results_root)
    raw_root = Path(raw_root) if raw_root else paths["raw"]

    runs = agg.collect_runs(raw_root)
    usable = agg.reportable(runs, track="unified")
    summary = agg.aggregate_by_condition(usable)

    copied = collect_into_layout(results_root, raw_root)

    agg.write_run_csv(runs, paths["tables"] / "all_runs.csv")
    (paths["tables"] / "all_runs.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True, default=str), encoding="utf-8")
    agg.write_markdown(summary, paths["tables"] / "benchmark_summary.md",
                       metrics=["auprc", "auroc", "macro_f1", "fraud_recall"], ratio=ratio)
    agg.write_latex(summary, paths["tables"] / "benchmark_table.tex",
                    metric="auprc", ratio=ratio)
    agg.write_ledger(runs, paths["tables"] / "run_ledger.csv", models)
    per_dataset_tables(summary, paths["tables"], models, ratio)
    efficiency_table(runs, paths["tables"], ratio)

    # The label-scarcity figure only means anything with several training
    # ratios. Under a TR40-only protocol it would be a single point per series.
    ratios_present = {r.get("ratio") for r in usable if r.get("ratio")}

    figures: List[Path] = []
    try:
        figures += metric_comparison_plots(summary, paths["plots"], ratio)
        heatmap = agg.plot_metric_heatmap(summary, paths["plots"] / f"metric_heatmap_{ratio}.png",
                                          ratio=ratio)
        if heatmap:
            figures.append(heatmap)
        runtime = agg.plot_runtime(summary, paths["plots"] / f"runtime_{ratio}.png", ratio=ratio)
        if runtime:
            figures.append(runtime)
        if len(ratios_present) > 1:
            scarcity = agg.plot_label_scarcity(
                summary, paths["plots"] / "label_scarcity_auprc.png", metric="auprc")
            if scarcity:
                figures.append(scarcity)
    except ImportError:
        pass

    matrix = status_matrix(runs, models, paths["reports"], skipped)
    manifest = reproducibility_manifest(results_root, runs, dataset_manifest,
                                        tuning_records)

    return {
        "runs_found": len(runs),
        "runs_reportable": len(usable),
        "copied": copied,
        "figures": [str(f) for f in figures],
        "experiment_matrix": str(matrix),
        "manifest": str(manifest),
        "directories": {name: str(path) for name, path in paths.items()},
    }
