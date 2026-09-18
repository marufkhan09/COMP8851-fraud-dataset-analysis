"""Aggregate COMP8851 run artefacts into tables and comparison figures.

Walks ``results/`` for ``summary.json`` files, collects every completed run, and
writes the reporting bundle required by master plan v4.4 section 10:

    results/reports/benchmark_results.csv      one row per run
    results/reports/benchmark_results.json     machine-readable, same content
    results/reports/benchmark_summary.md       mean +/- sd across seeds
    results/reports/benchmark_table.tex        LaTeX version of the same table
    results/reports/run_ledger.csv             status of every attempted cell
    results/plots/*.png                        comparison figures

Only runs whose ``status`` is COMPLETE contribute to the metric tables. Failed
and partial runs are still listed in the ledger, with their status and error,
so a missing cell is always visibly a failure rather than a silent gap.

Usage:
    python shared/comp8851/aggregate.py
    python shared/comp8851/aggregate.py --results-root results --metric auprc
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.comp8851 import evaluator, protocol  # noqa: E402

# --------------------------------------------------------------------------
# Palette
#
# Slots 1 and 2 of the validated reference categorical palette, used verbatim.
# That palette's first three slots are documented as clearing the all-pairs
# CVD and normal-vision floors in both modes, so no re-validation is needed as
# long as these hexes are not altered. Changing them means re-running
# scripts/validate_palette.js before shipping.
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834"]          # blue, orange
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

RATIO_ORDER = ["TR40", "TR30", "TR20", "TR10"]
DATASET_ORDER = list(protocol.DATASETS)


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def collect_runs(results_root: Path) -> List[Dict[str, Any]]:
    """Read every ``summary.json`` under ``results_root``."""
    runs: List[Dict[str, Any]] = []
    for path in sorted(Path(results_root).rglob("summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[aggregate] WARNING could not read {path}: {exc}")
            continue

        test_metrics = payload.get("test_metrics") or {}
        identity = payload.get("protocol_identity") or {}
        timing = payload.get("timing") or {}
        configuration = payload.get("configuration") or {}

        row: Dict[str, Any] = {
            "run_id": payload.get("run_id"),
            "status": payload.get("status"),
            "model": payload.get("model"),
            "dataset": payload.get("dataset"),
            "track": payload.get("track"),
            "ratio": (payload.get("ratio") or "").upper(),
            "seed": payload.get("train_seed"),
            "protocol_version": identity.get("protocol_version"),
            "hardware_profile_id": identity.get("hardware_profile_id"),
            "dataset_view_id": identity.get("dataset_view_id"),
            "split_sha256": identity.get("split_sha256"),
            "config_sha256": identity.get("config_sha256"),
            "selection_metric": payload.get("selection_metric"),
            "best_validation_epoch": payload.get("best_validation_epoch"),
            "selected_threshold": payload.get("selected_threshold"),
            "run_mode": configuration.get("run_mode"),
            "graph_backend": configuration.get("graph_backend"),
            "evidence_class": test_metrics.get("evidence_class", "benchmark"),
            "epochs_completed": timing.get("training_epochs_completed"),
            "mean_train_epoch_seconds": timing.get("mean_train_epoch_seconds"),
            "median_train_epoch_seconds": timing.get("median_train_epoch_seconds"),
            "std_train_epoch_seconds": timing.get("std_train_epoch_seconds"),
            "total_train_epoch_seconds": timing.get("total_train_epoch_seconds"),
            "peak_gpu_memory_mb": timing.get("peak_gpu_memory_mb"),
            "trainable_parameters": configuration.get("trainable_parameters"),
            "test_inference_seconds": test_metrics.get("inference_seconds"),
            "error": (payload.get("error") or {}).get("message"),
            "path": str(path.parent),
        }
        for key in evaluator.METRIC_KEYS:
            row[key] = test_metrics.get(key)
        runs.append(row)
    return runs


#: Run modes that are evidence of code correctness, not benchmark performance.
#: A run tagged with any of these is excluded from the metric tables, because
#: it was produced outside the frozen hardware and protocol conditions.
NON_BENCHMARK_RUN_MODES = {
    "local-cpu-validation",
    "smoke",
    "author-reproduction",
    "tuning",
}


def reportable(runs: List[Dict[str, Any]], track: str = "unified",
               allow_run_modes: Optional[set] = None) -> List[Dict[str, Any]]:
    """Runs eligible for the metric tables.

    A run qualifies only if it completed, belongs to the requested track, is not
    a smoke test, and was produced under a controlled run mode. Smoke metrics
    and local validation runs are feasibility or correctness evidence; letting
    either into a benchmark table would misrepresent it as a measured result.
    """
    excluded = NON_BENCHMARK_RUN_MODES - (allow_run_modes or set())
    keep = []
    for run in runs:
        if run["status"] != "COMPLETE":
            continue
        if track and run["track"] != track:
            continue
        if run.get("evidence_class") == "feasibility_only":
            continue
        if (run.get("run_mode") or "") in excluded:
            continue
        if run.get("auroc") is None:
            continue
        keep.append(run)
    return keep


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def write_run_csv(runs: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not runs:
        path.write_text("", encoding="utf-8")
        return
    fields = list(runs[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def aggregate_by_condition(runs: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    """Group runs by (model, dataset, ratio) and summarise across seeds."""
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["model"], run["dataset"], run["ratio"]), []).append(run)

    summary: Dict[tuple, Dict[str, Any]] = {}
    for key, group in grouped.items():
        entry: Dict[str, Any] = {
            "seeds": sorted(r["seed"] for r in group if r["seed"] is not None),
            "runs": len(group),
        }
        entry.update(evaluator.summarise_seeds(group))
        times = [r["mean_train_epoch_seconds"] for r in group
                 if r.get("mean_train_epoch_seconds") is not None]
        if times:
            entry["mean_train_epoch_seconds"] = float(np.mean(times))
        memory = [r["peak_gpu_memory_mb"] for r in group
                  if r.get("peak_gpu_memory_mb") is not None]
        if memory:
            entry["peak_gpu_memory_mb"] = float(np.max(memory))
        summary[key] = entry
    return summary


def _format(entry: Optional[Dict[str, Any]], metric: str) -> str:
    if not entry or metric not in entry:
        return "n/a"
    stats = entry[metric]
    if stats["n"] > 1:
        return f"{stats['mean']:.4f} ± {stats['std']:.4f}"
    return f"{stats['mean']:.4f}"


def write_markdown(summary: Dict[tuple, Dict[str, Any]], path: Path,
                   metrics: List[str], ratio: str = "TR40") -> None:
    models = sorted({key[0] for key in summary})
    lines = [
        f"# COMP8851 benchmark summary ({ratio}, unified track)",
        "",
        f"Protocol {protocol.PROTOCOL_VERSION}. Values are mean ± sample standard "
        f"deviation across training seeds {', '.join(str(s) for s in protocol.TRAIN_SEEDS)}. "
        "`n/a` means the cell was not run or did not complete; see the run ledger "
        "for its status.",
        "",
    ]
    for metric in metrics:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(models) + " |")
        lines.append("|---|" + "---|" * len(models))
        for dataset in DATASET_ORDER:
            cells = [_format(summary.get((model, dataset, ratio)), metric) for model in models]
            lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(summary: Dict[tuple, Dict[str, Any]], path: Path,
                metric: str = "auprc", ratio: str = "TR40") -> None:
    models = sorted({key[0] for key in summary})
    lines = [
        "% Generated by shared/comp8851/aggregate.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Test {metric.upper()} at {ratio} under COMP8851 protocol "
        f"{protocol.PROTOCOL_VERSION}. Mean $\\pm$ standard deviation over "
        f"seeds {', '.join(str(s) for s in protocol.TRAIN_SEEDS)}.}}",
        f"\\label{{tab:comp8851-{metric}-{ratio.lower()}}}",
        "\\begin{tabular}{l" + "r" * len(models) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(models) + " \\\\",
        "\\midrule",
    ]
    for dataset in DATASET_ORDER:
        cells = [_format(summary.get((model, dataset, ratio)), metric).replace("±", "$\\pm$")
                 for model in models]
        lines.append(f"{dataset} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ledger(runs: List[Dict[str, Any]], path: Path, models: List[str],
                 ratios: Optional[List[str]] = None) -> None:
    """Every required cell of the matrix with its current status.

    ``ratios`` defaults to the ratios actually present in the results, falling
    back to TR40. Listing ratios the protocol never asked for would fill the
    ledger with NOT_RUN rows that are not real gaps.
    """
    if ratios is None:
        seen = sorted({r.get("ratio") for r in runs if r.get("ratio")},
                      key=lambda value: RATIO_ORDER.index(value)
                      if value in RATIO_ORDER else 99)
        ratios = seen or ["TR40"]

    index = {(r["model"], r["dataset"], r["ratio"], r["seed"]): r for r in runs}
    rows = []
    for model in models:
        for dataset in DATASET_ORDER:
            for ratio in ratios:
                for seed in protocol.TRAIN_SEEDS:
                    run = index.get((model, dataset, ratio, seed))
                    rows.append({
                        "model": model, "dataset": dataset, "ratio": ratio, "seed": seed,
                        "status": run["status"] if run else "NOT_RUN",
                        "evidence_class": run.get("evidence_class") if run else "",
                        "auprc": run.get("auprc") if run else "",
                        "auroc": run.get("auroc") if run else "",
                        "error": run.get("error") if run else "",
                        "path": run.get("path") if run else "",
                    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def plot_model_comparison(summary: Dict[tuple, Dict[str, Any]], path: Path,
                          metric: str = "auprc", ratio: str = "TR40") -> Optional[Path]:
    """Grouped bars: one group per dataset, one bar per model, seed error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({key[0] for key in summary})
    if not models:
        return None

    datasets = [d for d in DATASET_ORDER
                if any((m, d, ratio) in summary for m in models)]
    if not datasets:
        return None

    figure, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(datasets), dtype=float)

    for index, model in enumerate(models):
        means, errors = [], []
        for dataset in datasets:
            entry = summary.get((model, dataset, ratio))
            stats = entry.get(metric) if entry else None
            means.append(stats["mean"] if stats else np.nan)
            errors.append(stats["std"] if stats else 0.0)
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(positions + offset, means, width * 0.92,
                      label=model, color=SERIES[index % len(SERIES)],
                      zorder=3, linewidth=0)
        ax.errorbar(positions + offset, means, yerr=errors, fmt="none",
                    ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=4)
        # Direct labels: with only two series every bar can carry its value.
        for bar, value in zip(bars, means):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                        f"{value:.3f}", ha="center", va="bottom",
                        color=INK_SECONDARY, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(datasets, color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, 1.05)
    # With two or more series identity must not rest on colour alone, so a
    # legend is always present. A single series is named by the title instead.
    title = f"Test {metric.upper()} by dataset ({ratio})"
    if len(models) == 1:
        title = f"{models[0]}: test {metric.upper()} by dataset ({ratio})"
    _style(ax, title=title, ylabel=metric.upper())
    if len(models) > 1:
        legend = ax.legend(frameon=False, fontsize=10, ncol=len(models), loc="upper right")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    figure.text(0.01, 0.01,
                f"Mean ± sd over seeds {', '.join(str(s) for s in protocol.TRAIN_SEEDS)}; "
                f"protocol {protocol.PROTOCOL_VERSION}. Missing bars were not run.",
                color=INK_MUTED, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=[0, 0.03, 1, 1])
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_metric_heatmap(summary: Dict[tuple, Dict[str, Any]], path: Path,
                        ratio: str = "TR40") -> Optional[Path]:
    """Sequential heatmap of every metric for every model-dataset cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    models = sorted({key[0] for key in summary})
    if not models:
        return None
    metrics = list(evaluator.METRIC_KEYS)
    rows = [(model, dataset) for model in models for dataset in DATASET_ORDER
            if (model, dataset, ratio) in summary]
    if not rows:
        return None

    matrix = np.full((len(rows), len(metrics)), np.nan)
    for r, (model, dataset) in enumerate(rows):
        entry = summary.get((model, dataset, ratio), {})
        for c, metric in enumerate(metrics):
            if metric in entry:
                matrix[r, c] = entry[metric]["mean"]

    cmap = LinearSegmentedColormap.from_list("comp8851_blue", SEQUENTIAL)
    cmap.set_bad(color="#f0efec")

    figure, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 2.4), facecolor=SURFACE)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1,
                      aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.upper() for m in metrics], rotation=35, ha="right",
                       color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{model}  ·  {dataset}" for model, dataset in rows],
                       color=INK_SECONDARY, fontsize=9)

    # Every cell is labelled, so magnitude never rests on colour alone.
    for r in range(len(rows)):
        for c in range(len(metrics)):
            value = matrix[r, c]
            if np.isfinite(value):
                ax.text(c, r, f"{value:.3f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if value > 0.55 else INK_PRIMARY)

    ax.set_title(f"Test metrics ({ratio}, unified track)", color=INK_PRIMARY,
                 fontsize=12, pad=12, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_label_scarcity(summary: Dict[tuple, Dict[str, Any]], path: Path,
                        metric: str = "auprc") -> Optional[Path]:
    """Small multiples: one panel per dataset, metric against training ratio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({key[0] for key in summary})
    datasets = [d for d in DATASET_ORDER
                if any((m, d, r) in summary for m in models for r in RATIO_ORDER)]
    if not datasets or not models:
        return None

    columns = min(3, len(datasets))
    rows = int(np.ceil(len(datasets) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows),
                                facecolor=SURFACE, squeeze=False)

    # Ratios ascend left to right so the x-axis reads as "more labels".
    ordered = list(reversed(RATIO_ORDER))
    x = np.arange(len(ordered))

    for index, dataset in enumerate(datasets):
        ax = axes[index // columns][index % columns]
        for model_index, model in enumerate(models):
            values, errors = [], []
            for ratio in ordered:
                entry = summary.get((model, dataset, ratio))
                stats = entry.get(metric) if entry else None
                values.append(stats["mean"] if stats else np.nan)
                errors.append(stats["std"] if stats else 0.0)
            colour = SERIES[model_index % len(SERIES)]
            ax.errorbar(x, values, yerr=errors, color=colour, linewidth=2.0,
                        marker="o", markersize=6, capsize=3, label=model,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, color=INK_SECONDARY, fontsize=9)
        ax.set_ylim(0, 1.05)
        _style(ax, title=dataset, ylabel=metric.upper() if index % columns == 0 else "")

    for blank in range(len(datasets), rows * columns):
        axes[blank // columns][blank % columns].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, frameon=False, fontsize=10,
                           ncol=len(models), loc="lower center")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.suptitle(f"Label scarcity: test {metric.upper()} from TR10 to TR40",
                    color=INK_PRIMARY, fontsize=13, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=[0, 0.05, 1, 0.96])
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_runtime(summary: Dict[tuple, Dict[str, Any]], path: Path,
                 ratio: str = "TR40") -> Optional[Path]:
    """Mean training-epoch seconds per model and dataset, log scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({key[0] for key in summary})
    datasets = [d for d in DATASET_ORDER
                if any((m, d, ratio) in summary for m in models)]
    if not datasets or not models:
        return None

    figure, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(datasets), dtype=float)

    any_value = False
    for index, model in enumerate(models):
        values = []
        for dataset in datasets:
            entry = summary.get((model, dataset, ratio), {})
            value = entry.get("mean_train_epoch_seconds")
            values.append(value if value else np.nan)
            any_value = any_value or bool(value)
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(positions + offset, values, width * 0.92, label=model,
               color=SERIES[index % len(SERIES)], zorder=3, linewidth=0)

    if not any_value:
        plt.close(figure)
        return None

    ax.set_xticks(positions)
    ax.set_xticklabels(datasets, color=INK_SECONDARY, fontsize=10)
    ax.set_yscale("log")
    _style(ax, title=f"Mean training time per epoch ({ratio})",
           ylabel="seconds per epoch (log scale)")
    legend = ax.legend(frameon=False, fontsize=10, ncol=len(models))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    figure.text(0.01, 0.01,
                "Comparable only within one frozen hardware class; see hardware.json "
                "for each run.", color=INK_MUTED, fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=[0, 0.04, 1, 1])
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate COMP8851 results")
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--metric", default="auprc",
                        help="Headline metric for figures and the LaTeX table")
    parser.add_argument("--ratio", default="TR40")
    parser.add_argument("--track", default="unified", choices=list(protocol.TRACKS) + [""])
    parser.add_argument("--models", nargs="*", default=["CARE-GNN", "GHRN"])
    parser.add_argument("--allow-run-mode", nargs="*", default=[],
                        help="Include run modes normally excluded from benchmark "
                             "tables, e.g. local-cpu-validation. Use only to inspect "
                             "correctness evidence, never for reported results.")
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    reports_dir = Path(args.reports_dir) if args.reports_dir else results_root / "reports"
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_root / "plots"

    if not results_root.exists():
        print(f"[aggregate] results root does not exist: {results_root}")
        return 1

    runs = collect_runs(results_root)
    print(f"[aggregate] found {len(runs)} run record(s) under {results_root}")
    if not runs:
        print("[aggregate] nothing to aggregate yet")
        return 0

    usable = reportable(runs, track=args.track, allow_run_modes=set(args.allow_run_mode))
    if args.allow_run_mode:
        print(f"[aggregate] WARNING including non-benchmark run modes: "
              f"{', '.join(args.allow_run_mode)}. These are correctness evidence, "
              "not measured benchmark results.")
    statuses: Dict[str, int] = {}
    for run in runs:
        statuses[run["status"] or "UNKNOWN"] = statuses.get(run["status"] or "UNKNOWN", 0) + 1
    print(f"[aggregate] status counts: {statuses}")
    print(f"[aggregate] reportable runs ({args.track} track, excluding smoke): {len(usable)}")

    write_run_csv(runs, reports_dir / "benchmark_results.csv")
    (reports_dir / "benchmark_results.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    write_ledger(runs, reports_dir / "run_ledger.csv", args.models)

    summary = aggregate_by_condition(usable)
    serialisable = {f"{model}|{dataset}|{ratio}": value
                    for (model, dataset, ratio), value in summary.items()}
    (reports_dir / "benchmark_aggregated.json").write_text(
        json.dumps(serialisable, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    write_markdown(summary, reports_dir / "benchmark_summary.md",
                   metrics=["auprc", "auroc", "macro_f1", "fraud_recall"], ratio=args.ratio)
    write_latex(summary, reports_dir / "benchmark_table.tex",
                metric=args.metric, ratio=args.ratio)
    print(f"[aggregate] tables written to {reports_dir}")

    try:
        produced = [
            plot_model_comparison(summary, plots_dir / f"model_comparison_{args.metric}_{args.ratio}.png",
                                  metric=args.metric, ratio=args.ratio),
            plot_metric_heatmap(summary, plots_dir / f"metric_heatmap_{args.ratio}.png",
                                ratio=args.ratio),
            plot_label_scarcity(summary, plots_dir / f"label_scarcity_{args.metric}.png",
                                metric=args.metric),
            plot_runtime(summary, plots_dir / f"runtime_{args.ratio}.png", ratio=args.ratio),
        ]
        for item in produced:
            if item:
                print(f"[aggregate] figure: {item}")
    except ImportError:
        print("[aggregate] matplotlib is not installed; tables were written, figures skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
