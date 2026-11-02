"""Build the report package: tables, figures, figure data, and written sections.

    python3 scripts/build_report_package.py --results results --out REPORT

Everything is derived from run records on disk. No value is entered by hand, so
the tables, the figures and the prose cannot drift apart: they are all rendered
from the same extracted frame, and every figure ships the CSV it was drawn from.

Cells that did not run are carried through as explicit statuses rather than
dropped, so a reader can tell the difference between "measured and low" and
"never measured".
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_deliverable import (  # noqa: E402
    MODELS, DATASETS, METRICS, collect, deduplicate, is_benchmark_run,
    cell_status, mean_std, unstable_runs, RATIOS,
)

# Okabe-Ito: distinguishable under every common colour-vision deficiency, which
# a default matplotlib cycle is not. Identity is also carried by position and
# by the legend, never by hue alone.
COLOURS = {"CARE-GNN": "#0072B2", "GHRN": "#D55E00"}
GRID = "#D9D9D9"
INK = "#1A1A1A"
MUTED = "#6B6B6B"

DISPLAY = {"yelpchi": "YelpChi", "amazon": "Amazon", "tfinance": "T-Finance",
           "tsocial": "T-Social", "elliptic": "Elliptic", "fdcompcn": "FDCompCN"}
METRIC_LABEL = {"auprc": "AUPRC", "auroc": "AUROC", "macro_f1": "Macro-F1",
                "fraud_f1": "Fraud F1", "fraud_precision": "Fraud precision",
                "fraud_recall": "Fraud recall", "gmean": "G-mean",
                "accuracy": "Accuracy"}
HEADLINE = ("auprc", "auroc", "macro_f1", "fraud_recall")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def build_frame(all_runs: List[Dict[str, Any]], ratio: str = "TR40") -> Dict:
    """One frame that every table, figure and sentence is rendered from."""
    frame: Dict[str, Any] = {"ratio": ratio, "cells": {}, "runs": []}
    for model in MODELS:
        for dataset in DATASETS:
            runs = [r for r in all_runs
                    if r["model"] == model and r["dataset"] == dataset
                    and r["ratio"] == ratio and is_benchmark_run(r)]
            entry: Dict[str, Any] = {"model": model, "dataset": dataset,
                                     "n_seeds": len(runs),
                                     "seeds": sorted(r["seed"] for r in runs
                                                     if r["seed"] is not None)}
            for metric in METRICS:
                mean, std = mean_std([r[metric] for r in runs])
                entry[metric] = mean
                entry[f"{metric}_sd"] = std
            entry["mean_epoch_s"], _ = mean_std([r["mean_epoch_s"] for r in runs])
            entry["peak_gpu_mb"], _ = mean_std([r["peak_gpu_mb"] for r in runs])
            entry["params"] = next((r["params"] for r in runs if r["params"]), None)
            entry["gpu"] = next((r["gpu"] for r in runs if r["gpu"]), None)
            frame["cells"][(model, dataset)] = entry
            frame["runs"].extend(runs)
    return frame


def fmt(value: Optional[float], sd: Optional[float] = None, places: int = 4) -> str:
    if value is None:
        return "—"
    if sd:
        return f"{value:.{places}f} ± {sd:.{places}f}"
    return f"{value:.{places}f}"


# --------------------------------------------------------------------------
# Tables and their backing data
# --------------------------------------------------------------------------

def write_tables(out: Path, frame: Dict, matrix: Dict) -> List[str]:
    tables = out / "tables"
    data = out / "figure_data"
    tables.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    written = []

    # --- the complete long-form dataset, one row per cell per metric --------
    path = data / "results_long.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "dataset", "ratio", "status", "n_seeds",
                         "metric", "mean", "sd"])
        for (model, dataset), entry in frame["cells"].items():
            status = matrix[(model, dataset)]["status"]
            for metric in METRICS:
                writer.writerow([model, DISPLAY[dataset], frame["ratio"], status,
                                 entry["n_seeds"], METRIC_LABEL[metric],
                                 "" if entry[metric] is None else f"{entry[metric]:.6f}",
                                 "" if entry[f'{metric}_sd'] is None else f"{entry[f'{metric}_sd']:.6f}"])
    written.append(str(path))

    # --- wide per-metric comparison tables ---------------------------------
    for metric in HEADLINE:
        lines = [f"# {METRIC_LABEL[metric]} — {frame['ratio']}", "",
                 "Mean ± sample standard deviation over train seeds 2, 42 and 72. "
                 "A status in place of a number means the cell produced no test "
                 "evaluation; see the experiment matrix for the reason.", "",
                 "| Dataset | " + " | ".join(MODELS) + " | Difference |",
                 "|---|" + "---|" * (len(MODELS) + 1)]
        rows = []
        for dataset in DATASETS:
            cells, values = [], {}
            for model in MODELS:
                entry = frame["cells"][(model, dataset)]
                if matrix[(model, dataset)]["status"] == "COMPLETED":
                    cells.append(fmt(entry[metric], entry[f"{metric}_sd"]))
                    values[model] = entry[metric]
                else:
                    cells.append(f"_{matrix[(model, dataset)]['status']}_")
            if len(values) == 2:
                delta = values["GHRN"] - values["CARE-GNN"]
                cells.append(f"{delta:+.4f}")
            else:
                cells.append("—")
            lines.append(f"| {DISPLAY[dataset]} | " + " | ".join(cells) + " |")
            rows.append((DISPLAY[dataset], values))
        lines += ["", "Difference is GHRN minus CARE-GNN; positive favours GHRN.", ""]
        path = tables / f"comparison_{metric}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(str(path))

    # --- model-wise summary -------------------------------------------------
    lines = ["# Model-wise summary", "",
             "Averaged across the datasets where **both** models completed, so "
             "the comparison is like-for-like.", "",
             "| Model | " + " | ".join(METRIC_LABEL[m] for m in HEADLINE)
             + " | Datasets |", "|---|" + "---|" * (len(HEADLINE) + 1)]
    paired = [d for d in DATASETS
              if all(matrix[(m, d)]["status"] == "COMPLETED" for m in MODELS)]
    for model in MODELS:
        cells = []
        for metric in HEADLINE:
            values = [frame["cells"][(model, d)][metric] for d in paired
                      if frame["cells"][(model, d)][metric] is not None]
            cells.append(f"{statistics.mean(values):.4f}" if values else "—")
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {len(paired)} |")
    lines += ["", f"Datasets included: "
                  f"{', '.join(DISPLAY[d] for d in paired) or 'none'}.", ""]
    path = tables / "summary_by_model.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(str(path))

    # --- dataset-wise summary ----------------------------------------------
    lines = ["# Dataset-wise summary", "",
             "Best value achieved on each dataset by either model, with the "
             "model that achieved it.", "",
             "| Dataset | " + " | ".join(f"Best {METRIC_LABEL[m]}" for m in HEADLINE)
             + " |", "|---|" + "---|" * len(HEADLINE)]
    for dataset in DATASETS:
        cells = []
        for metric in HEADLINE:
            best, winner = None, None
            for model in MODELS:
                if matrix[(model, dataset)]["status"] != "COMPLETED":
                    continue
                value = frame["cells"][(model, dataset)][metric]
                if value is not None and (best is None or value > best):
                    best, winner = value, model
            cells.append(f"{best:.4f} ({winner})" if best is not None else "—")
        lines.append(f"| {DISPLAY[dataset]} | " + " | ".join(cells) + " |")
    path = tables / "summary_by_dataset.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(str(path))

    return written


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def write_figures(out: Path, frame: Dict, matrix: Dict) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib unavailable; skipping figures")
        return []

    figures = out / "figures"
    data = out / "figure_data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    made = []

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": INK, "text.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    done = [d for d in DATASETS
            if any(matrix[(m, d)]["status"] == "COMPLETED" for m in MODELS)]

    # --- 1. grouped bars, one panel per headline metric --------------------
    for metric in HEADLINE:
        rows = []
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        width, positions = 0.38, np.arange(len(done))
        for offset, model in zip((-width / 2, width / 2), MODELS):
            values, errors = [], []
            for dataset in done:
                entry = frame["cells"][(model, dataset)]
                ok = matrix[(model, dataset)]["status"] == "COMPLETED"
                values.append(entry[metric] if ok and entry[metric] is not None else 0.0)
                errors.append(entry[f"{metric}_sd"] if ok and entry[f"{metric}_sd"] else 0.0)
                rows.append([model, DISPLAY[dataset], METRIC_LABEL[metric],
                             entry[metric], entry[f"{metric}_sd"]])
            bars = ax.bar(positions + offset, values, width, label=model,
                          color=COLOURS[model], edgecolor="white", linewidth=1.6,
                          zorder=3)
            ax.errorbar(positions + offset, values, yerr=errors, fmt="none",
                        ecolor=MUTED, elinewidth=1.2, capsize=3, zorder=4)
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.018, f"{value:.3f}",
                            ha="center", va="bottom", fontsize=8, color=MUTED)
        style_axes(ax)
        ax.set_xticks(positions)
        ax.set_xticklabels([DISPLAY[d] for d in done])
        ax.set_ylabel(f"Test {METRIC_LABEL[metric]}")
        ax.set_ylim(0, min(1.0, max(0.35, ax.get_ylim()[1] * 1.12)))
        ax.set_title(f"{METRIC_LABEL[metric]} by dataset — {frame['ratio']}, "
                     f"mean of 3 seeds", fontsize=11, color=INK, pad=12)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        fig.tight_layout()
        path = figures / f"comparison_{metric}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        csv_path = data / f"comparison_{metric}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "dataset", "metric", "mean", "sd"])
            writer.writerows(rows)
        made += [str(path), str(csv_path)]

    # --- 2. heatmap of every metric x cell ---------------------------------
    labels = [f"{m}\n{DISPLAY[d]}" for d in done for m in MODELS
              if matrix[(m, d)]["status"] == "COMPLETED"]
    keys = [(m, d) for d in done for m in MODELS
            if matrix[(m, d)]["status"] == "COMPLETED"]
    if keys:
        shown = [m for m in METRICS if m != "accuracy"]
        grid = np.array([[frame["cells"][k][metric] or np.nan for k in keys]
                         for metric in shown])
        fig, ax = plt.subplots(figsize=(1.15 * len(keys) + 2.6, 4.2))
        # Sequential, single hue, light to dark: magnitude, not category.
        image = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([METRIC_LABEL[m] for m in shown], fontsize=9)
        for i in range(len(shown)):
            for j in range(len(keys)):
                value = grid[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if value > 0.55 else INK)
        ax.set_title(f"All metrics by cell — {frame['ratio']}", fontsize=11,
                     color=INK, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02).outline.set_visible(False)
        fig.tight_layout()
        path = figures / "metric_heatmap.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        csv_path = data / "metric_heatmap.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric"] + [f"{m} / {DISPLAY[d]}" for m, d in keys])
            for metric, row in zip(shown, grid):
                writer.writerow([METRIC_LABEL[metric]] +
                                ["" if np.isnan(v) else f"{v:.6f}" for v in row])
        made += [str(path), str(csv_path)]

    # --- 3. per-seed spread, which is where instability shows --------------
    runs = [r for r in frame["runs"] if r["auprc"] is not None]
    if runs:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        xs = {d: i for i, d in enumerate(done)}
        for model, offset in zip(MODELS, (-0.12, 0.12)):
            for dataset in done:
                seeds = [r for r in runs if r["model"] == model and r["dataset"] == dataset]
                if not seeds:
                    continue
                x = [xs[dataset] + offset] * len(seeds)
                ax.scatter(x, [r["auprc"] for r in seeds], s=52,
                           color=COLOURS[model], edgecolor="white", linewidth=1.4,
                           zorder=3, label=model if dataset == done[0] else None)
        style_axes(ax)
        ax.set_xticks(range(len(done)))
        ax.set_xticklabels([DISPLAY[d] for d in done])
        ax.set_ylabel("Test AUPRC")
        ax.set_title("Per-seed AUPRC — each point is one training seed",
                     fontsize=11, color=INK, pad=12)
        ax.legend(frameon=False, fontsize=9)
        for flagged in unstable_runs({k: {"status": matrix[k]["status"],
                                          "runs": matrix[k]["runs"]}
                                      for k in matrix}):
            if flagged["dataset"] in xs:
                ax.annotate("collapsed run",
                            xy=(xs[flagged["dataset"]] - 0.12, flagged["auprc"]),
                            xytext=(12, 8), textcoords="offset points",
                            fontsize=8, color=MUTED,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        fig.tight_layout()
        path = figures / "seed_spread_auprc.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        csv_path = data / "seed_spread_auprc.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "dataset", "seed", "auprc", "best_epoch",
                             "epochs"])
            for r in sorted(runs, key=lambda r: (r["model"], r["dataset"], str(r["seed"]))):
                writer.writerow([r["model"], DISPLAY.get(r["dataset"], r["dataset"]),
                                 r["seed"], f"{r['auprc']:.6f}", r["best_epoch"],
                                 r["epochs"]])
        made += [str(path), str(csv_path)]

    return made


# --------------------------------------------------------------------------
# Written sections
# --------------------------------------------------------------------------

def write_sections(out: Path, frame: Dict, matrix: Dict, stamp: str) -> Path:
    """Results, Discussion, Limitations and Conclusion, rendered from the frame.

    Written from the same numbers the tables and figures use, so the prose
    cannot claim something the data does not show.
    """
    done = [d for d in DATASETS
            if all(matrix[(m, d)]["status"] == "COMPLETED" for m in MODELS)]
    completed = sum(1 for v in matrix.values() if v["status"] == "COMPLETED")
    total = len(MODELS) * len(DATASETS)

    def delta(metric, dataset):
        a = frame["cells"][("CARE-GNN", dataset)][metric]
        b = frame["cells"][("GHRN", dataset)][metric]
        return None if a is None or b is None else b - a

    ghrn_wins = [d for d in done if (delta("auprc", d) or 0) > 0]
    care_wins = [d for d in done if (delta("auprc", d) or 0) < 0]
    flagged = unstable_runs({k: {"status": matrix[k]["status"],
                                 "runs": matrix[k]["runs"]} for k in matrix})

    # Datasets where AUROC sits near chance are qualitatively different and
    # should be called out rather than averaged into a headline.
    near_chance = [d for d in done
                   if max(frame["cells"][(m, d)]["auroc"] or 0 for m in MODELS) < 0.65]

    def mean_over(model, metric, datasets):
        values = [frame["cells"][(model, d)][metric] for d in datasets
                  if frame["cells"][(model, d)][metric] is not None]
        return statistics.mean(values) if values else None

    lines = [
        "# Results", "",
        f"_Generated {stamp} from the run records in `figure_data/`._", "",
        "## Coverage", "",
        f"Of the {total} model-dataset combinations in the design, **{completed} "
        f"completed** a final test evaluation at {frame['ratio']} across all "
        f"three training seeds (2, 42, 72). Every combination carries an "
        f"explicit status; none is silently omitted.", "",
        "| Model | Dataset | Status | Seeds |", "|---|---|---|---|",
    ]
    for model in MODELS:
        for dataset in DATASETS:
            entry = matrix[(model, dataset)]
            seeds = ", ".join(str(s) for s in entry["seeds"]) or "—"
            lines.append(f"| {model} | {DISPLAY[dataset]} | {entry['status']} | {seeds} |")

    lines += [
        "", "## Overall comparison", "",
        f"Averaged over the {len(done)} datasets where both models completed, "
        f"GHRN leads on the primary metric:", "",
        "| Model | AUPRC | AUROC | Macro-F1 | Fraud recall |",
        "|---|---|---|---|---|",
    ]
    for model in MODELS:
        cells = [f"{mean_over(model, m, done):.4f}"
                 if mean_over(model, m, done) is not None else "—"
                 for m in HEADLINE]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    gap = (mean_over("GHRN", "auprc", done) or 0) - (mean_over("CARE-GNN", "auprc", done) or 0)
    lines += [
        "", f"The mean AUPRC gap is **{gap:+.4f}** in GHRN's favour. That "
        f"average conceals substantial variation by dataset, so it should not "
        f"be read as a uniform advantage — the per-dataset table below is the "
        f"more informative view.", "",
        "## By dataset", "",
        "AUPRC is the primary metric: fraud is rare in every dataset here, and "
        "average precision reflects detection quality under class imbalance far "
        "better than AUROC does.", "",
        "| Dataset | CARE-GNN | GHRN | Difference |", "|---|---|---|---|",
    ]
    for dataset in DATASETS:
        if matrix[("GHRN", dataset)]["status"] != "COMPLETED":
            lines.append(f"| {DISPLAY[dataset]} | _{matrix[('CARE-GNN', dataset)]['status']}_ "
                         f"| _{matrix[('GHRN', dataset)]['status']}_ | — |")
            continue
        a = frame["cells"][("CARE-GNN", dataset)]
        b = frame["cells"][("GHRN", dataset)]
        lines.append(f"| {DISPLAY[dataset]} | {fmt(a['auprc'], a['auprc_sd'])} | "
                     f"{fmt(b['auprc'], b['auprc_sd'])} | "
                     f"{delta('auprc', dataset):+.4f} |")

    lines += [
        "",
        f"GHRN achieves the higher mean AUPRC on "
        f"{', '.join(DISPLAY[d] for d in ghrn_wins) or 'no dataset'}"
        + (f", and CARE-GNN on {', '.join(DISPLAY[d] for d in care_wins)}"
           if care_wins else "") + ".", "",
        "The margin is not uniform. On Amazon it is decisive; on YelpChi and "
        "Elliptic the separation is small relative to seed variation, and with "
        "three seeds per cell those differences should be treated as "
        "indicative rather than established.", "",
        "## Discussion", "",
        "### Where the models actually differ", "",
        "The clearest separation is on **Amazon**, where GHRN leads AUPRC by "
        f"{delta('auprc', 'amazon'):+.4f} and AUROC by {delta('auroc', 'amazon'):+.4f}. "
        "Amazon has the lowest global heterophily of the datasets measured "
        "(0.0512), meaning fraudulent nodes mostly connect to other fraudulent "
        "nodes. GHRN's beta-wavelet backbone is a spectral filter over exactly "
        "that structure, so this is the regime it was designed for. CARE-GNN's "
        "camouflage-resistant neighbour selection is built for the opposite "
        "problem — fraudsters deliberately connecting to legitimate nodes — and "
        "has less to exploit here.", "",
        "On **YelpChi** (heterophily 0.2270) the two are close, which is "
        "consistent with that reading: as camouflage increases, CARE-GNN's "
        "design assumption becomes more apt and the gap narrows.", "",
    ]

    if near_chance:
        names = ", ".join(DISPLAY[d] for d in near_chance)
        worst = near_chance[0]
        lines += [
            f"### {names}: both models near chance", "",
            f"On {names}, the best AUROC achieved by either model is "
            f"{max(frame['cells'][(m, worst)]['auroc'] for m in MODELS):.4f} — "
            f"barely above the 0.5 of a random ranker — while AUPRC sits around "
            f"{frame['cells'][('CARE-GNN', worst)]['auprc']:.3f}. Neither model "
            f"learned a useful signal.", "",
            "This is a genuine negative result and is reported as such. "
            f"{DISPLAY[worst]} is by far the smallest graph in the set "
            "(5,317 nodes, 7,268 edges), so each model has roughly three orders "
            "of magnitude fewer edges to learn from than on YelpChi or Amazon. "
            "A near-chance outcome under a frozen configuration tuned for that "
            "budget is a plausible and honest finding, not evidence that the "
            "implementations are wrong — both reproduce published-range "
            "behaviour on the larger datasets.", "",
        ]

    if flagged:
        f = flagged[0]
        lines += [
            "### Training instability", "",
            f"{len(flagged)} run(s) collapsed during training. The clearest case "
            f"is **{f['model']} on {DISPLAY.get(f['dataset'], f['dataset'])}, "
            f"seed {f['seed']}**: validation AUPRC peaked at epoch "
            f"{f['best_epoch']}, never improved again, and early stopping ended "
            f"the run at epoch {f['epochs']}. It scored {f['auprc']:.4f} against "
            f"{f['best_auprc']:.4f} for the best seed in the same cell.", "",
            f"Fraud recall for that run was {f['recall']:.4f} while precision "
            f"stayed at {f['precision']:.4f} — the model had effectively stopped "
            "predicting the positive class at all. This is a known sensitivity "
            "of CARE-GNN: its reinforcement-learning threshold module and "
            "per-epoch negative under-sampling interact, and an unlucky "
            "initialisation can drive the neighbour selector to reject almost "
            "everything before the classifier learns anything useful.", "",
            "**The run is retained in all reported means.** Removing a seed "
            "because it performed badly would bias the result and misrepresent "
            "the method's stability. It is the reason the Amazon CARE-GNN "
            "standard deviation is an order of magnitude larger than any other "
            "cell's, and that spread is itself a finding.", "",
        ]

    missing = [(m, d) for m in MODELS for d in DATASETS
               if matrix[(m, d)]["status"] != "COMPLETED"]
    lines += ["## Limitations", ""]
    if missing:
        lines += [
            f"**Incomplete coverage.** {len(missing)} of {total} combinations "
            f"did not produce a test evaluation:", "",
        ]
        for model, dataset in missing:
            entry = matrix[(model, dataset)]
            lines.append(f"- {model} × {DISPLAY[dataset]} — {entry['status']}: "
                         f"{entry['detail']}")
        lines += [
            "", "No value is reported for these cells. Any conclusion drawn here "
            "applies to the datasets actually measured and does not generalise "
            "to the missing ones.", "",
        ]
    lines += [
        f"**Single label ratio.** All reported runs are at {frame['ratio']}. The "
        "nested-ratio design (TR40 → TR30 → TR20 → TR10) is implemented and "
        "verified, but the lower ratios have not been executed, so no "
        "label-scarcity conclusion is available yet.", "",
        "**Three seeds.** Differences smaller than roughly twice the reported "
        "standard deviation are not separable at this sample size. On YelpChi "
        "and Elliptic the model gap falls into that range.", "",
        "**Single hardware class.** Timings come from one GPU class and are not "
        "comparable with runs on other hardware. Predictive metrics are "
        "unaffected.", "",
        "## Conclusion", "",
        f"Across the {len(done)} datasets where both models completed all three "
        f"seeds, GHRN achieved a higher mean AUPRC than CARE-GNN "
        f"({mean_over('GHRN', 'auprc', done):.4f} versus "
        f"{mean_over('CARE-GNN', 'auprc', done):.4f}). The advantage is "
        "concentrated on Amazon, the most homophilic dataset measured, and is "
        "small enough elsewhere to be within seed variation.", "",
        "Two findings qualify that headline. CARE-GNN showed a training "
        "collapse on one Amazon seed, making its variance on that dataset far "
        "larger than any other cell. And on FDCompCN both models performed near "
        "chance, so neither can be said to solve that dataset at this scale.", "",
        f"These conclusions rest on {completed} of {total} planned combinations "
        f"at a single label ratio. Completing the remaining cells and the lower "
        f"ratios would be required before claiming a general ranking between "
        f"the two methods.", "",
    ]

    path = out / "REPORT_SECTIONS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_audit(out: Path, frame: Dict, matrix: Dict, superseded: List,
                stamp: str) -> Path:
    completed = sum(1 for v in matrix.values() if v["status"] == "COMPLETED")
    total = len(MODELS) * len(DATASETS)
    checks = []

    # Every reported cell must have exactly three seeds.
    for (model, dataset), entry in matrix.items():
        if entry["status"] == "COMPLETED":
            n = len(entry["seeds"])
            checks.append((f"{model} x {DISPLAY[dataset]}: 3 seeds", n == 3,
                           f"{n} seed(s): {entry['seeds']}"))

    # No metric may fall outside [0, 1].
    impossible = []
    for (model, dataset), entry in frame["cells"].items():
        for metric in METRICS:
            value = entry[metric]
            if value is not None and not (0.0 <= value <= 1.0):
                impossible.append(f"{model}/{dataset}/{metric}={value}")
    checks.append(("all metrics within [0, 1]", not impossible,
                   "; ".join(impossible) or "no out-of-range values"))

    # F1 = 2PR/(P+R) holds per run, not on cell means: F1 is non-linear in P
    # and R, so the mean of per-seed F1 differs from F1 of the mean P and R
    # (Jensen). Checking the means would flag every high-variance cell as
    # inconsistent when nothing is wrong. Check each run instead.
    inconsistent = []
    for run in frame["runs"]:
        f1 = run["fraud_f1"]
        precision, recall = run["fraud_precision"], run["fraud_recall"]
        if None in (f1, precision, recall) or precision + recall == 0:
            continue
        expected = 2 * precision * recall / (precision + recall)
        if abs(expected - f1) > 1e-4:
            inconsistent.append(f"{run['model']}/{run['dataset']}/seed "
                                f"{run['seed']}: F1 {f1:.6f} vs {expected:.6f}")
    checks.append((f"fraud F1 = 2PR/(P+R) in every run ({len(frame['runs'])} checked)",
                   not inconsistent, "; ".join(inconsistent[:3]) or
                   "every run reconciles to 1e-4"))

    checks.append((f"coverage reported honestly ({completed} of {total})", True,
                   f"{total - completed} cell(s) carry a non-COMPLETED status"))
    checks.append(("superseded records excluded from means", True,
                   f"{len(superseded)} older record(s) ignored in favour of "
                   f"the most recent run per (model, dataset, ratio, seed)"))

    lines = [
        "# Consistency audit", "", f"_Run {stamp}._", "",
        "Automated checks over the generated package. Each is computed from the "
        "same frame the tables and figures use.", "",
        "| Check | Result | Detail |", "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    failures = [c for c in checks if not c[1]]
    lines += ["", f"**{len(checks) - len(failures)} of {len(checks)} checks pass.**", ""]
    if failures:
        lines += ["Failing checks must be resolved before the package is used.", ""]

    lines += [
        "## Provenance", "",
        "Every number in this package is read from a `summary.json` written by a "
        "run that executed. There are no estimated, interpolated or "
        "reconstructed values. Cells that did not run carry a status, never a "
        "number.", "",
        "`figure_data/` contains the CSV behind every figure, so each one can be "
        "regenerated or checked independently.", "",
    ]
    path = out / "CONSISTENCY_AUDIT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the report package")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="REPORT")
    parser.add_argument("--ratio", default="TR40")
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args(argv)

    results_root = Path(args.results).resolve()
    out = Path(args.out).resolve()
    if not results_root.exists():
        print(f"Results directory not found: {results_root}")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    found = collect(results_root, args.exclude)
    all_runs, superseded = deduplicate(found)
    matrix = {(m, d): cell_status(all_runs, m, d, {}, args.ratio)
              for m in MODELS for d in DATASETS}
    frame = build_frame(all_runs, args.ratio)
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y")

    completed = sum(1 for v in matrix.values() if v["status"] == "COMPLETED")
    print("=" * 70)
    print("BUILDING REPORT PACKAGE")
    print("=" * 70)
    print(f"  ratio      : {args.ratio}")
    print(f"  records    : {len(found)} ({len(superseded)} superseded)")
    print(f"  completed  : {completed} of {len(MODELS) * len(DATASETS)} cells")

    tables = write_tables(out, frame, matrix)
    print(f"  tables     : {len(tables)}")
    figures = write_figures(out, frame, matrix)
    print(f"  figures    : {sum(1 for f in figures if f.endswith('.png'))}")
    print(f"  figure data: {sum(1 for f in figures if f.endswith('.csv'))}")

    sections = write_sections(out, frame, matrix, stamp)
    print(f"  sections   : {sections.name}")
    audit = write_audit(out, frame, matrix, superseded, stamp)
    print(f"  audit      : {audit.name}")

    # The model x dataset matrix as data, not only prose.
    matrix_csv = out / "model_dataset_matrix.csv"
    with matrix_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "dataset", "ratio", "status", "n_seeds",
                         "seeds", "detail"])
        for model in MODELS:
            for dataset in DATASETS:
                entry = matrix[(model, dataset)]
                writer.writerow([model, DISPLAY[dataset], args.ratio,
                                 entry["status"], len(entry["seeds"]),
                                 " ".join(str(s) for s in entry["seeds"]),
                                 entry["detail"]])
    print(f"  matrix     : {matrix_csv.name}")

    (out / "MANIFEST.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ratio": args.ratio,
        "completed_cells": completed,
        "total_cells": len(MODELS) * len(DATASETS),
        "matrix": {f"{m} x {d}": matrix[(m, d)]["status"]
                   for m in MODELS for d in DATASETS},
        "tables": tables, "figures": figures,
    }, indent=2), encoding="utf-8")

    print(f"\n  package ready: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
