import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/workspace/comp8851_two_models")
EVIDENCE = ROOT / "evidence" / "final_runs" / "amazon"
TABLES = EVIDENCE / "tables"
FIGURES = EVIDENCE / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = {
    "lexgnn": "LEX-GNN",
    "consisgad": "ConsisGAD",
}

RATIOS = [10, 20, 30, 40]
SEEDS = [2, 42, 72]

REQUIRED_ARTIFACTS = [
    "summary.json",
    "run_config.json",
    "hardware.json",
    "split_summary.json",
    "protocol_identity.json",
    "epoch_times.csv",
    "validation_metrics.csv",
    "validation_final.json",
    "test_metrics.json",
    "predictions.npz",
    "checkpoint.pt",
]

METRICS = [
    "auroc",
    "auprc",
    "macro_f1",
    "fraud_f1",
    "fraud_precision",
    "fraud_recall",
    "g_mean",
    "accuracy",
]

EFFICIENCY = [
    "training_seconds_total",
    "training_seconds_mean",
    "final_validation_inference_seconds",
    "test_inference_seconds",
    "complete_wall_seconds",
    "peak_gpu_memory_mb",
    "total_parameters",
]

EXPECTED = {
    (model, ratio, seed)
    for model in MODELS.values()
    for ratio in RATIOS
    for seed in SEEDS
}

records = []
found_keys = set()
verification_rows = []

summary_paths = []

for folder_name, expected_model in MODELS.items():
    result_root = (
        ROOT / "runs" / folder_name / "amazon" / "final"
    )

    for summary_path in sorted(result_root.rglob("summary.json")):
        summary_paths.append(
            (folder_name, expected_model, summary_path)
        )

if len(summary_paths) != 24:
    raise AssertionError(
        f"Expected 24 final summary files, found {len(summary_paths)}"
    )

print(f"Final summaries discovered: {len(summary_paths)}/24")

for number, (folder_name, expected_model, summary_path) in enumerate(
    summary_paths,
    start=1,
):
    run_dir = summary_path.parent

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    run_mode = str(summary.get("run_mode", "")).lower()
    ratio_match = re.search(r"tr(10|20|30|40)", run_mode)

    if ratio_match is None:
        raise AssertionError(
            f"Cannot determine training ratio: {run_dir}"
        )

    ratio = int(ratio_match.group(1))
    seed = int(summary["seed"])
    model = str(summary["model"])
    key = (model, ratio, seed)

    filled = int(30 * number / 24)
    bar = "#" * filled + "-" * (30 - filled)

    print()
    print(
        f"[{bar}] {number}/24 "
        f"({100 * number / 24:6.2f}%)"
    )
    print(f"Checking: {model} | TR{ratio} | seed={seed}")

    if model != expected_model:
        raise AssertionError(
            f"Incorrect model identity in {summary_path}"
        )

    if key not in EXPECTED:
        raise AssertionError(f"Unexpected run identity: {key}")

    if key in found_keys:
        raise AssertionError(f"Duplicate final run: {key}")

    found_keys.add(key)

    for artifact in REQUIRED_ARTIFACTS:
        artifact_path = run_dir / artifact

        if not artifact_path.is_file():
            raise AssertionError(
                f"Missing {artifact}: {run_dir}"
            )

        if artifact_path.stat().st_size == 0:
            raise AssertionError(
                f"Empty {artifact}: {run_dir}"
            )

    with (run_dir / "run_config.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    with (run_dir / "protocol_identity.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        protocol = json.load(file)

    with (run_dir / "test_metrics.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        test_metrics_file = json.load(file)

    if summary.get("status") != "PASS":
        raise AssertionError(f"Run status is not PASS: {run_dir}")

    if summary.get("dataset") != "Amazon":
        raise AssertionError(f"Incorrect dataset: {run_dir}")

    if summary.get("test_evaluated") is not True:
        raise AssertionError(
            f"Final test was not evaluated: {run_dir}"
        )

    if protocol.get("architecture_changed") is not False:
        raise AssertionError(
            f"Architecture-change flag failed: {run_dir}"
        )

    if protocol.get("test_evaluations_during_training") != 0:
        raise AssertionError(
            f"Test leakage detected: {run_dir}"
        )

    if protocol.get("test_evaluation_after_training") is not True:
        raise AssertionError(
            f"Final-test timing flag failed: {run_dir}"
        )

    expected_split_name = f"amazon_TR{ratio}_seed2.npz"

    if expected_split_name not in json.dumps(config):
        raise AssertionError(
            f"Incorrect frozen split in configuration: {run_dir}"
        )

    if int(config.get("seed", -1)) != seed:
        raise AssertionError(
            f"Training seed mismatch: {run_dir}"
        )

    summary_metrics = summary.get("test_metrics")

    if not isinstance(summary_metrics, dict):
        raise AssertionError(
            f"Missing summary test metrics: {run_dir}"
        )

    for metric in METRICS:
        if metric not in summary_metrics:
            raise AssertionError(
                f"Missing test metric {metric}: {run_dir}"
            )

        value = float(summary_metrics[metric])

        if not math.isfinite(value):
            raise AssertionError(
                f"Non-finite metric {metric}: {run_dir}"
            )

        file_value = float(test_metrics_file[metric])

        if not math.isclose(
            value,
            file_value,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"Metric mismatch for {metric}: {run_dir}"
            )

    predictions = np.load(
        run_dir / "predictions.npz",
        allow_pickle=False,
    )

    numeric_arrays = 0

    for array_name in predictions.files:
        array = predictions[array_name]

        if np.issubdtype(array.dtype, np.number):
            numeric_arrays += 1

            if not np.isfinite(array).all():
                raise AssertionError(
                    f"Invalid prediction values in "
                    f"{array_name}: {run_dir}"
                )

    if numeric_arrays == 0:
        raise AssertionError(
            f"No numeric prediction arrays: {run_dir}"
        )

    with (run_dir / "validation_metrics.csv").open(
        "r",
        encoding="utf-8",
    ) as file:
        validation_rows = list(csv.DictReader(file))

    with (run_dir / "epoch_times.csv").open(
        "r",
        encoding="utf-8",
    ) as file:
        epoch_rows = list(csv.DictReader(file))

    if not validation_rows:
        raise AssertionError(
            f"No validation rows: {run_dir}"
        )

    if not epoch_rows:
        raise AssertionError(
            f"No epoch timing rows: {run_dir}"
        )

    record = {
        "model": model,
        "training_ratio": ratio,
        "seed": seed,
        "run_id": summary["run_id"],
        "result_directory": str(run_dir),
        "epochs_completed": int(summary["epochs_completed"]),
        "best_validation_epoch": int(
            summary["best_validation_epoch"]
        ),
        "best_validation_auroc": float(
            summary["best_validation_auroc"]
        ),
        "selected_threshold": float(
            summary["selected_threshold"]
        ),
    }

    for metric in METRICS:
        record[metric] = float(summary_metrics[metric])

    for field in EFFICIENCY:
        record[field] = float(summary[field])

    records.append(record)

    verification_rows.append({
        "model": model,
        "training_ratio": f"TR{ratio}",
        "seed": seed,
        "status": "PASS",
        "test_evaluations_during_training": 0,
        "architecture_changed": False,
        "artifacts_verified": len(REQUIRED_ARTIFACTS),
        "result_directory": str(run_dir),
    })

    print("Artifacts: PASS")
    print("Protocol and test isolation: PASS")
    print(
        "Test metrics: "
        f"AUROC={record['auroc']:.6f}, "
        f"AUPRC={record['auprc']:.6f}, "
        f"Macro-F1={record['macro_f1']:.6f}, "
        f"Fraud-F1={record['fraud_f1']:.6f}, "
        f"G-Mean={record['g_mean']:.6f}"
    )

missing = EXPECTED - found_keys
unexpected = found_keys - EXPECTED

if missing:
    raise AssertionError(f"Missing final runs: {sorted(missing)}")

if unexpected:
    raise AssertionError(
        f"Unexpected final runs: {sorted(unexpected)}"
    )

raw = pd.DataFrame(records).sort_values(
    ["model", "training_ratio", "seed"]
)

verification = pd.DataFrame(verification_rows).sort_values(
    ["model", "training_ratio", "seed"]
)

raw_path = TABLES / "amazon_final_runs_raw.csv"
verification_path = TABLES / "amazon_run_verification.csv"

raw.to_csv(raw_path, index=False)
verification.to_csv(verification_path, index=False)

aggregate_fields = (
    METRICS
    + [
        "best_validation_auroc",
        "training_seconds_total",
        "training_seconds_mean",
        "final_validation_inference_seconds",
        "test_inference_seconds",
        "complete_wall_seconds",
        "peak_gpu_memory_mb",
        "total_parameters",
    ]
)

aggregate_rows = []
formatted_rows = []

for model in sorted(raw["model"].unique()):
    for ratio in RATIOS:
        group = raw[
            (raw["model"] == model)
            & (raw["training_ratio"] == ratio)
        ]

        if len(group) != 3:
            raise AssertionError(
                f"Expected three seeds for {model}, TR{ratio}"
            )

        formatted = {
            "model": model,
            "training_ratio": f"TR{ratio}",
            "seeds": "2, 42, 72",
        }

        for field in aggregate_fields:
            values = group[field].astype(float).to_numpy()
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            sem = std / math.sqrt(3)
            ci95 = 4.3026527299 * sem

            aggregate_rows.append({
                "model": model,
                "training_ratio": ratio,
                "metric": field,
                "n": 3,
                "mean": mean,
                "standard_deviation": std,
                "ci95_lower": mean - ci95,
                "ci95_upper": mean + ci95,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            })

            formatted[field] = f"{mean:.6f} ± {std:.6f}"

        formatted_rows.append(formatted)

aggregate = pd.DataFrame(aggregate_rows)
formatted = pd.DataFrame(formatted_rows)

aggregate_path = (
    TABLES / "amazon_three_seed_aggregate_long.csv"
)
formatted_path = (
    TABLES / "amazon_three_seed_summary_mean_sd.csv"
)

aggregate.to_csv(aggregate_path, index=False)
formatted.to_csv(formatted_path, index=False)

plot_metrics = [
    ("auroc", "ROC-AUC"),
    ("auprc", "PR-AUC / AUPRC"),
    ("macro_f1", "Macro-F1"),
    ("fraud_f1", "Fraud-class F1"),
    ("g_mean", "G-Mean"),
    ("accuracy", "Accuracy"),
]

colors = {
    "LEX-GNN": "#2166AC",
    "ConsisGAD": "#B2182B",
}

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9),
    constrained_layout=True,
)

for axis, (metric, title) in zip(
    axes.flatten(),
    plot_metrics,
):
    for model in ["LEX-GNN", "ConsisGAD"]:
        means = []
        stds = []

        for ratio in RATIOS:
            values = raw[
                (raw["model"] == model)
                & (raw["training_ratio"] == ratio)
            ][metric].to_numpy()

            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)))

        axis.errorbar(
            RATIOS,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=4,
            label=model,
            color=colors[model],
        )

    axis.set_title(title)
    axis.set_xlabel("Training ratio (%)")
    axis.set_ylabel("Score")
    axis.set_xticks(RATIOS)
    axis.grid(alpha=0.25)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
)

fig.suptitle(
    "Amazon final performance: mean ± SD across seeds 2, 42 and 72",
    fontsize=15,
)

performance_figure = (
    FIGURES / "amazon_performance_mean_sd.png"
)

fig.savefig(
    performance_figure,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

efficiency_plots = [
    ("training_seconds_total", "Training time (seconds)"),
    ("test_inference_seconds", "Test inference time (seconds)"),
    ("peak_gpu_memory_mb", "Peak GPU memory (MB)"),
    ("total_parameters", "Model parameters"),
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
    constrained_layout=True,
)

for axis, (metric, title) in zip(
    axes.flatten(),
    efficiency_plots,
):
    positions = np.arange(len(RATIOS))
    width = 0.36

    for offset, model in zip(
        [-width / 2, width / 2],
        ["LEX-GNN", "ConsisGAD"],
    ):
        means = []
        stds = []

        for ratio in RATIOS:
            values = raw[
                (raw["model"] == model)
                & (raw["training_ratio"] == ratio)
            ][metric].to_numpy()

            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)))

        axis.bar(
            positions + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=model,
            color=colors[model],
            alpha=0.88,
        )

    axis.set_title(title)
    axis.set_xlabel("Training ratio")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"TR{ratio}" for ratio in RATIOS]
    )
    axis.grid(axis="y", alpha=0.25)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
)

fig.suptitle(
    "Amazon final efficiency: mean ± SD across three seeds",
    fontsize=15,
)

efficiency_figure = (
    FIGURES / "amazon_efficiency_mean_sd.png"
)

fig.savefig(
    efficiency_figure,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

verification_record = {
    "status": "PASS",
    "dataset": "Amazon",
    "models": ["LEX-GNN", "ConsisGAD"],
    "training_ratios": ["TR40", "TR30", "TR20", "TR10"],
    "training_seeds": [2, 42, 72],
    "split_seed": 2,
    "expected_runs": 24,
    "verified_runs": len(raw),
    "failed_runs": 0,
    "test_evaluations_during_training": 0,
    "architecture_changed": False,
    "aggregation": "mean, sample SD and 95% t confidence interval",
    "test_re_evaluated_during_aggregation": False,
    "raw_results": str(raw_path),
    "aggregate_results": str(aggregate_path),
    "formatted_summary": str(formatted_path),
    "performance_figure": str(performance_figure),
    "efficiency_figure": str(efficiency_figure),
}

verification_json = (
    EVIDENCE / "amazon_final_verification.json"
)

verification_json.write_text(
    json.dumps(
        verification_record,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 110)
print("AMAZON THREE-SEED PERFORMANCE SUMMARY")
print("=" * 110)

display_columns = [
    "model",
    "training_ratio",
    "auroc",
    "auprc",
    "macro_f1",
    "fraud_f1",
    "g_mean",
]

print(
    formatted[display_columns].to_string(index=False)
)

print()
print("All 24 final runs: PASS")
print("Three seeds per model/ratio: PASS")
print("Test leakage detected: NO")
print("Test data re-evaluated: NO")
print("Mean and sample SD calculated: PASS")
print("95% confidence intervals calculated: PASS")
print("Performance graph created: PASS")
print("Efficiency graph created: PASS")
print("FINAL AMAZON VERIFICATION: PASS")
