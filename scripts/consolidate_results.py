"""Consolidate the COMP8851 CARE-GNN / GHRN benchmark into one set of tables.

Scope: 2 models (CARE-GNN, GHRN) x 6 datasets = 12 required combinations, TR40,
training seeds 2/42/72, split seed 2.

Reads the authoritative per-run record, normalises it, and emits the per-seed CSV,
the mean/SD CSV, the experiment matrix, per-dataset and per-metric comparison
tables, and the efficiency tables under ../tables/.

Run:  python consolidate_results.py
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = os.path.dirname(HERE)
PROJECT = os.path.dirname(DELIVERABLE)
V2 = os.path.join(PROJECT, "CLIENT_DELIVERABLE_v2")
OUT = os.path.join(DELIVERABLE, "tables")

SOURCE = os.path.join(V2, "tables", "all_benchmark_runs.csv")

MODELS = ["CARE-GNN", "GHRN"]
DATASETS = ["YelpChi", "Amazon", "T-Finance", "T-Social", "Elliptic", "FDCompCN"]
SEEDS = [2, 42, 72]
RATIO = "TR40"

METRICS = [
    ("auprc", "AUPRC"),
    ("auroc", "AUROC"),
    ("macro_f1", "Macro-F1"),
    ("fraud_f1", "Fraud F1"),
    ("fraud_precision", "Fraud precision"),
    ("fraud_recall", "Fraud recall"),
    ("gmean", "G-mean"),
    ("accuracy", "Accuracy"),
]

CANON = {
    "yelpchi": "YelpChi", "amazon": "Amazon", "tfinance": "T-Finance",
    "tsocial": "T-Social", "elliptic": "Elliptic", "fdcompcn": "FDCompCN",
}

# Cells with no completed run, and the verified reason for each. Both are T-Social,
# and neither was skipped: each was attempted repeatedly and every attempt left an
# artefact bundle or a recorded console trace. See tables/tsocial_attempts.md.
UNCOMPLETED = {
    ("CARE-GNN", "T-Social"): (
        "FAILED",
        "Attempted repeatedly and never reached a test evaluation. Hyperparameter "
        "tuning completed two full trials (roughly 71 minutes each, best validation "
        "AUPRC 0.0859). The final run then trained for 16 epochs over 1 h 55 m on a "
        "Tesla T4 and was learning — validation AUPRC climbed to 0.1006 at epoch 14 — "
        "before it hit the 180-minute session cap and was interrupted inside "
        "layers.py during validation prediction. The cost is structural: validation "
        "took about 336 s per epoch against 62 s of training, because CARE-GNN filters "
        "neighbours per node in Python across all 1,156,213 validation nodes.",
    ),
    ("GHRN", "T-Social"): (
        "FAILED",
        "Attempted repeatedly across a five-rung escalation ladder that varied width, "
        "filter order and device (hid_dim 32/64, order 1/2, both cuda:0 and CPU). "
        "Every rung failed. The recorded error is a DGL index-dtype mismatch — "
        "'Expect argument \"u\" to have data type torch.int32. But got torch.int64' — "
        "raised because T-Social is large enough that its graph carries int32 indices "
        "while torch.arange and torch.argsort produce int64. Tuning trial 0 failed "
        "after 7.2 minutes with no validation AUPRC recorded.",
    ),
}


def canon(name):
    return CANON[name.strip().lower().replace("-", "").replace("_", "")]


def fnum(value):
    text = "" if value is None else str(value).strip()
    return float(text) if text else None


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def load_per_seed():
    rows = []
    for src in read_csv(SOURCE):
        if src["status"].strip().upper() != "COMPLETE":
            continue
        row = OrderedDict(
            model=src["model"].strip(),
            dataset=canon(src["dataset"]),
            ratio=src["ratio"].strip(),
            split_seed=2,
            train_seed=int(src["seed"]),
            gpu=src["gpu"].strip(),
        )
        for key, _ in METRICS:
            row[key] = fnum(src[key])
        row["threshold"] = fnum(src["threshold"])
        row["best_epoch"] = fnum(src["best_epoch"])
        row["epochs"] = fnum(src["epochs"])
        row["mean_epoch_s"] = fnum(src["mean_epoch_s"])
        row["peak_gpu_mb"] = fnum(src["peak_gpu_mb"])
        row["params"] = fnum(src["params"])
        rows.append(row)
    rows.sort(key=lambda r: (MODELS.index(r["model"]),
                             DATASETS.index(r["dataset"]), r["train_seed"]))
    return rows


def mean_sd(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    n = len(clean)
    mean = sum(clean) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return mean, math.sqrt(var)


def aggregate(per_seed):
    groups = OrderedDict()
    for row in per_seed:
        groups.setdefault((row["model"], row["dataset"]), []).append(row)

    out = []
    for (model, dataset), rows in groups.items():
        record = OrderedDict(
            model=model, dataset=dataset, ratio=RATIO, n_seeds=len(rows),
            seeds=" ".join(str(r["train_seed"]) for r in rows),
            gpu=rows[0]["gpu"],
        )
        for key, _ in METRICS:
            mean, sd = mean_sd([r[key] for r in rows])
            record[f"{key}_mean"] = mean
            record[f"{key}_sd"] = sd
        for key in ("mean_epoch_s", "peak_gpu_mb", "params", "epochs"):
            mean, _ = mean_sd([r[key] for r in rows])
            record[key] = mean
        out.append(record)
    out.sort(key=lambda r: (MODELS.index(r["model"]), DATASETS.index(r["dataset"])))
    return out


def build_matrix(agg):
    done = {(r["model"], r["dataset"]): r for r in agg}
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            key = (model, dataset)
            if key in done:
                rec = done[key]
                rows.append(OrderedDict(
                    model=model, dataset=dataset, ratio=RATIO, status="COMPLETED",
                    n_seeds=rec["n_seeds"], seeds=rec["seeds"],
                    detail=f"{rec['n_seeds']} final run(s), each with exactly one test "
                           f"evaluation, on {rec['gpu']}.",
                ))
            else:
                status, detail = UNCOMPLETED[key]
                rows.append(OrderedDict(
                    model=model, dataset=dataset, ratio=RATIO, status=status,
                    n_seeds=0, seeds="", detail=detail,
                ))
    return rows


def fmt(mean, sd, places=4):
    if mean is None:
        return "—"
    return f"{mean:.{places}f} ± {sd:.{places}f}"


def status_lookup(matrix):
    return {(r["model"], r["dataset"]): r["status"] for r in matrix}


def write_metric_tables(agg, matrix):
    status = status_lookup(matrix)
    index = {(r["model"], r["dataset"]): r for r in agg}
    for key, label in METRICS:
        lines = [
            f"# {label} — TR40", "",
            "Mean ± sample standard deviation over training seeds 2, 42 and 72. A status "
            "in place of a number means the cell produced no test evaluation; see "
            "`../02_EXPERIMENT_MATRIX.md` for the reason.", "",
            "| Dataset | CARE-GNN | GHRN | Difference |", "|---|---|---|---|",
        ]
        for dataset in DATASETS:
            cells, values = [], {}
            for model in MODELS:
                rec = index.get((model, dataset))
                if rec:
                    cells.append(fmt(rec[f"{key}_mean"], rec[f"{key}_sd"]))
                    values[model] = rec[f"{key}_mean"]
                else:
                    cells.append(f"_{status[(model, dataset)]}_")
            if len(values) == 2:
                delta = values["GHRN"] - values["CARE-GNN"]
                cells.append(f"{delta:+.4f}")
            else:
                cells.append("—")
            lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
        lines += ["", "Difference is GHRN minus CARE-GNN; positive favours GHRN."]
        write_text(os.path.join(OUT, f"metric_{key}.md"), lines)


def write_dataset_tables(agg, matrix):
    status = status_lookup(matrix)
    index = {(r["model"], r["dataset"]): r for r in agg}
    for dataset in DATASETS:
        present = [m for m in MODELS if (m, dataset) in index]
        lines = [f"# {dataset} — TR40", ""]
        if not present:
            lines += [
                "No model produced a completed test evaluation on this dataset.", "",
                "| Model | Status |", "|---|---|",
            ]
            lines += [f"| {m} | **{status[(m, dataset)]}** |" for m in MODELS]
            lines += ["", "See `../02_EXPERIMENT_MATRIX.md` for the reason behind each status."]
        else:
            lines += [
                "Mean ± sample standard deviation over training seeds 2, 42 and 72.", "",
                "| Metric | " + " | ".join(present) + " |",
                "|" + "---|" * (len(present) + 1),
            ]
            for key, label in METRICS:
                cells = [fmt(index[(m, dataset)][f"{key}_mean"],
                             index[(m, dataset)][f"{key}_sd"]) for m in present]
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            gpus = {index[(m, dataset)]["gpu"] for m in present}
            lines += ["", f"Hardware: {', '.join(sorted(gpus))}."]
        write_text(os.path.join(OUT, f"comparison_{dataset.lower().replace('-', '')}.md"), lines)


def write_efficiency(agg):
    # Keep the unrounded value alongside the rounded one. Formatting the CSV's
    # 4dp figure again at 3dp double-rounds: 0.028477 -> 0.0285 -> "0.029",
    # which is not the correct 3dp rendering of the measurement.
    records = []
    for r in agg:
        records.append(OrderedDict(
            gpu=r["gpu"], model=r["model"], dataset=r["dataset"],
            mean_epoch_s=round(r["mean_epoch_s"], 4) if r["mean_epoch_s"] is not None else None,
            peak_gpu_mb=round(r["peak_gpu_mb"], 1) if r["peak_gpu_mb"] is not None else None,
            params=int(r["params"]) if r["params"] is not None else None,
            mean_epochs=round(r["epochs"], 1) if r["epochs"] is not None else None,
            _raw_epoch_s=r["mean_epoch_s"], _raw_gpu_mb=r["peak_gpu_mb"],
        ))
    records.sort(key=lambda r: (r["gpu"], MODELS.index(r["model"]),
                                DATASETS.index(r["dataset"])))
    write_csv(os.path.join(OUT, "efficiency.csv"),
              [OrderedDict((k, v) for k, v in r.items() if not k.startswith("_"))
               for r in records])

    lines = [
        "# Efficiency — TR40", "",
        "Mean training time per epoch, peak GPU memory and trainable parameter count, "
        "averaged over the three training seeds.", "",
        "> Timings and memory are comparable **only within** one GPU block. The two "
        "blocks must not be pooled. Predictive metrics are unaffected by GPU class and "
        "remain comparable everywhere.", "",
    ]
    for gpu in sorted({r["gpu"] for r in records}):
        lines += [
            f"## {gpu}", "",
            "| Model | Dataset | Mean epoch (s) | Peak GPU (MB) | Parameters | Mean epochs |",
            "|---|---|---|---|---|---|",
        ]
        for r in [x for x in records if x["gpu"] == gpu]:
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r['_raw_epoch_s']:.3f} | "
                f"{r['_raw_gpu_mb']:.1f} | {r['params']:,} | {r['mean_epochs']:.0f} |")
        lines.append("")
    write_text(os.path.join(OUT, "efficiency.md"), lines)
    return records


def write_headline(agg):
    """Model-wise mean over the datasets where both models completed."""
    index = {(r["model"], r["dataset"]): r for r in agg}
    shared = [d for d in DATASETS if all((m, d) in index for m in MODELS)]
    rows = []
    for model in MODELS:
        record = OrderedDict(model=model, datasets=len(shared),
                             dataset_list=" ".join(shared))
        for key, label in METRICS:
            record[key] = round(
                sum(index[(model, d)][f"{key}_mean"] for d in shared) / len(shared), 6)
        rows.append(record)
    write_csv(os.path.join(OUT, "summary_by_model.csv"), rows)

    lines = [
        "# Model-wise summary", "",
        f"Averaged across the {len(shared)} datasets where **both** models completed, so "
        "the comparison is like-for-like: " + ", ".join(shared) + ".", "",
        "| Model | AUPRC | AUROC | Macro-F1 | Fraud recall |", "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['auprc']:.4f} | {r['auroc']:.4f} | "
                     f"{r['macro_f1']:.4f} | {r['fraud_recall']:.4f} |")
    write_text(os.path.join(OUT, "summary_by_model.md"), lines)
    return rows, shared


def write_seed_stability(per_seed, agg):
    """Flag runs whose best validation epoch is <= 3, the training-collapse signature."""
    flagged = []
    for row in per_seed:
        if row["best_epoch"] is not None and row["best_epoch"] <= 3:
            flagged.append(OrderedDict(
                model=row["model"], dataset=row["dataset"], seed=row["train_seed"],
                auprc=round(row["auprc"], 4), best_epoch=int(row["best_epoch"]),
                epochs=int(row["epochs"]), threshold=row["threshold"],
                fraud_recall=round(row["fraud_recall"], 4),
                fraud_precision=round(row["fraud_precision"], 4),
            ))
    write_csv(os.path.join(OUT, "seed_stability_flags.csv"), flagged)

    spread = []
    index = {(r["model"], r["dataset"]): r for r in agg}
    for (model, dataset), rec in index.items():
        spread.append(OrderedDict(
            model=model, dataset=dataset,
            auprc_mean=round(rec["auprc_mean"], 4),
            auprc_sd=round(rec["auprc_sd"], 4),
            auprc_cv=round(rec["auprc_sd"] / rec["auprc_mean"], 4) if rec["auprc_mean"] else None,
        ))
    spread.sort(key=lambda r: -(r["auprc_cv"] or 0))
    write_csv(os.path.join(OUT, "seed_spread.csv"), spread)
    return flagged, spread


def main():
    os.makedirs(OUT, exist_ok=True)
    per_seed = load_per_seed()
    write_csv(os.path.join(OUT, "master_per_seed.csv"), per_seed)

    agg = aggregate(per_seed)
    write_csv(os.path.join(OUT, "master_mean_sd.csv"), agg)

    matrix = build_matrix(agg)
    write_csv(os.path.join(OUT, "experiment_matrix.csv"), matrix)

    write_metric_tables(agg, matrix)
    write_dataset_tables(agg, matrix)
    efficiency = write_efficiency(agg)
    headline, shared = write_headline(agg)
    flagged, spread = write_seed_stability(per_seed, agg)

    counts = {}
    for row in matrix:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    summary = OrderedDict(
        models=MODELS,
        datasets=DATASETS,
        required_cells=len(MODELS) * len(DATASETS),
        matrix_counts=counts,
        final_runs=len(per_seed),
        completed_cells=len(agg),
        shared_datasets=shared,
        headline={r["model"]: {"auprc": r["auprc"], "auroc": r["auroc"],
                               "macro_f1": r["macro_f1"],
                               "fraud_recall": r["fraud_recall"]} for r in headline},
        collapse_flagged_runs=len(flagged),
        highest_seed_variance=spread[0] if spread else None,
        efficiency_rows=len(efficiency),
    )
    with open(os.path.join(OUT, "consolidation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
