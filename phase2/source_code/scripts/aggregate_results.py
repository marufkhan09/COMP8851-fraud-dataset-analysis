"""Aggregate run evidence into CSV/JSON/Markdown tables.

    python scripts/aggregate_results.py              # final runs (results/raw)
    python scripts/aggregate_results.py --phase smoke

Outputs (results/aggregated/<phase>/):
    per_seed.csv / per_seed.json        one row per model x dataset x ratio x seed, every status
    mean_sd.csv / mean_sd.json          mean and sample SD (ddof=1) over COMPLETE seeds
    status_matrix.md                    explicit status for all 18 model-dataset pairs
    comparison_table6.md                AUPRC next to the previous 10-model Table 6 (final phase only)

Only COMPLETE runs enter means. INCOMPLETE/FAILED/missing cells are listed
with their reason; nothing is imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from shared.comp8851.evaluator import METRIC_KEYS  # noqa: E402
except ImportError:
    METRIC_KEYS = ("auroc", "auprc", "macro_f1", "fraud_f1", "fraud_precision", "fraud_recall", "gmean", "accuracy")

MODELS = {"sagad": "SAGAD", "spacegnn": "SpaceGNN", "croc": "CRoC"}
DATASETS = ["yelpchi", "amazon", "tfinance", "elliptic", "fdcompcn", "tsocial"]
DATASET_LABEL = {"yelpchi": "YelpChi", "amazon": "Amazon", "tfinance": "T-Finance", "elliptic": "Elliptic",
                 "fdcompcn": "FDCompCN", "tsocial": "T-Social"}
SEEDS = [2, 42, 72]
PHASE_ROOT = {"final": ROOT / "results" / "raw", "smoke": ROOT / "results" / "smoke", "pilot": ROOT / "results" / "pilot"}

#: Table 6 of COMP8851_Assessment2_Report_FINAL-069.pdf (test AUPRC at TR40,
#: mean +- sample SD over three seeds). Markers as in the report:
#: '*' single native-protocol run, '+' non-canonical split or subsample.
#: '#' marks a finding of this audit: PC-GNN's YelpChi evidence uses
#: validation/test partitions of 9,099/18,474 nodes, not the registry's 9,191/18,382.
TABLE6 = {
    "CARE-GNN": ["0.510 ±0.013", "0.715 ±0.232", "0.712 ±0.040", "0.484 ±0.027", "0.134 ±0.013", "FAILED"],
    "FRAUDRE": ["0.512 ±0.006", "0.850 ±0.004", "0.801 ±0.170", "0.021 ±0.002 +", "0.156 ±0.028", "0.925 ±0.070"],
    "PC-GNN": ["0.524 ±0.011 #", "0.882 ±0.006", "0.751 ±0.002", "0.465 ±0.063", "0.246 ±0.020", "-"],
    "GAGA": ["0.795 *", "0.871 *", "0.845 *", "0.599 *", "0.228 *", "-"],
    "DGA-GNN": ["0.922 *", "0.927 *", "0.904 *", "0.813 *", "0.298 ±0.147 +", "0.981 *"],
    "GAAP": ["0.899 ±0.008", "0.909 ±0.008", "0.887 ±0.016", "0.756 ±0.023", "0.329 ±0.013", "0.994 ±0.031"],
    "BWGNN": ["0.537 ±0.018", "0.881 ±0.013", "0.705 ±0.134", "0.444 ±0.037", "0.158 ±0.004", "0.714 ±0.059"],
    "GHRN": ["0.525 ±0.012", "0.876 ±0.024", "0.676 ±0.206", "0.498 ±0.057", "0.120 ±0.003", "FAILED"],
    "PMP": ["0.758 ±0.003", "0.908 ±0.004", "0.901 ±0.010", "0.458 ±0.073", "0.123 ±0.003", "0.919 ±0.052"],
    "SplitGNN": ["0.496 ±0.005 +", "0.550 ±0.016 +", "0.798 ±0.069 +", "0.577 ±0.028 +", "0.110 ±0.022 +", "0.437 ±0.019 +"],
}
#: Fraud prevalence of each frozen TR40 *test* partition (= random-ranker AUPRC):
#: 2671/18382, 329/3456, 721/15744, 1417/18625, 223/2128, 69712/2312426.
PREVALENCE = {"yelpchi": 0.1453, "amazon": 0.0952, "tfinance": 0.0458, "elliptic": 0.0761, "fdcompcn": 0.1048, "tsocial": 0.0301}


def load_runs(root: Path, ratios: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for slug, label in MODELS.items():
        for ds in DATASETS:
            for ratio in ratios:
                for seed in SEEDS if root.name != "smoke" else [2]:
                    d = root / slug / ds / "unified" / ratio.lower() / f"seed_{seed}"
                    row: Dict[str, Any] = {"model": label, "dataset": ds, "ratio": ratio, "seed": seed,
                                           "result_dir": str(d.relative_to(ROOT)) if d.exists() else ""}
                    path = d / "summary.json"
                    if not path.exists():
                        row["status"] = "NOT_RUN"
                        rows.append(row)
                        continue
                    s = json.loads(path.read_text(encoding="utf-8"))
                    t = s.get("test_metrics") or {}
                    timing = s.get("timing") or {}
                    cfg = s.get("configuration") or {}
                    row.update({
                        "status": s.get("status"), "failure_reason": s.get("failure_reason", ""),
                        "stop_reason": s.get("stop_reason", ""), "run_mode": s.get("run_mode", ""),
                        "best_epoch": (s.get("best_validation_epoch", -1) or 0) + 1,
                        "threshold": s.get("selected_threshold"),
                        "val_auprc": (s.get("best_validation_metrics") or {}).get("auprc"),
                        **{f"test_{k}": t.get(k) for k in METRIC_KEYS},
                        "test_prevalence": t.get("test_prevalence"),
                        "auprc_lift": t.get("auprc_lift_over_prevalence"),
                        "epochs_run": timing.get("training_epochs_completed"),
                        "train_seconds": timing.get("total_train_epoch_seconds"),
                        "validation_seconds": timing.get("total_validation_seconds"),
                        "setup_seconds": timing.get("setup_seconds"),
                        "total_wall_seconds": timing.get("total_wall_seconds"),
                        "peak_gpu_memory_mb": timing.get("peak_gpu_memory_mb"),
                        "peak_rss_mb": timing.get("peak_rss_mb"),
                        "device": cfg.get("resolved_device"),
                        "hyperparameter_source": cfg.get("hyperparameter_source"),
                        "hyperparameters": json.dumps(cfg.get("hyperparameters", {}), sort_keys=True),
                        "config_sha256": (s.get("protocol_identity") or {}).get("config_sha256"),
                        "split_sha256": (s.get("protocol_identity") or {}).get("split_sha256"),
                        "model_source_commit": (s.get("protocol_identity") or {}).get("model_source_commit"),
                        "predictions_file": str((d / "predictions.npz").relative_to(ROOT)) if (d / "predictions.npz").exists() else "",
                        "predictions_sha256": s.get("predictions_sha256", ""),
                        "checkpoint": next((str(p.relative_to(ROOT)) for p in (d / "checkpoints").glob("*_best.pt")), ""),
                        "reload_check_passed": (s.get("checkpoint_reload_check") or {}).get("passed"),
                        "artifacts_complete": (s.get("artifact_verification") or {}).get("complete"),
                    })
                    rows.append(row)
    return rows


def mean_sd(rows: List[Dict[str, Any]], ratios: List[str]) -> List[Dict[str, Any]]:
    out = []
    for slug, label in MODELS.items():
        for ds in DATASETS:
            for ratio in ratios:
                cell = [r for r in rows if r["model"] == label and r["dataset"] == ds and r["ratio"] == ratio]
                done = [r for r in cell if r["status"] == "COMPLETE" or r.get("test_auprc") is not None]
                entry: Dict[str, Any] = {"model": label, "dataset": ds, "ratio": ratio,
                                         "seeds_complete": len(done), "seeds_expected": len(cell),
                                         "seeds_complete_list": ",".join(str(r["seed"]) for r in done)}
                if len(done) == len(cell) and done:
                    entry["status"] = "COMPLETED"
                elif done:
                    entry["status"] = "PARTIAL"
                elif any(r["status"] == "FAILED" for r in cell):
                    entry["status"] = "FAILED"
                elif any(r["status"] == "INCOMPLETE" for r in cell):
                    entry["status"] = "INCOMPLETE"
                else:
                    entry["status"] = "NOT_RUN"
                entry["reasons"] = "; ".join(sorted({f"seed {r['seed']}: {r.get('failure_reason') or r['status']}"
                                                     for r in cell if r["status"] != "COMPLETE"}))
                for key in METRIC_KEYS:
                    vals = [float(r[f"test_{key}"]) for r in done if r.get(f"test_{key}") is not None]
                    if vals:
                        entry[f"{key}_mean"] = float(np.mean(vals))
                        entry[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                entry["test_prevalence"] = PREVALENCE[ds]
                if "auprc_mean" in entry:
                    entry["auprc_lift"] = entry["auprc_mean"] / PREVALENCE[ds]
                out.append(entry)
    return out


def fmt(entry: Dict[str, Any]) -> str:
    if "auprc_mean" not in entry:
        return entry["status"]
    sd = entry.get("auprc_sd")
    text = f"{entry['auprc_mean']:.3f}" + (f" ±{sd:.3f}" if sd is not None and not math.isnan(sd) else "")
    if entry["status"] != "COMPLETED":
        text += f" ({entry['seeds_complete']}/{entry['seeds_expected']} seeds)"
    return text


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="final", choices=list(PHASE_ROOT))
    parser.add_argument("--ratios", nargs="*", default=["TR40"])
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args(argv)
    base = Path(args.results_dir).resolve()
    root = base / {"final": "raw", "smoke": "smoke", "pilot": "pilot"}[args.phase]
    out = base / "aggregated" / args.phase
    out.mkdir(parents=True, exist_ok=True)

    rows = load_runs(root, args.ratios)
    agg = mean_sd(rows, args.ratios)
    write_csv(out / "per_seed.csv", rows)
    (out / "per_seed.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    write_csv(out / "mean_sd.csv", agg)
    (out / "mean_sd.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")

    evidence = {"final": "FINAL protocol runs", "smoke": "SMOKE TESTS (2 epochs; feasibility only - NOT results)",
                "pilot": "PILOT runs (seed 2 only)"}[args.phase]
    lines = [f"# Status matrix - {evidence}", "", "| Model | " + " | ".join(DATASET_LABEL[d] for d in DATASETS) + " |",
             "|---|" + "---|" * len(DATASETS)]
    for ratio in args.ratios:
        for label in MODELS.values():
            cells = [next(e for e in agg if e["model"] == label and e["dataset"] == d and e["ratio"] == ratio) for d in DATASETS]
            lines.append(f"| {label} ({ratio}) | " + " | ".join(f"{e['status']}<br>{fmt(e)}" for e in cells) + " |")
    lines += ["", "Cells show status and test AUPRC (mean ± sample SD over completed seeds).",
              "Reasons for every non-completed seed:", ""]
    lines += [f"- {e['model']} / {e['dataset']} / {e['ratio']}: {e['reasons']}" for e in agg if e["reasons"]]
    (out / "status_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.phase == "final":
        t = ["# Test AUPRC at TR40 - previous ten models (report Table 6) and the three new models", "",
             "| Model | " + " | ".join(DATASET_LABEL[d] for d in DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
        for model, values in TABLE6.items():
            t.append(f"| {model} | " + " | ".join(values) + " |")
        for label in MODELS.values():
            cells = [next(e for e in agg if e["model"] == label and e["dataset"] == d and e["ratio"] == "TR40") for d in DATASETS]
            t.append(f"| **{label}** | " + " | ".join(fmt(e) for e in cells) + " |")
        t += ["", "| Random-ranker AUPRC (fraud prevalence) | " + " | ".join(f"{PREVALENCE[d]:.4f}" for d in DATASETS) + " |",
              "", "Comparability: the new rows use the frozen registry splits (byte-identical data and TR40 partitions to "
              "CARE-GNN/GHRN), seeds 2/42/72, validation-AUPRC checkpointing and the shared evaluator, so they are "
              "directly comparable with the unmarked three-seed rows. '*' single native run and '+' non-canonical split "
              "(report markers) are not comparable. '#' audit finding: PC-GNN's YelpChi partitions differ from the registry."]
        (out / "comparison_table6.md").write_text("\n".join(t) + "\n", encoding="utf-8")

    counts: Dict[str, int] = {}
    for e in agg:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    print(f"[{args.phase}] {len(rows)} seed rows; pair status: {counts}")
    print(f"tables written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
