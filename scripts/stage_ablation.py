"""GHRN refinement ablation: stage 1 (unrefined) against stage 2 (refined).

GHRN runs a two-stage procedure. Stage 1 trains the beta-wavelet backbone on the
unpruned graph; its predictions then drive the edge pruning, and stage 2 retrains on
the refined graph. Each stage recorded its own best-validation metrics, which gives a
within-model ablation of the refinement step at no extra cost.

Everything here is scored on **validation**, never on test, so the ablation does not
touch the held-out set.

Writes ../tables/stage_ablation.csv / .md

Run:  python stage_ablation.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = os.path.dirname(HERE)
EVIDENCE = os.path.join(DELIVERABLE, "evidence")
OUT = os.path.join(DELIVERABLE, "tables")

DATASETS = ["YelpChi", "Amazon", "T-Finance", "T-Social", "Elliptic", "FDCompCN"]
CANON = {
    "yelpchi": "YelpChi", "amazon": "Amazon", "tfinance": "T-Finance",
    "tsocial": "T-Social", "elliptic": "Elliptic", "fdcompcn": "FDCompCN",
}


def mean(values):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(EVIDENCE)):
        path = os.path.join(EVIDENCE, name, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("model") != "GHRN":
            continue
        s1, s2 = data.get("stage1") or {}, data.get("stage2") or {}
        if not s1 or not s2:
            continue
        v1 = s1.get("best_selection")
        v2 = s2.get("best_selection")
        rows.append(OrderedDict(
            dataset=CANON[data["dataset"].strip().lower()],
            seed=data.get("train_seed"),
            stage1_val_auprc=round(v1, 6) if v1 is not None else None,
            stage2_val_auprc=round(v2, 6) if v2 is not None else None,
            delta_stage2_minus_stage1=round(v2 - v1, 6) if None not in (v1, v2) else None,
            refinement_helped=(v2 > v1) if None not in (v1, v2) else None,
            selected_stage=data.get("selected_stage"),
            stage1_epochs=s1.get("epochs_run"),
            stage2_epochs=s2.get("epochs_run"),
        ))

    rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), r["seed"]))
    with open(os.path.join(OUT, "stage_ablation.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)

    lines = [
        "# GHRN refinement ablation — stage 1 against stage 2", "",
        "GHRN trains in two stages: **stage 1** fits the beta-wavelet backbone on the "
        "unpruned graph, and its predictions drive the edge pruning; **stage 2** retrains "
        "on the refined graph. Comparing the two isolates what the heterophily refinement "
        "contributes, independently of any comparison with CARE-GNN.", "",
        "All values are **best validation AUPRC**, the protocol's selection metric. "
        "Nothing here touches the test set.", "",
        "| Dataset | Seed | Stage 1 (unrefined) | Stage 2 (refined) | Δ | Refinement helped? |",
        "|---|---|---|---|---|---|",
    ]
    for dataset in DATASETS:
        for r in by_dataset.get(dataset, []):
            mark = "yes" if r["refinement_helped"] else "**no**"
            lines.append(
                f"| {r['dataset']} | {r['seed']} | {r['stage1_val_auprc']:.4f} | "
                f"{r['stage2_val_auprc']:.4f} | {r['delta_stage2_minus_stage1']:+.4f} | {mark} |")

    lines += ["", "## Mean effect per dataset", "",
              "| Dataset | Stage 1 | Stage 2 | Mean Δ | Seeds helped |", "|---|---|---|---|---|"]
    helped_total = 0
    for dataset in DATASETS:
        group = by_dataset.get(dataset)
        if not group:
            continue
        m1 = mean([r["stage1_val_auprc"] for r in group])
        m2 = mean([r["stage2_val_auprc"] for r in group])
        helped = sum(1 for r in group if r["refinement_helped"])
        helped_total += helped
        lines.append(f"| {dataset} | {m1:.4f} | {m2:.4f} | {m2 - m1:+.4f} | {helped} of {len(group)} |")

    lines += [
        "", f"Across all {len(rows)} GHRN runs, refinement improved validation AUPRC in "
        f"**{helped_total} of {len(rows)}**.", "",
        "`selected_stage` in each run's `summary.json` records which stage supplied the "
        "final checkpoint.",
    ]

    with open(os.path.join(OUT, "stage_ablation.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(json.dumps({
        "runs": len(rows),
        "helped": helped_total,
        "hurt_or_equal": len(rows) - helped_total,
        "selected_stage_counts": {
            s: sum(1 for r in rows if r["selected_stage"] == s)
            for s in sorted({r["selected_stage"] for r in rows})},
    }, indent=2))
    for dataset in DATASETS:
        group = by_dataset.get(dataset)
        if group:
            m1 = mean([r["stage1_val_auprc"] for r in group])
            m2 = mean([r["stage2_val_auprc"] for r in group])
            print(f"{dataset:<10} stage1={m1:.4f} stage2={m2:.4f} delta={m2 - m1:+.4f}")


if __name__ == "__main__":
    main()
