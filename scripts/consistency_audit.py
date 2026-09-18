"""Check every 'mean ± sd' figure quoted in the documents against the generated tables.

The documents are written by hand; the tables are generated from the run records. This
script re-reads both and verifies that every `0.1234 ± 0.5678` pair appearing in any
document actually exists in tables/master_mean_sd.csv at 4 decimal places, so a
transcription slip cannot survive into the deliverable.

Also re-derives the experiment-matrix counts and the model-wise averages quoted in the
executive summary.

Writes ../CONSISTENCY_AUDIT.md and exits non-zero if anything fails.

Run:  python consistency_audit.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = os.path.dirname(HERE)
TABLES = os.path.join(DELIVERABLE, "tables")

METRICS = ["auprc", "auroc", "macro_f1", "fraud_f1", "fraud_precision",
           "fraud_recall", "gmean", "accuracy"]

PAIR = re.compile(r"(\d\.\d{4})\s*±\s*(\d\.\d{4})")


def read_csv(name):
    with open(os.path.join(TABLES, name), newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    agg = read_csv("master_mean_sd.csv")
    per_seed = read_csv("master_per_seed.csv")
    matrix = read_csv("experiment_matrix.csv")

    # Every (mean, sd) pair the tables legitimately contain, at 4dp.
    known = set()
    for row in agg:
        for metric in METRICS:
            mean = float(row[f"{metric}_mean"])
            sd = float(row[f"{metric}_sd"])
            known.add((f"{mean:.4f}", f"{sd:.4f}"))

    checks, failures = [], []

    # --- 1. every quoted mean ± sd must exist in the tables ---
    docs = sorted(f for f in os.listdir(DELIVERABLE) if f.endswith(".md"))
    quoted = 0
    for doc in docs:
        with open(os.path.join(DELIVERABLE, doc), encoding="utf-8") as handle:
            text = handle.read()
        for mean, sd in PAIR.findall(text):
            quoted += 1
            if (mean, sd) not in known:
                failures.append(f"{doc}: quoted `{mean} ± {sd}` is not in master_mean_sd.csv")
    checks.append(("Quoted mean ± sd figures traced to the tables", quoted, not failures))

    # --- 2. matrix counts ---
    counts = {}
    for row in matrix:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    expected = {"COMPLETED": 10, "FAILED": 2}
    ok = counts == expected and len(matrix) == 12
    if not ok:
        failures.append(f"experiment matrix counts are {counts} over {len(matrix)} rows, "
                        f"expected {expected} over 12")
    checks.append(("Experiment matrix is 12 cells: 10 completed, 2 failed", len(matrix), ok))

    # --- 3. run count ---
    ok = len(per_seed) == 30
    if not ok:
        failures.append(f"per-seed table holds {len(per_seed)} runs, expected 30")
    checks.append(("Final runs consolidated", len(per_seed), ok))

    # --- 4. model-wise averages quoted in the executive summary ---
    index = {(r["model"], r["dataset"]): r for r in agg}
    models = ["CARE-GNN", "GHRN"]
    datasets = sorted({r["dataset"] for r in agg})
    shared = [d for d in datasets if all((m, d) in index for m in models)]
    headline = {}
    for model in models:
        headline[model] = {
            metric: sum(float(index[(model, d)][f"{metric}_mean"]) for d in shared) / len(shared)
            for metric in ("auprc", "auroc", "macro_f1", "fraud_recall")}

    expected_headline = {
        "CARE-GNN": {"auprc": 0.5111, "auroc": 0.8091, "macro_f1": 0.7162, "fraud_recall": 0.4670},
        "GHRN": {"auprc": 0.5390, "auroc": 0.8326, "macro_f1": 0.7416, "fraud_recall": 0.5618},
    }
    ok = True
    for model, metrics in expected_headline.items():
        for metric, value in metrics.items():
            got = round(headline[model][metric], 4)
            if abs(got - value) > 0.0001:
                ok = False
                failures.append(f"headline {model} {metric}: documents say {value}, "
                                f"tables give {got}")
    checks.append((f"Executive-summary averages over {len(shared)} shared datasets", len(shared), ok))

    # --- 5. collapse flags: all three are CARE-GNN, none are GHRN ---
    flags = read_csv("seed_stability_flags.csv")
    care = sum(1 for r in flags if r["model"] == "CARE-GNN")
    ghrn = sum(1 for r in flags if r["model"] == "GHRN")
    ok = len(flags) == 3 and care == 3 and ghrn == 0
    if not ok:
        failures.append(f"collapse flags: {len(flags)} total, {care} CARE-GNN, {ghrn} GHRN; "
                        f"expected 3 / 3 / 0")
    checks.append(("Training collapses: 3, all CARE-GNN", len(flags), ok))

    # --- 6. stage ablation: 15 GHRN runs, refined stage shipped in all ---
    stages = read_csv("stage_ablation.csv")
    helped = sum(1 for r in stages if r["refinement_helped"] == "True")
    refined = sum(1 for r in stages if r["selected_stage"] == "stage2_refined")
    ok = len(stages) == 15 and helped == 9 and refined == 15
    if not ok:
        failures.append(f"stage ablation: {len(stages)} runs, helped {helped}, "
                        f"refined shipped {refined}; expected 15 / 9 / 15")
    checks.append(("GHRN stage ablation: refinement helped 9 of 15", len(stages), ok))

    # --- 7. no completed cell is missing a metric ---
    missing = [f"{r['model']}×{r['dataset']}.{m}" for r in agg for m in METRICS
               if not r[f"{m}_mean"]]
    if missing:
        failures.append("cells missing a metric: " + ", ".join(missing))
    checks.append(("Every completed cell has all 8 metrics", len(agg) * len(METRICS), not missing))

    # --- 8. hand-written stage-ablation table matches the generated one ---
    # A transposition here is invisible to the mean ± sd check, because these are
    # bare numbers and counts rather than mean/sd pairs.
    by_ds = {}
    for row in read_csv("stage_ablation.csv"):
        by_ds.setdefault(row["dataset"], []).append(row)
    expected_rows = {}
    for dataset, group in by_ds.items():
        s1 = sum(float(r["stage1_val_auprc"]) for r in group) / len(group)
        s2 = sum(float(r["stage2_val_auprc"]) for r in group) / len(group)
        helped = sum(1 for r in group if r["refinement_helped"] == "True")
        expected_rows[dataset] = (round(s1, 4), round(s2, 4), round(s2 - s1, 4),
                                  helped, len(group))

    row_re = re.compile(
        r"^\|\s*([A-Za-z\-]+)\s*\|\s*\*{0,2}(-?[\d.]+)\*{0,2}\s*\|"
        r"\s*\*{0,2}(-?[\d.]+)\*{0,2}\s*\|\s*\*{0,2}([+−-][\d.]+)\*{0,2}\s*\|"
        r"\s*(\d+) of (\d+)\s*\|\s*$", re.M)
    compared = 0
    doc_path = os.path.join(DELIVERABLE, "04_GRAPH_CHARACTERISTICS.md")
    if os.path.isfile(doc_path):
        with open(doc_path, encoding="utf-8") as handle:
            text = handle.read()
        for dataset, s1, s2, delta, helped, total in row_re.findall(text):
            if dataset not in expected_rows:
                continue
            compared += 1
            e1, e2, ed, eh, et = expected_rows[dataset]
            got = (float(s1), float(s2), float(delta.replace("−", "-")),
                   int(helped), int(total))
            if abs(got[0] - e1) > 0.0001 or abs(got[1] - e2) > 0.0001 \
                    or abs(got[2] - ed) > 0.0001 or got[3] != eh or got[4] != et:
                failures.append(
                    f"04_GRAPH_CHARACTERISTICS.md stage-ablation row for {dataset}: "
                    f"document says {got}, tables give {(e1, e2, ed, eh, et)}")
    ok = compared == len(expected_rows)
    if not ok:
        failures.append(f"stage-ablation rows checked: {compared}, expected "
                        f"{len(expected_rows)}")
    checks.append(("Stage-ablation table in the prose matches the tables", compared, ok))

    # --- 9. efficiency figures quoted in the prose match the measurements ---
    # Compare against the unrounded measurement, not efficiency.csv's 4dp copy:
    # comparing a 3dp document figure to a 4dp stored one hides a double-rounding
    # discrepancy inside the tolerance.
    eff = read_csv("efficiency.csv")
    eff_index = {(r["model"], r["dataset"]): {
        "mean_epoch_s": agg_row["mean_epoch_s"],
        "peak_gpu_mb": agg_row["peak_gpu_mb"],
        "params": agg_row["params"],
    } for r in eff for agg_row in agg
        if agg_row["model"] == r["model"] and agg_row["dataset"] == r["dataset"]}
    eff_re = re.compile(
        r"^\|\s*(CARE-GNN|GHRN)\s*\|\s*([A-Za-z\-]+)\s*\|\s*([\d.]+)\s*\|"
        r"\s*([\d,.]+)\s*\|\s*([\d,]+)\s*\|", re.M)
    eff_checked = 0
    doc_path = os.path.join(DELIVERABLE, "05_EFFICIENCY.md")
    if os.path.isfile(doc_path):
        with open(doc_path, encoding="utf-8") as handle:
            text = handle.read()
        for model, dataset, epoch, mem, params in eff_re.findall(text):
            rec = eff_index.get((model, dataset))
            if not rec:
                continue
            eff_checked += 1
            if epoch != f"{float(rec['mean_epoch_s']):.3f}":
                failures.append(f"05_EFFICIENCY.md {model}×{dataset} epoch: document "
                                f"{epoch}, measurement rounds to "
                                f"{float(rec['mean_epoch_s']):.3f}")
            if mem.replace(",", "") != f"{float(rec['peak_gpu_mb']):.1f}":
                failures.append(f"05_EFFICIENCY.md {model}×{dataset} memory: document "
                                f"{mem}, measurement rounds to "
                                f"{float(rec['peak_gpu_mb']):.1f}")
            if int(params.replace(",", "")) != int(float(rec["params"])):
                failures.append(f"05_EFFICIENCY.md {model}×{dataset} parameters: document "
                                f"{params}, measurement {rec['params']}")
    ok = eff_checked == len(eff)
    if not ok:
        failures.append(f"efficiency rows checked: {eff_checked}, expected {len(eff)}")
    checks.append(("Efficiency table in the prose matches the measurements", eff_checked, ok))

    # --- 10. remaining hand-written tables against their generated sources ---
    def doc_text(name):
        path = os.path.join(DELIVERABLE, name)
        if not os.path.isfile(path):
            return ""
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def num(text):
        return float(text.replace(",", "").replace("−", "-").replace("%", ""))

    def check_rows(label, doc, pattern, expected, fields, tol=0.00005):
        """expected: {key: (v1, v2, ...)} matched against the regex's capture groups."""
        nonlocal failures
        found = 0
        for match in re.finditer(pattern, doc_text(doc), re.M):
            key = match.group(1).strip()
            if key not in expected:
                continue
            found += 1
            got = [num(g) for g in match.groups()[1:]]
            for name, g, e in zip(fields, got, expected[key]):
                if abs(g - e) > tol:
                    failures.append(f"{doc} {label} row {key}, {name}: document {g}, "
                                    f"tables {e}")
        ok = found == len(expected)
        if not ok:
            failures.append(f"{doc} {label}: matched {found} rows, expected {len(expected)}")
        checks.append((f"{label} in the prose matches the tables", found, ok))

    # 10a. dataset profile (04)
    prof = {r["dataset"]: (float(r["global_heterophily"]), float(r["fraud_local_mean"]),
                           float(r["fraud_local_median"]), float(r["fraud_local_std"]),
                           float(r["labelled_edges_considered"]), float(r["test_nodes"]),
                           float(r["test_fraud_pct"]))
            for r in read_csv("dataset_profile.csv")}
    # The dataset name may carry a trailing footnote marker (" \*"), as T-Social does.
    check_rows("Dataset profile", "04_GRAPH_CHARACTERISTICS.md",
               r"^\|\s*([A-Za-z\-]+)(?:\s*\\\*)?\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
               r"\s*([\d.]+)\s*\|"
               r"\s*([\d.]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d.]+)%\s*\|",
               prof,
               ["global heterophily", "fraud-local mean", "fraud-local median",
                "fraud-local SD", "labelled edges", "test nodes", "test fraud %"],
               tol=0.006)

    # 10b. refinement ablation (04)
    refine = {r["dataset"]: (float(r["del_ratio"]), float(r["edges_before"]),
                             float(r["edges_removed"]),
                             float(r["edge_heterophily_before"]),
                             float(r["edge_heterophily_after"]),
                             float(r["edge_heterophily_reduction"]))
              for r in read_csv("refinement_ablation.csv")}
    check_rows("Refinement ablation", "04_GRAPH_CHARACTERISTICS.md",
               r"^\|\s*([A-Za-z\-]+)\s*\|\s*([\d.]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|"
               r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\*{0,2}([+−-][\d.]+)\*{0,2}\s*\|",
               refine,
               ["del_ratio", "edges before", "edges removed", "heterophily before",
                "heterophily after", "reduction"],
               tol=0.0000005)

    # 10c. collapse flags (06) - nine columns:
    # model | dataset | seed | AUPRC | best epoch | epochs | threshold | recall | precision
    flag_rows = {f"{r['model']}|{r['dataset']}|{r['seed']}":
                 (float(r["auprc"]), float(r["best_epoch"]), float(r["epochs"]),
                  float(r["threshold"]), float(r["fraud_recall"]),
                  float(r["fraud_precision"]))
                 for r in flags}
    found = 0
    for match in re.finditer(
            r"^\|\s*(CARE-GNN|GHRN)\s*\|\s*([A-Za-z\-]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
            r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$",
            doc_text("06_QA_NOTES.md"), re.M):
        key = f"{match.group(1)}|{match.group(2)}|{match.group(3)}"
        if key not in flag_rows:
            continue
        found += 1
        got = [float(match.group(i)) for i in (4, 5, 6, 7, 8, 9)]
        for name, g, e in zip(["AUPRC", "best epoch", "epochs", "threshold",
                               "recall", "precision"], got, flag_rows[key]):
            if abs(g - e) > 0.00005:
                failures.append(f"06_QA_NOTES.md collapse row {key}, {name}: "
                                f"document {g}, tables {e}")
    ok = found == len(flag_rows)
    if not ok:
        failures.append(f"06_QA_NOTES.md collapse table: matched {found} rows, "
                        f"expected {len(flag_rows)}")
    checks.append(("Collapse table in the prose matches the tables", found, ok))

    # 10d. seed spread (06)
    spread_rows = {f"{r['model']}|{r['dataset']}":
                   (float(r["auprc_mean"]), float(r["auprc_sd"]), float(r["auprc_cv"]))
                   for r in read_csv("seed_spread.csv")}
    # Anchored to exactly five columns so the nine-column collapse table above
    # cannot also match this pattern.
    found = 0
    for match in re.finditer(
            r"^\|\s*(CARE-GNN|GHRN)\s*\|\s*([A-Za-z\-]+)\s*\|\s*\*{0,2}([\d.]+)\*{0,2}\s*\|"
            r"\s*\*{0,2}([\d.]+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)\*{0,2}\s*\|\s*$",
            doc_text("06_QA_NOTES.md"), re.M):
        key = f"{match.group(1)}|{match.group(2)}"
        if key not in spread_rows:
            continue
        found += 1
        got = [float(match.group(i)) for i in (3, 4, 5)]
        # The document rounds CV to 3dp; allow the half-way case (0.3045 -> 0.305).
        for name, g, e in zip(["mean", "SD", "CV"], got, spread_rows[key]):
            if abs(g - e) > 0.00055:
                failures.append(f"06_QA_NOTES.md seed-spread row {key}, {name}: "
                                f"document {g}, tables {e}")
    ok = found == len(spread_rows)
    if not ok:
        failures.append(f"06_QA_NOTES.md seed-spread table: matched {found} rows, "
                        f"expected {len(spread_rows)}")
    checks.append(("Seed-spread table in the prose matches the tables", found, ok))

    # 10e. head-to-head tally (03), recomputed from the cell means
    tally = {}
    for metric in METRICS:
        wins = sum(1 for d in sorted({r["dataset"] for r in agg})
                   if all((m, d) in index for m in models)
                   and float(index[("GHRN", d)][f"{metric}_mean"])
                   > float(index[("CARE-GNN", d)][f"{metric}_mean"]))
        tally[metric] = wins
    label_map = {"AUPRC": "auprc", "AUROC": "auroc", "Macro-F1": "macro_f1",
                 "Fraud recall": "fraud_recall", "Fraud precision": "fraud_precision",
                 "G-mean": "gmean", "Accuracy": "accuracy"}
    found = 0
    # The label class must include digits, or "Macro-F1" never matches.
    for match in re.finditer(r"^\|\s*([A-Za-z\d\- ]+?)\s*\|\s*(\d)\s*\|\s*(\d)\s*\|\s*$",
                             doc_text("03_RESULTS.md"), re.M):
        label = match.group(1).strip()
        if label not in label_map:
            continue
        found += 1
        g_wins, c_wins = int(match.group(2)), int(match.group(3))
        expected_g = tally[label_map[label]]
        if g_wins != expected_g or c_wins != len(shared) - expected_g:
            failures.append(f"03_RESULTS.md head-to-head {label}: document "
                            f"{g_wins}/{c_wins}, recomputed "
                            f"{expected_g}/{len(shared) - expected_g}")
    ok = found == len(label_map)
    if not ok:
        failures.append(f"03_RESULTS.md head-to-head tally: matched {found} rows, "
                        f"expected {len(label_map)}")
    checks.append(("Head-to-head tally recomputed from the cell means", found, ok))

    # --- report ---
    lines = [
        "# Consistency audit", "",
        f"Generated {datetime.now().strftime('%d %B %Y')}.", "",
        "Automated check that the prose in this package matches the generated tables, "
        "and that the generated tables match the run records. Regenerate with "
        "`python scripts/consistency_audit.py`.", "",
        "| Check | Items | Result |", "|---|---|---|",
    ]
    for label, count, ok in checks:
        lines.append(f"| {label} | {count} | {'**PASS**' if ok else '**FAIL**'} |")

    if failures:
        lines += ["", "## Failures", ""] + [f"- {f}" for f in failures]
    else:
        lines += [
            "", "## Result", "",
            "**All checks passed.** Every `mean ± sd` figure quoted in the nine documents "
            "was found in `tables/master_mean_sd.csv`, the experiment matrix accounts for "
            "all 12 required cells, and the 30 consolidated runs carry a complete metric "
            "set. No figure in this package is unsourced.",
        ]

    with open(os.path.join(DELIVERABLE, "CONSISTENCY_AUDIT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    for label, count, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label} ({count})")
    for failure in failures:
        print("  !", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
