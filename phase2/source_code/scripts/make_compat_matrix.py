"""Write audit/COMPATIBILITY_MATRIX.{md,csv} for all 18 model-dataset pairs.

Static facts (native support, adapter actions) come from the audited official
code; execution status is read from smoke evidence if present
(results/aggregated/smoke/per_seed.json, or evidence/local_cpu/... when given
via --smoke-json), never typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["yelpchi", "amazon", "tfinance", "elliptic", "fdcompcn", "tsocial"]

NATIVE = {
    "SAGAD": {"yelpchi": "native (GADBench 'yelp')", "amazon": "native", "tfinance": "native", "elliptic": "native",
              "fdcompcn": "not in official config", "tsocial": "native (official MRQ impractical)"},
    "SpaceGNN": {"yelpchi": "native (GADBench 'yelp')", "amazon": "native", "tfinance": "native",
                 "elliptic": "unsupported (crash: stdneg undefined)", "fdcompcn": "unsupported (crash: stdneg undefined)",
                 "tsocial": "native"},
    "CRoC": {"yelpchi": "native", "amazon": "native", "tfinance": "native ('tfin')", "elliptic": "unsupported (NotImplementedError)",
             "fdcompcn": "unsupported (NotImplementedError)", "tsocial": "native ('tsoc')"},
}
ADAPTER = {
    "SAGAD": "canonical union view -> load_data graph ops; official semi_train.conf.yaml values ({}); exact vectorised MRQ",
    "SpaceGNN": "canonical union view + self-loops; feature_i propagation; {}",
    "CRoC": "{}",
}


def adapter_note(model: str, ds: str) -> str:
    if model == "SAGAD":
        return ADAPTER[model].format("default block" if ds == "fdcompcn" else "dataset block")
    if model == "SpaceGNN":
        return ADAPTER[model].format("argparse defaults + std 0.015 (B3)" if ds in ("elliptic", "fdcompcn")
                                     else "PARAMETERS / Table 4 values")
    if ds in ("yelpchi", "amazon", "fdcompcn"):
        extra = " (3 semantic relations; stored union 'homo' excluded, B8)" if ds == "fdcompcn" else ""
        return f"relational branch: one-hot relation types, min-max features, base model{extra}"
    return "single-relation branch: constant type, z-score features, _Large model" + (" (B5/B6)" if ds == "elliptic" else "")


def classify(native: str) -> str:
    if native.startswith("native"):
        return "PASS_WITH_SHARED_ADAPTER"
    return "ADAPTED (documented)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-json", default=str(ROOT / "results" / "aggregated" / "smoke" / "per_seed.json"))
    p.add_argument("--label", default="local CPU smoke (2 epochs)")
    args = p.parse_args(argv)
    smoke = {}
    path = Path(args.smoke_json)
    if path.exists():
        for row in json.loads(path.read_text(encoding="utf-8")):
            smoke[(row["model"], row["dataset"])] = row
    rows = []
    for model in ("SAGAD", "SpaceGNN", "CRoC"):
        for ds in DATASETS:
            s = smoke.get((model, ds), {})
            status = s.get("status", "NOT_RUN")
            detail = ""
            if status == "COMPLETE":
                detail = (f"reload check {'passed' if s.get('reload_check_passed') else 'FAILED'}; "
                          f"{s.get('epochs_run')} epochs; {float(s.get('total_wall_seconds') or 0):.0f}s on {s.get('device')}")
            elif status not in ("NOT_RUN",):
                detail = s.get("failure_reason", "")
            subset = None
            if s.get("result_dir"):
                summary_path = ROOT / s["result_dir"] / "summary.json"
                if summary_path.exists():
                    cfg = json.loads(summary_path.read_text(encoding="utf-8")).get("configuration") or {}
                    subset = cfg.get("smoke_subset_nodes") or None
            rows.append({"model": model, "dataset": ds, "official_support": NATIVE[model][ds],
                         "compatibility_status": classify(NATIVE[model][ds]), "adapter": adapter_note(model, ds),
                         "smoke_status": status, "smoke_detail": detail,
                         "subset": (f"{int(subset):,}-node induced subgraph (smoke only)" if subset
                                    else "full graph")})
    with (ROOT / "audit" / "COMPATIBILITY_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 18-pair compatibility matrix", "",
             "`PASS_WITH_SHARED_ADAPTER`: the official code supports the dataset; only the frozen canonical data and "
             "splits are fed through a documented adapter. `ADAPTED (documented)`: the official code has no entry for "
             "the dataset; the decision is recorded in audit/DEVIATIONS.md section B. No pair is UNSUPPORTED.", "",
             f"Execution evidence column: {args.label}. Smoke runs are feasibility tests, not results.", "",
             "| Model | Dataset | Official support | Status | Adapter | Smoke | Smoke detail |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['official_support']} | {r['compatibility_status']} | "
                     f"{r['adapter']} | {r['smoke_status']} | {r['smoke_detail']} ({r['subset']}) |")
    (ROOT / "audit" / "COMPATIBILITY_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
