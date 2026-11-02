"""Generate the frozen CARE-GNN and GHRN configuration files.

Run once to (re)create ``models/<model>/configs/{author,unified}/*.yml``. The
generated files are the artefacts that runs actually consume; this script
exists so the hyperparameter precedence rule is auditable in one place rather
than being retyped across 18 YAML files.

Precedence rule (run guide section 11):
  1. the paper's dataset-specific configuration when clearly stated
  2. otherwise the official repository's dataset-specific configuration
  3. for a non-native dataset, the closest paper-supported transferable
     configuration, documented before running

Usage:
    python shared/comp8851/make_configs.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# CARE-GNN -- official defaults from https://github.com/YingtongDou/CARE-GNN
# (train.py argparse). YelpChi and Amazon are the paper's datasets.
# --------------------------------------------------------------------------
CARE_AUTHOR: Dict[str, Dict[str, Any]] = {
    "yelpchi": {"batch_size": 1024, "lr": 0.01, "lambda_1": 2.0, "lambda_2": 1e-3,
                "emb_size": 64, "epochs": 31, "step_size": 2e-2, "under_sample": 1.0,
                "seed": 72, "inter": "GNN"},
    "amazon": {"batch_size": 256, "lr": 0.01, "lambda_1": 2.0, "lambda_2": 1e-3,
               "emb_size": 64, "epochs": 31, "step_size": 2e-2, "under_sample": 1.0,
               "seed": 72, "inter": "GNN"},
}

# --------------------------------------------------------------------------
# GHRN -- official defaults from https://github.com/blacksingular/GHRN
# (main.py argparse). YelpChi, Amazon, T-Finance and T-Social are native.
# The deletion ratio is the one parameter the paper varies per dataset.
# --------------------------------------------------------------------------
GHRN_AUTHOR: Dict[str, Dict[str, Any]] = {
    "yelpchi": {"hid_dim": 64, "order": 2, "lr": 0.01, "epochs": 100,
                "del_ratio": 0.015, "seed": 72},
    "amazon": {"hid_dim": 64, "order": 2, "lr": 0.01, "epochs": 100,
               "del_ratio": 0.015, "seed": 72},
    "tfinance": {"hid_dim": 64, "order": 2, "lr": 0.01, "epochs": 100,
                 "del_ratio": 0.015, "seed": 72},
    "tsocial": {"hid_dim": 64, "order": 2, "lr": 0.01, "epochs": 100,
                "del_ratio": 0.015, "seed": 72},
}

DATASETS = ("yelpchi", "amazon", "tfinance", "tsocial", "elliptic", "fdcompcn")

# Batch size for CARE-GNN on non-native datasets: the author code uses 1024 for
# the larger graph (YelpChi) and 256 for the smaller (Amazon). Transfer that
# rule by graph size rather than inventing a new value.
CARE_BATCH_BY_DATASET = {
    "yelpchi": 1024, "amazon": 256, "tfinance": 1024,
    "tsocial": 1024, "elliptic": 1024, "fdcompcn": 256,
}

NON_NATIVE_NOTE = {
    "care-gnn": {
        "tfinance": "Non-native. T-Finance is natively single-relation, so CARE-GNN's "
                    "inter-relation aggregator degenerates to one channel. Expect "
                    "CONDITIONAL (A2 mechanism risk), not PASS.",
        "tsocial": "Non-native and very large. Same single-relation caveat as T-Finance, "
                   "plus CARE-GNN's per-batch Python neighbour filtering scales poorly. "
                   "Treat OOM or timeout as a recorded feasibility outcome.",
        "elliptic": "Non-native, single-relation and temporal. The chronological split "
                    "applies. CARE-GNN has no temporal mechanism; interpret accordingly.",
        "fdcompcn": "Non-native but genuinely multi-relational (C-I-C, C-P-C, C-S-C), so "
                    "CARE-GNN's mechanism is intact. Closest transferable configuration "
                    "is the Amazon one, matching its small node count.",
    },
    "ghrn": {
        "elliptic": "Non-native and temporal. Chronological split applies; the 166-feature "
                    "locked view is used.",
        "fdcompcn": "Non-native and multi-relational. GHRN is homogeneous, so it receives "
                    "the frozen union view: PASS_WITH_SHARED_ADAPTER (A1).",
    },
}


def unified_common(dataset: str, seed: int, ratio: str) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "ratio": ratio,
        "seed": seed,
        "track": "unified",
        "selection_metric": "auprc",
        "epochs": 100,
        "patience": 20,
        "valid_every": 1,
        "protocol_version": "vast-v4.4",
    }


def write(path: Path, payload: Dict[str, Any], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    path.write_text(f"# {header}\n" + body, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    # ---- CARE-GNN author configs ----
    for dataset, values in CARE_AUTHOR.items():
        payload = dict(values)
        payload.update({"dataset": dataset, "track": "author", "model": "CARE-GNN"})
        write(
            REPO_ROOT / "models" / "care-gnn" / "configs" / "author" / f"care-gnn_{dataset}.yml",
            payload,
            "CARE-GNN author-mode configuration. Source: official repository "
            "https://github.com/YingtongDou/CARE-GNN train.py defaults.",
        )

    # ---- CARE-GNN unified configs ----
    for dataset in DATASETS:
        payload = unified_common(dataset, seed=2, ratio="TR40")
        payload.update({
            "model": "CARE-GNN",
            "emb_size": 64,
            "inter": "GNN",
            "lambda_1": 2.0,
            "lambda_2": 1e-3,
            "step_size": 2e-2,
            "under_sample": 1.0,
            "lr": 0.01,
            "batch_size": CARE_BATCH_BY_DATASET[dataset],
        })
        note = NON_NATIVE_NOTE["care-gnn"].get(dataset)
        if note:
            payload["compatibility_note"] = note
        write(
            REPO_ROOT / "models" / "care-gnn" / "configs" / "unified" / f"care-gnn_{dataset}_TR40.yml",
            payload,
            "CARE-GNN unified-track configuration under COMP8851 protocol v4.4. "
            "Architecture and model hyperparameters follow the official repository; "
            "split, seeds, stopping, selection, threshold and evaluator are protocol-fixed.",
        )

    # ---- GHRN author configs ----
    for dataset, values in GHRN_AUTHOR.items():
        payload = dict(values)
        payload.update({"dataset": dataset, "track": "author", "model": "GHRN"})
        write(
            REPO_ROOT / "models" / "ghrn" / "configs" / "author" / f"ghrn_{dataset}.yml",
            payload,
            "GHRN author-mode configuration. Source: official repository "
            "https://github.com/blacksingular/GHRN main.py defaults.",
        )

    # ---- GHRN unified configs ----
    for dataset in DATASETS:
        payload = unified_common(dataset, seed=2, ratio="TR40")
        payload.update({
            "model": "GHRN",
            "hid_dim": 64,
            "order": 2,
            "lr": 0.01,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "del_ratio": 0.015,
            "refine_mode": "post_aggregation",
        })
        note = NON_NATIVE_NOTE["ghrn"].get(dataset)
        if note:
            payload["compatibility_note"] = note
        write(
            REPO_ROOT / "models" / "ghrn" / "configs" / "unified" / f"ghrn_{dataset}_TR40.yml",
            payload,
            "GHRN unified-track configuration under COMP8851 protocol v4.4. "
            "Architecture and model hyperparameters follow the official repository; "
            "split, seeds, stopping, selection, threshold and evaluator are protocol-fixed.",
        )


if __name__ == "__main__":
    main()
