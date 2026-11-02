"""Explicit model configuration: official defaults, budgets and search spaces.

Every hyperparameter a run uses comes from ``configs/<slug>.yaml`` (with its
documented source) or from an explicit ``--hp KEY=VALUE`` override recorded in
``run_config.yml``. There are no hidden constants in the adapters.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from shared.comp8851 import protocol

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = BUNDLE_ROOT / "configs"


def load_model_config(slug: str, path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML configuration file for the given model slug (e.g., 'sagad', 'spacegnn', 'croc')."""
    cfg_path = Path(path) if path else CONFIG_DIR / f"{slug}.yaml"
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_path"] = str(cfg_path)
    return cfg


def official_defaults(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Merge the model's global default block with the dataset-specific official defaults and provenance string."""
    block = cfg["official_defaults"]
    merged = copy.deepcopy(block.get("default", {}))
    entry = copy.deepcopy(block.get(dataset, {}))
    source = entry.pop("_source", None) or block.get("default", {}).get("_source", "default block")
    merged.pop("_source", None)
    merged.update(entry)
    merged["_source"] = source
    return merged


def parse_override(text: str):
    """Parse a single CLI hyperparameter override string ('KEY=VALUE') using YAML type inference."""
    if "=" not in text:
        raise ValueError(f"--hp expects KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def resolve_hparams(cfg: Dict[str, Any], dataset: str, overrides: List[str]) -> Dict[str, Any]:
    """Resolve the final hyperparameter dictionary by applying validated CLI overrides on top of official defaults."""
    hp = official_defaults(cfg, dataset)
    # Only allow keys explicitly declared in official_defaults, search_space, or extra_keys
    allowed = set(hp) | set(cfg.get("search_space", {})) | set(cfg.get("extra_keys", []))
    applied = {}
    for text in overrides or []:
        key, value = parse_override(text)
        if key not in allowed:
            raise KeyError(f"unknown hyperparameter {key!r}; allowed: {sorted(k for k in allowed if not k.startswith('_'))}")
        hp[key] = value
        applied[key] = value
    if applied:
        hp["_source"] = f"{hp['_source']} + overrides {json.dumps(applied, sort_keys=True, default=str)}"
    return hp


def resolve_budget(cfg: Dict[str, Any], dataset: str, mode: str,
                   epochs: Optional[int] = None, patience: Optional[int] = None) -> Dict[str, Any]:
    """Resolve training epoch and early-stopping patience rules for 'protocol' (100/20) or 'official' mode."""
    if mode == "protocol":
        budget = {"epochs": protocol.MAX_EPOCHS, "patience": protocol.EARLY_STOPPING_PATIENCE,
                  "source": "protocol vast-v4.4 unified stopping rule (MAX_EPOCHS, EARLY_STOPPING_PATIENCE)"}
    else:
        block = cfg["budget"]["official"]
        budget = dict(block.get("default", {}))
        budget.update(block.get(dataset, {}))
        budget["source"] = f"official budget ({budget.pop('_source', 'author code/paper')})"
    if epochs is not None:
        budget["epochs"] = int(epochs)
        budget["source"] += f"; epochs overridden to {epochs}"
    if patience is not None:
        budget["patience"] = int(patience)
        budget["source"] += f"; patience overridden to {patience}"
    return budget
