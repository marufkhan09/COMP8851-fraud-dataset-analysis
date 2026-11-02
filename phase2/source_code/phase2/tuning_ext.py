"""Protocol tuning with per-dataset official defaults.

Identical rules to ``shared.comp8851.tuning`` (<= 12 trials, TR40, train seed
2, validation AUPRC, ``--skip-test`` on every trial, trial 0 = the published
configuration, deterministic sampling with the split seed) - extended only so
that trial 0 is the *per-dataset* official configuration, which all three new
models publish, rather than one global default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from shared.comp8851.protocol import MAX_TUNING_TRIALS, SPLIT_SEED, TUNING_RATIO, TUNING_SEED

from phase2 import configs as pconfigs


def build_trials(cfg: Dict[str, Any], dataset: str, budget: int = MAX_TUNING_TRIALS,
                 seed: int = SPLIT_SEED) -> List[Dict[str, Any]]:
    """Generate up to `budget` deterministic hyperparameter trial configurations.

    Trial 0 is always the dataset-specific official configuration; subsequent trials
    are sampled deterministically without replacement from the declared search space.
    """
    if not 1 <= budget <= MAX_TUNING_TRIALS:
        raise ValueError(f"budget must be in 1..{MAX_TUNING_TRIALS}")
    defaults = {k: v for k, v in pconfigs.official_defaults(cfg, dataset).items() if not k.startswith("_")}
    space = cfg.get("search_space", {})
    trials = [dict(defaults)]
    seen = {json.dumps(trials[0], sort_keys=True, default=str)}
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(trials) < budget and attempts < budget * 200:
        attempts += 1
        candidate = dict(defaults)
        for name, values in space.items():
            candidate[name] = values[int(rng.integers(len(values)))]
        key = json.dumps(candidate, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            trials.append(candidate)
    return trials


def overrides_for(config: Dict[str, Any]) -> List[str]:
    """Format a hyperparameter dictionary into CLI ``--hp KEY=VALUE`` arguments."""
    args = []
    for key, value in sorted(config.items()):
        args += ["--hp", f"{key}={json.dumps(value)}"]
    return args


def read_trial(summary_path: Path) -> Dict[str, Any]:
    """Read the completion status and best validation AUPRC from a trial's summary.json."""
    if not summary_path.exists():
        return {"status": "MISSING"}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    value = (payload.get("best_validation_metrics") or {}).get("auprc")
    return {"status": payload.get("status"), "validation_auprc": value,
            "failure_reason": payload.get("failure_reason", ""),
            "duration_seconds": (payload.get("timing") or {}).get("total_wall_seconds")}


def select(records: List[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Select the optimal hyperparameter configuration based strictly on highest validation AUPRC.

    Trials stopped by the wall-clock cap (INCOMPLETE) still have a validation-selected
    checkpoint and compete, while FAILED trials are excluded. Test labels are never accessed.
    """
    ok = [r for r in records if r.get("status") in ("COMPLETE", "INCOMPLETE") and r.get("validation_auprc") is not None]
    if ok:
        # Tie-break in favor of earlier trial indices (preferring Trial 0 official defaults on exact ties)
        best = max(ok, key=lambda r: (r["validation_auprc"], -r["trial"]))
        return {"selected_trial": best["trial"], "selected_hyperparameters": best["hyperparameters"],
                "selected_validation_auprc": best["validation_auprc"], "selected_trial_status": best["status"],
                "trials_time_capped": sum(1 for r in records if r.get("status") == "INCOMPLETE"),
                "selection_reason": "highest validation AUPRC"}
    return {"selected_trial": None, "selected_hyperparameters": defaults, "selected_validation_auprc": None,
            "selection_reason": "all trials failed; published configuration used"}


TUNING_CONSTANTS = {"ratio": TUNING_RATIO, "seed": TUNING_SEED, "max_trials": MAX_TUNING_TRIALS,
                    "selection_metric": "validation AUPRC", "test_used": False}
