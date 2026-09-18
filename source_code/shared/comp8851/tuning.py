"""Hyperparameter tuning under the COMP8851 controlled protocol.

Rules this module enforces, none of them optional:

* **At most 12 trials** per model-dataset pair, and the *same* budget for every
  pair. Neither model may receive a larger search than the other.
* Tuning runs on **TR40 with train seed 2 only**.
* Trials are selected on **validation AUPRC**. Nothing else decides.
* **The test set is never touched during tuning.** Every trial is launched with
  ``--skip-test``, so no test evaluation can occur even by accident.
* Every trial is recorded: number, hyperparameters, seed, dataset, validation
  AUPRC, duration, status. A failed trial stays in the record as failed.

The search space comes from the ranges the run guide permits (learning rate
1e-4 to 1e-2 on a log scale, weight decay {0, 1e-5, 1e-4, 1e-3}, dropout
{0, 0.3, 0.5}, hidden dimension {32, 64, 128}), restricted to the parameters
each published model actually exposes. Trial 0 is always the paper's own
configuration, so the search can never do worse than the published defaults.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .protocol import MAX_TUNING_TRIALS, SPLIT_SEED, TUNING_RATIO, TUNING_SEED

#: Paper/repository defaults. Trial 0 always uses these.
AUTHOR_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "CARE-GNN": {"lr": 0.01, "emb_size": 64, "lambda_2": 1e-3, "step_size": 0.02},
    "GHRN": {"lr": 0.01, "hid_dim": 64, "weight_decay": 0.0, "dropout": 0.0,
             "del_ratio": 0.015, "order": 2},
}

#: Candidate values, drawn only from the permitted ranges.
SEARCH_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "CARE-GNN": {
        "lr": [1e-2, 5e-3, 1e-3, 5e-4],
        "emb_size": [32, 64, 128],
        "lambda_2": [0.0, 1e-5, 1e-4, 1e-3],
    },
    "GHRN": {
        "lr": [1e-2, 5e-3, 1e-3, 5e-4],
        "hid_dim": [32, 64, 128],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
        "dropout": [0.0, 0.3, 0.5],
        "del_ratio": [0.005, 0.015, 0.03],
    },
}

#: Command-line flag for each tunable parameter.
FLAGS: Dict[str, str] = {
    "lr": "--lr",
    "emb_size": "--emb-size",
    "hid_dim": "--hid-dim",
    "lambda_1": "--lambda-1",
    "lambda_2": "--lambda-2",
    "step_size": "--step-size",
    "under_sample": "--under-sample",
    "batch_size": "--batch-size",
    "weight_decay": "--weight-decay",
    "dropout": "--dropout",
    "del_ratio": "--del-ratio",
    "order": "--order",
    "refine_mode": "--refine-mode",
}


def assert_flags_cover(model: str) -> None:
    """Fail loudly if a tuned parameter has no command-line flag.

    Without this, an unmapped parameter is silently dropped from the command and
    the trial quietly runs with the runner's default instead of the value the
    search chose, which would make the recorded configuration a lie.
    """
    names = set(AUTHOR_DEFAULTS.get(model, {})) | set(SEARCH_SPACE.get(model, {}))
    missing = sorted(name for name in names if name not in FLAGS)
    if missing:
        raise KeyError(
            f"{model}: tuned parameter(s) {missing} have no entry in FLAGS, so they "
            "would be dropped from the trial command. Add them before tuning."
        )


def build_trials(model: str, budget: int, seed: int = SPLIT_SEED) -> List[Dict[str, Any]]:
    """Generate at most ``budget`` distinct configurations, deterministically.

    Trial 0 is the published configuration. The rest are drawn from the search
    space with a fixed RNG, so the same budget always yields the same trials and
    the search is reproducible from the manifest alone.
    """
    if budget < 1:
        raise ValueError("Tuning budget must be at least 1.")
    if budget > MAX_TUNING_TRIALS:
        raise ValueError(
            f"Tuning budget {budget} exceeds the protocol maximum of {MAX_TUNING_TRIALS}."
        )
    if model not in SEARCH_SPACE:
        raise KeyError(f"No search space defined for {model!r}.")
    assert_flags_cover(model)

    space = SEARCH_SPACE[model]
    defaults = AUTHOR_DEFAULTS[model]

    trials: List[Dict[str, Any]] = [dict(defaults)]
    seen = {json.dumps(trials[0], sort_keys=True)}
    rng = np.random.default_rng(seed)

    # Bounded sampling; duplicates are rejected so the budget buys distinct
    # configurations rather than repeats.
    attempts = 0
    while len(trials) < budget and attempts < budget * 200:
        attempts += 1
        candidate = dict(defaults)
        for name, values in space.items():
            candidate[name] = values[int(rng.integers(len(values)))]
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            trials.append(candidate)

    return trials


def trial_command(python: str, script: str, model: str, dataset: str,
                  data_path: str, results_root: str, config: Dict[str, Any],
                  trial_index: int, epochs: int, patience: int,
                  max_minutes: float, no_cuda: bool,
                  extra: Optional[List[str]] = None) -> List[str]:
    """Build the command for one tuning trial.

    ``--skip-test`` is not optional here: it is what makes test isolation during
    tuning a property of the command rather than a promise.
    """
    command = [
        python, script,
        "--dataset", dataset,
        "--data-path", str(data_path),
        "--ratio", TUNING_RATIO,
        "--seed", str(TUNING_SEED),
        "--track", "unified",
        "--results-root", str(results_root),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--max-minutes", str(max_minutes),
        "--run-mode", f"tuning-trial-{trial_index:02d}",
        "--skip-test",
        "--no-strict-stats",
    ]
    for name, value in config.items():
        flag = FLAGS.get(name)
        if flag:
            command += [flag, str(value)]
    if no_cuda:
        command.append("--no-cuda")
    if extra:
        command += extra
    return command


def read_validation_auprc(results_root: Path, model_slug: str, dataset: str,
                          ratio: str, seed: int) -> Optional[float]:
    """Read the best validation AUPRC a trial achieved, or None if it failed."""
    summary = (Path(results_root) / model_slug / dataset / "unified" /
               ratio.lower() / f"seed_{seed}" / "summary.json")
    if not summary.exists():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "COMPLETE":
        return None
    metrics = payload.get("best_validation_metrics") or {}
    value = metrics.get("auprc")
    if value is None:
        value = payload.get("best_selection_value")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def run_tuning(model: str, model_slug: str, script: str, dataset: str,
               data_path: str, tuning_root: Path, budget: int,
               epochs: int, patience: int, max_minutes: float,
               no_cuda: bool, python: str = sys.executable,
               cwd: Optional[Path] = None, extra: Optional[List[str]] = None,
               verbose: bool = True) -> Dict[str, Any]:
    """Run the tuning search for one model-dataset pair.

    Returns a record containing every trial and the selected configuration.
    Trials that fail are kept in the record with their status, because a search
    that silently dropped its failures would misrepresent the budget actually
    spent.
    """
    tuning_root = Path(tuning_root)
    trials = build_trials(model, budget)
    records: List[Dict[str, Any]] = []

    if verbose:
        print(f"Tuning {model} on {dataset}: {len(trials)} trial(s), "
              f"selection metric validation AUPRC, test set untouched")

    for index, config in enumerate(trials):
        # Each trial writes into its own directory so trials never overwrite
        # one another's evidence.
        trial_results = tuning_root / f"trial_{index:02d}"

        # Resume: a trial that already completed is reused rather than repeated.
        # Long searches routinely outlive a hosted session's time limit.
        existing = read_validation_auprc(trial_results, model_slug, dataset,
                                         TUNING_RATIO, TUNING_SEED)
        if existing is not None:
            records.append({
                "trial": index, "model": model, "dataset": dataset,
                "ratio": TUNING_RATIO, "seed": TUNING_SEED,
                "hyperparameters": dict(config), "validation_auprc": existing,
                "training_duration_seconds": 0.0, "status": "COMPLETE",
                "error": "", "results_path": str(trial_results), "resumed": True,
            })
            if verbose:
                print(f"  trial {index:2d}/{len(trials) - 1}  RESUMED   "
                      f"val AUPRC {existing:.6f}  (already complete)")
            continue

        command = trial_command(python, script, model, dataset, data_path,
                                str(trial_results), config, index, epochs,
                                patience, max_minutes, no_cuda, extra)

        started = time.time()
        try:
            completed = subprocess.run(command, cwd=str(cwd) if cwd else None,
                                       capture_output=True, text=True)
            status = "COMPLETE" if completed.returncode == 0 else "FAILED"
            error = "" if status == "COMPLETE" else (
                completed.stdout.strip().splitlines()[-1][:300]
                if completed.stdout.strip() else f"exit {completed.returncode}")
        except Exception as exc:  # noqa: BLE001
            status, error = "FAILED", f"{type(exc).__name__}: {exc}"
        duration = time.time() - started

        auprc = read_validation_auprc(trial_results, model_slug, dataset,
                                      TUNING_RATIO, TUNING_SEED)
        if auprc is None and status == "COMPLETE":
            status, error = "FAILED", "no validation AUPRC recorded"

        record = {
            "trial": index,
            "model": model,
            "dataset": dataset,
            "ratio": TUNING_RATIO,
            "seed": TUNING_SEED,
            "hyperparameters": dict(config),
            "validation_auprc": auprc,
            "training_duration_seconds": duration,
            "status": status,
            "error": error,
            "results_path": str(trial_results),
        }
        records.append(record)

        if verbose:
            shown = ", ".join(f"{k}={v}" for k, v in config.items())
            score = f"{auprc:.6f}" if auprc is not None else "n/a"
            print(f"  trial {index:2d}/{len(trials) - 1}  {status:8s}  "
                  f"val AUPRC {score}  {duration / 60:5.1f} min  [{shown}]")

    successful = [r for r in records if r["status"] == "COMPLETE"
                  and r["validation_auprc"] is not None]
    if successful:
        best = max(successful, key=lambda r: r["validation_auprc"])
        selected, selected_trial = dict(best["hyperparameters"]), best["trial"]
        selected_auprc = best["validation_auprc"]
        reason = "highest validation AUPRC"
    else:
        # Nothing succeeded. Fall back to the published configuration and say so
        # rather than pretending a search happened.
        selected = dict(AUTHOR_DEFAULTS[model])
        selected_trial, selected_auprc = None, None
        reason = "all trials failed; falling back to the published configuration"

    if verbose:
        print(f"  selected: trial {selected_trial} ({reason})")

    return {
        "model": model,
        "dataset": dataset,
        "budget": budget,
        "protocol_max_trials": MAX_TUNING_TRIALS,
        "tuning_ratio": TUNING_RATIO,
        "tuning_seed": TUNING_SEED,
        "selection_metric": "validation AUPRC",
        "test_used_during_tuning": False,
        "trials": records,
        "trials_attempted": len(records),
        "trials_successful": len(successful),
        "selected_trial": selected_trial,
        "selected_hyperparameters": selected,
        "selected_validation_auprc": selected_auprc,
        "selection_reason": reason,
    }


def write_tuning_record(record: Dict[str, Any], metrics_dir: Path) -> Path:
    """Persist a tuning record as JSON plus a flat CSV of the trial table."""
    import csv

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{record['model'].lower().replace('-', '')}_{record['dataset']}"

    json_path = metrics_dir / f"tuning_{slug}.json"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str),
                         encoding="utf-8")

    csv_path = metrics_dir / f"tuning_{slug}.csv"
    fields = ["trial", "model", "dataset", "ratio", "seed", "validation_auprc",
              "training_duration_seconds", "status", "error", "hyperparameters"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for trial in record["trials"]:
            row = dict(trial)
            row["hyperparameters"] = json.dumps(trial["hyperparameters"], sort_keys=True)
            writer.writerow({field: row.get(field, "") for field in fields})

    return json_path


def final_command(python: str, script: str, dataset: str, data_path: str,
                  results_root: str, config: Dict[str, Any], seed: int,
                  ratio: str, epochs: int, patience: int, max_minutes: float,
                  no_cuda: bool, extra: Optional[List[str]] = None) -> List[str]:
    """Build the command for a final run: frozen config, test evaluated once."""
    command = [
        python, script,
        "--dataset", dataset,
        "--data-path", str(data_path),
        "--ratio", ratio,
        "--seed", str(seed),
        "--track", "unified",
        "--results-root", str(results_root),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--max-minutes", str(max_minutes),
        "--run-mode", "controlled",
        "--no-strict-stats",
    ]
    for name, value in config.items():
        flag = FLAGS.get(name)
        if flag:
            command += [flag, str(value)]
    if no_cuda:
        command.append("--no-cuda")
    if extra:
        command += extra
    return command
