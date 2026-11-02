"""Resumable experiment orchestration for the 3 x 6 COMP8851 Phase 2 benchmark matrix.

Phases (each idempotent and resumable; COMPLETE cells are never re-run):

    python scripts/run_matrix.py smoke   [--models ..] [--datasets ..]
    python scripts/run_matrix.py pilot   # one full TR40 seed-2 run per model on T-Social + FDCompCN
    python scripts/run_matrix.py tune    # <= 12 trials per pair, TR40 seed 2, --skip-test
    python scripts/run_matrix.py final   # tuned config, seeds 2/42/72, TR40 (add --ratios for the ladder)
    python scripts/run_matrix.py all     # tune then final
    python scripts/run_matrix.py status

Common options: --concurrency N (parallel processes), --gpus 0,1 (round-robin
CUDA_VISIBLE_DEVICES), --max-retries (FAILED / INCOMPLETE cells are retried
with --resume), --dry-run.

The ledger results/ledger.csv is rewritten atomically after every state
change. Status vocabulary: PENDING, RUNNING, COMPLETED, INCOMPLETE, FAILED,
UNSUPPORTED, BLOCKED_DATA.
SIGINT/SIGTERM stop new launches and forward SIGINT to running children, which
checkpoint and exit INCOMPLETE; re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from phase2 import configs as pconfigs  # noqa: E402
from phase2 import tuning_ext  # noqa: E402

RESULTS = ROOT / "results"
LEDGER = RESULTS / "ledger.csv"
META = RESULTS / "matrix_meta.json"
FIELDS = ["job_id", "phase", "model", "dataset", "ratio", "seed", "trial", "status", "attempts",
          "started_utc", "finished_utc", "duration_seconds", "result_dir", "failure_reason", "command"]
POLICY = yaml.safe_load((ROOT / "configs" / "matrix.yaml").read_text(encoding="utf-8"))
STOP = {"flag": False}
RUNNING: List[Dict[str, Any]] = []


def utc() -> str:
    """Return the current UTC timestamp formatted to ISO-8601 seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_ledger() -> List[Dict[str, Any]]:
    """Load all existing job rows from the persistent CSV ledger (`results/ledger.csv`)."""
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_ledger(rows: List[Dict[str, Any]]) -> None:
    """Atomically write the updated job ledger to disk using a temporary file and os.replace."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(LEDGER.parent), suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, LEDGER)


def read_meta() -> Dict[str, Any]:
    """Read accumulated wall-clock execution metadata from `results/matrix_meta.json`."""
    return json.loads(META.read_text(encoding="utf-8")) if META.exists() else {"accumulated_wall_seconds": 0.0}


def write_meta(meta: Dict[str, Any]) -> None:
    """Persist accumulated wall-clock execution metadata to `results/matrix_meta.json`."""
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def policy_value(model: str, dataset: str, key: str):
    """Resolve an execution policy setting (`max_minutes`, `tuning_max_minutes`) from `configs/matrix.yaml`."""
    value = POLICY["defaults"].get(key)
    value = POLICY.get("per_dataset", {}).get(dataset, {}).get(key, value)
    value = POLICY.get("per_model_dataset", {}).get(model, {}).get(dataset, {}).get(key, value)
    return value


def trials_for(model: str, dataset: str, default: int) -> int:
    """Return the number of hyperparameter tuning trials configured for a given (model, dataset) pair."""
    override = (POLICY.get("tuning_trials_override") or {}).get(model, {}).get(dataset)
    return int(override) if override else default


def data_available(dataset: str) -> bool:
    """Check whether the verified canonical `.npz` dataset cache exists on disk."""
    return (ROOT / "data" / "canonical" / f"{dataset}_canonical.npz").exists()


def summary_status(result_dir: Path) -> Dict[str, Any]:
    """Read and parse `summary.json` from a run directory, returning `status='MISSING'` if absent."""
    path = result_dir / "summary.json"
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "MISSING"}
    return payload


def tuned_hparams(model: str, dataset: str) -> Optional[Dict[str, Any]]:
    """Load the best validation-selected hyperparameters from `tuning_record.json` if tuning completed."""
    path = RESULTS / "tuning" / model / dataset / "tuning_record.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["selected_hyperparameters"]


# --------------------------------------------------------------------------
# Job construction
# --------------------------------------------------------------------------

def job(phase: str, model: str, dataset: str, ratio: str, seed: int, trial: str, root: Path,
        extra: List[str]) -> Dict[str, Any]:
    """Construct a structured ledger job entry and CLI command array for a single experiment cell."""
    result_dir = root / model / dataset / "unified" / ratio.lower() / f"seed_{seed}"
    cmd = [sys.executable, str(ROOT / "models" / model / "run_one.py"), "--dataset", dataset,
           "--ratio", ratio, "--seed", str(seed), "--results-root", str(root)] + extra
    job_id = f"{phase}:{model}:{dataset}:{ratio}:{seed}:{trial}"
    return {"job_id": job_id, "phase": phase, "model": model, "dataset": dataset, "ratio": ratio,
            "seed": str(seed), "trial": trial, "status": "PENDING", "attempts": "0",
            "result_dir": str(result_dir), "command": json.dumps(cmd)}


def plan_jobs(args) -> List[Dict[str, Any]]:
    """Build the ordered list of jobs for the requested execution phase (`smoke`, `pilot`, `tune`, or `final`)."""
    jobs: List[Dict[str, Any]] = []
    common = list(args.extra or [])
    if args.no_cuda:
        common.append("--no-cuda")
    elif args.require_cuda:
        common.append("--require-cuda")
    for model in args.models:
        for dataset in args.datasets:
            if args.phase == "smoke":
                extra = common + ["--smoke", "--smoke-epochs", str(POLICY["smoke"]["epochs"])]
                subset = POLICY["smoke"]["subset_nodes"].get(dataset)
                if subset and args.smoke_subsets:
                    extra += ["--smoke-subset-nodes", str(subset)]
                jobs.append(job("smoke", model, dataset, "TR40", 2, "-", RESULTS / "smoke", extra))
            elif args.phase == "pilot":
                extra = common + ["--max-minutes", str(policy_value(model, dataset, "max_minutes"))]
                jobs.append(job("pilot", model, dataset, "TR40", 2, "-", RESULTS / "pilot", extra))
            elif args.phase == "tune":
                cfg = pconfigs.load_model_config(model)
                for index, trial in enumerate(tuning_ext.build_trials(cfg, dataset, trials_for(model, dataset, args.trials))):
                    extra = common + ["--skip-test", "--run-mode", f"tuning-trial-{index:02d}",
                                      "--max-minutes", str(policy_value(model, dataset, "tuning_max_minutes"))]
                    extra += tuning_ext.overrides_for(trial)
                    root = RESULTS / "tuning" / model / dataset / f"trial_{index:02d}"
                    jobs.append(job("tune", model, dataset, "TR40", 2, f"{index:02d}", root, extra))
            elif args.phase == "final":
                hp = tuned_hparams(model, dataset)
                source = "tuned"
                if hp is None:
                    if not args.allow_untuned:
                        continue  # planned later, once tuning finished
                    cfg = pconfigs.load_model_config(model)
                    hp = {k: v for k, v in pconfigs.official_defaults(cfg, dataset).items() if not k.startswith("_")}
                    source = "official-untuned"
                for ratio in args.ratios:
                    for seed in args.seeds:
                        extra = common + ["--max-minutes", str(policy_value(model, dataset, "max_minutes")),
                                          "--run-mode", f"controlled-{source}"] + tuning_ext.overrides_for(hp)
                        jobs.append(job("final", model, dataset, ratio, seed, source, RESULTS / "raw", extra))
    return jobs


def merge(ledger: List[Dict[str, Any]], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge newly planned jobs into the existing ledger while preserving completed job history."""
    index = {row["job_id"]: row for row in ledger}
    for j in jobs:
        if j["job_id"] not in index:
            ledger.append(j)
            index[j["job_id"]] = j
        else:
            index[j["job_id"]]["command"] = j["command"]  # keep newest command (e.g. tuned hp)
    return ledger


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def estimate_seconds(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> float:
    """Estimate expected job runtime from median historical durations or policy max_minutes."""
    same = [float(r["duration_seconds"]) for r in rows
            if r["status"] in ("COMPLETED", "INCOMPLETE") and r["duration_seconds"]
            and r["model"] == row["model"] and r["dataset"] == row["dataset"] and r["phase"] == row["phase"]]
    if same:
        return statistics.median(same)
    return 60.0 * float(policy_value(row["model"], row["dataset"], "max_minutes"))


def execute(args, rows: List[Dict[str, Any]], selected: List[Dict[str, Any]]) -> None:
    """Launch and monitor queued matrix jobs up to the configured concurrency limit."""
    meta = read_meta()
    running = RUNNING
    gpus = [g for g in (args.gpus or "").split(",") if g != ""]
    next_gpu = 0
    session_start = time.time()
    base_wall = float(meta.get("accumulated_wall_seconds", 0.0))

    def finish(entry, code):
        """Record final status, duration, and failure reason into the persistent CSV ledger."""
        row = entry["row"]
        result_dir = Path(row["result_dir"])
        payload = summary_status(result_dir)
        status = payload.get("status", "MISSING")
        row["status"] = {"COMPLETE": "COMPLETED", "INCOMPLETE": "INCOMPLETE", "FAILED": "FAILED"}.get(status, "FAILED")
        if code == 0 and "already COMPLETE" in entry.get("note", ""):
            row["status"] = "COMPLETED"
        row["failure_reason"] = payload.get("failure_reason") or ("" if row["status"] == "COMPLETED" else f"exit {code}")
        row["finished_utc"] = utc()
        row["duration_seconds"] = f"{time.time() - entry['t0']:.1f}"
        entry["log"].close()
        write_ledger(rows)
        print(f"[{utc()}] {row['status']:10s} {row['job_id']}  ({row['duration_seconds']}s) {row['failure_reason']}")

    queue = [r for r in selected]
    while queue or running:
        # Poll active child processes for completion
        for entry in list(running):
            code = entry["proc"].poll()
            if code is not None:
                running.remove(entry)
                finish(entry, code)
        # Launch new jobs from the queue while below concurrency limit
        while queue and len(running) < args.concurrency and not STOP["flag"]:
            row = queue.pop(0)
            if row["status"] == "COMPLETED":
                continue
            if not data_available(row["dataset"]):
                row["status"], row["failure_reason"] = "BLOCKED_DATA", "canonical cache missing (scripts/prepare_data.py)"
                write_ledger(rows)
                continue
            if int(row["attempts"] or 0) > args.max_retries:
                continue
            cmd = json.loads(row["command"])
            if row.get("failure_reason") == "CUDA_OOM":
                # Memory-only recovery: smaller inference batches (recorded in the
                # config hash), fresh start (the failed attempt is archived).
                cmd += ["--eval-batch-size", "1024"]
            elif int(row["attempts"] or 0) > 0 or (Path(row["result_dir"]) / "checkpoints" / "last.pt").exists():
                cmd.append("--resume")
            env = dict(os.environ, DGLBACKEND="pytorch")
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = gpus[next_gpu % len(gpus)]
                next_gpu += 1
            log_dir = RESULTS / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = open(log_dir / (row["job_id"].replace(":", "__") + ".log"), "a", encoding="utf-8")
            log.write(f"\n===== {utc()} launch: {' '.join(cmd)}\n")
            log.flush()
            if args.dry_run:
                print("DRY-RUN", " ".join(cmd))
                log.close()
                continue
            row["status"], row["started_utc"] = "RUNNING", utc()
            row["attempts"] = str(int(row["attempts"] or 0) + 1)
            write_ledger(rows)
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            running.append({"row": row, "proc": proc, "t0": time.time(), "log": log})
            print(f"[{utc()}] RUNNING    {row['job_id']}  (attempt {row['attempts']})")
        if STOP["flag"] and not running:
            break
        time.sleep(2)
    meta["accumulated_wall_seconds"] = base_wall + (time.time() - session_start)
    write_meta(meta)


def write_tuning_records(args) -> None:
    for model in args.models:
        cfg = pconfigs.load_model_config(model)
        for dataset in args.datasets:
            base = RESULTS / "tuning" / model / dataset
            n_trials = trials_for(model, dataset, args.trials)
            trials = tuning_ext.build_trials(cfg, dataset, n_trials)
            records = []
            for index, hp in enumerate(trials):
                result = tuning_ext.read_trial(base / f"trial_{index:02d}" / model / dataset / "unified" / "tr40" / "seed_2" / "summary.json")
                records.append({"trial": index, "hyperparameters": hp, **result})
            if all(r["status"] == "MISSING" for r in records):
                continue
            defaults = trials[0]
            selection = tuning_ext.select(records, defaults)
            payload = {"model": model, "dataset": dataset, **tuning_ext.TUNING_CONSTANTS,
                       "trials_planned": len(trials), "trials": records, **selection,
                       "deviation": (f"tuning budget reduced to {n_trials} trial(s) by configs/matrix.yaml "
                                     "tuning_trials_override" if n_trials != args.trials else None),
                       "complete": all(r["status"] in ("COMPLETE", "FAILED", "INCOMPLETE") for r in records)}
            base.mkdir(parents=True, exist_ok=True)
            if payload["complete"]:
                (base / "tuning_record.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                print(f"[tune] {model}/{dataset}: selected trial {selection['selected_trial']} "
                      f"(val AUPRC {selection['selected_validation_auprc']})")
            else:
                (base / "tuning_progress.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def print_status(rows: List[Dict[str, Any]]) -> None:
    from collections import Counter

    by_phase: Dict[str, Counter] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], Counter())[r["status"]] += 1
    for phase, counts in by_phase.items():
        print(f"{phase:6s} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    meta = read_meta()
    print(f"accumulated matrix wall-clock: {meta.get('accumulated_wall_seconds', 0) / 3600:.2f} h")
    for r in rows:
        if r["status"] in ("FAILED", "INCOMPLETE", "BLOCKED_DATA", "SKIPPED_BUDGET"):
            print(f"  {r['status']:14s} {r['job_id']}  {r['failure_reason']}")


def main(argv=None) -> int:
    global RESULTS, LEDGER, META
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["smoke", "pilot", "tune", "final", "all", "status"])
    parser.add_argument("--models", nargs="*", default=POLICY["models"])
    parser.add_argument("--datasets", nargs="*", default=POLICY["datasets"])
    parser.add_argument("--seeds", nargs="*", type=int, default=POLICY["final_seeds"])
    parser.add_argument("--ratios", nargs="*", default=POLICY["final_ratios"])
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-hourly-usd", type=float, default=0.0)
    parser.add_argument("--budget-usd", type=float, default=0.0, help="0 = no cap")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--allow-untuned", action="store_true",
                        help="final runs with official defaults when no tuning record exists (recorded as such)")
    parser.add_argument("--smoke-subsets", action="store_true", help="use the documented T-Social smoke subset")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS),
                        help="root for ledger, smoke/pilot/tuning/raw runs (default results/)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed through to run_one.py")
    args = parser.parse_args(argv)
    RESULTS = Path(args.results_dir).resolve()
    LEDGER = RESULTS / "ledger.csv"
    META = RESULTS / "matrix_meta.json"

    def on_signal(signum, frame):  # noqa: ARG001
        STOP["flag"] = True
        print(f"[{utc()}] signal {signum}: no new launches; children will checkpoint (re-run to resume)")
        for entry in RUNNING:
            try:
                entry["proc"].send_signal(signal.CTRL_C_EVENT if os.name == "nt" else signal.SIGINT)
            except Exception:  # noqa: BLE001
                pass
    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    rows = read_ledger()
    if args.phase == "status":
        print_status(rows)
        return 0
    phases = ["tune", "final"] if args.phase == "all" else [args.phase]
    for phase in phases:
        args.phase = phase
        jobs = plan_jobs(args)
        rows = merge(rows, jobs)
        write_ledger(rows)
        wanted = {j["job_id"] for j in jobs}
        selected = [r for r in rows if r["job_id"] in wanted and r["status"] != "COMPLETED"]
        # Cheapest cells first, so a spending cap leaves the most completed pairs:
        # T-Social last, SpaceGNN (46k optimiser steps/epoch on T-Social) at the very end.
        heavy = {"tsocial": 3, "tfinance": 1, "yelpchi": 1}
        selected.sort(key=lambda r: (heavy.get(r["dataset"], 0) + (2 if r["model"] == "spacegnn" else 0)
                                     + (4 if r["dataset"] == "tsocial" and r["model"] == "spacegnn" else 0)))
        print(f"[{utc()}] phase {phase}: {len(jobs)} cells planned, {len(selected)} to run")
        execute(args, rows, selected)
        if phase == "tune":
            write_tuning_records(args)
        if STOP["flag"]:
            break
    print_status(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
