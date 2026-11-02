"""Resumable run queue for the COMP8851 benchmark matrix.

Drives ``models/<model>/scripts/run_one.py`` across the full matrix of
model x dataset x ratio x seed, writing status atomically after every cell so an
interrupted session resumes from the next incomplete row rather than starting
over (master plan v4.4, section 8.2).

The queue lives at ``shared/run_matrix.csv`` with the required fields:
``run_id, model, dataset, track, ratio, seed, config_id, status, host_id,
result_path``.

Typical use on the benchmark host, inside tmux::

    # 1. Build the queue for the two models this owner is responsible for
    python shared/comp8851/run_matrix.py plan \
        --models CARE-GNN GHRN --ratios TR40 --seeds 2 42 72

    # 2. Two-epoch compatibility smoke over all pairs, before any full run
    python shared/comp8851/run_matrix.py smoke --models CARE-GNN GHRN

    # 3. Execute, resuming automatically if it stops
    python shared/comp8851/run_matrix.py run --host-id a6000-ref-v1

    # 4. See what is left
    python shared/comp8851/run_matrix.py status

A row is only marked COMPLETE when the child process exits zero *and* its
summary.json reports COMPLETE. Anything else is FAILED with the error retained,
because a silently skipped cell is indistinguishable from a passing one in the
final table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.comp8851 import protocol  # noqa: E402

QUEUE_FIELDS = [
    "run_id", "model", "dataset", "track", "ratio", "seed", "config_id",
    "status", "host_id", "result_path", "started_utc", "finished_utc",
    "duration_seconds", "error",
]

MODEL_SCRIPTS = {
    "CARE-GNN": REPO_ROOT / "models" / "care-gnn" / "scripts" / "run_one.py",
    "GHRN": REPO_ROOT / "models" / "ghrn" / "scripts" / "run_one.py",
}

DEFAULT_QUEUE = REPO_ROOT / "shared" / "run_matrix.csv"


# --------------------------------------------------------------------------
# Queue persistence
# --------------------------------------------------------------------------

def read_queue(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_queue(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Atomic write: a crash mid-write must not corrupt the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", delete=False, dir=str(path.parent)
    )
    try:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in QUEUE_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def row_key(row: Dict[str, Any]) -> tuple:
    return (row["model"], row["dataset"], row["track"], row["ratio"], str(row["seed"]))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def command_plan(args) -> int:
    queue_path = Path(args.queue)
    existing = {row_key(row): row for row in read_queue(queue_path)}

    rows: List[Dict[str, Any]] = []
    for model in args.models:
        if model not in MODEL_SCRIPTS:
            raise SystemExit(f"Unknown model {model!r}; known: {sorted(MODEL_SCRIPTS)}")
        for dataset in args.datasets:
            for ratio in args.ratios:
                for seed in args.seeds:
                    row = {
                        "run_id": "",
                        "model": model,
                        "dataset": dataset,
                        "track": args.track,
                        "ratio": ratio,
                        "seed": seed,
                        "config_id": f"{model.lower()}_{dataset}_{ratio}",
                        "status": "PENDING",
                        "host_id": "",
                        "result_path": "",
                        "started_utc": "",
                        "finished_utc": "",
                        "duration_seconds": "",
                        "error": "",
                    }
                    key = row_key(row)
                    # Never overwrite a finished cell when re-planning.
                    rows.append(existing.get(key, row) if key in existing else row)

    for key, row in existing.items():
        if key not in {row_key(r) for r in rows}:
            rows.append(row)

    write_queue(queue_path, rows)
    pending = sum(1 for row in rows if row["status"] == "PENDING")
    print(f"[plan] queue at {queue_path}")
    print(f"[plan] {len(rows)} row(s) total, {pending} pending")
    return 0


def _dataset_path(args, dataset: str) -> str:
    override = getattr(args, f"path_{dataset}", None)
    if override:
        return override
    return str(Path(args.data_root) / dataset)


def _build_command(row: Dict[str, Any], args, extra: List[str]) -> List[str]:
    script = MODEL_SCRIPTS[row["model"]]
    command = [
        sys.executable, str(script),
        "--dataset", row["dataset"],
        "--data-path", _dataset_path(args, row["dataset"]),
        "--ratio", row["ratio"],
        "--seed", str(row["seed"]),
        "--track", row["track"],
        "--results-root", args.results_root,
    ]
    if args.host_id:
        command += ["--hardware-profile-id", args.host_id]
    if args.container_digest:
        command += ["--container-digest", args.container_digest]
    if args.no_cuda:
        command.append("--no-cuda")
    elif args.require_cuda:
        command.append("--require-cuda")
    if args.max_minutes:
        command += ["--max-minutes", str(args.max_minutes)]
    return command + extra


def _execute(row: Dict[str, Any], args, extra: List[str]) -> Dict[str, Any]:
    command = _build_command(row, args, extra)
    print("-" * 78)
    print(f"[run] {row['model']} | {row['dataset']} | {row['ratio']} | seed {row['seed']}")
    print(f"[run] {' '.join(command)}")

    started = time.time()
    row["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    row["status"] = "RUNNING"
    row["host_id"] = args.host_id or ""

    try:
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        returncode = completed.returncode
        error = "" if returncode == 0 else f"exit code {returncode}"
    except KeyboardInterrupt:
        row["status"] = "PENDING"
        row["error"] = "interrupted"
        raise
    except Exception as exc:  # noqa: BLE001
        returncode, error = -1, f"{type(exc).__name__}: {exc}"

    duration = time.time() - started
    row["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row["duration_seconds"] = f"{duration:.3f}"

    run_dir = protocol.result_dir(args.results_root, row["model"].lower(), row["dataset"],
                                  row["track"], row["ratio"], int(row["seed"]))
    row["result_path"] = str(run_dir)

    # Trust the written record over the exit code alone.
    summary_path = run_dir / "summary.json"
    summary_status = None
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_status = payload.get("status")
            row["run_id"] = payload.get("run_id", "")
            if payload.get("error"):
                error = error or payload["error"].get("message", "")
        except (OSError, json.JSONDecodeError) as exc:
            error = error or f"unreadable summary.json: {exc}"

    if returncode == 0 and summary_status == "COMPLETE":
        row["status"] = "COMPLETE"
        row["error"] = ""
    else:
        row["status"] = "FAILED"
        row["error"] = error or f"summary status {summary_status}"
        print(f"[run] FAILED: {row['error']}")

    return row


def command_run(args) -> int:
    queue_path = Path(args.queue)
    rows = read_queue(queue_path)
    if not rows:
        print(f"[run] queue is empty; run 'plan' first: {queue_path}")
        return 1

    targets = [row for row in rows
               if row["status"] in ("PENDING", "FAILED" if args.retry_failed else "PENDING")]
    if args.only_model:
        targets = [row for row in targets if row["model"] in args.only_model]
    if args.only_dataset:
        targets = [row for row in targets if row["dataset"] in args.only_dataset]
    if args.limit:
        targets = targets[:args.limit]

    print(f"[run] {len(targets)} cell(s) to execute of {len(rows)} in the queue")
    extra = args.extra or []

    for index, row in enumerate(targets, start=1):
        print(f"\n[run] === cell {index}/{len(targets)} ===")
        try:
            _execute(row, args, extra)
        except KeyboardInterrupt:
            write_queue(queue_path, rows)
            print("\n[run] interrupted; queue saved, resume with the same command")
            return 130
        finally:
            # Persist after every cell so progress survives a crash.
            write_queue(queue_path, rows)

    completed = sum(1 for row in rows if row["status"] == "COMPLETE")
    failed = sum(1 for row in rows if row["status"] == "FAILED")
    print(f"\n[run] complete {completed} | failed {failed} | total {len(rows)}")
    return 0 if failed == 0 else 1


def command_smoke(args) -> int:
    """Two-epoch feasibility attempt on every model-dataset pair."""
    print("[smoke] two-epoch compatibility attempt over all pairs")
    print("[smoke] smoke metrics are feasibility evidence only, never benchmark results")
    failures: List[str] = []

    for model in args.models:
        for dataset in args.datasets:
            row = {
                "model": model, "dataset": dataset, "track": "unified",
                "ratio": "TR40", "seed": 2, "status": "PENDING",
                "run_id": "", "config_id": "", "host_id": "", "result_path": "",
                "started_utc": "", "finished_utc": "", "duration_seconds": "", "error": "",
            }
            outcome = _execute(row, args, ["--smoke"] + (args.extra or []))
            if outcome["status"] != "COMPLETE":
                failures.append(f"{model} x {dataset}: {outcome['error']}")

    print("\n" + "=" * 78)
    print("[smoke] compatibility summary")
    total = len(args.models) * len(args.datasets)
    print(f"[smoke] attempted {total}, failed {len(failures)}")
    for failure in failures:
        print(f"  FAILED  {failure}")
    if failures:
        print("[smoke] a failure here is a feasibility finding; record its status "
              "(FAILED_TECHNICAL or NOT_ELIGIBLE) with the log, do not silently drop the pair")
    return 0


def command_status(args) -> int:
    rows = read_queue(Path(args.queue))
    if not rows:
        print("[status] queue is empty")
        return 0

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[status] {len(rows)} row(s): " +
          ", ".join(f"{status} {count}" for status, count in sorted(counts.items())))

    failed = [row for row in rows if row["status"] == "FAILED"]
    if failed:
        print("\n[status] failed cells:")
        for row in failed:
            print(f"  {row['model']:10s} {row['dataset']:10s} {row['ratio']:5s} "
                  f"seed {row['seed']:3s}  {row['error']}")

    pending = [row for row in rows if row["status"] == "PENDING"]
    if pending:
        print(f"\n[status] {len(pending)} pending; next up:")
        for row in pending[:10]:
            print(f"  {row['model']:10s} {row['dataset']:10s} {row['ratio']:5s} seed {row['seed']}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"),
                        help="Directory holding one subdirectory per dataset")
    for dataset in protocol.DATASETS:
        parser.add_argument(f"--path-{dataset}", default=None,
                            help=f"Explicit path for {dataset}, overriding --data-root")
    parser.add_argument("--host-id", default=None, help="Frozen hardware profile ID")
    parser.add_argument("--container-digest", default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Extra arguments passed through to run_one.py")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COMP8851 resumable run matrix")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Create or extend the run queue")
    plan.add_argument("--models", nargs="+", default=["CARE-GNN", "GHRN"])
    plan.add_argument("--datasets", nargs="+", default=list(protocol.DATASETS))
    plan.add_argument("--ratios", nargs="+", default=["TR40"], choices=list(protocol.TRAIN_RATIOS))
    plan.add_argument("--seeds", nargs="+", type=int, default=list(protocol.TRAIN_SEEDS))
    plan.add_argument("--track", default="unified", choices=list(protocol.TRACKS))
    plan.add_argument("--queue", default=str(DEFAULT_QUEUE))
    plan.set_defaults(func=command_plan)

    run = sub.add_parser("run", help="Execute pending cells, resuming automatically")
    add_common(run)
    run.add_argument("--only-model", nargs="*", default=None)
    run.add_argument("--only-dataset", nargs="*", default=None)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=command_run)

    smoke = sub.add_parser("smoke", help="Two-epoch compatibility attempt on every pair")
    add_common(smoke)
    smoke.add_argument("--models", nargs="+", default=["CARE-GNN", "GHRN"])
    smoke.add_argument("--datasets", nargs="+", default=list(protocol.DATASETS))
    smoke.set_defaults(func=command_smoke)

    status = sub.add_parser("status", help="Show queue status")
    status.add_argument("--queue", default=str(DEFAULT_QUEUE))
    status.set_defaults(func=command_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
