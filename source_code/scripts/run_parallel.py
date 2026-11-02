"""Run the benchmark across every available GPU, to finish in hours not days.

    tmux new -s comp8851
    python3 scripts/run_parallel.py

What this does, and what it deliberately does not do
----------------------------------------------------
Each unit of work is one (model, dataset) pair: its tuning search plus its three
final seeds. Those pairs are completely independent, so they are handed out to
workers, one worker per GPU, each pinned with ``CUDA_VISIBLE_DEVICES``.

This is **parallelism across independent runs**, not multi-GPU training of a
single model. Every individual run still sees exactly one GPU and is
bit-identical to running it alone on that card. Data-parallel training would
change the computation and break the controlled specification; this does not.

The one honest caveat: workers share the host's CPU, RAM and disk. Predictive
metrics are unaffected, but wall-clock timings can be slightly inflated by
contention. If per-epoch timing is a headline number in your report, either run
the timing set on an idle single-GPU box, or state that timings were collected
under N-way parallelism. The manifest records the GPU; this script records the
worker count in ``results/parallel_manifest.json``.

Thread caps are divided among workers so the box is not oversubscribed.

Sharding
--------
Default is by (model, dataset), which gives up to 12 independent units and
therefore scales to 12 GPUs. With fewer GPUs, units are queued and picked up as
workers free up, longest-first so the big datasets start early.

Options:
    --gpus 4                use this many GPUs (default: all detected)
    --models / --datasets   restrict the matrix
    --trials 12             tuning budget per pair
    --epochs 100            maximum epochs
    --reduced               smaller search and epoch budget
    --dry-run               print the plan and exit
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODELS = {"CARE-GNN": "care-gnn", "GHRN": "ghrn"}
# Largest first: starting the heavy graphs early keeps the tail short.
DATASETS_BY_COST = ("tsocial", "elliptic", "yelpchi", "tfinance", "amazon", "fdcompcn")

print_lock = threading.Lock()


def say(*parts) -> None:
    with print_lock:
        print(*parts, flush=True)


def detect_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def no_gpu_reason() -> str:
    """Why no GPU was found: a missing environment, or genuinely no device.

    Launching from a shell that never activated the project's virtualenv looks
    identical to having no GPU, because torch simply is not importable. Saying
    which one it is turns a confusing stop into a one-line fix.
    """
    try:
        import torch
    except ImportError:
        return (f"torch is not importable from {sys.executable}.\n"
                "  This usually means the shell did not activate the project "
                "environment\n  (a fresh tmux pane does not inherit it). Either "
                "activate it:\n"
                "      source /venv/main/bin/activate\n"
                "  or name the interpreter directly:\n"
                "      /venv/main/bin/python scripts/run_parallel.py ...")
    if not torch.cuda.is_available():
        return (f"torch {torch.__version__} is installed but reports no CUDA "
                "device.\n  Check `nvidia-smi -L` and CUDA_VISIBLE_DEVICES "
                f"(currently {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')!r}).")
    return "torch reports CUDA available but zero devices."


def gpu_names() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def find_dataset(name: str, data_root: Path) -> Path | None:
    for candidate in (data_root / "canonical" / f"{name}_canonical.npz", data_root / name):
        if candidate.exists():
            return candidate
    return None


def human(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60}m" if hours else f"{rest // 60}m"


def worker(gpu: int, work: "queue.Queue[tuple[str, str]]", args,
           threads: int, outcomes: list, failures: list) -> None:
    """Drain the queue on one GPU."""
    while True:
        try:
            model, dataset = work.get_nowait()
        except queue.Empty:
            return

        label = f"[gpu {gpu}] {model} x {dataset}"
        say(f"{label}: starting")
        started = time.time()

        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            environment[name] = str(threads)

        command = [sys.executable, "scripts/run_benchmark.py",
                   "--models", model, "--datasets", dataset,
                   "--data-root", args.data_root, "--results-root", args.results_root,
                   "--seeds", *[str(s) for s in args.seeds],
                   "--trials", str(args.trials),
                   "--epochs", str(args.epochs), "--patience", str(args.patience),
                   "--max-minutes", str(args.max_minutes),
                   "--no-archive", "--no-report"]
        if getattr(args, "ratios", None):
            command += ["--ratios", *args.ratios]
        if args.tune_epochs is not None:
            command += ["--tune-epochs", str(args.tune_epochs)]
        if args.tune_patience is not None:
            command += ["--tune-patience", str(args.tune_patience)]
        if args.skip_tuning:
            command.append("--skip-tuning")

        log_dir = Path(args.results_root) / "worker_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gpu{gpu}_{MODELS[model]}_{dataset}.log"

        try:
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=str(REPO_ROOT), env=environment,
                                        stdout=handle, stderr=subprocess.STDOUT)
            elapsed = time.time() - started
            if result.returncode == 0:
                say(f"{label}: done in {human(elapsed)}")
                outcomes.append((model, dataset, "COMPLETE", elapsed))
            else:
                say(f"{label}: FAILED after {human(elapsed)}  (see {log_path.name})")
                outcomes.append((model, dataset, "FAILED", elapsed))
                failures.append((model, dataset, str(log_path)))
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - started
            say(f"{label}: EXCEPTION {type(exc).__name__}: {exc}")
            outcomes.append((model, dataset, "FAILED", elapsed))
            failures.append((model, dataset, str(exc)))
        finally:
            work.task_done()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COMP8851 parallel benchmark")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all detected)")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS_BY_COST))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--seeds", nargs="*", type=int, default=[2, 42, 72])
    parser.add_argument("--ratios", nargs="*", default=["TR40"], metavar="RATIO",
                        help="Label ratios for final runs (TR40 TR30 TR20 TR10). "
                             "Tuning stays at TR40; the frozen config is reused.")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tune-epochs", type=int, default=None)
    parser.add_argument("--tune-patience", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.reduced:
        args.trials = min(args.trials, 4)
        args.epochs = min(args.epochs, 60)
        args.patience = min(args.patience, 15)
        args.max_minutes = min(args.max_minutes, 25.0)

    if args.trials > 12:
        print(f"Refusing {args.trials} trials: the protocol maximum is 12.")
        return 1

    os.chdir(REPO_ROOT)
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    detected = detect_gpus()
    workers = args.gpus if args.gpus else detected
    names = gpu_names()

    print("=" * 72)
    print("COMP8851 PARALLEL BENCHMARK")
    print("=" * 72)
    print(f"  GPUs detected : {detected}")
    for index, name in enumerate(names):
        print(f"     gpu {index}: {name}")
    if detected == 0:
        if not args.dry_run:
            print("\n  No GPU detected. A controlled run must not proceed on CPU;")
            print("  it would waste rental time and produce incomparable timings.")
            print(f"\n  Reason: {no_gpu_reason()}")
            return 1
        # --dry-run must work anywhere, so the plan can be checked before renting.
        print("\n  No GPU detected, but --dry-run only prints the plan.")
        workers = args.gpus or 1
    if detected and workers > detected:
        print(f"  requested {workers} workers but only {detected} GPUs; using {detected}")
        workers = detected

    distinct = {name.split()[-1] for name in names[:workers]}
    if len(distinct) > 1:
        print(f"\n  WARNING: the GPUs are not identical ({sorted(distinct)}).")
        print("  The specification fixes one GPU class. Runs would not be comparable.")
        print("  Use --gpus to restrict to matching cards, or rent a uniform box.")

    # ---- build the work queue ----
    available, missing = {}, []
    for name in args.datasets:
        path = find_dataset(name, data_root)
        if path:
            available[name] = path
        else:
            missing.append(name)

    units = [(model, dataset)
             for dataset in DATASETS_BY_COST if dataset in available
             for model in args.models if model in MODELS]

    threads_each = max(2, (os.cpu_count() or 8) // max(workers, 1))

    print(f"\n  models        : {', '.join(args.models)}")
    print(f"  datasets      : {', '.join(sorted(available)) or 'none'}")
    if missing:
        print(f"  unavailable   : {', '.join(missing)}  (skipped, recorded as BLOCKED)")
    print(f"  seeds         : {args.seeds}")
    print(f"  trials        : {args.trials} per pair (protocol maximum 12)")
    print(f"  epochs        : {args.epochs}, patience {args.patience}")
    print(f"  work units    : {len(units)}  ((model, dataset) pairs)")
    print(f"  workers       : {workers}, one per GPU")
    print(f"  threads each  : {threads_each}")

    if not units:
        print("\n  Nothing to run. Check that datasets exist under data/.")
        return 1

    print("\n  schedule (longest first):")
    for index, (model, dataset) in enumerate(units):
        print(f"     {index + 1:2d}. {model:9s} x {dataset}")

    if args.dry_run:
        print("\n  --dry-run: stopping here")
        return 0

    # ---- run ----
    work: "queue.Queue[tuple[str, str]]" = queue.Queue()
    for unit in units:
        work.put(unit)

    outcomes: list = []
    failures: list = []
    started = time.time()
    print(f"\n  started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("  per-worker output goes to results/worker_logs/\n")

    threads = []
    for gpu in range(workers):
        thread = threading.Thread(target=worker,
                                  args=(gpu, work, args, threads_each, outcomes, failures),
                                  daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(2)      # stagger starts so downloads and caches do not collide

    for thread in threads:
        thread.join()

    elapsed = time.time() - started
    print(f"\n{'=' * 72}")
    print(f"  all workers finished in {human(elapsed)}")
    complete = [o for o in outcomes if o[2] == "COMPLETE"]
    print(f"  {len(complete)}/{len(units)} unit(s) complete")
    for model, dataset, status, seconds in sorted(outcomes, key=lambda o: -o[3]):
        print(f"     {status:9s} {human(seconds):>7s}  {model:9s} x {dataset}")
    if failures:
        print(f"\n  {len(failures)} failure(s); logs:")
        for model, dataset, detail in failures:
            print(f"     {model} x {dataset}: {detail}")

    # ---- record how the parallelism was done ----
    (results_root / "parallel_manifest.json").write_text(json.dumps({
        "workers": workers,
        "gpus": names[:workers],
        "threads_per_worker": threads_each,
        "work_units": [{"model": m, "dataset": d, "status": s, "seconds": sec}
                       for m, d, s, sec in outcomes],
        "wall_clock_seconds": elapsed,
        "note": ("Independent runs executed in parallel, one GPU each. No run used "
                 "more than one GPU. Predictive metrics are unaffected; wall-clock "
                 "timings may be inflated by host contention."),
    }, indent=2, default=str), encoding="utf-8")

    # ---- build the deliverable once, now that nothing is writing ----
    print(f"\n{'=' * 72}\n  building the deliverable\n{'=' * 72}")
    from shared.comp8851 import report

    manifest, records = {}, []
    for model, slug in MODELS.items():
        model_results = results_root / f"results_{slug}"
        if not model_results.exists():
            continue
        dataset_manifest = model_results / "reports" / "dataset_manifest.json"
        if dataset_manifest.exists():
            try:
                manifest.update(json.loads(dataset_manifest.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        for record in sorted((model_results / "metrics").glob("tuning_*.json")):
            try:
                records.append(json.loads(record.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        outcome = report.build(results_root=model_results, models=[model], ratio="TR40",
                               dataset_manifest=manifest, tuning_records=records,
                               skipped={d: "dataset unavailable" for d in missing},
                               raw_root=model_results / "raw")
        print(f"  {model}: {outcome['runs_reportable']} reportable run(s)")

    # combined
    combined = results_root / "results_combined"
    raw = combined / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    merged = 0
    for slug in MODELS.values():
        source = results_root / f"results_{slug}" / "raw"
        if not source.exists():
            continue
        for summary in source.rglob("summary.json"):
            target = raw / summary.parent.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(summary.parent, target, dirs_exist_ok=True)
            merged += 1
    outcome = report.build(results_root=combined, models=list(MODELS), ratio="TR40",
                           dataset_manifest=manifest, tuning_records=records,
                           skipped={d: "dataset unavailable" for d in missing},
                           raw_root=raw)
    print(f"  combined: {merged} bundle(s), {outcome['runs_reportable']} reportable")

    matrix = combined / "reports" / "experiment_matrix.md"
    if matrix.exists():
        print()
        print(matrix.read_text(encoding="utf-8"))

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = shutil.make_archive(str(REPO_ROOT / f"comp8851_vastai_{stamp}"),
                                  "zip", root_dir=str(results_root))
    print(f"\n  archive: {archive}  ({Path(archive).stat().st_size / 1024**2:.1f} MB)")
    print("\n  COPY IT OFF THIS HOST BEFORE DESTROYING THE INSTANCE:")
    print(f"      scp -P <port> root@<host>:{archive} .")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
