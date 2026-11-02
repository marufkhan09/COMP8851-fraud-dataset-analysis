"""Preflight: prove the benchmark will work before spending money on it.

    python3 scripts/preflight.py

Runs a two-epoch attempt for every model-dataset pair and reports a clear
go / no-go. Takes roughly ten to twenty minutes and can save many hours of
rental time, because every expensive failure mode shows up here first:

  * a dataset that did not download or will not load
  * a model that cannot consume a particular dataset
  * out of memory, which is the expected T-Social outcome on a 16 GB T4
  * a broken environment, a missing GPU, an unusable DGL

Nothing here is benchmark evidence. Every run is launched with ``--smoke``, so
the results are tagged ``feasibility_only`` and the aggregator excludes them
from metric tables. The point is feasibility, not performance.

Exit code 0 means every attempted pair trained. Non-zero means at least one
pair failed, and the report says which and why.

Options:
    --datasets a b c    restrict the check
    --models CARE-GNN   restrict the check
    --epochs 2          epochs per probe
    --timeout 10        minutes allowed per probe
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODELS = {
    "CARE-GNN": ("care-gnn", "models/care-gnn/scripts/run_one.py"),
    "GHRN": ("ghrn", "models/ghrn/scripts/run_one.py"),
}
DATASETS = ("yelpchi", "amazon", "fdcompcn", "tfinance", "elliptic", "tsocial")


def find_dataset(name: str, data_root: Path) -> Path | None:
    """Locate a dataset, preferring the library-free canonical cache."""
    candidates = [data_root / "canonical" / f"{name}_canonical.npz", data_root / name]
    return next((c for c in candidates if c.exists()), None)


def human(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f}s"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COMP8851 preflight")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Minutes allowed per probe")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the probe artefacts instead of deleting them")
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    data_root = Path(args.data_root)
    probe_root = REPO_ROOT / "results" / "_preflight"

    print("=" * 72)
    print("COMP8851 PREFLIGHT")
    print("=" * 72)

    # ---- environment ----
    print("\n[environment]")
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        vram = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3) if cuda else 0
        print(f"  torch {torch.__version__} | CUDA {torch.version.cuda} | available {cuda}")
        if cuda:
            print(f"  GPU  {gpu} ({vram:.1f} GB)")
            if "t4" not in gpu.lower():
                print(f"  note: '{gpu}' is not a Tesla T4; timings will not be "
                      "comparable with T4 numbers")
        else:
            print("  NO GPU. Stop here: a controlled run must not proceed on CPU,")
            print("  and running it on CPU would only waste rental time.")
            return 1
    except ImportError:
        print("  torch is not installed. Run scripts/vastai_setup.py first.")
        return 1

    try:
        import dgl

        dgl.graph(([0], [1]))
        print(f"  DGL  {dgl.__version__} (usable)")
    except Exception:
        print("  DGL  not usable; DGL-format datasets rely on the isolated decoder")

    free_gb = shutil.disk_usage(REPO_ROOT).free / 1024 ** 3
    print(f"  disk {free_gb:.1f} GB free")
    if free_gb < 20:
        print("  warning: under 20 GB free. T-Social alone needs more than this.")

    # ---- datasets ----
    print("\n[datasets]")
    available: dict[str, Path] = {}
    missing: dict[str, str] = {}
    for name in args.datasets:
        path = find_dataset(name, data_root)
        if path is None:
            missing[name] = "not found under data/"
            print(f"  {name:10s} MISSING")
            continue
        try:
            from shared.comp8851 import datasets as ds

            started = time.perf_counter()
            dataset = ds.load_dataset(name, path, strict=False)
            stats = dataset.statistics()
            available[name] = path
            print(f"  {name:10s} ok  {stats['nodes']:>9,} nodes | "
                  f"{stats['feature_dimension']:>3} feat | "
                  f"fraud {stats['fraud_percentage']:5.2f}% | "
                  f"{human(time.perf_counter() - started)}")
        except Exception as exc:
            missing[name] = f"{type(exc).__name__}: {exc}"
            print(f"  {name:10s} FAILED TO LOAD: {str(exc)[:80]}")

    if not available:
        print("\nNo dataset is usable. Fix acquisition before renting more time.")
        return 1

    # ---- probes ----
    print(f"\n[training probes] {args.epochs} epoch(s) each, "
          f"{args.timeout:.0f} min cap, test set never touched")
    print("  These are feasibility checks. Their metrics are NOT results.\n")

    results: list[dict] = []
    for model in args.models:
        slug, script = MODELS[model]
        for name in args.datasets:
            if name not in available:
                results.append({"model": model, "dataset": name,
                                "status": "BLOCKED", "detail": missing.get(name, "unavailable"),
                                "seconds": 0.0})
                continue

            label = f"{model:9s} x {name:10s}"
            print(f"  {label} ... ", end="", flush=True)
            target = probe_root / slug / name
            command = [sys.executable, script,
                       "--dataset", name, "--data-path", str(available[name]),
                       "--ratio", "TR40", "--seed", "2", "--track", "unified",
                       "--results-root", str(target),
                       "--epochs", str(args.epochs), "--patience", "0",
                       "--max-minutes", str(args.timeout),
                       "--run-mode", "preflight", "--smoke",
                       "--skip-test", "--no-strict-stats"]

            started = time.time()
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=args.timeout * 60 + 120)
                elapsed = time.time() - started
                output = completed.stdout + completed.stderr
                if completed.returncode == 0:
                    status, detail = "PASS", ""
                else:
                    status = "FAIL"
                    lowered = output.lower()
                    if "out of memory" in lowered or "outofmemory" in lowered:
                        status, detail = "OOM", "out of memory"
                    elif "--max-nodes" in output:
                        status = "TOO_LARGE"
                        detail = "graph exceeds the model's node guard"
                    else:
                        lines = [l for l in output.strip().splitlines() if l.strip()]
                        detail = lines[-1][:140] if lines else f"exit {completed.returncode}"
            except subprocess.TimeoutExpired:
                elapsed = time.time() - started
                status, detail = "TIMEOUT", f"exceeded {args.timeout:.0f} min"

            marker = {"PASS": "PASS", "OOM": "OOM ", "TOO_LARGE": "BIG ",
                      "TIMEOUT": "SLOW", "FAIL": "FAIL"}[status]
            print(f"{marker}  {human(elapsed)}  {detail[:60]}")
            results.append({"model": model, "dataset": name, "status": status,
                            "detail": detail, "seconds": elapsed})

    # ---- verdict ----
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    passed = [r for r in results if r["status"] == "PASS"]
    oom = [r for r in results if r["status"] in ("OOM", "TOO_LARGE")]
    slow = [r for r in results if r["status"] == "TIMEOUT"]
    broken = [r for r in results if r["status"] == "FAIL"]
    blocked = [r for r in results if r["status"] == "BLOCKED"]

    print(f"\n  will train      : {len(passed)} of {len(results)} pairs")
    for item in passed:
        print(f"     {item['model']:9s} x {item['dataset']}")

    if oom:
        print(f"\n  too large       : {len(oom)}")
        for item in oom:
            print(f"     {item['model']:9s} x {item['dataset']}  ({item['detail']})")
        print("     Expected for T-Social on a 16 GB T4. This is a legitimate,")
        print("     recorded feasibility outcome, not something to hide or work")
        print("     around by shrinking the graph.")

    if slow:
        print(f"\n  too slow        : {len(slow)}")
        for item in slow:
            print(f"     {item['model']:9s} x {item['dataset']}")
        print("     Raise --max-minutes for the real run, or drop the pair.")

    if blocked:
        print(f"\n  dataset missing : {len(blocked)}")
        for item in blocked:
            print(f"     {item['model']:9s} x {item['dataset']}  ({item['detail'][:70]})")

    if broken:
        print(f"\n  BROKEN          : {len(broken)}")
        for item in broken:
            print(f"     {item['model']:9s} x {item['dataset']}")
            print(f"        {item['detail']}")
        print("     Investigate before starting the benchmark; these are real defects.")

    estimate = sum(r["seconds"] for r in passed) / max(args.epochs, 1)
    print(f"\n  rough cost signal: {human(estimate)} per epoch across all passing pairs")
    print("  A 100-epoch full protocol with 12 trials is roughly")
    print(f"  {estimate * 100 * (12 + 3) / 3600:.1f} GPU-hours; at $0.15/hr that is about "
          f"${estimate * 100 * (12 + 3) / 3600 * 0.15:.2f}.")
    print("  Early stopping at patience 20 usually cuts this substantially.")

    report_path = REPO_ROOT / "results" / "preflight_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  report: {report_path}")

    if not args.keep and probe_root.exists():
        shutil.rmtree(probe_root, ignore_errors=True)
        print("  probe artefacts removed (pass --keep to retain them)")

    print()
    if broken:
        print("  NO-GO. Fix the broken pairs first.")
        return 1
    if not passed:
        print("  NO-GO. Nothing trains.")
        return 1
    print(f"  GO. {len(passed)} pair(s) will train. Start the benchmark with:")
    print("      tmux new -s comp8851")
    print("      python3 scripts/run_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
