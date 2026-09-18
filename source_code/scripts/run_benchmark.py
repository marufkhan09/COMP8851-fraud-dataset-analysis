"""COMP8851 controlled benchmark: tuning, final runs, and the full deliverable.

    tmux new -s comp8851
    python3 scripts/run_benchmark.py

For each model, per dataset:

  1. Tuning on TR40 with train seed 2, at most 12 trials, the same budget for
     every dataset, selected on validation AUPRC. Every trial runs with
     ``--skip-test``, so no tuning run can read, score or write test data.
  2. Final runs with the frozen configuration across train seeds 2, 42 and 72,
     each evaluating the test set exactly once.
  3. The deliverable: the seven required directories, the experiment matrix and
     the reproducibility manifest.

Then the combined cross-model comparison.

Everything is resumable. Completed tuning trials and finished runs are detected
and skipped, so an interruption costs only the run that was in flight. That
matters on rented hardware: re-running after a dropped connection is free.

Options:
    --models CARE-GNN GHRN      which models to run
    --datasets yelpchi amazon   which datasets
    --trials 12                 tuning budget per pair (protocol maximum 12)
    --epochs 100                maximum epochs
    --patience 20               early-stopping patience
    --seeds 2 42 72             training seeds
    --reduced                   4 trials and lower epochs; all other rules unchanged
    --skip-tuning               use published defaults, no search
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

from shared.comp8851 import protocol  # noqa: E402  (needs REPO_ROOT on the path)

MODELS = {
    "CARE-GNN": ("care-gnn", "models/care-gnn/scripts/run_one.py"),
    "GHRN": ("ghrn", "models/ghrn/scripts/run_one.py"),
}
DATASETS = ("yelpchi", "amazon", "fdcompcn", "tfinance", "elliptic", "tsocial")


def human(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes = rest // 60
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def find_dataset(name: str, data_root: Path) -> Path | None:
    for candidate in (data_root / "canonical" / f"{name}_canonical.npz", data_root / name):
        if candidate.exists():
            return candidate
    return None


def run_model(model: str, args, data_root: Path, results_root: Path) -> dict:
    from shared.comp8851 import datasets as ds, report, tuning

    slug, script = MODELS[model]
    model_results = results_root / f"results_{slug}"
    raw = model_results / "raw"
    tuning_root = model_results / "tuning"
    metrics = model_results / "metrics"
    for directory in (raw, tuning_root, metrics, model_results / "reports"):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 72}\n# {model}\n{'#' * 72}")

    no_cuda = False
    try:
        import torch

        no_cuda = not torch.cuda.is_available()
    except ImportError:
        no_cuda = True
    if no_cuda:
        print("  WARNING: no GPU. Timings will not be comparable.")

    # ---- usable datasets ----
    available: dict[str, Path] = {}
    skipped: dict[str, str] = {}
    manifest: dict[str, dict] = {}
    print("\n[datasets]")
    for name in args.datasets:
        path = find_dataset(name, data_root)
        if path is None:
            skipped[name] = "not present under data/"
            print(f"  {name:10s} SKIPPED: not found")
            continue
        try:
            dataset = ds.load_dataset(name, path, strict=False)
            stats, hetero = dataset.statistics(), dataset.heterophily()
            available[name] = path
            manifest[name] = {
                "source_path": str(path),
                "source_sha256": dataset.source_sha256,
                "statistics": stats, "heterophily": hetero,
                "view_relation": dataset.view_manifest("relation")["view_sha256"],
                "view_homogeneous": dataset.view_manifest("homogeneous")["view_sha256"],
            }
            value = hetero.get("global_heterophily")
            suffix = f"heterophily {value:.4f}" if isinstance(value, float) else ""
            print(f"  {name:10s} ok  {stats['nodes']:>9,} nodes | "
                  f"fraud {stats['fraud_percentage']:5.2f}% | {suffix}")
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
            print(f"  {name:10s} SKIPPED: {str(exc)[:70]}")

    (model_results / "reports" / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    if not available:
        print("\n  nothing to train on")
        return {"model": model, "trained": 0, "skipped": skipped}

    # ---- tuning ----
    records = []
    if args.skip_tuning:
        print("\n[tuning] skipped; using the published configuration for every dataset")
        for name in sorted(available):
            records.append({
                "model": model, "dataset": name, "budget": 0,
                "selected_trial": None, "trials": [], "trials_attempted": 0,
                "trials_successful": 0, "test_used_during_tuning": False,
                "selected_hyperparameters": dict(tuning.AUTHOR_DEFAULTS[model]),
                "selected_validation_auprc": None,
                "selection_reason": "tuning skipped by request",
            })
    else:
        print(f"\n[tuning] {args.trials} trial(s) per dataset on TR40, train seed 2")
        print("  selection metric: validation AUPRC; the test set is untouchable here")
        for name in sorted(available):
            path = metrics / f"tuning_{model.lower().replace('-', '')}_{name}.json"
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
                print(f"\n  [{name}] already tuned; reusing trial "
                      f"{records[-1]['selected_trial']}")
                continue
            print(f"\n  [{name}]")
            record = tuning.run_tuning(
                model=model, model_slug=slug, script=script, dataset=name,
                data_path=str(available[name]), tuning_root=tuning_root / f"{slug}_{name}",
                budget=args.trials, epochs=args.tune_epochs,
                patience=args.tune_patience, max_minutes=args.max_minutes,
                no_cuda=no_cuda, python=sys.executable, cwd=REPO_ROOT, verbose=True,
                extra=["--allow-large"] if getattr(args, "allow_large", False)
                else None)
            tuning.write_tuning_record(record, metrics)
            records.append(record)

        leaked = list(tuning_root.rglob("test_metrics.json"))
        print(f"\n  test isolation: {len(leaked)} test evaluation(s) under tuning/ "
              f"{'FAIL - investigate before reporting' if leaked else '(clean)'}")

    # ---- final runs ----
    best = {r["dataset"]: r["selected_hyperparameters"] for r in records}
    print(f"\n[final runs] seeds {args.seeds}, {args.epochs} epochs, "
          f"patience {args.patience}, one test evaluation each")

    # The nested-ratio design: tune once at TR40, then reuse that frozen
    # configuration unchanged at every lower label ratio. Only the training
    # split shrinks, so the degradation curve isolates label scarcity rather
    # than confounding it with a fresh hyperparameter search per ratio.
    ratios = [r.upper() for r in getattr(args, "ratios", None) or ["TR40"]]
    unknown = [r for r in ratios if r not in protocol.TRAIN_RATIOS]
    if unknown:
        print(f"\nUnknown ratio(s) {unknown}; valid: "
              f"{', '.join(protocol.TRAIN_RATIOS)}")
        return {"model": model, "trained": 0, "skipped": skipped,
                "records": records, "manifest": manifest}

    trained = 0
    for name in sorted(available):
        for ratio in ratios:
            for seed in args.seeds:
                summary = (raw / slug / name / "unified" / ratio.lower() /
                           f"seed_{seed}" / "summary.json")
                if summary.exists():
                    try:
                        payload = json.loads(summary.read_text(encoding="utf-8"))
                        if payload.get("status") == "COMPLETE" and payload.get("test_metrics"):
                            print(f"  [{name} {ratio} seed {seed}] already complete")
                            trained += 1
                            continue
                    except (OSError, json.JSONDecodeError):
                        pass

                print(f"\n  [{name} {ratio} seed {seed}]")
                command = tuning.final_command(
                    sys.executable, script, name, str(available[name]), str(raw),
                    best.get(name, {}), seed=seed, ratio=ratio, epochs=args.epochs,
                    patience=args.patience, max_minutes=args.max_minutes,
                    no_cuda=no_cuda,
                    extra=["--allow-large"] if getattr(args, "allow_large", False)
                    else None)
                started = time.time()
                result = subprocess.run(command)
                elapsed = time.time() - started
                if result.returncode == 0:
                    trained += 1
                    print(f"    COMPLETE in {human(elapsed)}")
                else:
                    print(f"    FAILED in {human(elapsed)}; recorded with its traceback")

    # ---- deliverable ----
    if getattr(args, "no_report", False):
        print(f"\n[deliverable] skipped for {model}; the parallel runner builds it once")
    else:
        print(f"\n[deliverable] {model}")
        outcome = report.build(results_root=model_results, models=[model], ratio="TR40",
                               dataset_manifest=manifest, tuning_records=records,
                               skipped=skipped, raw_root=raw)
        print(f"  reportable runs: {outcome['runs_reportable']}")
        print(f"  figures        : {len(outcome['figures'])}")
        print(f"  matrix         : {outcome['experiment_matrix']}")
        print(f"  manifest       : {outcome['manifest']}")
    return {"model": model, "trained": trained, "skipped": skipped,
            "records": records, "manifest": manifest}


def combine(results_root: Path, outcomes: list[dict]) -> None:
    from shared.comp8851 import report

    print(f"\n{'#' * 72}\n# Combined cross-model deliverable\n{'#' * 72}")
    combined = results_root / "results_combined"
    raw = combined / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    manifest, records, merged = {}, [], 0
    for model in MODELS:
        slug = MODELS[model][0]
        source = results_root / f"results_{slug}" / "raw"
        if not source.exists():
            continue
        for summary in source.rglob("summary.json"):
            target = raw / summary.parent.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(summary.parent, target, dirs_exist_ok=True)
            merged += 1
    for outcome in outcomes:
        manifest.update(outcome.get("manifest") or {})
        records.extend(outcome.get("records") or [])

    print(f"  merged {merged} run bundle(s)")
    outcome = report.build(results_root=combined, models=list(MODELS), ratio="TR40",
                           dataset_manifest=manifest, tuning_records=records,
                           raw_root=raw)
    print(f"  reportable runs: {outcome['runs_reportable']}")

    matrix = combined / "reports" / "experiment_matrix.md"
    if matrix.exists():
        print()
        print(matrix.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COMP8851 controlled benchmark")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--seeds", nargs="*", type=int, default=[2, 42, 72])
    parser.add_argument("--ratios", nargs="*", default=["TR40"],
                        metavar="RATIO",
                        help="Label ratios for the final runs, e.g. "
                             "--ratios TR40 TR30 TR20 TR10. Tuning always "
                             "happens at TR40 and the frozen configuration is "
                             "reused unchanged at every lower ratio.")
    parser.add_argument("--allow-large", action="store_true",
                        help="Override a model's node guard. CARE-GNN refuses "
                             "graphs above 2M nodes because it materialises "
                             "per-relation Python adjacency sets; only pass this "
                             "on a host with very large RAM, and expect it to be "
                             "slow. The override is recorded in every run config.")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tune-epochs", type=int, default=None)
    parser.add_argument("--tune-patience", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--reduced", action="store_true",
                        help="4 trials and lower epochs; all other protocol rules unchanged")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip the deliverable build. Used by the parallel "
                             "runner, which builds it once after all shards finish "
                             "so concurrent workers cannot race on the same files.")
    args = parser.parse_args(argv)

    if args.reduced:
        args.trials = min(args.trials, 4)
        args.epochs = min(args.epochs, 60)
        args.patience = min(args.patience, 15)
        args.max_minutes = min(args.max_minutes, 25.0)
    if args.tune_epochs is None:
        args.tune_epochs = args.epochs
    if args.tune_patience is None:
        args.tune_patience = args.patience

    if args.trials > 12:
        print(f"Refusing {args.trials} trials: the protocol maximum is 12.")
        return 1

    os.chdir(REPO_ROOT)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "8")

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("COMP8851 CONTROLLED BENCHMARK")
    print("=" * 72)
    print(f"  models   : {', '.join(args.models)}")
    print(f"  datasets : {', '.join(args.datasets)}")
    print(f"  seeds    : {args.seeds}")
    print(f"  trials   : {args.trials} per pair (protocol maximum 12)")
    print(f"  epochs   : {args.epochs}, patience {args.patience}")
    print(f"  results  : {results_root.resolve()}")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    print(f"  started  : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    started = time.time()
    outcomes = []
    for model in args.models:
        if model not in MODELS:
            print(f"Unknown model {model!r}")
            return 1
        outcomes.append(run_model(model, args, data_root, results_root))

    if len(args.models) > 1:
        combine(results_root, outcomes)

    elapsed = time.time() - started
    print(f"\n{'=' * 72}")
    print(f"Total wall clock: {human(elapsed)}")
    for outcome in outcomes:
        print(f"  {outcome['model']:9s} {outcome['trained']} run(s) complete")

    if not args.no_archive:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        archive = shutil.make_archive(str(REPO_ROOT / f"comp8851_vastai_{stamp}"),
                                      "zip", root_dir=str(results_root))
        size = Path(archive).stat().st_size / 1024 ** 2
        print(f"\n  archive: {archive}  ({size:.1f} MB)")
        print("\n  COPY IT OFF THIS HOST BEFORE DESTROYING THE INSTANCE:")
        print(f"      scp -P <port> root@<host>:{archive} .")
        print("  A stopped instance still bills for storage; destroying it wipes the disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
