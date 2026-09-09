from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path


RATIOS = ["TR40", "TR30", "TR20", "TR10"]
SEEDS = [2, 42, 72]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream(command, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["DGLBACKEND"] = "pytorch"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            clean = line.strip()
            if (
                clean.startswith("Epoch ") or clean.startswith("Early stopping")
                or clean.startswith("Best validation") or clean.startswith("auroc:")
                or clean.startswith("auprc:") or clean.startswith("macro_f1:")
                or clean.startswith("fraud_f1:") or clean.startswith("gmean:")
                or clean.startswith("Mean epoch") or clean.startswith("RUN COMPLETED")
                or clean.startswith("T-FINANCE SPLITS")
            ):
                print(clean, flush=True)
        return process.wait()


def completed_summary(result_dir: Path):
    summaries = sorted(result_dir.rglob("summary.json")) if result_dir.exists() else []
    return summaries[-1] if summaries else None


def run_model(source: Path, data_path: Path, result_dir: Path, log_path: Path, *,
              split_path: Path | None, run_mode: str, seed: int, hid_dim: int,
              epochs: int, patience: int, skip_test=False,
              selection_metric="auroc"):
    existing = completed_summary(result_dir)
    if existing:
        print(f"SKIP: {run_mode}; existing summary {existing}")
        return existing
    command = [
        sys.executable, "-u", "main_instrumented.py",
        "--dataset", "tfinance", "--data-path", str(data_path),
        "--seed", str(seed), "--split-seed", "2",
        "--hid-dim", str(hid_dim), "--order", "2",
        "--lr", "0.01", "--weight-decay", "0.0",
        "--epochs", str(epochs), "--valid-every", "5",
        "--patience", str(patience), "--selection-metric", selection_metric,
        "--run-mode", run_mode, "--results-dir", str(result_dir),
    ]
    if split_path:
        command += ["--split-path", str(split_path)]
    else:
        command += ["--train-ratio", "0.4"]
    if skip_test:
        command.append("--skip-test")
    print("-" * 78)
    print(f"START: {run_mode}")
    print(f"Seed={seed} | hidden={hid_dim} | max epochs={epochs} | patience={patience}")
    print(f"Log: {log_path}")
    code = stream(command, source, log_path)
    summary = completed_summary(result_dir)
    if code != 0 or not summary:
        raise RuntimeError(f"Run failed: {run_mode}. Check {log_path}")
    print(f"COMPLETE: {run_mode}")
    return summary


def save_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def package(source: Path, output: Path):
    stage = source / "package_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    include = [
        "BWGNN.py", "dataset.py", "main.py", "main_instrumented.py",
        "generate_splits.py", "run_all.py", "README.md", "requirements.txt",
        "readme(1).md",
    ]
    for name in include:
        path = source / name
        if path.exists():
            shutil.copy2(path, stage / name)
    for folder in ["shared", "results", "logs", "reports"]:
        path = source / folder
        if path.exists():
            shutil.copytree(path, stage / folder)
    for path in list(stage.rglob("*")):
        if path.is_file() and ("checkpoints" in path.parts or path.suffix in {".pt", ".pth", ".pkl", ".pyc"}):
            path.unlink()
    for path in sorted(stage.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    manifest = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(stage)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    save_rows(stage / "FILE_MANIFEST.csv", manifest)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage))
    with zipfile.ZipFile(output) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Output ZIP failed CRC: {bad}")
    return len(manifest), sha256(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-zip", default="/kaggle/working/bwgnn_tfinance_complete_results.zip")
    args = parser.parse_args()

    source = Path(__file__).resolve().parent
    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    os.chdir(source)
    started = time.time()

    print("=" * 78)
    print("COMP8851 BWGNN + T-FINANCE COMPLETE WORKFLOW")
    print("=" * 78)
    print(f"Source: {source}")
    print(f"Dataset: {data_path}")
    print("Stages: splits -> smoke -> author-setting reference -> tuning -> 12 final runs -> package")
    print("This workflow is resumable. Existing completed summaries will be skipped.")

    split_dir = source / "shared" / "splits" / "tfinance"
    split40 = split_dir / "tfinance_tr40_split_seed2.npz"
    if not split40.exists():
        print("\n[1/6] Creating persistent nested split family")
        command = [sys.executable, "generate_splits.py", "--data-path", str(data_path), "--output-dir", str(split_dir), "--seed", "2"]
        code = stream(command, source, source / "logs" / "generate_splits.log")
        if code != 0 or not split40.exists():
            raise RuntimeError("Split generation failed")
    else:
        print("\n[1/6] Persistent splits already exist: SKIP")

    print("\n[2/6] GPU smoke test")
    run_model(source, data_path, source / "results" / "smoke", source / "logs" / "smoke.log",
              split_path=split_dir / "tfinance_tr10_split_seed2.npz", run_mode="smoke_tfinance_tr10_seed72",
              seed=72, hid_dim=64, epochs=2, patience=0)

    print("\n[3/6] Author-setting reference check")
    run_model(
              source, data_path, source / "results" / "author-setting-reference",
              source / "logs" / "author_setting_reference.log",
              split_path=None, run_mode="author_setting_reference_tfinance_tr40_seed72",
              seed=72, hid_dim=64, epochs=100, patience=0,
              selection_metric="macro_f1")

    print("\n[4/6] Fast validation-only hidden-dimension tuning")
    tuning = []
    for hidden in [32, 64]:
        summary_path = run_model(
            source, data_path, source / "results" / "tuning" / f"hidden_{hidden}",
            source / "logs" / f"tuning_hidden_{hidden}.log", split_path=split40,
            run_mode=f"tuning_tfinance_tr40_hidden{hidden}_seed72", seed=72,
            hid_dim=hidden, epochs=50, patience=15, skip_test=True,
        )
        summary = json.loads(summary_path.read_text())
        tuning.append({
            "hidden_dimension": hidden,
            "best_validation_auroc": summary["best_validation_auroc"],
            "best_validation_macro_f1": summary["best_validation_macro_f1"],
            "best_validation_epoch": summary["best_validation_epoch"],
            "epochs_completed": summary["timing"]["training_epochs"],
            "test_evaluated": False,
            "summary_path": str(summary_path.relative_to(source)),
        })
    selected = max(tuning, key=lambda row: (row["best_validation_auroc"], -row["hidden_dimension"]))
    selection = {
        "dataset": "T-Finance", "model": "BWGNN", "tuning_seed": 72,
        "split_seed": 2, "ratio": "TR40", "selection_metric": "validation AUROC",
        "test_metrics_used_for_selection": False, "trials": tuning,
        "selected_hidden_dimension": selected["hidden_dimension"],
        "selected_validation_auroc": selected["best_validation_auroc"],
        "tie_break": "smaller hidden dimension only if validation AUROC is exactly equal",
    }
    report_dir = source / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tuning_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    save_rows(report_dir / "tuning_selection.csv", tuning)
    print(f"Selected hidden dimension: {selected['hidden_dimension']}")
    print(f"Selected validation AUROC: {selected['best_validation_auroc']:.6f}")
    print("Test used for tuning: False")

    print("\n[5/6] Final TR40/TR30/TR20/TR10 × three-seed batch")
    registry = []
    for ratio in RATIOS:
        for seed in SEEDS:
            mode = f"final_tfinance_{ratio.lower()}_seed{seed}"
            summary_path = run_model(
                source, data_path, source / "results" / "final" / ratio.lower() / f"seed{seed}",
                source / "logs" / "final" / f"{ratio.lower()}_seed{seed}.log",
                split_path=split_dir / f"tfinance_{ratio.lower()}_split_seed2.npz", run_mode=mode,
                seed=seed, hid_dim=selected["hidden_dimension"], epochs=100, patience=20,
            )
            summary = json.loads(summary_path.read_text())
            metrics = summary["test_metrics"]
            timing = summary["timing"]
            registry.append({
                "ratio": ratio, "seed": seed, "hidden_dimension": selected["hidden_dimension"],
                "order": 2, "best_validation_epoch": summary["best_validation_epoch"],
                "best_validation_auroc": summary["best_validation_auroc"],
                "auroc": metrics["auroc"], "auprc": metrics["auprc"],
                "macro_f1": metrics["macro_f1"], "fraud_f1": metrics["fraud_f1"],
                "fraud_precision": metrics["fraud_precision"], "fraud_recall": metrics["fraud_recall"],
                "gmean": metrics["gmean"], "accuracy": metrics["accuracy"],
                "threshold": metrics["threshold"], "epochs_completed": timing["training_epochs"],
                "mean_epoch_seconds": timing["mean_train_epoch_seconds"],
                "fit_wall_seconds": timing["fit_wall_seconds_including_validation"],
                "inference_seconds": timing["test_inference_seconds"],
                "peak_gpu_memory_mb": timing["peak_gpu_memory_mb"],
                "summary_path": str(summary_path.relative_to(source)),
            })
    save_rows(report_dir / "final_results_registry.csv", registry)
    (report_dir / "final_results_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    aggregates = []
    metric_keys = ["auroc", "auprc", "macro_f1", "fraud_f1", "fraud_precision", "fraud_recall", "gmean", "accuracy", "mean_epoch_seconds", "fit_wall_seconds", "inference_seconds", "peak_gpu_memory_mb"]
    for ratio in RATIOS:
        rows = [row for row in registry if row["ratio"] == ratio]
        out = {"ratio": ratio, "n": len(rows)}
        for key in metric_keys:
            values = [float(row[key]) for row in rows]
            out[f"{key}_mean"] = statistics.fmean(values)
            out[f"{key}_sample_sd"] = statistics.stdev(values)
        aggregates.append(out)
    save_rows(report_dir / "final_results_mean_sd.csv", aggregates)
    (report_dir / "final_results_mean_sd.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")

    epoch_rows = 0
    for row in registry:
        summary_path = source / row["summary_path"]
        epoch_file = summary_path.parent / "epoch_times.csv"
        with epoch_file.open(encoding="utf-8") as handle:
            epoch_rows += sum(1 for _ in csv.DictReader(handle))
    final_status = {
        "status": "PASS", "final_runs_expected": 12, "final_runs_found": len(registry),
        "ratios": RATIOS, "seeds": SEEDS, "split_seed": 2,
        "selected_hidden_dimension": selected["hidden_dimension"], "order": 2,
        "optimizer": "Adam", "learning_rate": 0.01,
        "maximum_epochs": 100, "early_stopping_patience_epochs": 20,
        "selection_metric": "validation AUROC", "test_used_for_tuning": False,
        "fixed_validation_ids": True, "fixed_test_ids": True,
        "nested_training_sets": True, "final_epoch_rows": epoch_rows,
        "hardware_requirement": "one NVIDIA Tesla T4, GPU 0",
        "elapsed_minutes_this_execution": (time.time() - started) / 60,
    }
    (report_dir / "verification_status.json").write_text(json.dumps(final_status, indent=2), encoding="utf-8")
    print(f"Final runs recorded: {len(registry)}/12")
    print(f"Final epoch rows recorded: {epoch_rows}")

    print("\n[6/6] Creating complete evidence ZIP")
    output = Path(args.output_zip).resolve()
    file_count, zip_hash = package(source, output)
    print("=" * 78)
    print("BWGNN + T-FINANCE WORKFLOW COMPLETE")
    print("=" * 78)
    print(f"Final runs: {len(registry)}/12")
    print(f"Selected hidden dimension: {selected['hidden_dimension']}")
    print(f"Final epoch rows: {epoch_rows}")
    print(f"Package files: {file_count}")
    print(f"ZIP: {output}")
    print(f"ZIP size: {output.stat().st_size / 1024**2:.2f} MB")
    print(f"ZIP SHA256: {zip_hash}")
    print("COMPLETE WORKFLOW: TRUE")


if __name__ == "__main__":
    main()
