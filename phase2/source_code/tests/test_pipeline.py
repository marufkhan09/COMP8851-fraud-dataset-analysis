"""End-to-end pipeline tests on real data (FDCompCN), run as subprocesses
exactly as the matrix runner launches them.

Covers, for every model: forward/backward/optimiser step, validation,
checkpoint save, reload reproduction, prediction files and metric
recomputation; plus resume after an interruption, the non-finite-loss guard,
skip-test isolation and overwrite protection.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, needs_dataset

MODELS = ["sagad", "spacegnn", "croc"]


def run(model, results_root, *extra):
    cmd = [sys.executable, str(ROOT / "models" / model / "run_one.py"), "--dataset", "fdcompcn",
           "--seed", "2", "--results-root", str(results_root), "--no-cuda", "--num-workers", "0", *extra]
    env = dict(os.environ, DGLBACKEND="pytorch")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=3600)


def run_dir(results_root, model):
    return Path(results_root) / model / "fdcompcn" / "unified" / "tr40" / "seed_2"


@needs_dataset("fdcompcn")
@pytest.mark.parametrize("model", MODELS)
def test_smoke_run_and_independent_metric_recomputation(model, tmp_path):
    proc = run(model, tmp_path, "--smoke")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    d = run_dir(tmp_path, model)
    summary = json.loads((d / "summary.json").read_text())
    assert summary["status"] == "COMPLETE"
    assert summary["artifact_verification"]["complete"]
    assert summary["checkpoint_reload_check"]["passed"]
    pred = np.load(d / "predictions.npz")
    from shared.comp8851 import evaluator

    threshold, _ = evaluator.select_threshold(pred["valid_labels"], pred["valid_prob"])
    assert abs(threshold - float(pred["threshold"])) < 1e-12
    recomputed = evaluator.metric_bundle(pred["test_labels"], pred["test_prob"], threshold)
    stored = summary["test_metrics"]
    for key in evaluator.METRIC_KEYS:
        assert abs(recomputed[key] - stored[key]) < 1e-12, key
    # labels in the prediction file are the true labels of the frozen split
    from phase2 import data as pdata

    prep = pdata.prepare("fdcompcn", "TR40")
    assert np.array_equal(pred["test_idx"], prep.test_idx)
    assert np.array_equal(pred["test_labels"], prep.dataset.labels[prep.test_idx])


@needs_dataset("fdcompcn")
def test_resume_after_interruption(tmp_path):
    first = run("sagad", tmp_path, "--epochs", "6", "--patience", "0", "--inject-fault", "crash_after_epoch1")
    assert first.returncode == 3, first.stdout[-2000:]
    d = run_dir(tmp_path, "sagad")
    assert json.loads((d / "summary.json").read_text())["status"] == "INCOMPLETE"
    assert (d / "checkpoints" / "last.pt").exists()
    second = run("sagad", tmp_path, "--epochs", "6", "--patience", "0", "--resume")
    assert second.returncode == 0, second.stdout[-3000:]
    summary = json.loads((d / "summary.json").read_text())
    assert summary["status"] == "COMPLETE" and summary["resumed_from_epoch"] == 1
    assert summary["timing"]["training_epochs_completed"] == 6

    # An uninterrupted run with the same settings reaches the same weights.
    ref_root = tmp_path / "reference"
    third = run("sagad", ref_root, "--epochs", "6", "--patience", "0")
    assert third.returncode == 0
    a = np.load(d / "predictions.npz")["test_prob"]
    b = np.load(run_dir(ref_root, "sagad") / "predictions.npz")["test_prob"]
    assert np.max(np.abs(a - b)) < 1e-6


@needs_dataset("fdcompcn")
def test_non_finite_loss_fails_the_run(tmp_path):
    proc = run("sagad", tmp_path, "--smoke", "--inject-fault", "nan_loss")
    assert proc.returncode == 1
    summary = json.loads((run_dir(tmp_path, "sagad") / "summary.json").read_text())
    assert summary["status"] == "FAILED" and summary["failure_reason"] == "NON_FINITE_LOSS"
    assert summary["test_metrics"] is None
    assert not (run_dir(tmp_path, "sagad") / "predictions.npz").exists()


@needs_dataset("fdcompcn")
def test_out_of_memory_is_recorded_not_hidden(tmp_path):
    proc = run("sagad", tmp_path, "--smoke", "--inject-fault", "oom")
    assert proc.returncode == 1
    summary = json.loads((run_dir(tmp_path, "sagad") / "summary.json").read_text())
    assert summary["status"] == "FAILED" and summary["failure_reason"] == "CUDA_OOM"
    assert summary["test_metrics"] is None


@needs_dataset("fdcompcn")
def test_skip_test_never_scores_test(tmp_path):
    proc = run("croc", tmp_path, "--smoke", "--skip-test")
    assert proc.returncode == 0, proc.stdout[-2000:]
    d = run_dir(tmp_path, "croc")
    assert not (d / "test_metrics.json").exists()
    assert "test_prob" not in np.load(d / "predictions.npz").files


@needs_dataset("fdcompcn")
def test_complete_run_is_not_overwritten(tmp_path):
    assert run("sagad", tmp_path, "--smoke").returncode == 0
    d = run_dir(tmp_path, "sagad")
    digest = json.loads((d / "summary.json").read_text())["predictions_sha256"]
    again = run("sagad", tmp_path, "--smoke")
    assert again.returncode == 0 and "already COMPLETE" in again.stdout
    assert json.loads((d / "summary.json").read_text())["predictions_sha256"] == digest
    forced = run("sagad", tmp_path, "--smoke", "--force")
    assert forced.returncode == 0
    archived = list((tmp_path / "_attempts" / "sagad" / "fdcompcn").rglob("summary.json"))
    assert archived, "forced re-run must archive, not delete, the previous run"
