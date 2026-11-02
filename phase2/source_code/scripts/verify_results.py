"""Independent results-integrity check.

    python scripts/verify_results.py                 # final runs
    python scripts/verify_results.py --phase smoke
    python scripts/verify_results.py --deep          # also re-predict from each checkpoint

For every run directory with a summary.json it checks, independently of the
training code path:

1. predictions.npz exists and its SHA256 equals the one recorded at run time;
2. node IDs and labels in predictions.npz equal the frozen split and the
   canonical labels (so a score cannot be computed on the wrong population);
3. the threshold re-selected from the saved *validation* predictions with the
   shared evaluator equals the recorded threshold;
4. every test metric recomputed from the saved test predictions equals
   test_metrics.json (|diff| <= 1e-12);
5. the best checkpoint loads; with --deep, a freshly built model loaded from it
   reproduces the saved test predictions within max(1e-5, 3 x the run's
   recorded repeat noise) and the test AUPRC within 1e-4.

Writes results/aggregated/<phase>/integrity_report.json and exits non-zero on
any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.comp8851 import evaluator, protocol  # noqa: E402
from phase2 import data as pdata  # noqa: E402

PHASE_ROOT = {"final": ROOT / "results" / "raw", "smoke": ROOT / "results" / "smoke", "pilot": ROOT / "results" / "pilot"}
_split_cache = {}


def check_run(run_dir: Path, deep: bool) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = {"run_dir": str(run_dir.relative_to(ROOT)), "status": summary.get("status"), "checks": {}}
    if summary.get("status") != "COMPLETE":
        report["checks"]["skipped"] = f"status {summary.get('status')} ({summary.get('failure_reason')})"
        report["ok"] = True
        return report
    c = report["checks"]
    pred_path = run_dir / "predictions.npz"
    c["predictions_present"] = pred_path.exists()
    if not pred_path.exists():
        report["ok"] = False
        return report
    c["predictions_sha256_matches"] = protocol.sha256_file(pred_path) == summary.get("predictions_sha256")
    pred = np.load(pred_path)
    ds_name, ratio = summary["dataset"], summary["ratio"]
    key = (ds_name, ratio)
    if key not in _split_cache:
        ds = pdata.load_canonical_dataset(ds_name)
        arrays, _, digest = pdata.resolve_split(ds, ratio)
        _split_cache[key] = (ds.labels, arrays, digest)
    labels, arrays, digest = _split_cache[key]
    smoke_subset = "smoke_subset" in str(summary.get("configuration", {}).get("smoke_subset_nodes") or "") or \
        bool(summary.get("configuration", {}).get("smoke_subset_nodes"))
    if not smoke_subset:
        c["split_sha256_matches_registry"] = (summary.get("protocol_identity") or {}).get("split_sha256") == digest
        c["valid_ids_and_labels_match_split"] = (np.array_equal(pred["valid_idx"], arrays["valid_idx"])
                                                 and np.array_equal(pred["valid_labels"], labels[arrays["valid_idx"]]))
    threshold, _ = evaluator.select_threshold(pred["valid_labels"], pred["valid_prob"])
    c["threshold_reselected_from_validation"] = abs(threshold - float(pred["threshold"])) < 1e-12 and \
        abs(threshold - float(summary["selected_threshold"])) < 1e-12
    if "test_prob" in pred.files:
        if not smoke_subset:
            c["test_ids_and_labels_match_split"] = (np.array_equal(pred["test_idx"], arrays["test_idx"])
                                                    and np.array_equal(pred["test_labels"], labels[arrays["test_idx"]]))
        recomputed = evaluator.metric_bundle(pred["test_labels"], pred["test_prob"], threshold)
        stored = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        diffs = {k: abs(recomputed[k] - stored[k]) for k in evaluator.METRIC_KEYS}
        c["test_metrics_recomputed_equal"] = max(diffs.values()) <= 1e-12
        report["recomputed_test_auprc"] = recomputed["auprc"]
    ckpts = list((run_dir / "checkpoints").glob("*_best.pt"))
    c["best_checkpoint_present"] = bool(ckpts)
    if ckpts:
        import torch

        state = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        c["best_checkpoint_loads"] = "model_state" in state
    c["recorded_reload_check_passed"] = bool((summary.get("checkpoint_reload_check") or {}).get("passed"))
    if deep and "test_prob" in pred.files:
        c["deep_reprediction_within_1e-5"] = deep_repredict(run_dir, summary, pred)
    c.update({k: bool(v) for k, v in c.items() if isinstance(v, (bool, np.bool_))})
    report["ok"] = all(v is True for v in c.values())
    return report


def deep_repredict(run_dir: Path, summary: dict, pred) -> bool:
    """Rebuild the adapter from the recorded configuration and re-predict."""
    import importlib
    from types import SimpleNamespace

    import torch

    from phase2 import configs
    from phase2.harness import RunContext

    slug = {"SAGAD": "sagad", "SpaceGNN": "spacegnn", "CRoC": "croc"}[summary["model"]]
    sys.path.insert(0, str(ROOT / "models" / slug))
    module = importlib.import_module(f"{slug}_adapter")
    cls = next(v for k, v in vars(module).items() if k.endswith("Adapter") and k != "Adapter")
    cfg = summary["configuration"]
    hp = dict(cfg["hyperparameters"])
    overrides = [f"{k}={json.dumps(v)}" for k, v in hp.items()]
    hparams = configs.resolve_hparams(configs.load_model_config(slug), summary["dataset"], overrides)
    device = torch.device("cuda:0" if torch.cuda.is_available() and "cuda" in str(cfg.get("resolved_device")) else "cpu")
    prep = pdata.prepare(summary["dataset"], summary["ratio"])
    args = SimpleNamespace(seed=summary["train_seed"], eval_batch_size=cfg.get("eval_batch_size"),
                           ratio=summary["ratio"], epochs=cfg.get("epochs"), num_workers=0)
    adapter = cls(RunContext(args, prep, device, hparams, None))
    adapter.setup()
    ckpt = next((run_dir / "checkpoints").glob("*_best.pt"))
    adapter.load_model_state(torch.load(ckpt, map_location=device, weights_only=False)["model_state"])
    replay = adapter.predict(pred["test_idx"])
    from sklearn.metrics import average_precision_score

    noise = float((summary.get("checkpoint_reload_check") or {}).get("repeat_noise_max_abs_diff") or 0.0)
    tol = max(1e-5, 3.0 * noise + 1e-6)
    auprc_diff = abs(average_precision_score(pred["test_labels"], replay)
                     - average_precision_score(pred["test_labels"], pred["test_prob"]))
    return bool(float(np.max(np.abs(replay - pred["test_prob"]))) <= tol and auprc_diff <= 1e-4)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="final", choices=list(PHASE_ROOT))
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args(argv)
    base = Path(args.results_dir).resolve()
    root = base / {"final": "raw", "smoke": "smoke", "pilot": "pilot"}[args.phase]
    reports = [check_run(p.parent, args.deep) for p in sorted(root.rglob("summary.json")) if "_attempts" not in p.parts]
    out = base / "aggregated" / args.phase
    out.mkdir(parents=True, exist_ok=True)
    (out / "integrity_report.json").write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    bad = [r for r in reports if not r["ok"]]
    for r in reports:
        flag = "OK  " if r["ok"] else "FAIL"
        extra = f" AUPRC={r['recomputed_test_auprc']:.6f}" if "recomputed_test_auprc" in r else ""
        print(f"{flag} {r['status']:10s} {r['run_dir']}{extra}")
        if not r["ok"]:
            print("     ", {k: v for k, v in r["checks"].items() if v is not True})
    print(f"{len(reports)} runs checked, {len(bad)} failed integrity checks")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
