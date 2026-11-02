"""Model-agnostic run harness for SAGAD, SpaceGNN and CRoC under protocol vast-v4.4.

One invocation trains one (model, dataset, ratio, seed) cell. The harness owns
everything that must be identical across models:

* data and frozen splits (``phase2.data``), with labels isolated so model code
  never sees a validation or test label;
* the epoch loop, validation-AUPRC checkpoint selection, validation-only
  threshold selection, patience-based early stopping and the time cap, all via
  ``shared.comp8851.evaluator`` and ``protocol`` unchanged;
* the artefact bundle (``shared.comp8851.artifacts.RunRecorder``) plus the
  phase-2 additions: ``predictions.npz`` (validation and test probabilities
  with node IDs and labels), atomic best/last checkpoints, resume, a
  checkpoint-reload reproduction check, peak GPU memory and peak RSS;
* overwrite protection: a COMPLETE run is never re-run without ``--force``,
  and an incomplete run directory is archived (never deleted) before a fresh
  start.

Each model supplies an :class:`Adapter` with the authors' architecture, loss,
sampling and inference.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

# Deterministic cuBLAS must be configured before torch initialises CUDA. This
# changes no arithmetic; it only makes cuBLAS workspace use reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("DGLBACKEND", "pytorch")

import numpy as np  # noqa: E402

import shared.comp8851 as comp8851  # noqa: E402
from shared.comp8851 import artifacts, evaluator, protocol, runtime  # noqa: E402

from phase2 import __version__ as phase2_version  # noqa: E402
from phase2 import configs as pconfigs  # noqa: E402
from phase2 import data as pdata  # noqa: E402

runtime.apply_thread_caps()

import torch  # noqa: E402

RESULTS_ROOT = BUNDLE_ROOT / "results" / "raw"
TUNING_ROOT = BUNDLE_ROOT / "results" / "tuning"
ATTEMPTS_DIR = "_attempts"
RELOAD_TOLERANCE = 1e-5


class StopRun(Exception):
    """Raised by an adapter to abort with a recorded, non-crash reason."""


class Adapter:
    """Interface implemented by each model. One instance per run."""

    name = ""          # display name, e.g. "SAGAD"
    slug = ""          # directory slug, e.g. "sagad"
    repo = ""          # key in third_party/manifest.json
    view = "homogeneous"  # dataset view recorded in the manifest

    def __init__(self, ctx: "RunContext"):
        self.ctx = ctx

    # Called once. Builds graph views, precomputations, model and optimiser.
    def setup(self) -> None:
        raise NotImplementedError

    # One training epoch in the authors' own definition; returns mean loss.
    def train_epoch(self, epoch: int) -> float:
        raise NotImplementedError

    # Fraud probability (positive class 1) for the given global node IDs.
    def predict(self, node_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def model_state(self) -> Dict[str, Any]:
        return {"model": self.model.state_dict()}

    def load_model_state(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])

    def optimizer_state(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict()}

    def load_optimizer_state(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])

    # Rebuilds the network only (no precomputation) for the reload check.
    def fresh_copy_for_reload(self) -> "Adapter":
        raise NotImplementedError

    def invalidate_cache(self) -> None:
        """Drop any cached inference (adapters that cache all-node scores)."""

    def describe(self) -> Dict[str, Any]:
        return {}

    def num_parameters(self) -> Dict[str, int]:
        return runtime.count_parameters(self.model)


class RunContext:
    def __init__(self, args, prepared: pdata.PreparedData, device, hparams: Dict[str, Any],
                 recorder: Optional[artifacts.RunRecorder]):
        self.args = args
        self.data = prepared
        self.device = device
        self.hp = hparams
        self.recorder = recorder
        self.seed = int(args.seed)
        self.notes: Dict[str, Any] = {}

    def note(self, key: str, value: Any) -> None:
        """Attach a documented adaptation/observation to summary.json."""
        self.notes[key] = value
        print(f"[note] {key}: {value}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser(adapter_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{adapter_cls.name} runner (COMP8851 phase 2)")
    parser.add_argument("--dataset", required=True, choices=list(protocol.DATASETS))
    parser.add_argument("--data-path", default=None, help="canonical .npz or its directory (default data/canonical)")
    parser.add_argument("--split-dir", default=str(pdata.DEFAULT_SPLIT_DIR))
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--ratio", default="TR40", choices=list(protocol.TRAIN_RATIOS))
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--track", default="unified", choices=list(protocol.TRACKS))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--config", default=None, help="model YAML (default configs/<slug>.yaml)")
    parser.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                        help="hyperparameter override, YAML-typed; repeatable")
    parser.add_argument("--budget", default="protocol", choices=["protocol", "official"],
                        help="'protocol' = unified 100 epochs / patience 20 (as for the previous 10 "
                             "models); 'official' = the authors' epoch budget (sensitivity track)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument("--selection-metric", default=protocol.SELECTION_METRIC,
                        choices=["auprc", "auroc", "macro_f1"])
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="wall-clock cap for training; reaching it marks the run INCOMPLETE")
    parser.add_argument("--skip-test", action="store_true",
                        help="never load, score or write test predictions (tuning trials)")
    parser.add_argument("--run-mode", default="controlled")
    parser.add_argument("--smoke", action="store_true", help="feasibility run: few epochs, not evidence")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-subset-nodes", type=int, default=0,
                        help="smoke only: restrict to an induced subgraph of this many nodes")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--hardware-profile-id", default=None)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoints/last.pt if present")
    parser.add_argument("--force", action="store_true", help="re-run even if a COMPLETE run exists (old run archived)")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="data-loader workers where the official code uses them (default: official value)")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="inference batch size; does not change outputs (eval mode)")
    parser.add_argument("--inject-fault", default=None, choices=[None, "nan_loss", "crash_after_epoch1", "oom"],
                        help="testing only: deliberately trigger a failure path")
    return parser


def parse_args(adapter_cls, argv=None):
    args = build_parser(adapter_cls).parse_args(argv)
    return args


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def atomic_torch_save(obj: Any, path: Path) -> None:
    """Write to a temporary file in the same directory, fsync, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_dgl(seed: int) -> None:
    try:
        import dgl

        dgl.seed(int(seed) % (2 ** 31 - 1))
    except Exception:  # noqa: BLE001 - DGL optional for pure-torch paths
        pass


def peak_rss_mb() -> float:
    try:
        import resource  # POSIX only

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(value) / 1024.0  # Linux reports KiB
    except Exception:  # noqa: BLE001
        try:
            import psutil

            return float(psutil.Process().memory_info().rss) / (1024 ** 2)
        except Exception:  # noqa: BLE001
            return float("nan")


def check_probabilities(probs: np.ndarray, expected: int, where: str) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.shape[0] != expected:
        raise RuntimeError(f"{where}: {probs.shape[0]} predictions for {expected} nodes")
    if not np.isfinite(probs).all():
        raise RuntimeError(f"{where}: non-finite predictions")
    if probs.min() < -1e-6 or probs.max() > 1 + 1e-6:
        raise RuntimeError(f"{where}: probabilities outside [0, 1] ({probs.min()}, {probs.max()})")
    return np.clip(probs, 0.0, 1.0)


def archive_existing(run_dir: Path, results_root: Path, slug: str, dataset: str, ratio: str, seed: int) -> Optional[Path]:
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    target = (results_root / ATTEMPTS_DIR / slug / dataset / ratio.lower() / f"seed_{seed}" / runtime.utc_stamp())
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    print(f"[safety] previous incomplete attempt archived to {target}")
    return target


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    import csv

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def smoke_subset(prepared: pdata.PreparedData, n_nodes: int, seed: int) -> pdata.PreparedData:
    """Smoke tests only: induced subgraph on a random node sample (documented)."""
    from shared.comp8851.datasets import CanonicalDataset

    ds = prepared.dataset
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(ds.num_nodes, size=min(n_nodes, ds.num_nodes), replace=False))
    remap = -np.ones(ds.num_nodes, dtype=np.int64)
    remap[keep] = np.arange(keep.size)
    relations = {}
    for name, matrix in ds.relations.items():
        relations[name] = matrix.tocsr()[keep][:, keep].tocsr()
    eligible = remap[np.intersect1d(ds.eligible_idx, keep)]
    sub = CanonicalDataset(name=ds.name, features=ds.features[keep], labels=ds.labels[keep],
                           eligible_idx=np.sort(eligible), relations=relations,
                           time_steps=None if ds.time_steps is None else ds.time_steps[keep],
                           source_files=ds.source_files, source_sha256=ds.source_sha256,
                           notes={**ds.notes, "smoke_subset_nodes": int(keep.size)})
    split = {}
    for key, ids in prepared.split.items():
        ids = np.asarray(ids)
        split[key] = np.sort(remap[ids[remap[ids] >= 0]])
    for key in ("train_idx", "valid_idx", "test_idx"):
        labels = sub.labels[split[key]]
        if split[key].size == 0 or labels.min() == labels.max():
            raise StopRun(f"smoke subset of {n_nodes} nodes leaves {key} without both classes; use more nodes")
    return pdata.PreparedData(dataset=sub, split=split, split_path=prepared.split_path + f" [smoke subset {keep.size}]",
                              split_sha256=prepared.split_sha256,
                              train_labels_only=pdata.isolate_labels(sub.labels, split["train_idx"]))


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def main(adapter_cls, argv=None) -> int:
    args = parse_args(adapter_cls, argv)
    if args.smoke:
        args.run_mode = "smoke"
    results_root = Path(args.results_root)
    run_dir = protocol.result_dir(results_root, adapter_cls.slug, args.dataset, args.track, args.ratio, args.seed)

    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("status") == "COMPLETE":
            print(f"[skip] {run_dir} is already COMPLETE; pass --force to re-run (old run is archived).")
            return 0
    last_ckpt = run_dir / "checkpoints" / "last.pt"
    if args.force or not (args.resume and last_ckpt.exists()):
        archive_existing(run_dir, results_root, adapter_cls.slug, args.dataset, args.ratio, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    with runtime.capture_terminal(run_dir / "terminal.log"):
        return _execute(adapter_cls, args, run_dir)


def _execute(adapter_cls, args, run_dir: Path) -> int:
    started_wall = time.perf_counter()
    device = runtime.resolve_device(no_cuda=args.no_cuda, require_cuda=args.require_cuda)
    gpu = runtime.gpu_profile_id(device)
    run_id = protocol.run_name(adapter_cls.slug, args.dataset, args.track, args.ratio, args.seed, gpu, runtime.utc_stamp())
    recorder = artifacts.RunRecorder(run_dir=run_dir, model=adapter_cls.name, dataset=args.dataset,
                                     track=args.track, ratio=args.ratio, seed=args.seed,
                                     run_id=run_id, device=device)
    print("=" * 78)
    print(f"{adapter_cls.name} | {args.dataset} | {args.track} | {args.ratio} | seed {args.seed} | {args.run_mode}")
    print("=" * 78)
    print(f"Run ID: {run_id}   device: {device}   protocol: {protocol.PROTOCOL_VERSION}")
    if args.smoke:
        print("MODE: SMOKE TEST - feasibility only; metrics are NOT benchmark evidence")

    status, error, test_metrics = "COMPLETE", None, None
    stop_reason, failure_reason = "", ""
    timing: Dict[str, Any] = {}
    ctx = None
    try:
        # ---------------- data ----------------
        t0 = time.perf_counter()
        prepared = pdata.prepare(args.dataset, args.ratio, args.data_path, args.split_dir, args.split_path)
        if args.smoke and args.smoke_subset_nodes:
            prepared = smoke_subset(prepared, args.smoke_subset_nodes, args.seed)
            print(f"[smoke] induced subgraph of {prepared.dataset.num_nodes:,} nodes (documented subset)")
        timing["data_load_seconds"] = time.perf_counter() - t0
        ds = prepared.dataset
        print(f"Data: {ds.num_nodes:,} nodes, {ds.num_features} features, relations {sorted(ds.relations)}")
        print(f"Split: train {len(prepared.train_idx):,} | valid {len(prepared.valid_idx):,} | "
              f"test {len(prepared.test_idx):,}  ({Path(prepared.split_path).name}, sha256 {prepared.split_sha256[:16]})")
        recorder.record_split(prepared.split, ds.labels, prepared.split_path)
        labels_all = ds.labels  # held by the harness only; never handed to the adapter
        y_valid = labels_all[prepared.valid_idx]

        # ---------------- configuration ----------------
        model_cfg = pconfigs.load_model_config(adapter_cls.slug, args.config)
        hparams = pconfigs.resolve_hparams(model_cfg, args.dataset, args.hp)
        budget = pconfigs.resolve_budget(model_cfg, args.dataset, args.budget, args.epochs, args.patience)
        if args.smoke:
            budget = {"epochs": min(budget["epochs"], args.smoke_epochs), "patience": 0,
                      "source": f"smoke ({args.smoke_epochs} epochs)"}
        if "valid_every" in budget:
            args.valid_every = int(budget["valid_every"])
        print(f"Hyperparameters ({hparams.get('_source', '?')}): "
              f"{ {k: v for k, v in hparams.items() if not k.startswith('_')} }")
        print(f"Budget: {budget}")

        runtime.seed_everything(args.seed)
        seed_dgl(args.seed)
        ctx = RunContext(args, prepared, device, hparams, recorder)
        adapter = adapter_cls(ctx)

        t0 = time.perf_counter()
        runtime.reset_peak_memory(device)
        adapter.setup()
        timing["setup_seconds"] = time.perf_counter() - t0
        params = adapter.num_parameters()
        print(f"Setup done in {timing['setup_seconds']:.1f}s; parameters {params}")

        from phase2 import official

        config = {k: v for k, v in vars(args).items() if k not in ("hp",)}
        config.update({
            "model": adapter_cls.name,
            "hyperparameters": {k: v for k, v in hparams.items() if not k.startswith("_")},
            "hyperparameter_source": hparams.get("_source"),
            "budget": budget,
            "official_implementation": official.provenance(adapter_cls.repo),
            "adapter": adapter.describe(),
            "phase2_version": phase2_version,
            "resolved_device": str(device),
            **params,
        })
        cfg_hash = protocol.config_sha256({k: v for k, v in config.items()
                                           if k not in ("resume", "force", "results_root", "max_minutes",
                                                        "num_workers", "checkpoint_every",
                                                        "inject_fault", "hardware_profile_id")})
        view = ds.view_manifest(adapter_cls.view if adapter_cls.view in ("relation", "homogeneous") else "homogeneous")
        identity = protocol.ProtocolIdentity(
            hardware_profile_id=args.hardware_profile_id or gpu,
            model_source_commit=official.commit(adapter_cls.repo),
            dataset_manifest_version="comp8851-canonical-v1",
            source_sha256=next(iter(ds.source_sha256.values()), None) or "canonical-cache",
            dataset_view_id=view["dataset_view_id"],
            view_sha256=view["view_sha256"],
            split_registry_version=f"seed{protocol.SPLIT_SEED}",
            split_sha256=prepared.split_sha256,
            evaluator_commit=f"shared.comp8851=={comp8851.__version__}",
            adapter_version=f"phase2-{adapter_cls.slug}-{phase2_version}",
            config_sha256=cfg_hash,
            selection_metric=args.selection_metric,
            max_epochs=int(budget["epochs"]),
            early_stopping_patience=int(budget["patience"]),
        )
        recorder.begin(config, identity, ds.manifest(), view)

        # ---------------- resume ----------------
        ckpt_dir = recorder.checkpoint_dir
        last_path = ckpt_dir / "last.pt"
        best_path = recorder.checkpoint_path
        start_epoch, best = 0, {"selection": float("-inf"), "epoch": -1, "threshold": 0.5, "metrics": {}}
        no_improve = 0
        if args.resume and last_path.exists():
            state = torch.load(last_path, map_location=device, weights_only=False)
            if state.get("config_sha256") != cfg_hash:
                raise RuntimeError("resume refused: configuration differs from the interrupted run")
            adapter.load_model_state(state["model_state"])
            adapter.load_optimizer_state(state["optimizer_state"])
            set_rng_state(state["rng"])
            start_epoch = int(state["epoch"]) + 1
            best = state["best"]
            no_improve = int(state["no_improve"])
            recorder.epoch_rows = [r for r in read_csv_rows(run_dir / "epoch_times.csv") if int(r["epoch"]) < start_epoch]
            recorder.validation_rows = [r for r in read_csv_rows(run_dir / "validation_metrics.csv") if int(r["epoch"]) < start_epoch]
            recorder._cumulative_seconds = float(recorder.epoch_rows[-1]["cumulative_seconds"]) if recorder.epoch_rows else 0.0
            recorder.add("resumed_from_epoch", start_epoch)
            print(f"[resume] continuing at epoch {start_epoch + 1}; best so far epoch {best['epoch'] + 1} "
                  f"({args.selection_metric} {best['selection']:.6f})")

        # ---------------- training ----------------
        deadline = time.perf_counter() + args.max_minutes * 60 if args.max_minutes else None
        epochs, patience = int(budget["epochs"]), int(budget["patience"])
        epoch = start_epoch - 1
        fit_started = time.perf_counter()
        for epoch in range(start_epoch, epochs):
            seed_dgl(args.seed * 100003 + epoch)  # per-epoch DGL stream: resumable, reproducible
            runtime.synchronise(device)
            t_train = time.perf_counter()
            loss = float(adapter.train_epoch(epoch))
            if args.inject_fault == "nan_loss":
                loss = float("nan")
            if args.inject_fault == "oom":
                raise torch.cuda.OutOfMemoryError("[fault injection] simulated CUDA out of memory")
            runtime.synchronise(device)
            train_seconds = time.perf_counter() - t_train
            if not np.isfinite(loss):
                failure_reason = "NON_FINITE_LOSS"
                raise StopRun(f"training loss became non-finite at epoch {epoch + 1}: {loss}")

            validation_metrics, validation_seconds, is_best = None, 0.0, False
            if epoch == start_epoch or (epoch + 1) % args.valid_every == 0 or epoch == epochs - 1:
                t_val = time.perf_counter()
                probs = check_probabilities(adapter.predict(prepared.valid_idx), len(prepared.valid_idx), "validation")
                runtime.synchronise(device)
                validation_seconds = time.perf_counter() - t_val
                validation_metrics = evaluator.evaluate_validation(y_valid, probs)
                selection = evaluator.selection_value(validation_metrics, args.selection_metric)
                if selection > best["selection"] + 1e-12:
                    best = {"selection": selection, "epoch": epoch,
                            "threshold": float(validation_metrics["threshold"]),
                            "metrics": dict(validation_metrics)}
                    is_best = True
                    no_improve = 0
                    atomic_torch_save({"model_state": adapter.model_state(), "epoch": epoch,
                                       "config_sha256": cfg_hash}, best_path)
                else:
                    no_improve += args.valid_every
                recorder.record_validation(epoch, validation_metrics, validation_seconds, is_best)
                print(f"epoch {epoch + 1:3d}/{epochs} | loss {loss:.6f} | train {train_seconds:.2f}s | "
                      f"val AUPRC {validation_metrics['auprc']:.6f} AUROC {validation_metrics['auroc']:.6f} | "
                      f"best epoch {best['epoch'] + 1}")
            else:
                print(f"epoch {epoch + 1:3d}/{epochs} | loss {loss:.6f} | train {train_seconds:.2f}s")
            recorder.record_epoch(epoch, train_seconds, loss, validation_seconds, validation_metrics, is_best)

            if (epoch + 1) % args.checkpoint_every == 0 or epoch == epochs - 1:
                atomic_torch_save({"model_state": adapter.model_state(), "optimizer_state": adapter.optimizer_state(),
                                   "rng": rng_state(), "epoch": epoch, "best": best, "no_improve": no_improve,
                                   "config_sha256": cfg_hash}, last_path)
            if args.inject_fault == "crash_after_epoch1" and epoch == start_epoch:
                raise SystemExit("[fault injection] simulated interruption after one epoch")
            if patience and no_improve >= patience:
                stop_reason = f"early stopping: no validation improvement for {patience} epochs"
                break
            if deadline and time.perf_counter() > deadline:
                stop_reason = f"time cap of {args.max_minutes} minutes reached"
                status = "INCOMPLETE"
                break
        else:
            stop_reason = f"epoch budget of {epochs} exhausted"
        timing["fit_wall_seconds_including_validation"] = time.perf_counter() - fit_started
        print(f"Stopped: {stop_reason}")
        if best["epoch"] < 0:
            raise StopRun("no validation evaluation completed; nothing to select")

        # ---------------- test (once, best checkpoint) ----------------
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        adapter.load_model_state(best_state["model_state"])
        predictions: Dict[str, Any] = {
            "valid_idx": prepared.valid_idx, "valid_labels": y_valid,
            "valid_prob": check_probabilities(adapter.predict(prepared.valid_idx), len(prepared.valid_idx), "validation(best)"),
            "threshold": np.asarray(best["threshold"]), "best_epoch": np.asarray(best["epoch"]),
            "positive_label": np.asarray(evaluator.POSITIVE_LABEL),
        }
        reload_check: Dict[str, Any] = {}
        if args.skip_test:
            print("--skip-test: the test set was not loaded, scored or written (tuning trial).")
            recorder.add("test_skipped", True)
            recorder.add("evidence_class", "tuning_trial")
            check_ids = prepared.valid_idx
        else:
            t_inf = time.perf_counter()
            test_prob = check_probabilities(adapter.predict(prepared.test_idx), len(prepared.test_idx), "test")
            inference_seconds = time.perf_counter() - t_inf
            y_test = labels_all[prepared.test_idx]
            test_metrics = evaluator.metric_bundle(y_test, test_prob, best["threshold"])
            test_metrics["inference_seconds"] = inference_seconds
            test_metrics["test_prevalence"] = float(y_test.mean())
            test_metrics["auprc_lift_over_prevalence"] = float(test_metrics["auprc"] / max(y_test.mean(), 1e-12))
            if args.smoke:
                test_metrics["evidence_class"] = "feasibility_only"
            predictions.update({"test_idx": prepared.test_idx, "test_labels": y_test, "test_prob": test_prob})
            check_ids = prepared.test_idx
        pred_path = run_dir / "predictions.npz"
        np.savez_compressed(pred_path, **{k: np.asarray(v) for k, v in predictions.items()})
        recorder.add("predictions_file", pred_path.name)
        recorder.add("predictions_sha256", protocol.sha256_file(pred_path))

        # Checkpoint round trip. (1) Integrity: a freshly built network loaded
        # from disk must hold bit-identical parameters. (2) Reproduction: its
        # predictions must match the saved ones within the model's own repeat
        # noise (same model, same inputs, predicted twice), and AUPRC within
        # 1e-4. SpaceGNN's hyperbolic head amplifies float32 reduction-order
        # noise, so a fixed 1e-5 prediction tolerance would test the
        # arithmetic, not the checkpoint.
        reference = predictions["test_prob"] if not args.skip_test else predictions["valid_prob"]
        check_labels = labels_all[check_ids]
        adapter.invalidate_cache()  # the repeat must genuinely recompute
        repeat = check_probabilities(adapter.predict(check_ids), len(check_ids), "repeat")
        repeat_noise = float(np.max(np.abs(repeat - reference)))
        saved_state = torch.load(best_path, map_location=device, weights_only=False)["model_state"]
        twin = adapter.fresh_copy_for_reload()
        twin.load_model_state(saved_state)

        def _flatten(state):
            out = {}
            for group, sub in state.items():
                for name, tensor in (sub.items() if isinstance(sub, dict) else []):
                    out[f"{group}.{name}"] = tensor
            return out

        saved_flat, twin_flat = _flatten(saved_state), _flatten(twin.model_state())
        params_identical = saved_flat.keys() == twin_flat.keys() and all(
            torch.equal(saved_flat[k].detach().cpu(), twin_flat[k].detach().cpu()) for k in saved_flat)
        replay = check_probabilities(twin.predict(check_ids), len(check_ids), "reload")
        max_diff = float(np.max(np.abs(replay - reference)))
        from sklearn.metrics import average_precision_score
        auprc_diff = float(abs(average_precision_score(check_labels, replay) -
                               average_precision_score(check_labels, reference)))
        tolerance = max(RELOAD_TOLERANCE, 3.0 * repeat_noise + 1e-6)
        reload_check = {"checked_nodes": int(len(check_ids)), "parameters_bit_identical": bool(params_identical),
                        "max_abs_diff": max_diff, "repeat_noise_max_abs_diff": repeat_noise,
                        "prediction_tolerance": tolerance, "auprc_abs_diff": auprc_diff, "auprc_tolerance": 1e-4,
                        "passed": bool(params_identical and max_diff <= tolerance and auprc_diff <= 1e-4)}
        print(f"Checkpoint reload check: parameters identical {params_identical}; max |diff| {max_diff:.3e} "
              f"(repeat noise {repeat_noise:.3e}, tol {tolerance:.3e}); |dAUPRC| {auprc_diff:.2e} -> "
              f"{'PASS' if reload_check['passed'] else 'FAIL'}")
        recorder.add("checkpoint_reload_check", reload_check)
        if not reload_check["passed"]:
            status = "FAILED"
            failure_reason = "CHECKPOINT_RELOAD_MISMATCH"

        recorder.set_best(best["epoch"], best["metrics"], best["threshold"], args.selection_metric, best["selection"])
        if test_metrics is not None:
            print("-" * 78)
            for key in evaluator.METRIC_KEYS:
                print(f"test {key}: {test_metrics[key]:.6f}")
            print(f"test prevalence {test_metrics['test_prevalence']:.4f} | AUPRC lift {test_metrics['auprc_lift_over_prevalence']:.2f}")
        print(f"threshold {best['threshold']:.2f} (validation Macro-F1) | best epoch {best['epoch'] + 1}")
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        status, error, failure_reason = "FAILED", exc, "CUDA_OOM"
        print(f"RUN FAILED (CUDA out of memory): {exc}")
        traceback.print_exc()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    except StopRun as exc:
        status, error = "FAILED", exc
        failure_reason = failure_reason or "STOPPED"
        print(f"RUN FAILED: {exc}")
    except KeyboardInterrupt as exc:
        status, error, failure_reason = "INCOMPLETE", exc, "INTERRUPTED"
        print("RUN INTERRUPTED: resume with --resume")
    except SystemExit as exc:
        status, error, failure_reason = "INCOMPLETE", exc, "INTERRUPTED"
        print(f"RUN INTERRUPTED: {exc}; resume with --resume")
    except BaseException as exc:  # noqa: BLE001 - every failure is recorded
        status, error = "FAILED", exc
        failure_reason = failure_reason or type(exc).__name__
        print(f"RUN FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    timing["peak_rss_mb"] = peak_rss_mb()
    timing["total_wall_seconds"] = time.perf_counter() - started_wall
    recorder.add("stop_reason", stop_reason)
    recorder.add("failure_reason", failure_reason)
    recorder.add("run_mode", args.run_mode)
    recorder.add("phase2_status_vocabulary", "COMPLETE | INCOMPLETE | FAILED")
    if ctx is not None:
        recorder.add("adapter_notes", ctx.notes)
    verification = recorder.finalise(test_metrics=test_metrics, status=status, error=error, timing=timing)
    print("-" * 78)
    print(f"Artefacts: {run_dir}")
    print(f"Artefact bundle complete: {verification['complete']}  missing: {verification['missing_artifacts']}")
    print(f"RUN STATUS: {status}{' (' + failure_reason + ')' if failure_reason else ''}")
    return 0 if status == "COMPLETE" else (3 if status == "INCOMPLETE" else 1)
