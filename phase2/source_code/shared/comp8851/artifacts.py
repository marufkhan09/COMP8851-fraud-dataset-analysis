"""Result-artefact writer for COMP8851 benchmark protocol v4.4.

Master plan section 9.5 fixes the contents of every run directory:

``results/<model>/<dataset>/<track>/<ratio>/seed_<seed>/``

    run_config.yml            effective configuration after defaults + overrides
    protocol_identity.json    provenance and comparability fields
    environment.txt           software stack actually installed
    hardware.json             machine the run executed on
    dataset_manifest.json     canonical source identity
    dataset_view_manifest.json  model-ready representation identity
    split_indices.npz         exact node-ID partition used
    split_summary.json        partition sizes and class balance
    epoch_times.csv           one row per epoch, training and validation timed apart
    validation_metrics.csv    every validation evaluation performed
    test_metrics.json         single final test evaluation
    summary.json              machine-readable roll-up of the whole run
    terminal.log              complete stdout/stderr
    checkpoints/              best-validation checkpoint

A run that lacks any of these is not reportable. :meth:`RunRecorder.finalise`
returns the verification result so a caller can fail loudly rather than
publishing an incomplete record.
"""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import runtime
from .protocol import ProtocolIdentity, config_sha256

REQUIRED_ARTIFACTS = (
    "run_config.yml",
    "protocol_identity.json",
    "environment.txt",
    "hardware.json",
    "dataset_manifest.json",
    "dataset_view_manifest.json",
    "split_indices.npz",
    "split_summary.json",
    "epoch_times.csv",
    "validation_metrics.csv",
    "summary.json",
)

EPOCH_FIELDS = (
    "epoch",
    "train_seconds",
    "validation_seconds",
    "cumulative_seconds",
    "train_loss",
    "validation_auroc",
    "validation_auprc",
    "validation_macro_f1",
    "validation_threshold",
    "is_best_checkpoint",
    "timestamp_utc",
    "notes",
)


def _json_safe(value: Any) -> Any:
    """Coerce numpy and arbitrary objects into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write YAML if PyYAML is available, else fall back to JSON-in-.yml.

    JSON is valid YAML, so the file stays loadable either way.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    try:
        import yaml

        path.write_text(yaml.safe_dump(safe, sort_keys=True, default_flow_style=False),
                        encoding="utf-8")
    except ImportError:
        path.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]],
              fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still create the file with a header so its absence always means
        # "the run did not get that far", never "the writer was skipped".
        fieldnames = list(fieldnames or [])
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        return
    fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fieldnames})


@dataclass
class RunRecorder:
    """Collects and writes the artefact bundle for a single run.

    Typical use::

        recorder = RunRecorder(run_dir, model="CARE-GNN", dataset="yelpchi",
                               track="unified", ratio="TR40", seed=2)
        recorder.begin(config, identity, dataset_manifest, view_manifest)
        recorder.record_split(splits, labels, source)
        ...
        recorder.record_epoch(...)
        recorder.record_validation(...)
        recorder.finalise(test_metrics=...)
    """

    run_dir: Path
    model: str
    dataset: str
    track: str
    ratio: str
    seed: int
    run_id: str = ""
    device: Any = None

    epoch_rows: List[Dict[str, Any]] = field(default_factory=list)
    validation_rows: List[Dict[str, Any]] = field(default_factory=list)
    _config: Dict[str, Any] = field(default_factory=dict)
    _identity: Optional[ProtocolIdentity] = None
    _started_utc: str = ""
    _cumulative_seconds: float = 0.0
    _best: Dict[str, Any] = field(default_factory=dict)
    _extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.model.lower().replace('-', '')}_best.pt"

    @property
    def terminal_log(self) -> Path:
        return self.run_dir / "terminal.log"

    # -- lifecycle --------------------------------------------------------
    def begin(self, config: Dict[str, Any], identity: ProtocolIdentity,
              dataset_manifest: Dict[str, Any],
              view_manifest: Dict[str, Any]) -> None:
        """Write everything that is known before training starts."""
        self._started_utc = runtime.utc_now()
        self._config = dict(config)
        identity.config_sha256 = identity.config_sha256 or config_sha256(self._config)
        self._identity = identity

        write_yaml(self.run_dir / "run_config.yml", self._config)
        write_json(self.run_dir / "protocol_identity.json", identity.to_dict())
        write_json(self.run_dir / "hardware.json", runtime.hardware_metadata(self.device))
        write_json(self.run_dir / "dataset_manifest.json", dataset_manifest)
        write_json(self.run_dir / "dataset_view_manifest.json", view_manifest)

        env = runtime.environment_metadata()
        lines = [f"{key}: {value}" for key, value in sorted(env.items())]
        (self.run_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._extra["environment"] = env

        missing = identity.missing_fields()
        if missing:
            print(f"[protocol] WARNING unset identity fields: {', '.join(missing)}")

    def record_split(self, arrays: Dict[str, np.ndarray], labels: np.ndarray,
                     source: str) -> None:
        """Persist the exact node-ID partition and its summary."""
        np.savez_compressed(
            self.run_dir / "split_indices.npz",
            **{key: np.asarray(value, dtype=np.int64) for key, value in arrays.items()},
        )
        from .splits import split_summary

        summary = split_summary(arrays, labels, source)
        write_json(self.run_dir / "split_summary.json", summary)
        self._extra["split"] = summary

    def record_epoch(self, epoch: int, train_seconds: float, train_loss: float,
                     validation_seconds: float = 0.0,
                     validation_metrics: Optional[Dict[str, float]] = None,
                     is_best: bool = False, notes: str = "") -> None:
        """Append one epoch row. Training and validation time stay separate."""
        self._cumulative_seconds += float(train_seconds) + float(validation_seconds)
        row: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_seconds": float(train_seconds),
            "validation_seconds": float(validation_seconds) if validation_seconds else "",
            "cumulative_seconds": float(self._cumulative_seconds),
            "train_loss": float(train_loss),
            "is_best_checkpoint": bool(is_best),
            "timestamp_utc": runtime.utc_now(),
            "notes": notes,
        }
        if validation_metrics:
            row.update({
                "validation_auroc": float(validation_metrics.get("auroc", float("nan"))),
                "validation_auprc": float(validation_metrics.get("auprc", float("nan"))),
                "validation_macro_f1": float(validation_metrics.get("macro_f1", float("nan"))),
                "validation_threshold": float(validation_metrics.get("threshold", float("nan"))),
            })
        self.epoch_rows.append(row)
        # Flush after every epoch so a crash still leaves usable evidence.
        write_csv(self.run_dir / "epoch_times.csv", self.epoch_rows, EPOCH_FIELDS)

    def record_validation(self, epoch: int, metrics: Dict[str, float],
                          seconds: float, is_best: bool = False) -> None:
        """Append one validation evaluation to ``validation_metrics.csv``."""
        row = {"epoch": int(epoch), "validation_seconds": float(seconds),
               "is_best_checkpoint": bool(is_best), "timestamp_utc": runtime.utc_now()}
        row.update({k: v for k, v in metrics.items()})
        self.validation_rows.append(row)
        write_csv(self.run_dir / "validation_metrics.csv", self.validation_rows)

    def set_best(self, epoch: int, metrics: Dict[str, float], threshold: float,
                 selection_metric: str, selection_value: float) -> None:
        """Record which checkpoint was selected and why."""
        self._best = {
            "best_validation_epoch": int(epoch),
            "selection_metric": selection_metric,
            "best_selection_value": float(selection_value),
            "selected_threshold": float(threshold),
            "best_validation_metrics": dict(metrics),
        }

    def add(self, key: str, value: Any) -> None:
        """Attach an arbitrary extra field to ``summary.json``."""
        self._extra[key] = value

    def finalise(self, test_metrics: Optional[Dict[str, float]] = None,
                 status: str = "COMPLETE", error: Optional[BaseException] = None,
                 timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write ``test_metrics.json`` and ``summary.json``; verify the bundle.

        ``status`` must reflect what actually happened. A failed run is
        recorded as FAILED with its traceback preserved, never quietly dropped.
        """
        if test_metrics is not None:
            write_json(self.run_dir / "test_metrics.json", test_metrics)

        train_times = [float(row["train_seconds"]) for row in self.epoch_rows]
        validation_times = [
            float(row["validation_seconds"]) for row in self.epoch_rows
            if row.get("validation_seconds") not in ("", None)
        ]

        computed_timing: Dict[str, Any] = {
            "training_epochs_completed": len(train_times),
            "total_train_epoch_seconds": float(np.sum(train_times)) if train_times else 0.0,
            "mean_train_epoch_seconds": float(np.mean(train_times)) if train_times else 0.0,
            "median_train_epoch_seconds": float(np.median(train_times)) if train_times else 0.0,
            "std_train_epoch_seconds": (
                float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0
            ),
            "total_validation_seconds": float(np.sum(validation_times)) if validation_times else 0.0,
            "total_train_plus_validation_seconds": float(self._cumulative_seconds),
            "peak_gpu_memory_mb": runtime.peak_memory_mb(self.device),
        }
        if timing:
            computed_timing.update(timing)

        summary: Dict[str, Any] = {
            "schema_version": "2.0",
            "status": status,
            "run_id": self.run_id,
            "model": self.model,
            "dataset": self.dataset,
            "track": self.track,
            "ratio": self.ratio,
            "train_seed": int(self.seed),
            "run_started_utc": self._started_utc,
            "run_finished_utc": runtime.utc_now(),
            "configuration": self._config,
            "protocol_identity": self._identity.to_dict() if self._identity else {},
            "test_metrics": test_metrics,
            "timing": computed_timing,
        }
        summary.update(self._best)
        summary.update(self._extra)

        if error is not None:
            summary["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }

        # summary.json must exist before the bundle can be verified, and the
        # verification result belongs inside it. Write it, verify, then rewrite
        # with the verdict embedded so the stored record is self-describing.
        write_json(self.run_dir / "summary.json", summary)
        verification = self.verify(expect_test=test_metrics is not None)
        summary["artifact_verification"] = verification
        write_json(self.run_dir / "summary.json", summary)
        return verification

    def verify(self, expect_test: bool = True) -> Dict[str, Any]:
        """Check the required artefacts exist and epoch rows match epochs run."""
        required = list(REQUIRED_ARTIFACTS)
        if expect_test:
            required.append("test_metrics.json")
        missing = [name for name in required if not (self.run_dir / name).exists()]

        # terminal.log is written by the capture context, which may still be
        # open when finalise() runs; treat it as advisory.
        advisory = [] if self.terminal_log.exists() else ["terminal.log"]

        checkpoint_present = any(self.checkpoint_dir.glob("*"))
        return {
            "complete": not missing and (checkpoint_present or not expect_test),
            "missing_artifacts": missing,
            "advisory_missing": advisory,
            "checkpoint_present": bool(checkpoint_present),
            "epoch_rows": len(self.epoch_rows),
            "validation_rows": len(self.validation_rows),
        }
