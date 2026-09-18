"""Protocol identity and constants for COMP8851 benchmark protocol v4.4.

Every controlled run must record which protocol it was executed under. The
constants here are the machine-readable form of the rules frozen in
``COMP8851_VastAI_Master_Plan_FINAL_v4.4.docx``.

Nothing in this module may be changed for a single model or dataset. A change
here is a protocol amendment and requires a dated entry in the decision
register plus a re-run of every affected comparison.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------
# Frozen protocol constants (master plan v4.4, sections 7.3, 8.1 and 10)
# --------------------------------------------------------------------------

PROTOCOL_VERSION = "vast-v4.4"

#: Split-generation seed. Fixed for every model and dataset; never varies.
SPLIT_SEED = 2

#: Training seeds for final unified runs (weight init, dropout, sampling).
TRAIN_SEEDS = (2, 42, 72)

#: Controlled training ratios. Validation and test IDs are identical across all
#: four; only the labelled training pool shrinks. TR10 < TR20 < TR30 < TR40.
TRAIN_RATIOS = {
    "TR40": 0.40,
    "TR30": 0.30,
    "TR20": 0.20,
    "TR10": 0.10,
}

#: Fixed validation and test fractions of the whole eligible node set.
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.40

#: Unified stopping rule.
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20

#: Tuning budget. This is a hard ceiling: no model-dataset pair may exceed it,
#: and every pair must receive the same budget so neither model is advantaged.
MAX_TUNING_TRIALS = 12

#: Tuning runs on TR40 with train seed 2 only, and never evaluates test.
TUNING_RATIO = "TR40"
TUNING_SEED = 2

#: Controlled GPU class for the benchmark. Recorded in every run and checked at
#: launch; a mismatch is reported, never silently accepted.
#:
#: Master Plan v4.4 (12 September 2026) supersedes the earlier run guide and
#: locks the final benchmark to Vast.ai RTX A6000 48 GB, with A100 80 GB as the
#: sanctioned fallback, after the four-machine calibration bake-off (§3).
#: Earlier Kaggle Tesla T4 runs stay valid as pilot evidence, but v4.4 forbids
#: pooling their timings with A6000 timings in the hardware-normalised runtime
#: comparison, so they are tracked separately rather than merged.
CONTROLLED_GPU = "NVIDIA RTX A6000"
CONTROLLED_GPU_FALLBACKS = ("NVIDIA A100",)
PILOT_GPUS = ("Tesla T4",)

#: Checkpoint selection metric under protocol v4.4. NOT AUROC -- the team
#: pre-registered validation AUPRC for the imbalanced fraud task.
SELECTION_METRIC = "auprc"

#: Threshold-selection rule: sweep validation thresholds 0.01..0.99 in 0.01
#: steps, maximise Macro-F1, tie-break by fraud recall then closeness to 0.5.
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.01
THRESHOLD_OBJECTIVE = "macro_f1"

#: CPU thread cap applied identically on every benchmark machine.
CPU_THREAD_CAP = 8

#: The six datasets in scope. No substitutions, no additions.
DATASETS = ("yelpchi", "amazon", "tfinance", "tsocial", "elliptic", "fdcompcn")

#: Experimental tracks. These must never be merged in reporting.
TRACKS = ("author", "unified")

#: Compatibility statuses from master plan section 5.
COMPATIBILITY_STATUSES = (
    "PASS",
    "PASS_WITH_SHARED_ADAPTER",
    "CONDITIONAL",
    "FAILED_TECHNICAL",
    "NOT_ELIGIBLE",
)

#: Run status values used by the resumable run ledger.
RUN_STATUSES = ("PENDING", "RUNNING", "COMPLETE", "FAILED", "SKIPPED")

#: Datasets whose split is chronological rather than random-stratified.
CHRONOLOGICAL_DATASETS = ("elliptic",)


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 hex digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def config_sha256(config: Dict[str, Any]) -> str:
    """Hash a fully-resolved configuration dictionary.

    The hash is computed after defaults and command-line overrides have been
    merged, so two runs sharing a ``config_sha256`` used byte-identical
    effective hyperparameters.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(canonical.encode("utf-8"))


def git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Return the current 40-character git commit SHA, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


@dataclass
class ProtocolIdentity:
    """The identity fields required in every controlled run record.

    Master plan v4.4 section 9.6 makes each of these mandatory so that any two
    result rows can be checked for comparability without re-reading the code.
    """

    protocol_version: str = PROTOCOL_VERSION
    hardware_profile_id: Optional[str] = None
    container_digest: Optional[str] = None
    model_source_commit: Optional[str] = None
    dataset_manifest_version: Optional[str] = None
    source_sha256: Optional[str] = None
    dataset_view_id: Optional[str] = None
    view_sha256: Optional[str] = None
    split_registry_version: Optional[str] = None
    split_sha256: Optional[str] = None
    evaluator_commit: Optional[str] = None
    adapter_version: Optional[str] = None
    adapter_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    selection_metric: str = SELECTION_METRIC
    threshold_rule: str = (
        f"validation {THRESHOLD_OBJECTIVE} swept {THRESHOLD_MIN}-{THRESHOLD_MAX} "
        f"step {THRESHOLD_STEP}; tie-break fraud recall then closeness to 0.5"
    )
    max_epochs: int = MAX_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    split_seed: int = SPLIT_SEED
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def missing_fields(self) -> list[str]:
        """Return identity fields that are still unset.

        Used by the pre-run checklist: a final controlled run should not be
        reported while required provenance fields are empty.
        """
        required = [
            "hardware_profile_id",
            "model_source_commit",
            "source_sha256",
            "dataset_view_id",
            "split_sha256",
            "config_sha256",
        ]
        return [name for name in required if getattr(self, name) in (None, "")]


def run_name(model: str, dataset: str, track: str, ratio: str, seed: int,
             gpu: str, timestamp: str) -> str:
    """Build the canonical run name from master plan v4.4 section 9.2.

    Example: ``pcgnn_yelpchi_unified_tr40_seed2_a6000_20260912T093000Z``
    """
    safe_gpu = "".join(ch if ch.isalnum() else "-" for ch in gpu.lower()).strip("-")
    return (
        f"{model.lower()}_{dataset.lower()}_{track.lower()}_{ratio.lower()}"
        f"_seed{seed}_{safe_gpu}_{timestamp}"
    )


def result_dir(results_root: Path | str, model: str, dataset: str, track: str,
               ratio: str, seed: int) -> Path:
    """Return the required result directory from master plan v4.4 section 9.5.

    ``results/<model>/<dataset>/<track>/<ratio>/seed_<seed>/``
    """
    return (
        Path(results_root)
        / model.lower()
        / dataset.lower()
        / track.lower()
        / ratio.lower()
        / f"seed_{seed}"
    )
