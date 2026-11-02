"""COMP8851 unified split registry.

Split policy (master plan v4.4, section 4.9):

* Split-generation seed is fixed at 2 and never varies by model or seed.
* The base partition is 40% train / 20% validation / 40% test over the
  **eligible** (supervised-labelled) nodes.
* Training subsets are nested: TR10 subset of TR20 subset of TR30 subset of TR40.
* Validation and test node IDs are identical across all four ratios, across all
  training seeds and across all models.
* Nodes dropped from a smaller training pool become ``unused_idx``; they remain
  structurally present in the graph but carry no supervised label for that run.
* Ineligible nodes (Amazon's 3,305 unlabelled users, Elliptic's unknown
  transactions) are excluded from the partition entirely, while still existing
  in the graph for message passing.

Elliptic is the documented exception: its split is chronological, not random.
Validation and test occupy later time steps and only the earlier labelled
training region shrinks across ratios.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .protocol import (
    SPLIT_SEED,
    TEST_FRACTION,
    TRAIN_RATIOS,
    VALIDATION_FRACTION,
    sha256_file,
)

SPLIT_KEYS = ("train_idx", "valid_idx", "test_idx", "unused_idx")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def build_nested_splits(labels: np.ndarray,
                        eligible_idx: Optional[np.ndarray] = None,
                        seed: int = SPLIT_SEED) -> Dict[str, Dict[str, np.ndarray]]:
    """Build nested stratified TR40/TR30/TR20/TR10 splits over eligible nodes.

    ``labels`` is the full-graph label vector; ``eligible_idx`` selects the
    nodes that may take part in supervised learning. Returns a mapping from
    ratio name to arrays of global node IDs.
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if eligible_idx is None:
        eligible = np.arange(len(labels), dtype=np.int64)
    else:
        eligible = np.asarray(eligible_idx, dtype=np.int64)
    eligible_labels = labels[eligible]

    if set(np.unique(eligible_labels).tolist()) != {0, 1}:
        raise ValueError(
            "Eligible nodes must contain both classes with binary labels 0/1; "
            f"found {sorted(set(np.unique(eligible_labels).tolist()))}"
        )

    # Base partition: 40% train pool, 60% held out -> 1/3 validation, 2/3 test,
    # giving the frozen 40/20/40 proportions.
    train40, held_out = train_test_split(
        eligible,
        train_size=TRAIN_RATIOS["TR40"],
        random_state=seed,
        shuffle=True,
        stratify=eligible_labels,
    )
    held_out_labels = labels[held_out]
    valid, test = train_test_split(
        held_out,
        train_size=VALIDATION_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=seed,
        shuffle=True,
        stratify=held_out_labels,
    )

    # Nested training subsets. Each is drawn from the next larger pool so that
    # TR10 is a subset of TR20, and so on.
    train30, _ = train_test_split(
        train40, train_size=0.75, random_state=seed, shuffle=True,
        stratify=labels[train40],
    )
    train20, _ = train_test_split(
        train30, train_size=2.0 / 3.0, random_state=seed, shuffle=True,
        stratify=labels[train30],
    )
    train10, _ = train_test_split(
        train20, train_size=0.50, random_state=seed, shuffle=True,
        stratify=labels[train20],
    )

    pools = {"TR40": train40, "TR30": train30, "TR20": train20, "TR10": train10}
    base_pool = set(train40.tolist())
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for ratio, train in pools.items():
        unused = np.asarray(sorted(base_pool - set(train.tolist())), dtype=np.int64)
        splits[ratio] = {
            "train_idx": np.sort(np.asarray(train, dtype=np.int64)),
            "valid_idx": np.sort(np.asarray(valid, dtype=np.int64)),
            "test_idx": np.sort(np.asarray(test, dtype=np.int64)),
            "unused_idx": unused,
        }
    return splits


def build_chronological_splits(labels: np.ndarray,
                               time_steps: np.ndarray,
                               eligible_idx: Optional[np.ndarray] = None,
                               seed: int = SPLIT_SEED) -> Dict[str, Dict[str, np.ndarray]]:
    """Build the Elliptic chronological splits.

    Time order is preserved: the earliest time steps supply training, the middle
    supplies validation and the latest supplies test. Only the labelled training
    region shrinks across ratios, and it shrinks by dropping the *earliest*
    steps first so the remaining training data stays adjacent to validation.

    Nesting still holds (TR10 subset of TR20 subset of TR30 subset of TR40) and
    validation/test IDs never change.
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    time_steps = np.asarray(time_steps).reshape(-1)
    if eligible_idx is None:
        eligible = np.arange(len(labels), dtype=np.int64)
    else:
        eligible = np.asarray(eligible_idx, dtype=np.int64)

    order = eligible[np.argsort(time_steps[eligible], kind="stable")]
    total = len(order)
    n_train = int(round(total * TRAIN_RATIOS["TR40"]))
    n_valid = int(round(total * VALIDATION_FRACTION))

    train40 = order[:n_train]
    valid = order[n_train:n_train + n_valid]
    test = order[n_train + n_valid:]

    # Shrink the training window from the earliest end.
    pools = {"TR40": train40}
    for ratio in ("TR30", "TR20", "TR10"):
        keep = int(round(total * TRAIN_RATIOS[ratio]))
        pools[ratio] = train40[len(train40) - keep:]

    base_pool = set(train40.tolist())
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for ratio, train in pools.items():
        unused = np.asarray(sorted(base_pool - set(train.tolist())), dtype=np.int64)
        splits[ratio] = {
            "train_idx": np.sort(np.asarray(train, dtype=np.int64)),
            "valid_idx": np.sort(np.asarray(valid, dtype=np.int64)),
            "test_idx": np.sort(np.asarray(test, dtype=np.int64)),
            "unused_idx": unused,
        }
    return splits


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify_splits(splits: Dict[str, Dict[str, np.ndarray]],
                  eligible_idx: np.ndarray,
                  total_nodes: int) -> None:
    """Raise if any split invariant is violated.

    Checks: no duplicates, no overlap between partitions, IDs inside the graph,
    every partition inside the eligible set, exact coverage of the eligible set,
    fixed validation/test IDs across ratios, and training-pool nesting.
    """
    eligible = set(np.asarray(eligible_idx, dtype=np.int64).tolist())
    errors: list[str] = []

    reference_valid = None
    reference_test = None
    for ratio in sorted(splits):
        arrays = splits[ratio]
        missing = set(SPLIT_KEYS) - set(arrays)
        if missing:
            errors.append(f"{ratio}: missing arrays {sorted(missing)}")
            continue

        sets = {name: set(np.asarray(arrays[name]).tolist()) for name in SPLIT_KEYS}
        for name in SPLIT_KEYS:
            values = np.asarray(arrays[name])
            if len(values) != len(sets[name]):
                errors.append(f"{ratio}: duplicate IDs in {name}")
            if values.size and (values.min() < 0 or values.max() >= total_nodes):
                errors.append(f"{ratio}: out-of-range IDs in {name}")
            if not sets[name].issubset(eligible):
                errors.append(f"{ratio}: {name} contains ineligible node IDs")

        names = list(SPLIT_KEYS)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if sets[left] & sets[right]:
                    errors.append(f"{ratio}: overlap between {left} and {right}")

        covered = set().union(*sets.values())
        if covered != eligible:
            errors.append(
                f"{ratio}: partitions cover {len(covered)} eligible nodes, expected {len(eligible)}"
            )

        if not sets["train_idx"]:
            errors.append(f"{ratio}: empty training set")
        if not sets["valid_idx"] or not sets["test_idx"]:
            errors.append(f"{ratio}: empty validation or test set")

        if reference_valid is None:
            reference_valid, reference_test = sets["valid_idx"], sets["test_idx"]
        else:
            if sets["valid_idx"] != reference_valid:
                errors.append(f"{ratio}: validation IDs differ from other ratios")
            if sets["test_idx"] != reference_test:
                errors.append(f"{ratio}: test IDs differ from other ratios")

    present = [r for r in ("TR10", "TR20", "TR30", "TR40") if r in splits]
    for smaller, larger in zip(present, present[1:]):
        small = set(np.asarray(splits[smaller]["train_idx"]).tolist())
        large = set(np.asarray(splits[larger]["train_idx"]).tolist())
        if not small <= large:
            errors.append(f"training pools not nested: {smaller} is not a subset of {larger}")

    if errors:
        raise RuntimeError("Split verification failed:\n  " + "\n  ".join(errors))


def describe_partition(ids: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Summarise one partition: size and class balance."""
    ids = np.asarray(ids, dtype=np.int64)
    total = int(ids.size)
    fraud = int(np.asarray(labels).reshape(-1)[ids].sum()) if total else 0
    return {
        "nodes": total,
        "fraud_nodes": fraud,
        "normal_nodes": total - fraud,
        "fraud_percentage": (100.0 * fraud / total) if total else 0.0,
    }


def split_summary(arrays: Dict[str, np.ndarray], labels: np.ndarray,
                  source: str) -> Dict[str, object]:
    """Build the ``split_summary.json`` payload for one run."""
    return {
        "source": source,
        "train": describe_partition(arrays["train_idx"], labels),
        "validation": describe_partition(arrays["valid_idx"], labels),
        "test": describe_partition(arrays["test_idx"], labels),
        "unused_nodes": int(np.asarray(arrays["unused_idx"]).size),
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def split_filename(dataset: str, ratio: str, seed: int = SPLIT_SEED) -> str:
    """Canonical split filename, e.g. ``yelpchi_TR40_seed2.npz``."""
    return f"{dataset.lower()}_{ratio.upper()}_seed{seed}.npz"


def save_splits(splits: Dict[str, Dict[str, np.ndarray]], output_dir: Path | str,
                dataset: str, labels: np.ndarray, seed: int = SPLIT_SEED,
                extra_manifest: Optional[Dict[str, object]] = None) -> Path:
    """Write split archives plus a manifest, and return the manifest path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).reshape(-1)

    manifest: Dict[str, object] = {
        "dataset": dataset,
        "split_seed": int(seed),
        "convention": (
            "40/20/40 base partition over eligible nodes with nested stratified "
            "TR30/TR20/TR10 training subsets; validation and test IDs fixed"
        ),
        "splits": {},
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    for ratio, arrays in splits.items():
        path = output_dir / split_filename(dataset, ratio, seed)
        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.int64) for k, v in arrays.items()})
        manifest["splits"][ratio] = {  # type: ignore[index]
            "file": path.name,
            "file_sha256": sha256_file(path),
            "train": describe_partition(arrays["train_idx"], labels),
            "validation": describe_partition(arrays["valid_idx"], labels),
            "test": describe_partition(arrays["test_idx"], labels),
            "unused_nodes": int(np.asarray(arrays["unused_idx"]).size),
        }

    manifest_path = output_dir / f"{dataset.lower()}_split_manifest_seed{seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def load_split(path: Path | str) -> Dict[str, np.ndarray]:
    """Load one split archive, tolerating legacy files without ``unused_idx``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"train_idx", "valid_idx", "test_idx"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Split file {path.name} is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(data[name], dtype=np.int64) for name in data.files}
    arrays.setdefault("unused_idx", np.asarray([], dtype=np.int64))
    return {key: arrays[key] for key in SPLIT_KEYS}


def eligible_from_split(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Recover the eligible node set implied by a split archive."""
    union: Iterable[int] = set()
    for key in SPLIT_KEYS:
        union = set(union) | set(np.asarray(arrays[key], dtype=np.int64).tolist())
    return np.asarray(sorted(union), dtype=np.int64)


def assert_no_leakage(train_idx: Sequence[int], valid_idx: Sequence[int],
                      test_idx: Sequence[int]) -> None:
    """Fail loudly if the three supervised partitions intersect."""
    train, valid, test = (set(np.asarray(x).tolist()) for x in (train_idx, valid_idx, test_idx))
    overlaps = {
        "train/validation": train & valid,
        "train/test": train & test,
        "validation/test": valid & test,
    }
    offending = {name: len(ids) for name, ids in overlaps.items() if ids}
    if offending:
        raise RuntimeError(f"Split leakage detected: {offending}")
