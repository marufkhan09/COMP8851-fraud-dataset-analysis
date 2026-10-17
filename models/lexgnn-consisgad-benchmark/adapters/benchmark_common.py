from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import dgl
import numpy as np
import scipy.io
import scipy.sparse
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


AMAZON_RELATIONS = ("net_upu", "net_usu", "net_uvu")
YELPCHI_RELATIONS = ("net_rsr", "net_rtr", "net_rur")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_amazon_graph(mat_path: Path, normalize_for_lex: bool) -> dgl.DGLHeteroGraph:
    payload = scipy.io.loadmat(mat_path)
    labels = np.asarray(payload["label"]).reshape(-1).astype(np.int64)
    raw_features = payload["features"]
    if scipy.sparse.issparse(raw_features):
        raw_features = raw_features.toarray()
    features = np.asarray(raw_features, dtype=np.float32)

    graph_data = {}
    for relation in AMAZON_RELATIONS:
        adjacency = payload[relation].tocoo()
        graph_data[("user", relation, "user")] = (
            torch.from_numpy(adjacency.row.astype(np.int64, copy=False)),
            torch.from_numpy(adjacency.col.astype(np.int64, copy=False)),
        )

    graph = dgl.heterograph(
        graph_data,
        num_nodes_dict={"user": int(labels.size)},
    )
    graph.ndata["feature"] = torch.from_numpy(features)
    graph.ndata["label"] = torch.from_numpy(labels)

    if normalize_for_lex:
        transform = dgl.RowFeatNormalizer(
            subtract_min=True,
            node_feat_names=["feature"],
        )
        graph = transform(graph)

    return graph


def load_yelpchi_graph(mat_path: Path) -> dgl.DGLHeteroGraph:
    """Load the canonical relation-preserving YelpChi graph.

    LEX-GNN's author loader does not apply RowFeatNormalizer to YelpChi,
    so the canonical 32-dimensional features are retained unchanged for
    both benchmark adapters.
    """
    payload = scipy.io.loadmat(mat_path)
    labels = np.asarray(payload["label"]).reshape(-1).astype(np.int64)
    raw_features = payload["features"]
    if scipy.sparse.issparse(raw_features):
        raw_features = raw_features.toarray()
    features = np.asarray(raw_features, dtype=np.float32)

    graph_data = {}
    for relation in YELPCHI_RELATIONS:
        adjacency = payload[relation].tocoo()
        graph_data[("review", relation, "review")] = (
            torch.from_numpy(adjacency.row.astype(np.int64, copy=False)),
            torch.from_numpy(adjacency.col.astype(np.int64, copy=False)),
        )

    graph = dgl.heterograph(
        graph_data,
        num_nodes_dict={"review": int(labels.size)},
    )
    graph.ndata["feature"] = torch.from_numpy(features)
    graph.ndata["label"] = torch.from_numpy(labels)
    return graph


def _array_from_aliases(archive: np.lib.npyio.NpzFile, aliases: Iterable[str]) -> np.ndarray:
    for key in aliases:
        if key in archive.files:
            return np.asarray(archive[key], dtype=np.int64).reshape(-1)
    raise KeyError(f"None of {list(aliases)} found in {archive.files}")


def _load_and_validate_frozen_split(
    split_path: Path,
    expected: set[int],
    eligibility_description: str,
) -> Dict[str, np.ndarray]:
    aliases = {
        "train": ("train_idx", "train_indices", "train_ids", "train"),
        "validation": (
            "valid_idx",
            "val_idx",
            "validation_idx",
            "valid_indices",
            "validation",
            "val",
        ),
        "test": ("test_idx", "test_indices", "test_ids", "test"),
        "unused": (
            "unused_idx",
            "unused_indices",
            "unused_ids",
            "unused_nodes",
            "unused",
        ),
    }
    with np.load(split_path, allow_pickle=False) as archive:
        split = {
            name: _array_from_aliases(archive, keys)
            for name, keys in aliases.items()
        }

    observed_sets = {}
    for name, values in split.items():
        if len(values) != len(np.unique(values)):
            raise ValueError(f"Duplicate IDs in {name}")
        value_set = set(values.tolist())
        if not value_set.issubset(expected):
            raise ValueError(
                f"Ineligible node in {name}; expected "
                f"{eligibility_description}"
            )
        observed_sets[name] = value_set

    names = list(observed_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if not observed_sets[left].isdisjoint(observed_sets[right]):
                raise ValueError(f"Split overlap: {left}/{right}")
    if set().union(*observed_sets.values()) != expected:
        raise ValueError(
            "Frozen split does not cover all eligible nodes: "
            f"{eligibility_description}"
        )
    return split


def load_frozen_split(split_path: Path, node_count: int) -> Dict[str, np.ndarray]:
    """Load the frozen Amazon split (nodes 3305 onward are eligible)."""
    return _load_and_validate_frozen_split(
        split_path,
        set(range(3305, node_count)),
        "Amazon node IDs 3305 through node_count - 1",
    )


def load_yelpchi_frozen_split(
    split_path: Path,
    node_count: int,
) -> Dict[str, np.ndarray]:
    """Load the frozen YelpChi split (all graph nodes are eligible)."""
    return _load_and_validate_frozen_split(
        split_path,
        set(range(node_count)),
        "all YelpChi node IDs",
    )


def select_validation_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.linspace(0.01, 0.99, 99)
    ranked = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        ranked.append((macro_f1, -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    return max(ranked)[-1]


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "fraud_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "fraud_precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "g_mean": float(math.sqrt(sensitivity * specificity)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def hardware_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dgl": dgl.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_gpu_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        snapshot.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(properties.total_memory),
            }
        )
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,pstate",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        snapshot["nvidia_smi_query"] = query.stdout.strip()
    except Exception as exc:  # evidence collection must not abort training
        snapshot["nvidia_smi_error"] = repr(exc)
    return snapshot


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_summary(split: Mapping[str, np.ndarray], labels: torch.Tensor) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    labels_np = labels.detach().cpu().numpy().astype(int)
    for name, indices in split.items():
        values = labels_np[indices]
        result[name] = {
            "nodes": int(len(indices)),
            "fraud": int((values == 1).sum()),
            "normal": int((values == 0).sum()),
        }
    return result

