"""Dataset, split and graph-view layer for the phase-2 models.

Everything here sits on top of ``shared.comp8851.datasets`` / ``splits`` and
adds only what the three new models need:

* resolution of the frozen split registry (including the T-Finance file-name
  alias and the regenerated-and-verified Elliptic / FDCompCN registries),
* structural label isolation: model code only ever receives a label vector in
  which every non-training node is ``-1``,
* deterministic DGL graph views (homogeneous union, relation-typed multigraph).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from shared.comp8851 import datasets as cds
from shared.comp8851 import protocol, splits

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = BUNDLE_ROOT / "shared" / "splits"
DEFAULT_CANONICAL_DIR = BUNDLE_ROOT / "data" / "canonical"

#: Table 3 of the Assessment-2 report (canonical preprocessing). Used as
#: integrity assertions after freezing a dataset; a mismatch stops the run.
#: Amazon exception: Table 3 prints 4,398,392 edges, which is the count of the
#: ``homo`` key inside Amazon.mat. The canonical graph actually used by every
#: previous run is the union of the three relations, 4,417,576 undirected
#: edges (evidence/caregnn_amazon_seed2/dataset_manifest.json), so that value
#: is asserted here and the report discrepancy is recorded in the audit.
REPORT_TABLE3 = {
    "yelpchi": {"nodes": 45954, "eligible": 45954, "undirected_edges": 3846979, "features": 32, "fraud": 6677},
    "amazon": {"nodes": 11944, "eligible": 8639, "undirected_edges": 4417576, "features": 25, "fraud": 821},
    "tfinance": {"nodes": 39357, "eligible": 39357, "undirected_edges": 21222543, "features": 10, "fraud": 1804},
    "elliptic": {"nodes": 203769, "eligible": 46564, "undirected_edges": 234355, "features": 166, "fraud": 4545},
    "fdcompcn": {"nodes": 5317, "eligible": 5317, "undirected_edges": 3634, "features": 57, "fraud": 559},
    "tsocial": {"nodes": 5781065, "eligible": 5781065, "undirected_edges": 73105508, "features": 10, "fraud": 174280},
}

#: TR40 partition sizes and fraud counts recorded in the previous benchmark's
#: run evidence (COMP8851_FINAL_SUBMISSION/evidence/*/split_summary.json, used
#: identically by CARE-GNN and GHRN). Elliptic and FDCompCN registry *arrays*
#: are not present anywhere in the workspace, so their registries are
#: regenerated with the unchanged shared builder and must reproduce these
#: counts exactly before any run may use them.
PREVIOUS_TR40_COUNTS = {
    "elliptic": {"train": (18626, 1669), "validation": (9313, 1459), "test": (18625, 1417)},
    "fdcompcn": {"train": (2126, 224), "validation": (1063, 112), "test": (2128, 223)},
    "yelpchi": {"train": (18381, 2671), "validation": (9191, 1335), "test": (18382, 2671)},
    "amazon": {"train": (3455, 328), "validation": (1728, 164), "test": (3456, 329)},
    "tfinance": {"train": (15742, 722), "validation": (7871, 361), "test": (15744, 721)},
    "tsocial": {"train": (2312426, 69712), "validation": (1156213, 34856), "test": (2312426, 69712)},
}

#: View hashes recorded by the previous benchmark's runs
#: (COMP8851_FINAL_SUBMISSION/evidence/{ghrn,caregnn}_<ds>_seed2 and
#: evidence/tsocial/final_seed2). ``CanonicalDataset._view_hash`` covers
#: labels, features and edges, so equality proves the phase-2 models consume
#: byte-identical data to CARE-GNN and GHRN.
PREVIOUS_VIEW_SHA256 = {
    "yelpchi": {"homogeneous": "f7972775b0dc2ab27a19bf2e238d97ecfd9a9913d9837857cf9434878890eb16",
                "relation": "403ea176ea709f68aab34f1f36799527962dffcdbb6df39acc6a9fc000079b89"},
    "amazon": {"homogeneous": "71d2d6e02ce9b4177e26c57e50508717a7668d6d0351e0fd09c55056b38aebd3",
               "relation": "1bcfdb9592d2788b5644f7fddc922fd75ea6159db85ef80b10f373cbcb7b920f"},
    "tfinance": {"homogeneous": "e98c3bb6926235a1e49d023bec9c276af373fff3657b7a9dfbdeb943ed78c217",
                 "relation": "5b0b970c6df39bf341e25481f5d799959646c88a7d5e257032f1d2b1d7d3617c"},
    "elliptic": {"homogeneous": "a7eb1286ce4bd93fcbd7707e5092d26796eea402185ccd908fe9c1d09a4798e9",
                 "relation": "23c953e25b070612934615833a797a0992293dbbdf411d1ebda7d293bb5964aa"},
    "fdcompcn": {"homogeneous": "e18346217afff8351f822a3452832f5db889c95887bb30bc39ab81d9b49a130a",
                 "relation": "438889170ee3e9f809c4142f1f242ce50b77117914101e2ef88fd18345004a64"},
    "tsocial": {"relation": "36063bebb030ad7c92a1ac81152eaa023af97c70bf263ce2a88753003299c5ff"},
}

#: Raw-source hashes recorded by the previous runs (dataset_manifest.json).
PREVIOUS_SOURCE_SHA256_PREFIX = {
    "yelpchi": {"mat": "fedb35a8fa539b27"},
    "amazon": {"mat": "4b7e3f9cccc62b73"},
    "tfinance": {"graph": "b7d853ec4079e9f7"},
    "fdcompcn": {"graph": "e252b9a6b619b28a"},
    "elliptic": {"classes": "93e2e7b2405c735b", "edges": "a35053ba68a98e43", "features": "fd7f83573443c9e3"},
}

#: Registry files whose bytes were frozen by the dataset owners; their SHA256
#: must match the owners' split manifests.
REGISTRY_MANIFESTS = {
    "yelpchi": "yelpchi_split_manifest_seed2.json",
    "amazon": "amazon_nested_split_manifest_seed2.json",
    "tfinance": "tfinance_split_manifest_seed2.json",
    "tsocial": "tsocial_split_manifest_seed2.json",
}
TFINANCE_ORIGINAL_NAMES = {f"tfinance_{r}_seed2.npz": f"tfinance_{r.lower()}_split_seed2.npz"
                           for r in ("TR40", "TR30", "TR20", "TR10")}

MASKED_LABEL = -1


@dataclass
class PreparedData:
    """Everything a run needs, with labels already isolated."""

    dataset: cds.CanonicalDataset
    split: Dict[str, np.ndarray]
    split_path: str
    split_sha256: str
    train_labels_only: np.ndarray      # full-length, -1 outside the training set

    @property
    def train_idx(self) -> np.ndarray:
        return self.split["train_idx"]

    @property
    def valid_idx(self) -> np.ndarray:
        return self.split["valid_idx"]

    @property
    def test_idx(self) -> np.ndarray:
        return self.split["test_idx"]


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def canonical_path(name: str, canonical_dir: Path | str = DEFAULT_CANONICAL_DIR) -> Path:
    return Path(canonical_dir) / f"{name}_canonical.npz"


def load_canonical_dataset(name: str, data_path: Optional[str] = None) -> cds.CanonicalDataset:
    """Load a frozen canonical cache; fail with an actionable message if absent."""
    path = Path(data_path) if data_path else canonical_path(name)
    if path.is_dir():
        path = canonical_path(name, path)
    if not path.exists():
        raise FileNotFoundError(
            f"{name}: canonical cache {path} not found. Run "
            f"`python scripts/prepare_data.py --only {name}` first (see README, section Data). "
            "No substitute data is ever generated.")
    dataset = cds.load_dataset(name, path)
    if dataset.name != name:
        raise ValueError(f"canonical cache {path} holds dataset {dataset.name!r}, expected {name!r}")
    return dataset


def check_against_report(dataset: cds.CanonicalDataset) -> Dict[str, object]:
    """Compare a canonical dataset with Table 3 of the Assessment-2 report."""
    expected = REPORT_TABLE3[dataset.name]
    stats = dataset.statistics()
    observed = {
        "nodes": stats["nodes"],
        "eligible": stats["eligible_nodes"],
        "undirected_edges": stats["union_undirected_edges"],
        "features": stats["feature_dimension"],
        "fraud": stats["fraud_nodes"],
    }
    mismatches = {k: {"expected": expected[k], "observed": observed[k]}
                  for k in expected if expected[k] != observed[k]}
    return {"expected": expected, "observed": observed, "mismatches": mismatches,
            "matches_report": not mismatches}


def check_against_previous(dataset: cds.CanonicalDataset) -> Dict[str, object]:
    """Compare view hashes and raw-source hashes with the previous benchmark."""
    result: Dict[str, object] = {"views": {}, "sources": {}}
    for view, expected in PREVIOUS_VIEW_SHA256.get(dataset.name, {}).items():
        observed = dataset.view_manifest(view, add_self_loops=True)["view_sha256"]
        result["views"][view] = "identical" if observed == expected else f"DIFFERENT ({observed[:16]})"
    for key, prefix in PREVIOUS_SOURCE_SHA256_PREFIX.get(dataset.name, {}).items():
        observed = str(dataset.source_sha256.get(key, ""))
        result["sources"][key] = "identical" if observed.startswith(prefix) else f"DIFFERENT ({observed[:16]})"
    values = list(result["views"].values()) + list(result["sources"].values())
    result["identical_to_previous_benchmark"] = bool(values) and all(v == "identical" for v in values)
    return result


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def _partition_counts(arrays: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, Tuple[int, int]]:
    out = {}
    for name, key in (("train", "train_idx"), ("validation", "valid_idx"), ("test", "test_idx")):
        ids = np.asarray(arrays[key], dtype=np.int64)
        out[name] = (int(ids.size), int(labels[ids].sum()))
    return out


def verify_registry_bytes(name: str, split_dir: Path,
                          labels: Optional[np.ndarray] = None) -> Dict[str, str]:
    """Check every registry file of a dataset against its owner's manifest.

    Manifests with ``file_sha256`` are checked byte-for-byte. The Amazon
    manifest has no hashes, so its partitions are checked by size and fraud
    count per ratio instead (requires ``labels``).
    """
    manifest_name = REGISTRY_MANIFESTS.get(name)
    if manifest_name is None:
        return {}
    manifest = json.loads((split_dir / name / manifest_name).read_text(encoding="utf-8"))
    results = {}
    for ratio, entry in manifest["splits"].items():
        file_name = splits.split_filename(name, ratio)
        path = split_dir / name / file_name
        if not path.exists():
            results[file_name] = "MISSING"
            continue
        if "file_sha256" in entry:
            expected = entry["file_sha256"]
            observed = protocol.sha256_file(path)
            results[file_name] = "ok" if observed == expected else f"MISMATCH (expected {expected[:16]}, got {observed[:16]})"
            continue
        if labels is None:
            results[file_name] = "UNVERIFIED (manifest has no sha256 and no labels given)"
            continue
        observed_counts = _partition_counts(splits.load_split(path), labels)
        expected_counts = {part: (int(entry[part]["nodes"]), int(entry[part]["fraud_nodes"]))
                           for part in ("train", "validation", "test")}
        results[file_name] = ("ok (sizes and fraud counts; manifest carries no sha256)"
                              if observed_counts == expected_counts
                              else f"MISMATCH {observed_counts} vs {expected_counts}")
    return results


def ensure_regenerated_registry(dataset: cds.CanonicalDataset, split_dir: Path) -> Path:
    """Build the Elliptic/FDCompCN registry with the unchanged shared builder.

    The resulting TR40 counts must equal those recorded by the previous
    benchmark's runs; otherwise nothing is written and the run stops.
    """
    name = dataset.name
    target_dir = split_dir / name
    tr40 = target_dir / splits.split_filename(name, "TR40")
    if tr40.exists():
        return target_dir
    if name in protocol.CHRONOLOGICAL_DATASETS:
        generated = splits.build_chronological_splits(dataset.labels, dataset.time_steps,
                                                      dataset.eligible_idx, protocol.SPLIT_SEED)
    else:
        generated = splits.build_nested_splits(dataset.labels, dataset.eligible_idx, protocol.SPLIT_SEED)
    splits.verify_splits(generated, dataset.eligible_idx, dataset.num_nodes)
    observed = _partition_counts(generated["TR40"], dataset.labels)
    expected = {k: tuple(v) for k, v in PREVIOUS_TR40_COUNTS[name].items()}
    if observed != expected:
        raise RuntimeError(
            f"{name}: regenerated TR40 split {observed} does not reproduce the previous "
            f"benchmark's recorded partition {expected}. Refusing to write a registry.")
    splits.save_splits(generated, target_dir, name, dataset.labels, protocol.SPLIT_SEED,
                       extra_manifest={
                           "provenance": ("regenerated with shared/comp8851/splits.py (unchanged) because "
                                          "the frozen arrays are absent from the workspace; TR40 sizes and "
                                          "fraud counts verified equal to the previous benchmark's "
                                          "split_summary.json for CARE-GNN and GHRN"),
                           "verified_against": PREVIOUS_TR40_COUNTS[name],
                           "source_sha256": dataset.source_sha256,
                       })
    return target_dir


def resolve_split(dataset: cds.CanonicalDataset, ratio: str,
                  split_dir: Path | str = DEFAULT_SPLIT_DIR,
                  split_path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], str, str]:
    """Return ``(arrays, path, sha256)`` for the frozen split of ``dataset``.

    Registry datasets must already have their files (no silent regeneration);
    Elliptic and FDCompCN are regenerated once and verified.
    """
    split_dir = Path(split_dir)
    if split_path:
        path = Path(split_path)
    else:
        path = split_dir / dataset.name / splits.split_filename(dataset.name, ratio)
        if not path.exists():
            if dataset.name in REGISTRY_MANIFESTS:
                raise FileNotFoundError(
                    f"Frozen registry file {path} is missing. It must be copied from the "
                    "dataset owner's registry, never regenerated.")
            ensure_regenerated_registry(dataset, split_dir)
    arrays = splits.load_split(path)
    splits.verify_splits({ratio: arrays}, splits.eligible_from_split(arrays), dataset.num_nodes)
    splits.assert_no_leakage(arrays["train_idx"], arrays["valid_idx"], arrays["test_idx"])
    eligible = set(dataset.eligible_idx.tolist())
    for key in ("train_idx", "valid_idx", "test_idx"):
        if not set(np.asarray(arrays[key]).tolist()) <= eligible:
            raise RuntimeError(f"{dataset.name}: {key} contains nodes outside the eligible set")
    if ratio == "TR40":
        observed = _partition_counts(arrays, dataset.labels)
        expected = {k: tuple(v) for k, v in PREVIOUS_TR40_COUNTS[dataset.name].items()}
        if observed != expected:
            raise RuntimeError(f"{dataset.name}: TR40 split {observed} differs from the previous "
                               f"benchmark's partition {expected}")
    return arrays, str(path.resolve()), protocol.sha256_file(path)


def isolate_labels(labels: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Return a copy of ``labels`` with every non-training node set to -1.

    Model code receives only this vector. Validation and test labels are held
    by the harness and reach nothing but the shared evaluator.
    """
    masked = np.full(labels.shape[0], MASKED_LABEL, dtype=np.int64)
    masked[train_idx] = labels[train_idx]
    return masked


def prepare(name: str, ratio: str, data_path: Optional[str] = None,
            split_dir: Path | str = DEFAULT_SPLIT_DIR,
            split_path: Optional[str] = None) -> PreparedData:
    dataset = load_canonical_dataset(name, data_path)
    arrays, path, digest = resolve_split(dataset, ratio, split_dir, split_path)
    train_only = isolate_labels(dataset.labels, arrays["train_idx"])
    if (train_only[arrays["train_idx"]] < 0).any():
        raise RuntimeError("training labels missing after isolation")
    if set(np.unique(train_only[arrays["train_idx"]]).tolist()) != {0, 1}:
        raise RuntimeError(f"{name} {ratio}: training split must contain both classes")
    return PreparedData(dataset=dataset, split=arrays, split_path=path, split_sha256=digest,
                        train_labels_only=train_only)


# --------------------------------------------------------------------------
# Graph views
# --------------------------------------------------------------------------

def homogeneous_edges(dataset: cds.CanonicalDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric, binarised, self-loop-free union of all relations (row, col)."""
    union = dataset.homogeneous_adjacency(add_self_loops=False).tocoo()
    keep = union.row != union.col
    return union.row[keep].astype(np.int64), union.col[keep].astype(np.int64)


def semantic_relations(dataset: cds.CanonicalDataset):
    """Relation names excluding a stored union relation.

    The released FDCompCN graph (comp.dgl) carries a ``homo`` edge type next to
    its three semantic relations; it is exactly their union (verified here).
    Treating it as a fourth relation would duplicate every edge under an extra
    type, so relation-aware models receive only the semantic relations.
    """
    names = sorted(dataset.relations)
    if "homo" in names and len(names) > 1:
        others = [n for n in names if n != "homo"]
        union = None
        for name in others:
            m = dataset.relations[name].tocsr()
            union = m if union is None else union + m
        union.data[:] = 1
        homo = dataset.relations["homo"].tocsr().copy()
        homo.data[:] = 1
        diff = (homo - union)
        diff.eliminate_zeros()
        if diff.nnz != 0:
            raise RuntimeError(f"{dataset.name}: 'homo' relation is not the union of {others}")
        return others
    return names


def relation_order(dataset: cds.CanonicalDataset):
    """Relation order used by ``dgl.to_homogeneous``: canonical etypes sort
    alphabetically, so ``net_rsr < net_rtr < net_rur`` (YelpChi) etc."""
    return sorted(semantic_relations(dataset))


def typed_edges(dataset: cds.CanonicalDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-relation edges concatenated with an integer relation id.

    Duplicated (u, v) pairs across relations stay separate edges with
    different types, exactly as ``dgl.to_homogeneous`` keeps them.
    """
    rows, cols, types = [], [], []
    for type_id, name in enumerate(relation_order(dataset)):
        coo = dataset.relations[name].tocoo()
        keep = coo.row != coo.col
        rows.append(coo.row[keep].astype(np.int64))
        cols.append(coo.col[keep].astype(np.int64))
        types.append(np.full(int(keep.sum()), type_id, dtype=np.int64))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(types)
