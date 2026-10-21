import hashlib
import json
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace/COMP8851-fraud-dataset-analysis")
WORK = Path("/workspace/comp8851_elliptic_work")
RAW = Path(
    "/workspace/comp8851_elliptic_intake/"
    "elliptic_bitcoin_dataset"
)
SPLIT = (
    ROOT / "models/gaap/evidence/elliptic/shared/elliptic/"
    "elliptic_seed2_nested_splits.npz"
)

print("===== LOAD CANONICAL ELLIPTIC DATA =====", flush=True)

features = pd.read_csv(
    RAW / "elliptic_txs_features.csv",
    header=None,
    dtype={0: "string"},
)

classes = pd.read_csv(
    RAW / "elliptic_txs_classes.csv",
    dtype=str,
)

edges = pd.read_csv(
    RAW / "elliptic_txs_edgelist.csv",
    dtype=str,
)

assert features.shape == (203769, 167), features.shape
assert len(classes) == 203769
assert len(edges) == 234355

node_ids = pd.Index(features.iloc[:, 0].astype(str))
assert node_ids.is_unique

x = features.iloc[:, 1:].to_numpy(
    dtype=np.float32,
    copy=True,
)

assert x.shape == (203769, 166)
assert np.isfinite(x).all()

timesteps = x[:, 0].astype(np.int64)
assert np.all(x[:, 0] == timesteps)
assert timesteps.min() == 1
assert timesteps.max() == 49

print("PASS: Feature dimensions and timesteps", flush=True)

print("===== MAP LABELS BY TRANSACTION ID =====", flush=True)

class_ids = classes.iloc[:, 0].astype(str)
class_values = classes.iloc[:, 1].astype(str)

class_index = node_ids.get_indexer(class_ids)
assert np.all(class_index >= 0)
assert len(np.unique(class_index)) == len(class_index)

labels = np.full(len(node_ids), -1, dtype=np.int64)

for raw_value, mapped in (("1", 1), ("2", 0)):
    mask = (class_values == raw_value).to_numpy()
    labels[class_index[mask]] = mapped

assert set(class_values.unique()) == {"1", "2", "unknown"}
assert (labels == 1).sum() == 4545
assert (labels == 0).sum() == 42019
assert (labels == -1).sum() == 157205

print("PASS: Fraud, normal and unknown labels", flush=True)

print("===== MAP ORIGINAL DIRECTED EDGES =====", flush=True)

src = node_ids.get_indexer(edges.iloc[:, 0].astype(str))
dst = node_ids.get_indexer(edges.iloc[:, 1].astype(str))

assert np.all(src >= 0)
assert np.all(dst >= 0)
assert len(src) == 234355

graph = dgl.heterograph(
    {
        ("tx", "transaction", "tx"): (
            torch.from_numpy(src.astype(np.int64)),
            torch.from_numpy(dst.astype(np.int64)),
        )
    },
    num_nodes_dict={"tx": len(node_ids)},
)

assert graph.num_nodes() == 203769
assert graph.num_edges() == 234355

print("PASS: Original directed edges preserved", flush=True)

print("===== VALIDATE AVAILABLE GAAP SPLIT =====", flush=True)

with np.load(SPLIT, allow_pickle=False) as archive:
    split = {
        name: np.asarray(archive[name], dtype=np.int64)
        for name in ("TR10", "TR20", "TR30", "TR40", "val", "test")
    }

expected = {
    "TR10": 4543,
    "TR20": 9553,
    "TR30": 13670,
    "TR40": 18889,
    "val": 8726,
    "test": 18949,
}

for name, ids in split.items():
    assert len(ids) == expected[name]
    assert len(np.unique(ids)) == len(ids)
    assert np.all((ids >= 0) & (ids < len(node_ids)))
    assert np.all(labels[ids] >= 0), (
        f"Unknown labels found in {name}"
    )

for small, large in (
    ("TR10", "TR20"),
    ("TR20", "TR30"),
    ("TR30", "TR40"),
):
    assert np.isin(split[small], split[large]).all()

train = split["TR40"]
val = split["val"]
test = split["test"]

assert len(np.intersect1d(train, val)) == 0
assert len(np.intersect1d(train, test)) == 0
assert len(np.intersect1d(val, test)) == 0

assert timesteps[train].max() <= 20
assert timesteps[val].min() == 21
assert timesteps[val].max() == 31
assert timesteps[test].min() == 32

print("PASS: GAAP split labels, nesting and chronology", flush=True)

print("===== SAVE VERIFIED DATA ARRAYS =====", flush=True)

OUT = WORK / "data"
np.save(OUT / "features_original.npy", x)
np.save(OUT / "labels_with_unknown.npy", labels)
np.save(OUT / "timesteps.npy", timesteps)
np.save(OUT / "source_nodes.npy", src.astype(np.int64))
np.save(OUT / "target_nodes.npy", dst.astype(np.int64))

split_hash = hashlib.sha256(SPLIT.read_bytes()).hexdigest()

manifest = {
    "dataset": "Elliptic",
    "nodes": int(graph.num_nodes()),
    "directed_edges": int(graph.num_edges()),
    "feature_dimensions": int(x.shape[1]),
    "fraud": int((labels == 1).sum()),
    "normal": int((labels == 0).sum()),
    "unknown": int((labels == -1).sum()),
    "reverse_edges_added": False,
    "self_loops_added": False,
    "feature_normalization_applied": False,
    "split_status": "GAAP_PROVISIONAL_COMPATIBILITY_ONLY",
    "split_source": str(SPLIT),
    "split_sha256": split_hash,
    "splits": {
        name: {
            "count": len(ids),
            "fraud": int((labels[ids] == 1).sum()),
            "normal": int((labels[ids] == 0).sum()),
            "min_timestep": int(timesteps[ids].min()),
            "max_timestep": int(timesteps[ids].max()),
        }
        for name, ids in split.items()
    },
}

path = WORK / "evidence/elliptic_loader_manifest.json"
path.write_text(json.dumps(manifest, indent=2))

print("===== SPLIT SUMMARY =====", flush=True)

for name, info in manifest["splits"].items():
    print(
        f"{name}: {info['count']} nodes | "
        f"fraud={info['fraud']} | "
        f"timesteps={info['min_timestep']}-"
        f"{info['max_timestep']}",
        flush=True,
    )

print()
print("PASS: ELLIPTIC DATA LOADER VALIDATED", flush=True)
print("Saved:", path, flush=True)
print(
    "IMPORTANT: GAAP SPLIT IS NOT APPROVED FOR FINAL RUNS",
    flush=True,
)
