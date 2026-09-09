from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import dgl
import numpy as np
from sklearn.model_selection import train_test_split


RATIOS = {
    "TR40": 0.40,
    "TR30": 0.30,
    "TR20": 0.20,
    "TR10": 0.10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_from_graph(graph) -> np.ndarray:
    labels = graph.ndata["label"].detach().cpu().numpy()
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)
    labels = labels.reshape(-1).astype(np.int64)
    unique = set(np.unique(labels).tolist())
    if unique != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1; found {sorted(unique)}")
    return labels


def describe(ids: np.ndarray, labels: np.ndarray) -> dict:
    fraud = int(labels[ids].sum())
    total = int(len(ids))
    return {
        "nodes": total,
        "fraud_nodes": fraud,
        "normal_nodes": total - fraud,
        "fraud_percentage": 100.0 * fraud / total if total else 0.0,
    }


def split_nested(labels: np.ndarray, seed: int):
    ids = np.arange(len(labels), dtype=np.int64)
    train40, held_out = train_test_split(
        ids,
        train_size=0.40,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    valid, test = train_test_split(
        held_out,
        train_size=1.0 / 3.0,
        random_state=seed,
        shuffle=True,
        stratify=labels[held_out],
    )
    train30, _ = train_test_split(
        train40,
        train_size=0.75,
        random_state=seed,
        shuffle=True,
        stratify=labels[train40],
    )
    train20, _ = train_test_split(
        train30,
        train_size=2.0 / 3.0,
        random_state=seed,
        shuffle=True,
        stratify=labels[train30],
    )
    train10, _ = train_test_split(
        train20,
        train_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=labels[train20],
    )
    trains = {"TR40": train40, "TR30": train30, "TR20": train20, "TR10": train10}
    base_train = set(train40.tolist())
    result = {}
    for ratio, train in trains.items():
        unused = np.asarray(sorted(base_train - set(train.tolist())), dtype=np.int64)
        result[ratio] = {
            "train_idx": np.sort(np.asarray(train, dtype=np.int64)),
            "valid_idx": np.sort(np.asarray(valid, dtype=np.int64)),
            "test_idx": np.sort(np.asarray(test, dtype=np.int64)),
            "unused_idx": unused,
        }
    return result


def verify(splits: dict, total_nodes: int):
    errors = []
    reference_valid = splits["TR40"]["valid_idx"]
    reference_test = splits["TR40"]["test_idx"]
    for ratio, arrays in splits.items():
        sets = {name: set(values.tolist()) for name, values in arrays.items()}
        for name, values in arrays.items():
            if len(values) != len(sets[name]):
                errors.append(f"{ratio}: duplicate IDs in {name}")
        keys = list(sets)
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                if sets[left] & sets[right]:
                    errors.append(f"{ratio}: overlap between {left} and {right}")
        if len(set().union(*sets.values())) != total_nodes:
            errors.append(f"{ratio}: partitions do not account for all nodes")
        if not np.array_equal(arrays["valid_idx"], reference_valid):
            errors.append(f"{ratio}: validation IDs changed")
        if not np.array_equal(arrays["test_idx"], reference_test):
            errors.append(f"{ratio}: test IDs changed")
    train_sets = {ratio: set(splits[ratio]["train_idx"].tolist()) for ratio in RATIOS}
    if not (train_sets["TR10"] <= train_sets["TR20"] <= train_sets["TR30"] <= train_sets["TR40"]):
        errors.append("Training sets are not nested")
    if errors:
        raise RuntimeError("Split verification failed:\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="shared/splits/tfinance")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CREATING PERSISTENT T-FINANCE SPLITS")
    print("=" * 78)
    print(f"Dataset: {data_path}")
    print(f"Dataset SHA256: {sha256(data_path)}")
    graphs, _ = dgl.load_graphs(str(data_path))
    graph = graphs[0]
    labels = labels_from_graph(graph)
    print(f"Nodes: {graph.num_nodes():,}")
    print(f"Stored directed edge entries: {graph.num_edges():,}")
    print(f"Features: {graph.ndata['feature'].shape[1]}")
    print(f"Fraud nodes: {int(labels.sum()):,}")
    print(f"Normal nodes: {int((labels == 0).sum()):,}")

    splits = split_nested(labels, args.seed)
    verify(splits, len(labels))

    manifest = {
        "dataset": "T-Finance",
        "source_path": str(data_path),
        "source_sha256": sha256(data_path),
        "split_seed": args.seed,
        "convention": "Exact 40/20/40 base split with nested stratified TR30/TR20/TR10 subsets; fixed validation and test IDs",
        "total_nodes": int(len(labels)),
        "fraud_nodes": int(labels.sum()),
        "normal_nodes": int((labels == 0).sum()),
        "splits": {},
    }
    for ratio, arrays in splits.items():
        filename = f"tfinance_{ratio.lower()}_split_seed{args.seed}.npz"
        path = output_dir / filename
        np.savez_compressed(path, **arrays)
        manifest["splits"][ratio] = {
            "file": filename,
            "train": describe(arrays["train_idx"], labels),
            "validation": describe(arrays["valid_idx"], labels),
            "test": describe(arrays["test_idx"], labels),
            "unused_nodes": int(len(arrays["unused_idx"])),
            "file_sha256": sha256(path),
        }
        info = manifest["splits"][ratio]
        print(
            f"{ratio}: train={info['train']['nodes']:,} "
            f"(fraud={info['train']['fraud_nodes']:,}), "
            f"validation={info['validation']['nodes']:,}, "
            f"test={info['test']['nodes']:,}, unused={info['unused_nodes']:,}"
        )

    manifest_path = output_dir / f"tfinance_split_manifest_seed{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Fixed validation IDs: PASS")
    print("Fixed test IDs: PASS")
    print("TR10 contained in TR20 contained in TR30 contained in TR40: PASS")
    print("No split leakage: PASS")
    print(f"Manifest: {manifest_path}")
    print("T-FINANCE SPLITS READY: TRUE")


if __name__ == "__main__":
    main()
