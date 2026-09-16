#!/usr/bin/env python
"""Create the native DGA-GNN Elliptic graph from the upstream PyG pickle."""

import argparse
import json
import pickle
from pathlib import Path

import dgl
import numpy as np
import torch
import torch_geometric
from dgl.data.utils import load_graphs, save_graphs
from sklearn.preprocessing import StandardScaler


def distribution(values):
    keys, counts = torch.unique(values.cpu(), return_counts=True)
    return {int(key): int(count) for key, count in zip(keys, counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    output_path = Path(args.output)
    split_dir = Path(args.split_dir)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print("torch:", torch.__version__)
    print("dgl:", dgl.__version__)
    print("torch_geometric:", torch_geometric.__version__)
    print("raw_path:", raw_path)

    with raw_path.open("rb") as handle:
        data = pickle.load(handle)

    x = data.x.detach().cpu()
    y = data.y.detach().cpu().reshape(-1).long()
    edge_index = data.edge_index.detach().cpu().long()

    if x.ndim != 2:
        raise RuntimeError("Expected data.x to be a two-dimensional tensor")
    num_nodes = int(x.shape[0])
    if y.numel() != num_nodes:
        raise RuntimeError("Label count does not equal node count")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError("Expected edge_index shape [2, num_edges]")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise RuntimeError("edge_index contains an out-of-range node ID")

    masks = {}
    indices = {}
    source_names = {
        "trn_msk": "train_mask",
        "val_msk": "val_mask",
        "tst_msk": "test_mask",
    }
    for saved_name, source_name in source_names.items():
        mask = getattr(data, source_name).detach().cpu().reshape(-1).bool()
        if mask.numel() != num_nodes:
            raise RuntimeError("{} length does not equal node count".format(source_name))
        masks[saved_name] = mask
        indices[saved_name.replace("msk", "idx")] = torch.nonzero(
            mask, as_tuple=False
        ).reshape(-1).long()

    overlap = (
        masks["trn_msk"].to(torch.int8)
        + masks["val_msk"].to(torch.int8)
        + masks["tst_msk"].to(torch.int8)
    )
    if int((overlap > 1).sum()) != 0:
        raise RuntimeError("Native Elliptic masks overlap")
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            raise RuntimeError("{} is empty".format(name))

    x_numpy = x.numpy()
    if not np.isfinite(x_numpy).all():
        raise RuntimeError("Features contain NaN or infinity")
    x_standardized = StandardScaler().fit_transform(x_numpy).astype(np.float32)

    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    graph.ndata["feat"] = torch.from_numpy(x_standardized)
    graph.ndata["label"] = y
    for name, mask in masks.items():
        graph.ndata[name] = mask

    metadata = dict(masks)
    metadata.update(indices)
    save_graphs(str(output_path), [graph], metadata)

    loaded_graphs, loaded_metadata = load_graphs(str(output_path))
    if len(loaded_graphs) != 1:
        raise RuntimeError("Reloaded file does not contain exactly one graph")
    loaded = loaded_graphs[0]
    if loaded.num_nodes() != num_nodes or loaded.num_edges() != graph.num_edges():
        raise RuntimeError("Reloaded graph dimensions changed")
    if not torch.equal(loaded.ndata["label"], y):
        raise RuntimeError("Reloaded labels changed")
    for name, mask in masks.items():
        if not torch.equal(loaded.ndata[name], mask):
            raise RuntimeError("Reloaded {} changed".format(name))
    for name, index in indices.items():
        if name not in loaded_metadata or not torch.equal(loaded_metadata[name], index):
            raise RuntimeError("Reloaded {} is missing or changed".format(name))

    split_files = {}
    for name, index in indices.items():
        split_name = name.replace("trn", "train").replace("val", "validation").replace("tst", "test")
        split_path = split_dir / "elliptic_{}.npy".format(split_name)
        np.save(str(split_path), index.numpy())
        split_files[split_name] = str(split_path)

    covered = overlap > 0
    summary = {
        "dataset": "Elliptic",
        "source_object_type": "{}.{}".format(type(data).__module__, type(data).__name__),
        "nodes": num_nodes,
        "edges": int(graph.num_edges()),
        "features": [int(value) for value in graph.ndata["feat"].shape],
        "feature_dtype": str(graph.ndata["feat"].dtype),
        "all_labels": distribution(y),
        "train_nodes": int(masks["trn_msk"].sum()),
        "validation_nodes": int(masks["val_msk"].sum()),
        "test_nodes": int(masks["tst_msk"].sum()),
        "train_labels": distribution(y[masks["trn_msk"]]),
        "validation_labels": distribution(y[masks["val_msk"]]),
        "test_labels": distribution(y[masks["tst_msk"]]),
        "overlapping_nodes": int((overlap > 1).sum()),
        "covered_nodes": int(covered.sum()),
        "unassigned_nodes": int((~covered).sum()),
        "native_masks_preserved": True,
        "split_files": split_files,
        "reload_validation": "PASS",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ELLIPTIC PREPROCESSING AND RELOAD VALIDATION: PASS")


if __name__ == "__main__":
    main()
