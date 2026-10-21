"""Verified T-Finance graph and frozen-split loaders for COMP8851."""

from pathlib import Path
import sys

import dgl
import numpy as np
import torch

ADAPTERS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADAPTERS))

from benchmark_common import _load_and_validate_frozen_split


EXPECTED_NODES = 39357
EXPECTED_EDGES = 42445086
EXPECTED_FEATURES = 10
EXPECTED_FRAUD = 1804


def load_tfinance_graph(path):
    path = Path(path)

    graphs, _ = dgl.load_graphs(str(path))
    if len(graphs) != 1:
        raise ValueError("Expected exactly one T-Finance graph")

    source = graphs[0]

    if source.num_nodes() != EXPECTED_NODES:
        raise ValueError("Unexpected T-Finance node count")

    if source.num_edges() != EXPECTED_EDGES:
        raise ValueError("Unexpected T-Finance edge count")

    features = source.ndata["feature"].float().contiguous()
    raw_labels = source.ndata["label"]

    if tuple(features.shape) != (
        EXPECTED_NODES,
        EXPECTED_FEATURES,
    ):
        raise ValueError("Unexpected T-Finance feature shape")

    if tuple(raw_labels.shape) != (EXPECTED_NODES, 2):
        raise ValueError("Expected two-column one-hot labels")

    if not torch.all(
        (raw_labels == 0) | (raw_labels == 1)
    ):
        raise ValueError("Labels must be binary one-hot values")

    if not torch.all(raw_labels.sum(dim=1) == 1):
        raise ValueError("Invalid one-hot label rows")

    labels = raw_labels.argmax(dim=1).long().contiguous()

    if int(labels.sum()) != EXPECTED_FRAUD:
        raise ValueError("Unexpected fraud count")

    # Preserve the graph's original node IDs and edge topology.
    # This is a single-relation, homogeneous DGL graph.
    source.ndata["feature"] = features
    source.ndata["label"] = labels

    return source


def load_tfinance_frozen_split(path, node_count):
    return _load_and_validate_frozen_split(
        Path(path),
        set(range(node_count)),
        "all T-Finance node IDs",
    )
