from pathlib import Path

import dgl
import numpy as np
import torch


def load_elliptic_graph(dataset_path, normalize_for_lex=False):
    directory = Path(dataset_path).resolve()
    expected = Path(
        "/workspace/comp8851_elliptic_work/data"
    ).resolve()

    if directory != expected:
        raise ValueError(
            f"Unexpected Elliptic dataset directory: {directory}"
        )

    x = np.load(directory / "features_original.npy")
    y = np.load(directory / "labels_with_unknown.npy")
    src = np.load(directory / "source_nodes.npy")
    dst = np.load(directory / "target_nodes.npy")

    if x.shape != (203769, 166):
        raise ValueError("Incorrect feature dimensions")
    if y.shape != (203769,):
        raise ValueError("Incorrect label dimensions")
    if src.shape != (234355,) or dst.shape != (234355,):
        raise ValueError("Incorrect edge dimensions")

    graph = dgl.heterograph(
        {
            ("tx", "transaction", "tx"): (
                torch.from_numpy(src.astype(np.int64)),
                torch.from_numpy(dst.astype(np.int64)),
            )
        },
        num_nodes_dict={"tx": 203769},
    )

    graph.ndata["feature"] = torch.from_numpy(
        x.astype(np.float32)
    ).contiguous()
    graph.ndata["label"] = torch.from_numpy(
        y.astype(np.int64)
    ).contiguous()

    # For now, retain the original feature values for both models.
    # LEX-specific author normalization must be verified before training.
    if normalize_for_lex:
        graph = dgl.RowFeatNormalizer(
            subtract_min=True,
            node_feat_names=["feature"],
        )(graph)

    return graph


def load_elliptic_split(split_path, node_count):
    path = Path(split_path).resolve()

    provisional = path.name.endswith("_compat_only.npz")
    reconstructed_dir = Path(
        "/workspace/comp8851_elliptic_work/"
        "elliptic_reconstructed_v1/splits"
    ).resolve()

    reconstructed = (
        path.parent == reconstructed_dir
        and path.name in {
            "elliptic_TR10_seed2.npz",
            "elliptic_TR20_seed2.npz",
            "elliptic_TR30_seed2.npz",
            "elliptic_TR40_seed2.npz",
        }
    )

    if not (provisional or reconstructed):
        raise ValueError("Unapproved Elliptic split path")

    if reconstructed:
        import hashlib
        import json
        manifest = json.loads(
            (reconstructed_dir / "manifest.json").read_text()
        )
        ratio = path.stem.split("_")[1]
        expected = manifest["splits"][ratio]["sha256"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Frozen Elliptic split hash mismatch")

    with np.load(path, allow_pickle=False) as z:
        required = ("train", "validation", "test", "unused")
        result = {
            key: np.asarray(z[key], dtype=np.int64)
            for key in required
        }

    for name, ids in result.items():
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs in {name}")
        if np.any(ids < 0) or np.any(ids >= node_count):
            raise ValueError(f"Invalid node IDs in {name}")

    for a, b in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
        ("unused", "train"),
        ("unused", "validation"),
        ("unused", "test"),
    ):
        if np.intersect1d(result[a], result[b]).size:
            raise ValueError(f"Split overlap: {a}, {b}")

    return result
