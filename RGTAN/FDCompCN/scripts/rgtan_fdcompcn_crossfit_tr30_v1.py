#!/usr/bin/env python3
"""RGTAN FDCompCN TR30 cross-fitted risk features; no model training."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

import dgl
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace/comp8851")
DATA = Path("/workspace/uploads/rgtan_fdcompcn_i2_incoming")
OUT = ROOT / "assets/rgtan/fdcompcn"

ART = OUT / "fdcompcn_tr30_crossfit.npz"
META = OUT / "fdcompcn_tr30_crossfit_manifest.json"

SOURCE_SHA = (
    "e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa"
)
SPLIT_SHA = (
    "5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168"
)
PIN = "65e583f03ac5c5983e49cc35179fe7a65908b9c1"

RISK_NAMES = [
    "degree", "riskstat",
    "1hop_degree", "2hop_degree",
    "1hop_riskstat", "2hop_riskstat"
]

N = 5317

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(4 * 1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def main():
    start = time.perf_counter()

    assert not ART.exists()
    assert not META.exists()

    assert sha256(DATA / "comp.dgl") == SOURCE_SHA
    assert sha256(
        DATA / "fdcompcn_seed2_nested_splits.npz"
    ) == SPLIT_SHA

    commit = subprocess.check_output(
        [
            "git", "-C",
            str(ROOT / "repos/antifraud"),
            "rev-parse", "HEAD"
        ],
        text=True
    ).strip()

    assert commit == PIN

    print("PINNED_SOURCE_AND_HASHES=PASS", flush=True)

    # Frozen Assessment 2 partitions.
    with np.load(
        DATA / "fdcompcn_seed2_nested_splits.npz",
        allow_pickle=False
    ) as z:
        assert set(z.files) == {
            "TR40", "TR30", "TR20", "TR10",
            "val", "test", "seed"
        }
        assert int(z["seed"].item()) == 2

        ids = {
            key: np.array(z[key], dtype=np.int64, copy=True)
            for key in (
                "TR40", "TR30", "TR20",
                "TR10", "val", "test"
            )
        }

    train = ids["TR30"]
    val = ids["val"]
    test = ids["test"]

    assert (len(train), len(val), len(test)) == (
        1594, 1064, 2127
    )

    for group in ids.values():
        assert len(np.unique(group)) == len(group)
        assert np.all((group >= 0) & (group < N))

    assert len(np.unique(
        np.concatenate((train, val, test))
    )) == len(train) + len(val) + len(test)

    for smaller, larger in (
        ("TR10", "TR20"),
        ("TR20", "TR30"),
        ("TR30", "TR40")
    ):
        assert np.isin(
            ids[smaller], ids[larger]
        ).all()

    print("FROZEN_SPLIT_ISOLATION=PASS", flush=True)

    # Read the graph and numerical attributes.
    graphs, _ = dgl.load_graphs(
        str(DATA / "comp.dgl")
    )

    assert len(graphs) == 1
    graph = graphs[0]

    assert graph.num_nodes() == N
    assert graph.num_edges() == 30752
    assert graph.ntypes == ["company"]

    features = (
        graph.nodes["company"].data["feature"]
        .cpu().numpy()
        .astype(np.float32, copy=True)
    )

    assert features.shape == (N, 57)
    assert np.isfinite(features).all()

    # Extract labels from the frozen training IDs only.
    train_labels = (
        graph.nodes["company"].data["label"][train]
        .cpu().numpy()
        .astype(np.int64, copy=True)
    )

    assert set(np.unique(train_labels)) == {0, 1}
    assert int((train_labels == 1).sum()) == 167

    print("TRAINING_LABELS_ONLY=PASS", flush=True)

    # Homogeneous topology: do not copy edge
    # labels, edge masks or source split masks.
    u, v = graph.edges(
        etype=("company", "homo", "company")
    )

    adj = sp.csr_matrix(
        (
            np.ones(len(u), dtype=np.float32),
            (u.numpy(), v.numpy())
        ),
        shape=(N, N)
    )

    adj = (adj + adj.T).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0
    adj.setdiag(1.0)
    adj.eliminate_zeros()
    adj.sort_indices()

    assert adj.nnz == 12585
    assert (adj != adj.T).nnz == 0
    assert np.all(adj.diagonal() == 1)

    print("RGTAN_HOMOGENEOUS_GRAPH=PASS", flush=True)

    # Five stratified training folds.
    fold = np.full(N, -1, dtype=np.int8)

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2
    )

    for j, (_, held) in enumerate(
        splitter.split(
            np.zeros(len(train)), train_labels
        )
    ):
        fold[train[held]] = j

    assert set(fold[train]) == set(range(5))
    assert np.all(fold[val] == -1)
    assert np.all(fold[test] == -1)

    # Five OOF contexts plus one full-training
    # context for validation and later inference.
    context = np.full((6, N), 2, dtype=np.int8)
    positives = np.zeros((N, 6), dtype=np.float32)

    for j in range(5):
        usable = fold[train] != j
        source_ids = train[usable]
        source_labels = train_labels[usable]

        context[j, source_ids] = source_labels
        positives[source_ids, j] = (
            source_labels == 1
        ).astype(np.float32)

    context[5, train] = train_labels
    positives[train, 5] = (
        train_labels == 1
    ).astype(np.float32)

    for j in range(6):
        assert np.all(context[j, val] == 2)
        assert np.all(context[j, test] == 2)
        assert not np.any(positives[val, j])
        assert not np.any(positives[test, j])

    for j in range(5):
        held = train[fold[train] == j]
        assert np.all(context[j, held] == 2)
        assert not np.any(positives[held, j])

    print("FIVE_OOF_CONTEXTS=PASS", flush=True)
    print("VALIDATION_TEST_LABEL_ISOLATION=PASS", flush=True)

    # Exact two-hop reachability.
    # Exclude all directly adjacent nodes.
    two = (adj @ adj).tocsr()
    two.data[:] = 1.0

    two = (
        two - two.multiply(adj)
    ).tocsr()

    two.eliminate_zeros()
    two.sort_indices()

    assert (two != two.T).nnz == 0
    assert not np.any(two.diagonal())
    assert two.multiply(adj).nnz == 0

    # Independent CSR-reference checks.
    for node in (0, 123, 5316):
        first = set(
            adj.indices[
                adj.indptr[node]:adj.indptr[node + 1]
            ].tolist()
        )

        second = set()

        for neighbour in first:
            second.update(
                adj.indices[
                    adj.indptr[neighbour]:
                    adj.indptr[neighbour + 1]
                ].tolist()
            )

        second.difference_update(first)

        actual = set(
            two.indices[
                two.indptr[node]:two.indptr[node + 1]
            ].tolist()
        )

        assert actual == second

    print("TWO_HOP_REFERENCE_CHECK=PASS", flush=True)

    # Six Amazon-equivalent RGTAN risk features.
    degree = np.diff(adj.indptr).astype(np.float32)

    risk = (adj @ positives).astype(np.float32)

    first_degree = (
        adj @ degree
    ).astype(np.float32)

    second_degree = (
        two @ degree
    ).astype(np.float32)

    first_risk = (
        adj @ risk
    ).astype(np.float32)

    second_risk = (
        two @ risk
    ).astype(np.float32)

    raw_views = []

    for j in range(6):
        view = np.column_stack([
            degree,
            risk[:, j],
            first_degree,
            second_degree,
            first_risk[:, j],
            second_risk[:, j]
        ]).astype(np.float32)

        assert view.shape == (N, 6)
        assert np.isfinite(view).all()
        raw_views.append(view)

    print("SIX_RISK_FEATURES_COMPUTED=PASS", flush=True)

    # Fit scaler on out-of-fold training
    # features only, never validation/test.
    oof = np.empty(
        (len(train), 6), dtype=np.float32
    )

    for j in range(5):
        selected = fold[train] == j
        oof[selected] = raw_views[j][
            train[selected]
        ]

    scaler = StandardScaler().fit(oof)

    views = [
        scaler.transform(raw).astype(np.float32)
        for raw in raw_views
    ]

    assert all(
        view.shape == (N, 6)
        and np.isfinite(view).all()
        for view in views
    )

    print("TRAIN_ONLY_NORMALIZATION=PASS", flush=True)

    payload = {
        "features": features,
        "train_ids": train,
        "train_labels": train_labels,
        "val_ids": val,
        "fold_of_node": fold,
        "risk_feature_names": np.asarray(RISK_NAMES),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64)
    }

    for j in range(6):
        payload[f"risk_view_{j}"] = views[j]
        payload[f"label_context_{j}"] = context[j]

    OUT.mkdir(parents=True, exist_ok=True)

    temp = ART.with_name(
        ART.stem + ".partial.npz"
    )

    assert not temp.exists()
    np.savez_compressed(temp, **payload)

    # Independently check the serialized payload.
    with np.load(temp, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)

        for key, original in payload.items():
            assert np.array_equal(
                saved[key], original
            ), key

    os.replace(temp, ART)

    result = {
        "status": "RGTAN_CROSSFIT_ADAPTER_PASS",
        "model": "RGTAN",
        "dataset": "FDCompCN",
        "ratio": "TR30",
        "source_sha256": SOURCE_SHA,
        "split_sha256": SPLIT_SHA,
        "author_commit": PIN,
        "split_seed": 2,
        "fold_seed": 2,
        "node_count": N,
        "feature_dimensions": 57,
        "graph_edges": int(adj.nnz),
        "two_hop_edges": int(two.nnz),
        "training_nodes": len(train),
        "training_positives": int(
            (train_labels == 1).sum()
        ),
        "validation_nodes": len(val),
        "test_nodes": len(test),
        "training_contexts": 5,
        "validation_contexts": 1,
        "risk_feature_names": RISK_NAMES,
        "validation_test_labels_used_in_features": False,
        "source_edge_label_and_mask_fields_ignored": True,
        "source_node_split_masks_ignored": True,
        "training_target_fold_labels_excluded": True,
        "normalization_fit_on_oof_training_only": True,
        "risk_attention_heads": 3,
        "training_started": False,
        "test_evaluated": False,
        "artifact_sha256": sha256(ART),
        "builder_sha256": sha256(Path(__file__)),
        "duration_seconds": round(
            time.perf_counter() - start, 3
        )
    }

    temp_meta = META.with_name(
        META.name + ".partial"
    )

    assert not temp_meta.exists()

    temp_meta.write_text(
        json.dumps(
            result, indent=2, sort_keys=True
        ) + "\n"
    )

    os.replace(temp_meta, META)

    assert sha256(ART) == json.loads(
        META.read_text()
    )["artifact_sha256"]

    print("ADAPTER_ARTIFACT=", ART, flush=True)
    print("ADAPTER_SHA256=", sha256(ART), flush=True)
    print("ADAPTER_MANIFEST=", META, flush=True)
    print("TRAINING_STARTED=NO", flush=True)
    print("TEST_EVALUATED=NO", flush=True)
    print("RGTAN_FDCOMPCN_TR40_ADAPTER=PASS", flush=True)

if __name__ == "__main__":
    main()
