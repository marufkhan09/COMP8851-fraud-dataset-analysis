"""Offline self-test for the COMP8851 shared library and both model implementations.

This runs on a small synthetic multi-relational graph, so it needs no dataset
download, no GPU and no network. It is the check to run after any change to the
shared evaluator, split registry, dataset layer or model code, and on a fresh
machine before the first real run.

What it verifies
----------------
* split generation: nesting, fixed validation/test IDs, eligibility, leakage
* evaluator: threshold rule, metric bundle, single-class handling, aggregation
* dataset layer: canonical structure, views, statistics, heterophily
* CARE-GNN: forward, backward, finite loss, gradient flow, RL threshold updates
* GHRN: forward, backward, refinement, edge removal (skipped without DGL)
* artefact bundle: every required file is produced and verified

Usage:
    python shared/comp8851/selftest.py
    python shared/comp8851/selftest.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.comp8851 import artifacts, datasets, evaluator, protocol, runtime, splits  # noqa: E402

PASSED: List[str] = []
FAILED: List[Tuple[str, str]] = []
SKIPPED: List[Tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name} :: {detail}")


def skip(name: str, reason: str) -> None:
    SKIPPED.append((name, reason))
    print(f"  SKIP  {name} :: {reason}")


# --------------------------------------------------------------------------
# Synthetic dataset
# --------------------------------------------------------------------------

def make_synthetic(num_nodes: int = 400, num_features: int = 16,
                   fraud_rate: float = 0.15, num_relations: int = 3,
                   seed: int = 2) -> datasets.CanonicalDataset:
    """A small multi-relational graph with a learnable, imbalanced signal.

    Fraud nodes get a shifted feature mean so a working model should beat
    chance; if the self-test ever reports AUROC near 0.5 on this graph, the
    training path is broken rather than the data being hard.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(num_nodes) < fraud_rate).astype(np.int64)

    features = rng.normal(0.0, 1.0, size=(num_nodes, num_features)).astype(np.float32)
    features[labels == 1] += 1.2   # separable but overlapping

    relations: Dict[str, sp.spmatrix] = {}
    names = ["rur", "rtr", "rsr"][:num_relations]
    for index, name in enumerate(names):
        density = 0.02 + 0.01 * index
        matrix = sp.random(num_nodes, num_nodes, density=density,
                           random_state=seed + index, format="csr")
        matrix.data = np.ones_like(matrix.data)
        matrix = matrix.maximum(matrix.T)
        matrix.setdiag(0)
        matrix.eliminate_zeros()
        relations[name] = matrix.tocsr()

    return datasets.CanonicalDataset(
        name="synthetic",
        features=features,
        labels=labels,
        eligible_idx=np.arange(num_nodes, dtype=np.int64),
        relations=relations,
        source_files={"generator": "shared/comp8851/selftest.py"},
        notes={"purpose": "offline self-test only; never a benchmark dataset"},
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_splits(dataset: datasets.CanonicalDataset) -> Dict[str, Dict[str, np.ndarray]]:
    print("\n[splits]")
    generated = splits.build_nested_splits(dataset.labels, dataset.eligible_idx,
                                           protocol.SPLIT_SEED)

    try:
        splits.verify_splits(generated, dataset.eligible_idx, dataset.num_nodes)
        check("split invariants hold", True)
    except RuntimeError as exc:
        check("split invariants hold", False, str(exc))

    train10 = set(generated["TR10"]["train_idx"].tolist())
    train20 = set(generated["TR20"]["train_idx"].tolist())
    train30 = set(generated["TR30"]["train_idx"].tolist())
    train40 = set(generated["TR40"]["train_idx"].tolist())
    check("training pools nested", train10 <= train20 <= train30 <= train40)

    valid_sets = [tuple(generated[r]["valid_idx"].tolist()) for r in generated]
    test_sets = [tuple(generated[r]["test_idx"].tolist()) for r in generated]
    check("validation IDs identical across ratios", len(set(valid_sets)) == 1)
    check("test IDs identical across ratios", len(set(test_sets)) == 1)

    for ratio, arrays in generated.items():
        try:
            splits.assert_no_leakage(arrays["train_idx"], arrays["valid_idx"], arrays["test_idx"])
            leak_free = True
        except RuntimeError:
            leak_free = False
        check(f"{ratio} has no train/valid/test leakage", leak_free)

    tr40 = generated["TR40"]
    total = sum(len(tr40[key]) for key in splits.SPLIT_KEYS)
    check("TR40 partitions cover the eligible set exactly",
          total == dataset.eligible_idx.size,
          f"covered {total}, expected {dataset.eligible_idx.size}")

    fraction = len(tr40["train_idx"]) / dataset.eligible_idx.size
    check("TR40 train fraction is about 0.40", abs(fraction - 0.40) < 0.02,
          f"got {fraction:.4f}")

    # Chronological path
    time_steps = np.arange(dataset.num_nodes) // 10
    chrono = splits.build_chronological_splits(dataset.labels, time_steps,
                                               dataset.eligible_idx, protocol.SPLIT_SEED)
    latest_train = time_steps[chrono["TR40"]["train_idx"]].max()
    earliest_test = time_steps[chrono["TR40"]["test_idx"]].min()
    check("chronological split keeps test after train", earliest_test >= latest_train,
          f"train max step {latest_train}, test min step {earliest_test}")

    chrono_nested = (set(chrono["TR10"]["train_idx"].tolist())
                     <= set(chrono["TR40"]["train_idx"].tolist()))
    check("chronological training pools nested", chrono_nested)

    return generated


def test_evaluator() -> None:
    print("\n[evaluator]")
    rng = np.random.default_rng(0)
    labels = (rng.random(500) < 0.2).astype(np.int64)
    scores = rng.random(500) * 0.4 + labels * 0.4   # informative but imperfect

    grid = evaluator.threshold_grid()
    check("threshold grid is 0.01..0.99 in 0.01 steps",
          len(grid) == 99 and abs(grid[0] - 0.01) < 1e-9 and abs(grid[-1] - 0.99) < 1e-9,
          f"len={len(grid)} first={grid[0]} last={grid[-1]}")

    threshold, macro = evaluator.select_threshold(labels, scores)
    check("selected threshold lies on the grid",
          any(abs(threshold - value) < 1e-9 for value in grid))

    bundle = evaluator.metric_bundle(labels, scores, threshold)
    for key in evaluator.METRIC_KEYS:
        check(f"metric {key} present and finite",
              key in bundle and np.isfinite(bundle[key]), str(bundle.get(key)))
    check("reported macro_f1 matches the selection value",
          abs(bundle["macro_f1"] - macro) < 1e-9)
    check("confusion counts sum to the evaluated set",
          bundle["tp"] + bundle["tn"] + bundle["fp"] + bundle["fn"] == labels.size)

    # A perfect ranking must score 1.0, which catches an inverted positive class.
    perfect = labels.astype(np.float64)
    perfect_bundle = evaluator.metric_bundle(labels, perfect, 0.5)
    check("perfect ranking gives AUROC 1.0", abs(perfect_bundle["auroc"] - 1.0) < 1e-9,
          str(perfect_bundle["auroc"]))
    check("perfect ranking gives fraud recall 1.0",
          abs(perfect_bundle["fraud_recall"] - 1.0) < 1e-9)

    inverted_bundle = evaluator.metric_bundle(labels, 1.0 - perfect, 0.5)
    check("inverted ranking gives AUROC 0.0", abs(inverted_bundle["auroc"]) < 1e-9,
          "positive class may be inverted")

    single = evaluator.metric_bundle(np.ones(10, dtype=np.int64), np.full(10, 0.7), 0.5)
    check("single-class evaluation reports NaN AUROC rather than a fake number",
          np.isnan(single["auroc"]))

    aggregate = evaluator.summarise_seeds([bundle, perfect_bundle])
    check("seed aggregation reports mean and std",
          "auroc" in aggregate and {"mean", "std", "n"} <= set(aggregate["auroc"]))


def test_dataset_layer(dataset: datasets.CanonicalDataset) -> None:
    print("\n[dataset layer]")
    try:
        datasets.validate_dataset(dataset)
        check("synthetic dataset passes validation", True)
    except ValueError as exc:
        check("synthetic dataset passes validation", False, str(exc))

    adjacency = dataset.adjacency_lists(add_self_loops=True)
    check("one adjacency list per relation", len(adjacency) == len(dataset.relations))
    check("every node has at least one neighbour",
          all(len(adjacency[0][node]) >= 1 for node in range(dataset.num_nodes)))
    check("self-loops present in adjacency lists",
          all(node in adjacency[0][node] for node in range(0, dataset.num_nodes, 37)))

    union = dataset.homogeneous_adjacency(add_self_loops=True)
    check("union view is square", union.shape == (dataset.num_nodes, dataset.num_nodes))
    check("union view is symmetric", (abs(union - union.T)).nnz == 0)
    check("union view is binary", set(np.unique(union.data).tolist()) <= {1.0})

    relation_edges = sum(dataset.relations[name].nnz for name in dataset.relations)
    check("union is no larger than the sum of relations plus self-loops",
          union.nnz <= relation_edges + dataset.num_nodes)

    stats = dataset.statistics()
    check("statistics report node and feature counts",
          stats["nodes"] == dataset.num_nodes
          and stats["feature_dimension"] == dataset.num_features)
    check("statistics report fraud imbalance", stats["fraud_nodes"] > 0
          and stats["normal_nodes"] > 0)

    hetero = dataset.heterophily()
    check("heterophily computed in [0, 1]",
          hetero["global_heterophily"] is not None
          and 0.0 <= hetero["global_heterophily"] <= 1.0)

    relation_view = dataset.view_manifest("relation")
    homogeneous_view = dataset.view_manifest("homogeneous")
    check("view hashes differ between the two views",
          relation_view["view_sha256"] != homogeneous_view["view_sha256"])
    check("view hash is deterministic",
          dataset.view_manifest("relation")["view_sha256"] == relation_view["view_sha256"])


def test_care_gnn(dataset: datasets.CanonicalDataset,
                  split: Dict[str, np.ndarray], verbose: bool) -> None:
    print("\n[CARE-GNN]")
    try:
        import torch
    except ImportError:
        skip("CARE-GNN training", "torch is not installed")
        return

    sys.path.insert(0, str(REPO_ROOT / "models" / "care-gnn"))
    try:
        from caregnn.model import build_care_gnn
    except ModuleNotFoundError:
        # A single-model working tree is legitimate: each model ships its own
        # notebook, so CARE-GNN may simply not be present here.
        skip("CARE-GNN training", "the caregnn package is not present in this tree")
        return

    adjacency = dataset.adjacency_lists(add_self_loops=True)
    features = torch.tensor(dataset.features, dtype=torch.float32)
    model = build_care_gnn(features, adjacency, embed_dim=16, inter="GNN",
                           step_size=0.02, lambda_1=2.0, verbose=verbose)

    counts = runtime.count_parameters(model)
    check("CARE-GNN has trainable parameters", counts["trainable_parameters"] > 0)
    check("node feature table is frozen",
          not model.inter1.features.weight.requires_grad)

    train_idx = split["train_idx"]
    labels_np = dataset.labels
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-3
    )

    batch = [int(n) for n in train_idx[:64]]
    batch_labels = torch.tensor(labels_np[np.asarray(batch)], dtype=torch.long)

    model.train()
    model.inter1.batch_num = 1
    loss = model.loss(batch, batch_labels, train_flag=True)
    check("initial loss is finite", bool(torch.isfinite(loss)), str(loss.item()))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    with_grad = [name for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None and torch.any(p.grad != 0)]
    check("gradients reach the inter-relation weight",
          any("inter1.weight" in name for name in with_grad), str(with_grad))
    check("gradients reach the label classifier",
          any("label_clf" in name for name in with_grad))
    check("gradients reach the output classifier",
          any(name == "weight" for name in with_grad), str(with_grad))
    optimizer.step()

    # Train briefly and confirm the loss actually moves.
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        step_loss = model.loss(batch, batch_labels, train_flag=True)
        step_loss.backward()
        optimizer.step()
        losses.append(float(step_loss))
    check("loss decreases over a short training run", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    check("all losses finite", all(np.isfinite(losses)))

    # RL threshold module.
    initial = list(model.inter1.thresholds)
    model.inter1.batch_num = 2
    model.inter1.relation_score_log.clear()
    for _ in range(6):
        model.loss(batch, batch_labels, train_flag=True)
    moved = model.inter1.thresholds
    check("RL module recorded relation distances",
          len(model.inter1.relation_score_log) >= 4)
    check("RL thresholds stay inside [0.001, 0.999]",
          all(0.001 <= t <= 0.999 for t in moved), str(moved))
    check("RL thresholds updated away from the initial 0.5",
          moved != initial or len(model.inter1.thresholds_log) > 1,
          f"initial {initial} now {moved}")

    # Inference path and label orientation.
    model.eval()
    valid_idx = split["valid_idx"][:128]
    nodes = [int(n) for n in valid_idx]
    node_labels = torch.tensor(labels_np[valid_idx], dtype=torch.long)
    probabilities, _ = model.to_prob(nodes, node_labels, train_flag=False)
    values = probabilities[:, 1].detach().numpy()
    check("probabilities lie in [0, 1]", bool(np.all((values >= 0) & (values <= 1))))
    check("probability vector matches the node count", values.shape[0] == len(nodes))

    metrics = evaluator.evaluate_validation(labels_np[valid_idx], values)
    check("evaluation of CARE-GNN output produces finite AUROC",
          np.isfinite(metrics["auroc"]), str(metrics["auroc"]))

    # Checkpoint round trip.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "care.pt"
        torch.save(model.state_dict(), path)
        reloaded = build_care_gnn(features, adjacency, embed_dim=16, inter="GNN")
        reloaded.load_state_dict(torch.load(path, map_location="cpu"))
        reloaded.eval()
        again, _ = reloaded.to_prob(nodes, node_labels, train_flag=False)
        check("checkpoint reload reproduces predictions",
              bool(np.allclose(values, again[:, 1].detach().numpy(), atol=1e-6)))

    # The other three published aggregators must at least run.
    for variant in ("Mean", "Weight", "Att"):
        try:
            other = build_care_gnn(features, adjacency, embed_dim=16, inter=variant)
            other.inter1.batch_num = 1
            variant_loss = other.loss(batch, batch_labels, train_flag=True)
            ok = bool(torch.isfinite(variant_loss))
            detail = str(variant_loss.item())
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(f"CARE-{variant} aggregator runs", ok, detail)


def test_ghrn_edge_selection() -> None:
    """GHRN's pruning rule, tested on plain tensors so no graph library is needed."""
    print("\n[GHRN edge selection, library-free]")
    try:
        import torch
    except ImportError:
        skip("GHRN edge selection", "torch is not installed")
        return

    sys.path.insert(0, str(REPO_ROOT / "models" / "ghrn"))
    try:
        from ghrnlib.graph_refine import select_edges_to_delete
    except ModuleNotFoundError:
        skip("GHRN edge selection", "the ghrnlib package is not present in this tree")
        return

    # Ten edges with known scores; the two lowest are edges 3 and 7.
    scores = torch.tensor([0.9, 0.5, 0.4, -0.9, 0.8, 0.2, 0.6, -0.7, 0.3, 0.1])
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    chosen = select_edges_to_delete(scores, src, dst, del_ratio=0.2)
    check("selection removes exactly del_ratio of the edges", chosen.numel() == 2,
          f"removed {chosen.numel()}")
    check("selection removes the most negative scores first",
          set(chosen.tolist()) == {3, 7}, str(sorted(chosen.tolist())))

    none_chosen = select_edges_to_delete(scores, src, dst, del_ratio=0.0)
    check("del_ratio 0 deletes nothing", none_chosen.numel() == 0)

    # Self-loops must survive even when they score worst.
    loop_src = torch.tensor([0, 1, 2, 3])
    loop_dst = torch.tensor([0, 2, 3, 1])
    loop_scores = torch.tensor([-5.0, 0.4, 0.3, 0.9])
    protected = select_edges_to_delete(loop_scores, loop_src, loop_dst,
                                       del_ratio=0.5, protect_self_loops=True)
    check("self-loops are protected from deletion", 0 not in protected.tolist(),
          str(protected.tolist()))

    unprotected = select_edges_to_delete(loop_scores, loop_src, loop_dst,
                                         del_ratio=0.5, protect_self_loops=False)
    check("self-loop protection can be disabled", 0 in unprotected.tolist())

    try:
        select_edges_to_delete(scores, src, dst, del_ratio=1.5)
        rejected = False
    except ValueError:
        rejected = True
    check("an out-of-range del_ratio is rejected", rejected)


def test_ghrn(dataset: datasets.CanonicalDataset,
              split: Dict[str, np.ndarray], verbose: bool) -> None:
    print("\n[GHRN]")
    try:
        import torch
    except ImportError:
        skip("GHRN training", "torch is not installed")
        return

    sys.path.insert(0, str(REPO_ROOT / "models" / "ghrn"))
    try:
        from ghrnlib.backend import build_graph, dgl_available
        from ghrnlib.graph_refine import measure_pruning_quality, refine_graph
        from ghrnlib.model import build_ghrn
    except ModuleNotFoundError:
        skip("GHRN training", "the ghrnlib package is not present in this tree")
        return

    backend = "dgl" if dgl_available() else "torch-sparse"
    print(f"  (graph backend: {backend})")
    if backend != "dgl":
        skip("GHRN on the DGL backend",
             "DGL has no wheel for this Python/OS; the torch-sparse fallback is "
             "exercised instead. Re-run this self-test on the benchmark host to "
             "cover the DGL path and compare_backends().")

    features = torch.as_tensor(dataset.features, dtype=torch.float32)
    labels = torch.as_tensor(dataset.labels, dtype=torch.int64)
    src, dst = dataset.homogeneous_edges(add_self_loops=False)
    graph = build_graph(src, dst, dataset.num_nodes,
                        ndata={"feature": features, "label": labels})
    train_ids = torch.as_tensor(split["train_idx"], dtype=torch.long)

    model = build_ghrn(in_feats=features.shape[1], h_feats=16, d=2, del_ratio=0.05)
    counts = runtime.count_parameters(model)
    check("GHRN has trainable parameters", counts["trainable_parameters"] > 0)
    check("beta-wavelet bank has d+1 filters", len(model.backbone.conv) == 3,
          str(len(model.backbone.conv)))

    import torch.nn.functional as F

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fraud = int(labels[train_ids].sum())
    weight = torch.tensor([1.0, (len(train_ids) - fraud) / max(fraud, 1)])

    model.train()
    logits = model(graph, features)
    check("forward pass returns one row per node and two classes",
          logits.shape == (graph.num_nodes(), 2), str(tuple(logits.shape)))
    loss = F.cross_entropy(logits[train_ids], labels[train_ids], weight=weight)
    check("initial loss is finite", bool(torch.isfinite(loss)), str(loss.item()))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    with_grad = [name for name, p in model.named_parameters()
                 if p.grad is not None and torch.any(p.grad != 0)]
    check("gradients reach the backbone input projection",
          any("linear.weight" in name for name in with_grad), str(with_grad[:5]))
    check("gradients reach the output layer",
          any("linear4" in name for name in with_grad))
    optimizer.step()

    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        step_logits = model(graph, features)
        step_loss = F.cross_entropy(step_logits[train_ids], labels[train_ids], weight=weight)
        step_loss.backward()
        optimizer.step()
        losses.append(float(step_loss))
    check("loss decreases over a short training run", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    # Refinement.
    posteriors = model.posteriors(graph, features)
    check("posteriors are a valid distribution",
          bool(torch.allclose(posteriors.sum(dim=1), torch.ones(graph.num_nodes()), atol=1e-5)))

    refined, statistics = refine_graph(graph, posteriors, del_ratio=0.05,
                                       mode="post_aggregation")
    check("refinement removed edges", statistics["edges_removed"] > 0, str(statistics))
    check("refined graph is smaller than the original",
          refined.num_edges() < graph.num_edges(),
          f"{graph.num_edges()} -> {refined.num_edges()}")
    check("refined graph keeps every node", refined.num_nodes() == graph.num_nodes())
    check("refined graph retains node features", "feature" in refined.ndata)

    zero_refined, zero_statistics = refine_graph(graph, posteriors, del_ratio=0.0)
    check("del_ratio 0 leaves the graph untouched",
          zero_statistics["edges_removed"] == 0
          and zero_refined.num_edges() == graph.num_edges())

    kl_refined, kl_statistics = refine_graph(graph, posteriors, del_ratio=0.05, mode="kl")
    check("KL refinement mode runs and removes edges",
          kl_statistics["edges_removed"] > 0, str(kl_statistics))

    train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    train_mask[train_ids] = True
    quality = measure_pruning_quality(graph, refined, labels, train_mask)
    check("pruning diagnostic reports heterophily before and after",
          quality["edge_heterophily_before"] is not None
          and quality["edge_heterophily_after"] is not None, str(quality))

    # Forward pass on the refined graph, and checkpoint round trip.
    model.refine(graph, posteriors)
    refined_logits = model(graph, features)
    check("forward pass on the refined graph works",
          refined_logits.shape == (graph.num_nodes(), 2))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ghrn.pt"
        torch.save(model.state_dict(), path)
        reloaded = build_ghrn(in_feats=features.shape[1], h_feats=16, d=2, del_ratio=0.05)
        reloaded.load_state_dict(torch.load(path, map_location="cpu"))
        check("checkpoint reload succeeds", True)

    model.reset_parameters()
    check("reset_parameters runs before stage-2 retraining", True)


def test_artifacts(dataset: datasets.CanonicalDataset,
                   split: Dict[str, np.ndarray]) -> None:
    print("\n[artefact bundle]")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        recorder = artifacts.RunRecorder(
            run_dir=run_dir, model="SelfTest", dataset="synthetic", track="unified",
            ratio="TR40", seed=2, run_id="selftest",
        )
        identity = protocol.ProtocolIdentity(
            hardware_profile_id="cpu", model_source_commit="0" * 40,
            source_sha256="deadbeef", dataset_view_id="synthetic_relation_v1",
            split_sha256="cafebabe",
        )
        recorder.begin({"model": "SelfTest", "lr": 0.01}, identity,
                       dataset.manifest(), dataset.view_manifest("relation"))
        recorder.record_split(split, dataset.labels, "selftest")

        for epoch in range(3):
            metrics = {"auroc": 0.7 + epoch * 0.01, "auprc": 0.4, "macro_f1": 0.6,
                       "threshold": 0.5}
            recorder.record_epoch(epoch, 0.5, 0.3, 0.1, metrics, is_best=(epoch == 2))
            recorder.record_validation(epoch, metrics, 0.1, is_best=(epoch == 2))

        recorder.set_best(2, {"auprc": 0.4}, 0.5, "auprc", 0.4)
        (recorder.checkpoint_dir / "dummy.pt").write_bytes(b"checkpoint")
        recorder.terminal_log.write_text("selftest\n", encoding="utf-8")

        verification = recorder.finalise(test_metrics={"auroc": 0.72, "auprc": 0.41})

        check("artefact bundle reports complete", verification["complete"],
              str(verification))
        check("no required artefact missing", not verification["missing_artifacts"],
              str(verification["missing_artifacts"]))
        for name in artifacts.REQUIRED_ARTIFACTS:
            check(f"artefact {name} written", (run_dir / name).exists())
        check("test_metrics.json written", (run_dir / "test_metrics.json").exists())
        check("checkpoint directory populated", any(recorder.checkpoint_dir.iterdir()))

        import json

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        check("summary records the protocol version",
              summary["protocol_identity"]["protocol_version"] == protocol.PROTOCOL_VERSION)
        check("summary records status COMPLETE", summary["status"] == "COMPLETE")
        check("summary records epoch timing",
              summary["timing"]["training_epochs_completed"] == 3)

        # A failed run must still produce a record with the error preserved.
        fail_dir = Path(tmp) / "failed"
        failer = artifacts.RunRecorder(
            run_dir=fail_dir, model="SelfTest", dataset="synthetic", track="unified",
            ratio="TR40", seed=2, run_id="selftest-fail",
        )
        failer.begin({"model": "SelfTest"}, identity, dataset.manifest(),
                     dataset.view_manifest("relation"))
        failer.finalise(test_metrics=None, status="FAILED",
                        error=RuntimeError("synthetic failure"))
        failed_summary = json.loads((fail_dir / "summary.json").read_text(encoding="utf-8"))
        check("failed run records status FAILED", failed_summary["status"] == "FAILED")
        check("failed run preserves the traceback",
              "traceback" in failed_summary.get("error", {}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COMP8851 offline self-test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    runtime.apply_thread_caps()
    runtime.seed_everything(protocol.SPLIT_SEED)

    print("=" * 78)
    print("COMP8851 SELF-TEST (synthetic data, no download, no GPU)")
    print("=" * 78)
    print(f"Protocol: {protocol.PROTOCOL_VERSION}")
    for key, value in sorted(runtime.environment_metadata().items()):
        if key in ("python_version", "torch_version", "dgl_version", "numpy_version",
                   "sklearn_version"):
            print(f"  {key}: {value}")

    dataset = make_synthetic()
    generated = test_splits(dataset)
    test_evaluator()
    test_dataset_layer(dataset)
    test_care_gnn(dataset, generated["TR40"], args.verbose)
    test_ghrn_edge_selection()
    test_ghrn(dataset, generated["TR40"], args.verbose)
    test_artifacts(dataset, generated["TR40"])

    print("\n" + "=" * 78)
    print(f"PASSED {len(PASSED)}   FAILED {len(FAILED)}   SKIPPED {len(SKIPPED)}")
    if SKIPPED:
        print("\nSkipped:")
        for name, reason in SKIPPED:
            print(f"  - {name}: {reason}")
    if FAILED:
        print("\nFailures:")
        for name, detail in FAILED:
            print(f"  - {name}: {detail}")
        print("SELF-TEST RESULT: FAIL")
        return 1
    print("SELF-TEST RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
