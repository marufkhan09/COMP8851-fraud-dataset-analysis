"""Run a single GHRN experiment under COMP8851 protocol v4.4.

One invocation = one (dataset, track, ratio, seed) cell of the run matrix.

GHRN is trained with the published two-stage procedure:

  Stage 1  train the beta-wavelet backbone on the unrefined graph and take its
           class posteriors (equivalent to a BWGNN run, del_ratio = 0)
  Stage 2  score edges from those posteriors, delete the ``--del-ratio`` most
           heterophilic fraction, reinitialise and retrain on the refined graph

With ``--del-ratio 0`` only stage 1 runs, which is the BWGNN baseline the
paper compares against. Both stages obey the same stopping rule, selection
metric and evaluator, and stage 2's checkpoint is the one evaluated on test.

Example
-------
    python models/ghrn/scripts/run_one.py \
        --dataset tfinance --data-path data/tfinance \
        --ratio TR40 --seed 2 --del-ratio 0.015
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(MODEL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import shared.comp8851 as comp8851  # noqa: E402
from shared.comp8851 import artifacts, datasets, evaluator, protocol, runtime, splits  # noqa: E402

shared_version = comp8851.__version__

runtime.apply_thread_caps()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ghrnlib.backend import build_graph, dgl_available  # noqa: E402
from ghrnlib.graph_refine import measure_pruning_quality  # noqa: E402
from ghrnlib.model import build_ghrn  # noqa: E402

MODEL_NAME = "GHRN"
MODEL_SLUG = "ghrn"
UPSTREAM_REPO = "https://github.com/blacksingular/GHRN"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GHRN runner for COMP8851")

    parser.add_argument("--dataset", required=True, choices=list(protocol.DATASETS))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--split-dir", default=str(REPO_ROOT / "shared" / "splits"))
    parser.add_argument("--ratio", default="TR40", choices=list(protocol.TRAIN_RATIOS))
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--track", default="unified", choices=list(protocol.TRACKS))
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))

    # Model hyperparameters
    parser.add_argument("--hid-dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2,
                        help="Beta-wavelet bank order d; the bank holds d+1 filters")
    parser.add_argument("--del-ratio", type=float, default=0.0,
                        help="Fraction of most-heterophilic edges to delete (0 = BWGNN baseline)")
    parser.add_argument("--refine-mode", default="post_aggregation",
                        choices=["post_aggregation", "kl"])
    parser.add_argument("--stage-selection", default="validation",
                        choices=["validation", "always-refined"],
                        help="Which stage supplies the final checkpoint. "
                             "'validation' takes whichever of stage 1 and stage 2 "
                             "scored higher on the selection metric, as protocol "
                             "v4.4 requires. 'always-refined' reproduces the "
                             "published GHRN procedure, which ships stage 2 "
                             "regardless of how it validated.")
    parser.add_argument("--dropout", type=float, default=0.0)

    # Optimisation
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=protocol.MAX_EPOCHS)
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=protocol.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--selection-metric", default=protocol.SELECTION_METRIC,
                        choices=["auprc", "auroc", "macro_f1"])

    # Execution
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--run-mode", default="controlled")
    parser.add_argument("--hardware-profile-id", default=None)
    parser.add_argument("--container-digest", default=None)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not evaluate the test set. Required for tuning "
                             "trials: it makes test isolation a property of the "
                             "command rather than a promise.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--strict-stats", dest="strict_stats", action="store_true", default=True)
    parser.add_argument("--no-strict-stats", dest="strict_stats", action="store_false")
    return parser.parse_args(argv)


def resolve_split(args, dataset: datasets.CanonicalDataset):
    """Load the frozen split, or generate and cache the registry splits."""
    if args.split_path:
        arrays = splits.load_split(args.split_path)
        return arrays, str(Path(args.split_path).resolve()), protocol.sha256_file(args.split_path)

    split_dir = Path(args.split_dir) / args.dataset
    expected = split_dir / splits.split_filename(args.dataset, args.ratio, protocol.SPLIT_SEED)
    if not expected.exists():
        print(f"[split] generating registry splits for {args.dataset} into {split_dir}")
        if args.dataset in protocol.CHRONOLOGICAL_DATASETS:
            if dataset.time_steps is None:
                raise RuntimeError(f"{args.dataset} requires time steps for a chronological split.")
            generated = splits.build_chronological_splits(
                dataset.labels, dataset.time_steps, dataset.eligible_idx, protocol.SPLIT_SEED
            )
        else:
            generated = splits.build_nested_splits(
                dataset.labels, dataset.eligible_idx, protocol.SPLIT_SEED
            )
        splits.verify_splits(generated, dataset.eligible_idx, dataset.num_nodes)
        splits.save_splits(generated, split_dir, args.dataset, dataset.labels,
                           protocol.SPLIT_SEED,
                           extra_manifest={"source_files": dataset.source_files})
    arrays = splits.load_split(expected)
    return arrays, str(expected.resolve()), protocol.sha256_file(expected)


@torch.no_grad()
def predict(model, graph, features, node_ids: np.ndarray, device) -> Tuple[np.ndarray, float]:
    """Fraud probabilities for a node subset, with inference time measured."""
    model.eval()
    runtime.synchronise(device)
    started = time.perf_counter()
    logits = model(graph, features)
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    index = torch.as_tensor(node_ids, dtype=torch.long, device=device)
    selected = probabilities[index].detach().cpu().numpy()
    runtime.synchronise(device)
    return selected.astype(np.float64), time.perf_counter() - started


def train_stage(model, graph, features, labels, train_ids, valid_idx, labels_np,
                loss_weight, optimizer, args, recorder, device, stage: str,
                epoch_offset: int = 0) -> Dict[str, object]:
    """Train one stage to its best validation checkpoint.

    Returns the stage outcome: best epoch, selection value, threshold, metrics
    and the checkpoint path used.
    """
    checkpoint = recorder.checkpoint_dir / f"ghrn_{stage}.pt"
    best_selection = float("-inf")
    best_epoch = -1
    best_threshold = 0.5
    best_validation: Dict[str, float] = {}
    epochs_without_improvement = 0
    stopped_early = ""
    started = time.perf_counter()
    deadline = started + args.max_minutes * 60 if args.max_minutes else None

    for epoch in range(args.epochs):
        model.train()
        runtime.synchronise(device)
        epoch_started = time.perf_counter()

        logits = model(graph, features)
        loss = F.cross_entropy(logits[train_ids], labels[train_ids], weight=loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        runtime.synchronise(device)
        train_seconds = time.perf_counter() - epoch_started
        loss_value = float(loss.detach().cpu())
        if not np.isfinite(loss_value):
            raise RuntimeError(f"[{stage}] loss became non-finite at epoch {epoch}: {loss_value}")

        validation_metrics = None
        validation_seconds = 0.0
        is_best = False
        should_validate = (
            epoch == 0 or (epoch + 1) % args.valid_every == 0 or epoch == args.epochs - 1
        )

        if should_validate:
            probabilities, validation_seconds = predict(model, graph, features, valid_idx, device)
            validation_metrics = evaluator.evaluate_validation(labels_np[valid_idx], probabilities)
            selection = evaluator.selection_value(validation_metrics, args.selection_metric)
            recorder.record_validation(epoch_offset + epoch, validation_metrics, validation_seconds)

            if selection > best_selection + 1e-12:
                best_selection = selection
                best_epoch = epoch
                best_threshold = float(validation_metrics["threshold"])
                best_validation = dict(validation_metrics)
                is_best = True
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                epochs_without_improvement += args.valid_every

            print(f"[{stage}] epoch {epoch + 1:3d}/{args.epochs} | loss {loss_value:.6f} | "
                  f"train {train_seconds:.3f}s | val AUPRC {validation_metrics['auprc']:.6f} | "
                  f"val AUROC {validation_metrics['auroc']:.6f} | "
                  f"thr {validation_metrics['threshold']:.2f} | best epoch {best_epoch + 1}")
        else:
            print(f"[{stage}] epoch {epoch + 1:3d}/{args.epochs} | loss {loss_value:.6f} | "
                  f"train {train_seconds:.3f}s")

        recorder.record_epoch(epoch_offset + epoch, train_seconds, loss_value,
                              validation_seconds, validation_metrics, is_best,
                              notes=stage)

        if args.patience and should_validate and epochs_without_improvement >= args.patience:
            stopped_early = f"early stopping: no improvement for {args.patience} epochs"
            print(f"[{stage}] {stopped_early}")
            break
        if deadline and time.perf_counter() > deadline:
            stopped_early = f"time cap of {args.max_minutes} minutes reached"
            print(f"[{stage}] {stopped_early}")
            break

    if best_epoch < 0:
        raise RuntimeError(f"[{stage}] no validation evaluation completed; nothing to select.")

    return {
        "stage": stage,
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "best_threshold": best_threshold,
        "best_validation": best_validation,
        "checkpoint": checkpoint,
        "epochs_run": epoch + 1,
        "wall_seconds": time.perf_counter() - started,
        "stop_reason": stopped_early,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = 0

    runtime.seed_everything(args.seed)
    device = runtime.resolve_device(no_cuda=args.no_cuda, require_cuda=args.require_cuda)

    run_dir = protocol.result_dir(args.results_root, MODEL_SLUG, args.dataset,
                                  args.track, args.ratio, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    with runtime.capture_terminal(run_dir / "terminal.log"):
        return _execute(args, device, run_dir)


def _execute(args, device, run_dir: Path) -> int:
    timestamp = runtime.utc_stamp()
    gpu = runtime.gpu_profile_id(device)
    run_id = protocol.run_name(MODEL_SLUG, args.dataset, args.track, args.ratio,
                               args.seed, gpu, timestamp)

    print("=" * 78)
    print(f"{MODEL_NAME} | {args.dataset} | {args.track} | {args.ratio} | seed {args.seed}")
    print("=" * 78)
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Protocol: {protocol.PROTOCOL_VERSION}  selection={args.selection_metric}")
    print(f"Deletion ratio: {args.del_ratio} ({args.refine_mode})")
    if args.smoke:
        print("MODE: SMOKE TEST - feasibility only, metrics are not benchmark evidence")

    recorder = artifacts.RunRecorder(
        run_dir=run_dir, model=MODEL_NAME, dataset=args.dataset, track=args.track,
        ratio=args.ratio, seed=args.seed, run_id=run_id, device=device,
    )

    error: BaseException | None = None
    test_metrics = None
    status = "COMPLETE"
    fit_wall_seconds = 0.0
    data_load_seconds = 0.0

    try:
        load_started = time.perf_counter()
        dataset = datasets.load_dataset(args.dataset, args.data_path, strict=args.strict_stats)
        data_load_seconds = time.perf_counter() - load_started
        print(f"Loaded {dataset.name}: {dataset.num_nodes:,} nodes, "
              f"{dataset.num_features} features, {len(dataset.relations)} relation(s)")

        if dataset.is_multi_relational:
            print(f"[compatibility] NOTE {dataset.name} is multi-relational; GHRN is a "
                  "homogeneous spectral method and receives the frozen union view. "
                  "Record this pair as PASS_WITH_SHARED_ADAPTER (A1).")

        split_arrays, split_source, split_hash = resolve_split(args, dataset)
        splits.assert_no_leakage(split_arrays["train_idx"], split_arrays["valid_idx"],
                                 split_arrays["test_idx"])
        labels_np = dataset.labels
        recorder.record_split(split_arrays, labels_np, split_source)

        train_idx = split_arrays["train_idx"]
        valid_idx = split_arrays["valid_idx"]
        test_idx = split_arrays["test_idx"]
        print(f"Split: train {len(train_idx):,} | valid {len(valid_idx):,} | test {len(test_idx):,}")

        src, dst = dataset.homogeneous_edges(add_self_loops=False)
        features = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
        labels = torch.as_tensor(dataset.labels, dtype=torch.int64, device=device)
        index_dtype = {"int32": torch.int32, "int64": torch.int64,
                       "auto": None}[args.index_dtype]
        graph = build_graph(
            src.to(device), dst.to(device), dataset.num_nodes,
            ndata={"feature": features, "label": labels},
            idtype=index_dtype,
        )
        backend = "dgl" if dgl_available() else "torch-sparse"
        train_ids = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        graph_idtype = str(getattr(graph, "idtype", "n/a"))
        print(f"Graph: {graph.num_nodes():,} nodes, {graph.num_edges():,} stored "
              f"directed edges (backend: {backend}, index dtype: {graph_idtype})")
        recorder.add("graph_index_dtype", graph_idtype)
        if backend != "dgl":
            print("[environment] WARNING DGL is unavailable, so the torch-sparse "
                  "fallback backend is in use. This is valid for smoke tests, but "
                  "final controlled runs must use DGL, the authors' stack.")

        fraud = int(labels[train_ids].sum().item())
        normal = int(len(train_ids) - fraud)
        if fraud == 0:
            raise RuntimeError("Training split contains no fraud nodes.")
        class_weight = normal / fraud
        loss_weight = torch.tensor([1.0, class_weight], dtype=torch.float32, device=device)
        print(f"Training fraud/normal: {fraud:,}/{normal:,} | fraud weight {class_weight:.4f}")

        model = build_ghrn(
            in_feats=features.shape[1], h_feats=args.hid_dim, num_classes=2,
            d=args.order, del_ratio=args.del_ratio, dropout=args.dropout,
            refine_mode=args.refine_mode,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     betas=(0.9, 0.999), eps=1e-8)

        parameter_counts = runtime.count_parameters(model)
        print(f"Parameters: {parameter_counts['trainable_parameters']:,} trainable")

        config = {k: v for k, v in vars(args).items()}
        config.update({
            "model": MODEL_NAME,
            "upstream_repository": UPSTREAM_REPO,
            "resolved_device": str(device),
            "optimizer": "adam",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "architecture": "GHRN: beta-wavelet backbone with heterophily-aware graph refinement",
            "class_imbalance_handling": "weighted cross-entropy (normal/fraud ratio)",
            "graph_backend": backend,
            **parameter_counts,
        })
        identity = protocol.ProtocolIdentity(
            hardware_profile_id=args.hardware_profile_id or gpu,
            container_digest=args.container_digest,
            model_source_commit=protocol.git_commit(REPO_ROOT),
            source_sha256=next(iter(dataset.source_sha256.values()), None)
                          or "not-hashed-dgl-managed-download",
            dataset_manifest_version="comp8851-canonical-v1",
            dataset_view_id=f"{dataset.name}_homogeneous_v1",
            view_sha256=dataset.view_manifest("homogeneous")["view_sha256"],
            split_registry_version=f"seed{protocol.SPLIT_SEED}",
            split_sha256=split_hash,
            evaluator_commit=f"shared.comp8851=={shared_version}",
            adapter_version="homogeneous-union-view-v1",
            selection_metric=args.selection_metric,
        )
        recorder.begin(config, identity, dataset.manifest(),
                       dataset.view_manifest("homogeneous"))

        runtime.reset_peak_memory(device)
        fit_started = time.perf_counter()

        # ---------------- stage 1: unrefined backbone ----------------
        print("-" * 78)
        print("STAGE 1: training on the unrefined graph (del_ratio = 0)")
        stage1 = train_stage(model, graph, features, labels, train_ids, valid_idx,
                             labels_np, loss_weight, optimizer, args, recorder,
                             device, stage="stage1_unrefined")
        recorder.add("stage1", {k: v for k, v in stage1.items() if k != "checkpoint"})

        selected = stage1
        refinement_statistics: Dict[str, object] = {"del_ratio": args.del_ratio,
                                                    "applied": False}

        # ---------------- stage 2: refine and retrain ----------------
        if args.del_ratio > 0:
            print("-" * 78)
            print(f"STAGE 2: refining the graph (del_ratio = {args.del_ratio}) and retraining")

            model.load_state_dict(torch.load(stage1["checkpoint"], map_location=device))
            posteriors = model.posteriors(graph, features)
            refined = model.refine(graph, posteriors)
            refinement_statistics = dict(model.refinement_statistics)
            refinement_statistics["applied"] = True

            # Diagnostic only, restricted to training nodes so no validation or
            # test label ever informs the refinement.
            train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
            train_mask[train_ids] = True
            refinement_statistics.update(
                measure_pruning_quality(graph, refined, labels, train_mask)
            )
            print(f"Edges {refinement_statistics['edges_before']:,} -> "
                  f"{refinement_statistics['edges_after']:,} "
                  f"({refinement_statistics['edges_removed']:,} removed)")
            before = refinement_statistics.get("edge_heterophily_before")
            after = refinement_statistics.get("edge_heterophily_after")
            if before is not None and after is not None:
                print(f"Training-edge heterophily {before:.6f} -> {after:.6f}")

            # Fresh initialisation so stage 2 is a clean training run on the
            # refined graph rather than a continuation of stage 1.
            model.reset_parameters()
            runtime.seed_everything(args.seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         betas=(0.9, 0.999), eps=1e-8)
            stage2 = train_stage(model, refined, features, labels, train_ids, valid_idx,
                                 labels_np, loss_weight, optimizer, args, recorder,
                                 device, stage="stage2_refined",
                                 epoch_offset=stage1["epochs_run"])
            recorder.add("stage2", {k: v for k, v in stage2.items() if k != "checkpoint"})

            # Protocol v4.4 ships the configuration that validated best. The
            # published GHRN procedure always ships the refined stage, which
            # means that whenever refinement hurts, the run ships a model known
            # to be worse on validation. Keep the published behaviour reachable
            # via --stage-selection always-refined, but do not make it default.
            if args.stage_selection == "validation":
                selected = max((stage1, stage2), key=lambda s: s["best_selection"])
            else:
                selected = stage2

            recorder.add("stage_selection", {
                "rule": args.stage_selection,
                "metric": args.selection_metric,
                "stage1": stage1["best_selection"],
                "stage2": stage2["best_selection"],
                "chosen": selected["stage"],
                "margin": abs(stage2["best_selection"] - stage1["best_selection"]),
                "refinement_helped": stage2["best_selection"] > stage1["best_selection"],
            })
            print(f"Stage selection ({args.stage_selection}): "
                  f"stage1 {stage1['best_selection']:.6f} vs "
                  f"stage2 {stage2['best_selection']:.6f} "
                  f"-> {selected['stage']}")
        else:
            print("del_ratio = 0: stage 2 skipped; this run is the BWGNN-equivalent baseline.")

        fit_wall_seconds = time.perf_counter() - fit_started
        recorder.add("refinement", refinement_statistics)

        # ---------------- test ----------------
        # Under --skip-test the test split is never read, never scored and never
        # written. This is how tuning trials stay isolated from test data.
        # Restore the selected checkpoint in both branches. Without this, a
        # --skip-test run would persist the last-epoch weights as "best", so the
        # saved checkpoint would not be the one its reported validation score
        # came from.
        model.load_state_dict(torch.load(selected["checkpoint"], map_location=device))

        if args.skip_test:
            print("--skip-test: the test set was not loaded, scored or written.")
            print("This run is a tuning trial and reports validation metrics only.")
            recorder.add("test_skipped", True)
            recorder.add("evidence_class", "tuning_trial")
        else:
            evaluation_graph = model.active_graph(graph)
            probabilities, inference_seconds = predict(
                model, evaluation_graph, features, test_idx, device
            )
            test_metrics = evaluator.metric_bundle(
                labels_np[test_idx], probabilities, selected["best_threshold"]
            )
            test_metrics["inference_seconds"] = inference_seconds
            if args.smoke:
                test_metrics["evidence_class"] = "feasibility_only"

        recorder.set_best(selected["best_epoch"], selected["best_validation"],
                          selected["best_threshold"], args.selection_metric,
                          selected["best_selection"])
        recorder.add("selected_stage", selected["stage"])
        recorder.add("data_load_seconds", data_load_seconds)
        recorder.add("heterophily", dataset.heterophily())
        torch.save(model.state_dict(), recorder.checkpoint_path)

        print("-" * 78)
        if test_metrics is None:
            for key in ("auprc", "auroc", "macro_f1"):
                if key in selected["best_validation"]:
                    print(f"validation {key}: {selected['best_validation'][key]:.6f}")
        else:
            for key in evaluator.METRIC_KEYS:
                print(f"test {key}: {test_metrics[key]:.6f}")
        print(f"threshold: {selected['best_threshold']:.2f} (chosen on validation Macro-F1)")
        print(f"selected stage: {selected['stage']}, best epoch {selected['best_epoch'] + 1}")

    except BaseException as exc:  # noqa: BLE001 - failures must be recorded, not hidden
        error = exc
        status = "FAILED"
        print(f"RUN FAILED: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()

    verification = recorder.finalise(
        test_metrics=test_metrics, status=status, error=error,
        timing={"fit_wall_seconds_including_validation": fit_wall_seconds,
                "data_load_seconds": data_load_seconds},
    )

    print("-" * 78)
    print(f"Artefacts: {run_dir}")
    print(f"Artefact bundle complete: {verification['complete']}")
    if verification["missing_artifacts"]:
        print(f"Missing: {verification['missing_artifacts']}")
    print(f"RUN STATUS: {status}")
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
