"""Run a single CARE-GNN experiment under COMP8851 protocol v4.4.

One invocation = one (dataset, track, ratio, seed) cell of the run matrix, with
the complete artefact bundle written to
``results/care-gnn/<dataset>/<track>/<ratio>/seed_<seed>/``.

Example
-------
    python models/care-gnn/scripts/run_one.py \
        --dataset yelpchi --data-path data/yelpchi \
        --ratio TR40 --seed 2 --track unified

The author-mode training procedure is preserved: negatives are randomly
under-sampled each epoch rather than the loss being class-weighted, because
that is what the published implementation does. The unified protocol controls
the split, the seeds, the stopping rule, the selection metric, the threshold
rule and the evaluator; it does not rewrite the model's optimisation.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

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

from caregnn.model import build_care_gnn  # noqa: E402

MODEL_NAME = "CARE-GNN"
MODEL_SLUG = "care-gnn"
UPSTREAM_REPO = "https://github.com/YingtongDou/CARE-GNN"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARE-GNN runner for COMP8851")

    # Data and protocol
    parser.add_argument("--dataset", required=True, choices=list(protocol.DATASETS))
    parser.add_argument("--data-path", required=True,
                        help="Dataset location; see shared.comp8851.datasets.DATA_PATH_HELP")
    parser.add_argument("--split-path", default=None,
                        help="Frozen split .npz. If omitted, splits are generated "
                             "with the registry seed and cached under --split-dir.")
    parser.add_argument("--split-dir", default=str(REPO_ROOT / "shared" / "splits"))
    parser.add_argument("--ratio", default="TR40", choices=list(protocol.TRAIN_RATIOS))
    parser.add_argument("--seed", type=int, default=2, help="Training seed")
    parser.add_argument("--track", default="unified", choices=list(protocol.TRACKS))
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))

    # Model hyperparameters (author defaults from the official repository)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--inter", default="GNN", choices=["GNN", "Att", "Weight", "Mean"],
                        help="Inter-relation aggregator; GNN is the CARE-GNN variant")
    parser.add_argument("--lambda-1", type=float, default=2.0,
                        help="Weight of the label-aware similarity loss, Eq. (11)")
    parser.add_argument("--lambda-2", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--step-size", type=float, default=2e-2, help="RL action step size")
    parser.add_argument("--under-sample", type=float, default=1.0,
                        help="Negatives kept per positive during training")
    parser.add_argument("--no-rl", action="store_true",
                        help="Disable the RL threshold module (ablation only)")

    # Optimisation
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=protocol.MAX_EPOCHS)
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=protocol.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--selection-metric", default=protocol.SELECTION_METRIC,
                        choices=["auprc", "auroc", "macro_f1"])
    parser.add_argument("--eval-batch-size", type=int, default=1024)

    # Execution
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail instead of silently falling back to CPU")
    parser.add_argument("--run-mode", default="controlled")
    parser.add_argument("--hardware-profile-id", default=None)
    parser.add_argument("--container-digest", default=None)
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="Stop after this many minutes; 0 disables the cap")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not evaluate the test set. Required for tuning " 
                             "trials: it makes test isolation a property of the " 
                             "command rather than a promise.")
    parser.add_argument("--smoke", action="store_true",
                        help="Two-epoch feasibility run; never report its metrics")
    parser.add_argument("--max-nodes", type=int, default=2_000_000,
                        help="Refuse graphs larger than this. CARE-GNN materialises "
                             "per-relation Python adjacency sets, so memory grows with "
                             "nodes plus edges; refusing is a recorded feasibility "
                             "outcome and is safer than an OOM kill mid-notebook.")
    parser.add_argument("--allow-large", action="store_true",
                        help="Override --max-nodes. Expect heavy memory use.")
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


def sample_epoch_nodes(train_idx: np.ndarray, labels: np.ndarray,
                       under_sample: float) -> List[int]:
    """Author procedure: keep every positive, under-sample the negatives."""
    train_labels = labels[train_idx]
    positives = train_idx[train_labels == 1].tolist()
    negatives = train_idx[train_labels == 0].tolist()
    if not positives:
        raise RuntimeError("Training split contains no fraud nodes.")
    keep = min(len(negatives), max(1, int(len(positives) * under_sample)))
    sampled = positives + random.sample(negatives, keep)
    random.shuffle(sampled)
    return sampled


@torch.no_grad()
def predict(model, node_ids: np.ndarray, labels_np: np.ndarray, batch_size: int,
            device) -> tuple[np.ndarray, float]:
    """Fraud probabilities for a set of nodes, with inference time measured."""
    model.eval()
    probabilities: List[float] = []
    runtime.synchronise(device)
    started = time.perf_counter()
    for start in range(0, len(node_ids), batch_size):
        batch = [int(n) for n in node_ids[start:start + batch_size]]
        batch_labels = torch.tensor(labels_np[node_ids[start:start + batch_size]],
                                    dtype=torch.long, device=device)
        gnn_probabilities, _ = model.to_prob(batch, batch_labels, train_flag=False)
        probabilities.extend(gnn_probabilities[:, 1].detach().cpu().numpy().tolist())
    runtime.synchronise(device)
    return np.asarray(probabilities, dtype=np.float64), time.perf_counter() - started


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
    if args.smoke:
        print("MODE: SMOKE TEST - feasibility only, metrics are not benchmark evidence")

    recorder = artifacts.RunRecorder(
        run_dir=run_dir, model=MODEL_NAME, dataset=args.dataset, track=args.track,
        ratio=args.ratio, seed=args.seed, run_id=run_id, device=device,
    )

    error: BaseException | None = None
    test_metrics = None
    status = "COMPLETE"

    try:
        # ---------------- data ----------------
        load_started = time.perf_counter()
        dataset = datasets.load_dataset(args.dataset, args.data_path, strict=args.strict_stats)
        data_load_seconds = time.perf_counter() - load_started
        print(f"Loaded {dataset.name}: {dataset.num_nodes:,} nodes, "
              f"{dataset.num_features} features, {len(dataset.relations)} relation(s)")

        if not dataset.is_multi_relational:
            # CARE-GNN's mechanism is inter-relation aggregation. On a natively
            # single-relation graph it degenerates, which the compatibility
            # matrix must record rather than hide.
            print(f"[compatibility] NOTE {dataset.name} is single-relation; CARE-GNN's "
                  "inter-relation aggregator degenerates to one channel. "
                  "Record this pair as CONDITIONAL (A2 mechanism risk).")

        split_arrays, split_source, split_hash = resolve_split(args, dataset)
        splits.assert_no_leakage(split_arrays["train_idx"], split_arrays["valid_idx"],
                                 split_arrays["test_idx"])
        labels_np = dataset.labels
        recorder.record_split(split_arrays, labels_np, split_source)

        train_idx = split_arrays["train_idx"]
        valid_idx = split_arrays["valid_idx"]
        test_idx = split_arrays["test_idx"]
        print(f"Split: train {len(train_idx):,} | valid {len(valid_idx):,} | test {len(test_idx):,}")
        print(f"Training fraud nodes: {int(labels_np[train_idx].sum()):,}")

        if dataset.num_nodes > args.max_nodes and not args.allow_large:
            raise RuntimeError(
                f"{dataset.name} has {dataset.num_nodes:,} nodes, above the "
                f"--max-nodes guard of {args.max_nodes:,}. CARE-GNN builds a Python "
                "adjacency set per node per relation, so this would likely exhaust "
                "memory before the time cap could stop it. Record this pair as "
                "FAILED_TECHNICAL with this message, or re-run with --allow-large "
                "on a machine with enough RAM."
            )

        print("Building per-relation adjacency lists "
              f"({dataset.num_nodes:,} nodes x {len(dataset.relations)} relations)...")
        adjacency_started = time.perf_counter()
        adj_lists = dataset.adjacency_lists(add_self_loops=True)
        print(f"  built in {time.perf_counter() - adjacency_started:.1f}s")

        # ---------------- model ----------------
        features = torch.tensor(dataset.features, dtype=torch.float32)
        model = build_care_gnn(
            features=features,
            adj_lists=adj_lists,
            embed_dim=args.emb_size,
            inter=args.inter,
            step_size=args.step_size,
            lambda_1=args.lambda_1,
            rl_enabled=not args.no_rl,
            verbose=True,
        ).to(device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.lambda_2, betas=(0.9, 0.999), eps=1e-8,
        )

        parameter_counts = runtime.count_parameters(model)
        print(f"Parameters: {parameter_counts['trainable_parameters']:,} trainable")

        # ---------------- provenance ----------------
        config = {k: v for k, v in vars(args).items()}
        config.update({
            "model": MODEL_NAME,
            "upstream_repository": UPSTREAM_REPO,
            "resolved_device": str(device),
            "optimizer": "adam",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "architecture": "CARE-GNN one-layer camouflage-resistant aggregator",
            "class_imbalance_handling": "per-epoch negative under-sampling (author procedure)",
            **parameter_counts,
        })
        identity = protocol.ProtocolIdentity(
            hardware_profile_id=args.hardware_profile_id or gpu,
            container_digest=args.container_digest,
            model_source_commit=protocol.git_commit(REPO_ROOT),
            source_sha256=next(iter(dataset.source_sha256.values()), None)
                          or "not-hashed-dgl-managed-download",
            dataset_manifest_version="comp8851-canonical-v1",
            dataset_view_id=f"{dataset.name}_relation_v1",
            view_sha256=dataset.view_manifest("relation")["view_sha256"],
            split_registry_version=f"seed{protocol.SPLIT_SEED}",
            split_sha256=split_hash,
            evaluator_commit=f"shared.comp8851=={shared_version}",
            adapter_version="relation-view-v1",
            selection_metric=args.selection_metric,
        )
        recorder.begin(config, identity, dataset.manifest(),
                       dataset.view_manifest("relation"))

        # ---------------- training ----------------
        runtime.reset_peak_memory(device)
        best_selection = float("-inf")
        best_epoch = -1
        best_threshold = 0.5
        best_validation: Dict[str, float] = {}
        epochs_without_improvement = 0
        fit_started = time.perf_counter()
        deadline = fit_started + args.max_minutes * 60 if args.max_minutes else None
        stopped_early = ""

        for epoch in range(args.epochs):
            model.train()
            epoch_nodes = sample_epoch_nodes(train_idx, labels_np, args.under_sample)
            num_batches = max(1, int(np.ceil(len(epoch_nodes) / args.batch_size)))
            model.inter1.batch_num = num_batches

            runtime.synchronise(device)
            epoch_started = time.perf_counter()
            epoch_loss = 0.0

            for batch_index in range(num_batches):
                start = batch_index * args.batch_size
                batch_nodes = [int(n) for n in epoch_nodes[start:start + args.batch_size]]
                if not batch_nodes:
                    continue
                batch_labels = torch.tensor(labels_np[np.asarray(batch_nodes)],
                                            dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(batch_nodes, batch_labels, train_flag=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())

            runtime.synchronise(device)
            train_seconds = time.perf_counter() - epoch_started
            mean_loss = epoch_loss / num_batches

            if not np.isfinite(mean_loss):
                raise RuntimeError(f"Loss became non-finite at epoch {epoch}: {mean_loss}")

            validation_metrics = None
            validation_seconds = 0.0
            is_best = False
            should_validate = (
                epoch == 0 or (epoch + 1) % args.valid_every == 0 or epoch == args.epochs - 1
            )

            if should_validate:
                probabilities, validation_seconds = predict(
                    model, valid_idx, labels_np, args.eval_batch_size, device
                )
                validation_metrics = evaluator.evaluate_validation(
                    labels_np[valid_idx], probabilities
                )
                selection = evaluator.selection_value(validation_metrics, args.selection_metric)
                recorder.record_validation(epoch, validation_metrics, validation_seconds)

                if selection > best_selection + 1e-12:
                    best_selection = selection
                    best_epoch = epoch
                    best_threshold = float(validation_metrics["threshold"])
                    best_validation = dict(validation_metrics)
                    is_best = True
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), recorder.checkpoint_path)
                else:
                    epochs_without_improvement += args.valid_every

                print(f"Epoch {epoch + 1:3d}/{args.epochs} | loss {mean_loss:.6f} | "
                      f"train {train_seconds:.3f}s | val AUPRC {validation_metrics['auprc']:.6f} | "
                      f"val AUROC {validation_metrics['auroc']:.6f} | "
                      f"thr {validation_metrics['threshold']:.2f} | best epoch {best_epoch + 1}")
            else:
                print(f"Epoch {epoch + 1:3d}/{args.epochs} | loss {mean_loss:.6f} | "
                      f"train {train_seconds:.3f}s")

            recorder.record_epoch(epoch, train_seconds, mean_loss, validation_seconds,
                                  validation_metrics, is_best)

            if args.patience and should_validate and epochs_without_improvement >= args.patience:
                stopped_early = f"early stopping: no improvement for {args.patience} epochs"
                print(stopped_early)
                break
            if deadline and time.perf_counter() > deadline:
                stopped_early = f"time cap of {args.max_minutes} minutes reached"
                print(stopped_early)
                break

        fit_wall_seconds = time.perf_counter() - fit_started

        if best_epoch < 0:
            raise RuntimeError("No validation evaluation completed; nothing to select.")

        recorder.set_best(best_epoch, best_validation, best_threshold,
                          args.selection_metric, best_selection)
        recorder.add("rl_thresholds_final", model.thresholds())
        recorder.add("rl_threshold_history", model.inter1.thresholds_log[-20:])

        # ---------------- test ----------------
        # Under --skip-test the test split is never read, never scored and never
        # written. This is how tuning trials stay isolated from test data.
        if args.skip_test:
            print("--skip-test: the test set was not loaded, scored or written.")
            print("This run is a tuning trial and reports validation metrics only.")
            recorder.add("test_skipped", True)
            recorder.add("evidence_class", "tuning_trial")
        else:
            model.load_state_dict(torch.load(recorder.checkpoint_path, map_location=device))
            probabilities, inference_seconds = predict(
                model, test_idx, labels_np, args.eval_batch_size, device
            )
            test_metrics = evaluator.metric_bundle(labels_np[test_idx], probabilities,
                                                   best_threshold)
            test_metrics["inference_seconds"] = inference_seconds
            if args.smoke:
                test_metrics["evidence_class"] = "feasibility_only"

        recorder.add("data_load_seconds", data_load_seconds)
        recorder.add("heterophily", dataset.heterophily())
        if stopped_early:
            recorder.add("stop_reason", stopped_early)

        print("-" * 78)
        if test_metrics is None:
            for key in ("auprc", "auroc", "macro_f1"):
                if key in best_validation:
                    print(f"validation {key}: {best_validation[key]:.6f}")
        else:
            for key in evaluator.METRIC_KEYS:
                print(f"test {key}: {test_metrics[key]:.6f}")
        print(f"threshold: {best_threshold:.2f} (chosen on validation Macro-F1)")
        print(f"best validation epoch: {best_epoch + 1}")

    except BaseException as exc:  # noqa: BLE001 - failures must be recorded, not hidden
        error = exc
        status = "FAILED"
        print(f"RUN FAILED: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()

    verification = recorder.finalise(
        test_metrics=test_metrics, status=status, error=error,
        timing={"fit_wall_seconds_including_validation":
                locals().get("fit_wall_seconds", 0.0),
                "data_load_seconds": locals().get("data_load_seconds", 0.0)},
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
