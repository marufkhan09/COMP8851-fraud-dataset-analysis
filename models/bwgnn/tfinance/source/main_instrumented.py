from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from BWGNN import BWGNN


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def labels_from_graph(graph):
    labels = graph.ndata["label"]
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)
    return labels.long().reshape(-1)


def author_split(labels: np.ndarray, train_ratio: float, split_seed: int):
    ids = np.arange(len(labels), dtype=np.int64)
    train, rest = train_test_split(
        ids, train_size=train_ratio, random_state=split_seed,
        shuffle=True, stratify=labels,
    )
    valid, test = train_test_split(
        rest, test_size=0.67, random_state=split_seed,
        shuffle=True, stratify=labels[rest],
    )
    return {
        "train_idx": np.sort(train), "valid_idx": np.sort(valid),
        "test_idx": np.sort(test), "unused_idx": np.asarray([], dtype=np.int64),
    }, f"author_random_stratified_seed_{split_seed}"


def load_split(path: Path):
    with np.load(path) as data:
        required = {"train_idx", "valid_idx", "test_idx", "unused_idx"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Split file is missing: {sorted(missing)}")
        arrays = {name: np.asarray(data[name], dtype=np.int64) for name in required}
    return arrays, str(path.resolve())


def verify_split(arrays: dict, total_nodes: int):
    sets = {name: set(values.tolist()) for name, values in arrays.items()}
    errors = []
    for name, values in arrays.items():
        if len(values) != len(sets[name]):
            errors.append(f"duplicate IDs in {name}")
        if len(values) and (values.min() < 0 or values.max() >= total_nodes):
            errors.append(f"out-of-range IDs in {name}")
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if sets[left] & sets[right]:
                errors.append(f"overlap between {left} and {right}")
    if len(set().union(*sets.values())) != total_nodes:
        errors.append("partitions do not account for every node")
    if errors:
        raise RuntimeError("Invalid split: " + "; ".join(errors))


def describe(ids: np.ndarray, labels: np.ndarray):
    fraud = int(labels[ids].sum())
    return {"nodes": int(len(ids)), "fraud_nodes": fraud, "normal_nodes": int(len(ids) - fraud)}


def best_threshold(labels: np.ndarray, probabilities: np.ndarray):
    best_score, chosen = -1.0, 0.5
    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = (probabilities > threshold).astype(np.int64)
        score = f1_score(labels, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_score, chosen = float(score), float(threshold)
    return chosen, best_score


def metric_bundle(labels: np.ndarray, probabilities: np.ndarray, threshold: float):
    predictions = (probabilities > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "fraud_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "fraud_precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "gmean": float(np.sqrt(tpr * tnr)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


@torch.no_grad()
def predict_probabilities(model, features, ids, device):
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits = model(features)
    probabilities = logits.softmax(dim=1)[:, 1]
    index_tensor = torch.as_tensor(ids, dtype=torch.long, device=device)
    selected = probabilities[index_tensor].detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return selected, time.perf_counter() - started


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Instrumented BWGNN runner for COMP8851")
    parser.add_argument("--dataset", default="tfinance")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=2)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--hid-dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0, help="Epoch patience; 0 disables early stopping")
    parser.add_argument("--selection-metric", choices=["auroc", "macro_f1"], default="auroc")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the test set; use for tuning trials")
    parser.add_argument("--run-mode", default="controlled")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda" and not args.no_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")

    run_started_utc = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_tfinance_BWGNN_{args.run_mode}_seed{args.seed}"
    run_dir = Path(args.results_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoints" / "bwgnn.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BWGNN T-FINANCE RUN")
    print("=" * 78)
    print(f"Run ID: {run_id}")
    print(f"Mode: {args.run_mode}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seed: {args.seed}")
    print(f"Hidden dimension: {args.hid_dim}")
    print(f"Order: {args.order}")
    print(f"Maximum epochs: {args.epochs}")
    print(f"Patience: {args.patience if args.patience else 'disabled'}")

    load_started = time.perf_counter()
    graphs, _ = dgl.load_graphs(str(Path(args.data_path).resolve()))
    graph = graphs[0]
    labels_cpu = labels_from_graph(graph)
    graph.ndata["feature"] = graph.ndata["feature"].float()
    features_cpu = graph.ndata["feature"]
    data_load_seconds = time.perf_counter() - load_started
    labels_np = labels_cpu.numpy()

    if args.split_path:
        splits, split_source = load_split(Path(args.split_path))
    else:
        splits, split_source = author_split(labels_np, args.train_ratio, args.split_seed)
    verify_split(splits, graph.num_nodes())

    print(f"Nodes: {graph.num_nodes():,}")
    print(f"Stored directed edge entries: {graph.num_edges():,}")
    print(f"Features: {features_cpu.shape[1]}")
    print(f"Split source: {split_source}")
    print(f"Train/validation/test: {len(splits['train_idx']):,}/{len(splits['valid_idx']):,}/{len(splits['test_idx']):,}")

    graph = graph.to(device)
    features = features_cpu.to(device)
    labels = labels_cpu.to(device)
    train_ids = torch.as_tensor(splits["train_idx"], dtype=torch.long, device=device)
    valid_ids_np = splits["valid_idx"]
    test_ids_np = splits["test_idx"]

    model = BWGNN(features.shape[1], args.hid_dim, 2, graph, d=args.order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fraud = int(labels[train_ids].sum().item())
    normal = int(len(train_ids) - fraud)
    class_weight = normal / fraud
    loss_weight = torch.tensor([1.0, class_weight], dtype=torch.float32, device=device)
    print(f"Training fraud/normal: {fraud:,}/{normal:,}")
    print(f"Cross-entropy fraud weight: {class_weight:.6f}")

    configuration = vars(args).copy()
    configuration.update({
        "resolved_device": str(device), "resolved_run_id": run_id,
        "resolved_split_source": split_source, "optimizer": "adam",
        "architecture": "BWGNN homogeneous Beta wavelet graph neural network",
    })
    (run_dir / "run_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "split_indices.npz", **splits)
    split_summary = {
        "source": split_source,
        "train": describe(splits["train_idx"], labels_np),
        "validation": describe(splits["valid_idx"], labels_np),
        "test": describe(splits["test_idx"], labels_np),
        "unused_nodes": int(len(splits["unused_idx"])),
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_rows = []
    best_validation_auroc = -1.0
    best_validation_macro_f1 = -1.0
    best_selection_value = -1.0
    best_epoch = -1
    best_threshold_value = 0.5
    epochs_without_improvement = 0
    total_validation_seconds = 0.0
    fit_started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        logits = model(features)
        loss = F.cross_entropy(logits[train_ids], labels[train_ids], weight=loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started

        row = {
            "epoch": epoch, "loss": float(loss.detach().cpu()),
            "train_seconds": train_seconds, "validation_seconds": "",
            "validation_auroc": "", "validation_auprc": "",
            "validation_macro_f1": "", "validation_threshold": "",
            "timestamp_utc": utc_now(),
        }
        should_validate = epoch == 0 or (epoch + 1) % args.valid_every == 0 or epoch == args.epochs - 1
        if should_validate:
            validation_probabilities, validation_seconds = predict_probabilities(model, features, valid_ids_np, device)
            total_validation_seconds += validation_seconds
            validation_labels = labels_np[valid_ids_np]
            threshold, validation_macro_f1 = best_threshold(validation_labels, validation_probabilities)
            validation_auroc = roc_auc_score(validation_labels, validation_probabilities)
            validation_auprc = average_precision_score(validation_labels, validation_probabilities)
            row.update({
                "validation_seconds": validation_seconds,
                "validation_auroc": float(validation_auroc),
                "validation_auprc": float(validation_auprc),
                "validation_macro_f1": float(validation_macro_f1),
                "validation_threshold": float(threshold),
            })
            selection_value = (
                validation_auroc
                if args.selection_metric == "auroc"
                else validation_macro_f1
            )
            improved = selection_value > best_selection_value + 1e-12
            if improved:
                best_selection_value = float(selection_value)
                best_validation_auroc = float(validation_auroc)
                best_validation_macro_f1 = float(validation_macro_f1)
                best_epoch = epoch
                best_threshold_value = float(threshold)
                torch.save(model.state_dict(), checkpoint)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += args.valid_every
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | loss {loss.item():.6f} | "
                f"train {train_seconds:.3f}s | val AUROC {validation_auroc:.6f} | "
                f"val AUPRC {validation_auprc:.6f} | threshold {threshold:.2f} | "
                f"best epoch {best_epoch + 1}"
            )
        else:
            print(f"Epoch {epoch + 1:3d}/{args.epochs} | loss {loss.item():.6f} | train {train_seconds:.3f}s")
        epoch_rows.append(row)

        if args.patience and should_validate and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; validation AUROC has not improved for {args.patience} epochs.")
            break

    fit_wall_seconds = time.perf_counter() - fit_started
    write_csv(run_dir / "epoch_times.csv", epoch_rows)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    if args.skip_test:
        inference_seconds = None
        test_metrics = None
    else:
        test_probabilities, inference_seconds = predict_probabilities(model, features, test_ids_np, device)
        test_metrics = metric_bundle(labels_np[test_ids_np], test_probabilities, best_threshold_value)
        test_metrics["inference_seconds"] = inference_seconds
        write_csv(run_dir / "test_metrics.csv", [test_metrics])

    train_times = [float(row["train_seconds"]) for row in epoch_rows]
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    environment = {
        "platform": platform.platform(), "processor": platform.machine(),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "dgl_version": dgl.__version__, "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_used": device.type == "cuda",
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
    }
    summary = {
        "schema_version": "1.0", "run_id": run_id, "run_mode": args.run_mode,
        "model": "BWGNN", "dataset": "tfinance", "seed": args.seed,
        "run_started_utc": run_started_utc, "run_finished_utc": utc_now(),
        "configuration": configuration, "environment": environment,
        "split": split_summary, "best_validation_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_value": best_selection_value,
        "best_validation_auroc": best_validation_auroc,
        "best_validation_macro_f1": best_validation_macro_f1,
        "selected_threshold": best_threshold_value,
        "test_metrics": test_metrics,
        "timing": {
            "data_load_seconds": data_load_seconds,
            "training_epochs": len(epoch_rows),
            "total_train_epoch_seconds": float(sum(train_times)),
            "mean_train_epoch_seconds": float(np.mean(train_times)),
            "median_train_epoch_seconds": float(np.median(train_times)),
            "std_train_epoch_seconds": float(np.std(train_times)),
            "fit_wall_seconds_including_validation": fit_wall_seconds,
            "total_validation_seconds": total_validation_seconds,
            "test_inference_seconds": inference_seconds,
            "peak_gpu_memory_mb": peak_mb,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("FINAL RUN RESULTS")
    print("=" * 78)
    print(f"Best validation epoch: {best_epoch + 1}")
    print(f"Best validation AUROC: {best_validation_auroc:.6f}")
    if test_metrics is None:
        print("Test evaluation: SKIPPED (validation-only tuning trial)")
    else:
        for key in ["auroc", "auprc", "macro_f1", "fraud_f1", "fraud_precision", "fraud_recall", "gmean", "accuracy"]:
            print(f"{key}: {test_metrics[key]:.6f}")
    print(f"Epochs completed: {len(epoch_rows)}")
    print(f"Mean epoch time: {np.mean(train_times):.6f}s")
    print(f"Inference time: {inference_seconds:.6f}s" if inference_seconds is not None else "Inference time: not measured")
    print(f"Peak GPU memory: {peak_mb:.2f} MB")
    print(f"Artifacts: {run_dir}")
    print("RUN COMPLETED: TRUE")


if __name__ == "__main__":
    main()
