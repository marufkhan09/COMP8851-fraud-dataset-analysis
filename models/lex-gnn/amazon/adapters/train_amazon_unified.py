from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler

THIS_FILE = Path(__file__).resolve()
ADAPTER_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(ADAPTER_ROOT))

from benchmark_common import (  # noqa: E402
    evaluate_scores,
    hardware_snapshot,
    load_amazon_graph,
    load_frozen_split,
    parameter_count,
    select_validation_threshold,
    set_seed,
    sha256_file,
    split_summary,
    utc_now,
    write_csv,
    write_json,
)


def progress_text(current: int, total: int, width: int = 30) -> str:
    """Return a log-safe ASCII progress bar."""
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percentage = 100.0 * current / total
    return (
        f"[{bar}] "
        f"{current}/{total} "
        f"({percentage:6.2f}%)"
    )


def collect_scores(model, loader, device):
    labels = []
    scores = []
    model.eval()
    with torch.no_grad():
        for _, _, blocks in loader:
            blocks = [block.to(device) for block in blocks]
            target = blocks[-1].dstdata["y"].reshape(-1)
            logits, _ = model(blocks)
            probability = torch.softmax(logits, dim=1)[:, 1]
            labels.append(target.detach().cpu())
            scores.append(probability.detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def make_loaders(graph, split, layers, batch_size):
    labels = graph.ndata["label"]
    graph.ndata["y"] = labels
    masked_labels = torch.full_like(labels, 2)
    train_tensor = torch.as_tensor(split["train"], dtype=torch.int64)
    masked_labels[train_tensor] = labels[train_tensor]
    graph.ndata["y_mask"] = masked_labels
    graph.ndata["x"] = graph.ndata["feature"].float().contiguous()

    fanout = {etype: 50 for etype in graph.etypes}
    for canonical_etype in graph.canonical_etypes:
        source, _ = graph.edges(etype=canonical_etype)
        probability = torch.where(masked_labels[source] == 2, 0.5, 0.9).float()
        graph.edges[canonical_etype].data["prob"] = probability

    # DGL UVA requires every stored graph tensor to be contiguous.
    # This changes only memory layout, not values or model architecture.
    for key in list(graph.ndata.keys()):
        graph.ndata[key] = graph.ndata[key].contiguous()

    for canonical_etype in graph.canonical_etypes:
        edge_data = graph.edges[canonical_etype].data
        for key in list(edge_data.keys()):
            edge_data[key] = edge_data[key].contiguous()

    sampler = NeighborSampler([dict(fanout) for _ in range(layers)], prob="prob")

    def loader(indices, shuffle):
        return DataLoader(
            graph,
            torch.as_tensor(indices, dtype=torch.int64),
            sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            use_uva=True,
        )

    return (
        loader(split["train"], True),
        loader(split["validation"], False),
        loader(split["test"], False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    config_path = Path(options.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    root = Path(config["workspace_root"]).resolve()
    repo = root / "upstreams" / "LEX-GNN"
    sys.path.insert(0, str(repo))
    from model import LEXGNN  # noqa: E402
    from model_trainer import reset_model_parameters  # noqa: E402

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    set_seed(int(config["seed"]))

    mat_path = Path(config["dataset_path"])
    split_path = Path(config["split_path"])
    graph = load_amazon_graph(mat_path, normalize_for_lex=True)
    split = load_frozen_split(split_path, graph.num_nodes())
    train_loader, validation_loader, test_loader = make_loaders(
        graph,
        split,
        int(config["layers"]),
        int(config["batch_size"]),
    )

    evaluate_test = bool(config.get("evaluate_test", True))
    restart_threshold = float(config.get("restart_validation_auroc_threshold", 0.51))
    restart_check_epoch = int(config.get("restart_check_epoch", 3))
    max_restarts = int(config.get("max_initialization_restarts", 5))
    restart_seed_stride = int(config.get("restart_seed_stride", 1000))
    if restart_check_epoch < 1:
        raise ValueError("restart_check_epoch must be at least 1")
    if max_restarts < 0:
        raise ValueError("max_initialization_restarts cannot be negative")

    run_id = f"{utc_now()}_amazon_lexgnn_{config['run_mode']}_seed{config['seed']}"
    result_dir = Path(config["results_root"]) / run_id
    result_dir.mkdir(parents=True, exist_ok=False)

    write_json(result_dir / "run_config.json", config)
    write_json(result_dir / "hardware.json", hardware_snapshot())
    write_json(result_dir / "split_summary.json", split_summary(split, graph.ndata["label"]))
    write_json(
        result_dir / "protocol_identity.json",
        {
            "dataset_sha256": sha256_file(mat_path),
            "split_sha256": sha256_file(split_path),
            "source_commit": config["source_commit"],
            "selection_metric": "validation AUROC",
            "threshold_selection": "validation macro-F1 over thresholds 0.01..0.99",
            "test_evaluations_during_training": 0,
            "test_evaluation_after_training": evaluate_test,
            "architecture_changed": False,
            "feature_transform": "DGL RowFeatNormalizer(subtract_min=True), matching author code",
            "author_restart_rule": "restart when validation AUROC <= configured threshold",
            "restart_threshold": restart_threshold,
            "restart_check_epoch": restart_check_epoch,
            "maximum_restarts": max_restarts,
        },
    )

    torch.cuda.reset_peak_memory_stats(device)
    complete_start = time.perf_counter()
    base_seed = int(config["seed"])
    restart_records = []
    restart_overhead_seconds = 0.0
    attempt_number = 0

    while True:
        effective_seed = base_seed + attempt_number * restart_seed_stride
        set_seed(effective_seed)
        model = LEXGNN(
            graph.ndata["feature"].shape[1],
            2,
            int(config["hidden_dimension"]),
            int(config["layers"]),
            int(config["heads"]),
            float(config["dropout"]),
        ).to(device)
        model.apply(reset_model_parameters)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

        attempt_epoch_rows = []
        attempt_validation_rows = []
        best_auc = float("-inf")
        best_epoch = None
        best_state = None
        patience = 0
        rejected = False

        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            losses = []
            auxiliary_losses = []
            torch.cuda.synchronize(device)
            train_start = time.perf_counter()

            for _, _, blocks in train_loader:
                blocks = [block.to(device) for block in blocks]
                output_labels = blocks[-1].dstdata["y"].long().reshape(-1)
                known_source = blocks[-1].srcdata["y_mask"].long().reshape(-1) != 2
                auxiliary_labels = blocks[-1].srcdata["y"].long().reshape(-1)[known_source]

                optimizer.zero_grad(set_to_none=True)
                logits, auxiliary_logits = model(blocks)
                primary_loss = loss_function(logits, output_labels)
                auxiliary_loss = loss_function(auxiliary_logits[-1][known_source], auxiliary_labels)
                total_loss = primary_loss + float(config["beta"]) * auxiliary_loss
                total_loss.backward()
                optimizer.step()
                losses.append(float(primary_loss.detach().cpu()))
                auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))

            torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - train_start

            torch.cuda.synchronize(device)
            validation_start = time.perf_counter()
            validation_labels, validation_scores = collect_scores(model, validation_loader, device)
            torch.cuda.synchronize(device)
            validation_seconds = time.perf_counter() - validation_start
            validation_threshold = select_validation_threshold(validation_labels, validation_scores)
            validation_metrics = evaluate_scores(validation_labels, validation_scores, validation_threshold)

            improved = validation_metrics["auroc"] > best_auc
            if improved:
                best_auc = validation_metrics["auroc"]
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                patience = 0
            else:
                patience += 1

            attempt_epoch_rows.append(
                {
                    "attempt": attempt_number + 1,
                    "effective_seed": effective_seed,
                    "epoch": epoch,
                    "training_seconds": training_seconds,
                    "validation_seconds": validation_seconds,
                    "total_epoch_seconds": training_seconds + validation_seconds,
                    "primary_loss": float(np.mean(losses)),
                    "auxiliary_loss": float(np.mean(auxiliary_losses)),
                    "validation_auroc": validation_metrics["auroc"],
                    "validation_auprc": validation_metrics["auprc"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_threshold": validation_threshold,
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    "best_so_far": improved,
                }
            )
            attempt_validation_rows.append(
                {
                    "attempt": attempt_number + 1,
                    "effective_seed": effective_seed,
                    "epoch": epoch,
                    **validation_metrics,
                }
            )
            print(
                f"{progress_text(epoch, int(config['epochs']))} | "
                f"Attempt {attempt_number + 1} | "
                f"loss={np.mean(losses):.6f} | "
                f"train={training_seconds:.3f}s | "
                f"val_AUROC={validation_metrics['auroc']:.6f} | "
                f"val_AUPRC={validation_metrics['auprc']:.6f} | "
                f"val_MacroF1={validation_metrics['macro_f1']:.6f} | "
                f"val_FraudF1={validation_metrics['fraud_f1']:.6f} | "
                f"val_GMean={validation_metrics['g_mean']:.6f} | "
                f"best={improved}",
                flush=True,
            )

            if epoch == restart_check_epoch and validation_metrics["auroc"] <= restart_threshold:
                rejected = True
                break
            if patience > int(config["early_stop"]):
                break

        attempt_training_seconds = float(
            sum(row["training_seconds"] for row in attempt_epoch_rows)
        )
        restart_records.append(
            {
                "attempt": attempt_number + 1,
                "effective_seed": effective_seed,
                "status": "REJECTED" if rejected else "ACCEPTED",
                "epochs_completed": len(attempt_epoch_rows),
                "last_validation_auroc": float(attempt_epoch_rows[-1]["validation_auroc"]),
                "training_seconds": attempt_training_seconds,
            }
        )

        if rejected:
            restart_overhead_seconds += attempt_training_seconds
            attempt_number += 1
            if attempt_number > max_restarts:
                write_json(result_dir / "restart_attempts.json", restart_records)
                raise RuntimeError("LEX-GNN exceeded the configured initialization restart limit")
            print(
                f"Restarting LEX-GNN after validation AUROC "
                f"{validation_metrics['auroc']:.6f} <= {restart_threshold:.6f}",
                flush=True,
            )
            continue

        epoch_rows = attempt_epoch_rows
        validation_rows = attempt_validation_rows
        selected_effective_seed = effective_seed
        break

    if best_state is None:
        raise RuntimeError("No LEX-GNN validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)

    torch.cuda.synchronize(device)
    validation_start = time.perf_counter()
    validation_labels, validation_scores = collect_scores(model, validation_loader, device)
    torch.cuda.synchronize(device)
    final_validation_seconds = time.perf_counter() - validation_start
    threshold = select_validation_threshold(validation_labels, validation_scores)
    final_validation = evaluate_scores(validation_labels, validation_scores, threshold)

    test_seconds = 0.0
    test_metrics = None
    prediction_payload = {
        "validation_labels": validation_labels,
        "validation_scores": validation_scores,
    }
    if evaluate_test:
        torch.cuda.synchronize(device)
        test_start = time.perf_counter()
        test_labels, test_scores = collect_scores(model, test_loader, device)
        torch.cuda.synchronize(device)
        test_seconds = time.perf_counter() - test_start
        test_metrics = evaluate_scores(test_labels, test_scores, threshold)
        prediction_payload.update(
            {"test_labels": test_labels, "test_scores": test_scores}
        )
    complete_seconds = time.perf_counter() - complete_start

    torch.save(
        {"model_state_dict": best_state, "best_epoch": best_epoch, "config": config},
        result_dir / "checkpoint.pt",
    )
    np.savez_compressed(result_dir / "predictions.npz", **prediction_payload)
    write_csv(result_dir / "epoch_times.csv", epoch_rows)
    write_csv(result_dir / "validation_metrics.csv", validation_rows)
    write_json(result_dir / "validation_final.json", final_validation)
    write_json(result_dir / "restart_attempts.json", restart_records)
    if evaluate_test:
        write_json(result_dir / "test_metrics.json", test_metrics)
    accepted_training_seconds = float(sum(row["training_seconds"] for row in epoch_rows))
    summary = {
        "run_id": run_id,
        "status": "PASS",
        "model": "LEX-GNN",
        "dataset": "Amazon",
        "run_mode": config["run_mode"],
        "seed": base_seed,
        "effective_initialization_seed": selected_effective_seed,
        "initialization_restarts": attempt_number,
        "epochs_completed": len(epoch_rows),
        "best_validation_epoch": best_epoch,
        "best_validation_auroc": best_auc,
        "selected_threshold": threshold,
        "training_seconds_total": accepted_training_seconds,
        "restart_overhead_seconds": restart_overhead_seconds,
        "training_seconds_including_restarts": accepted_training_seconds + restart_overhead_seconds,
        "training_seconds_mean": float(np.mean([row["training_seconds"] for row in epoch_rows])),
        "final_validation_inference_seconds": final_validation_seconds,
        "test_inference_seconds": test_seconds,
        "complete_wall_seconds": complete_seconds,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "test_evaluated": evaluate_test,
        **parameter_count(model),
        "test_metrics": test_metrics,
        "result_directory": str(result_dir),
    }
    write_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"RESULT_DIRECTORY={result_dir}", flush=True)


if __name__ == "__main__":
    main()

