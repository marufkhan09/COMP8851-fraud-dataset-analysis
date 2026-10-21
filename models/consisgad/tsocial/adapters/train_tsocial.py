# T-SOCIAL ADAPTER (PROVISIONAL SPLIT)
# Not authorised for final benchmark results.
# Uses GAAP chronological splits until team splits recovered.
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

THIS_FILE = Path(__file__).resolve()
ADAPTER_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(ADAPTER_ROOT))
from tsocial_data import load_tsocial_graph, load_tsocial_split  # noqa

from benchmark_common import (  # noqa: E402
    evaluate_scores,
    hardware_snapshot,
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


def loader(indices, batch_size, shuffle, seed, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return TorchDataLoader(
        torch.as_tensor(indices, dtype=torch.int64),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        generator=generator,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    config_path = Path(options.config).resolve()
    args = json.loads(config_path.read_text(encoding="utf-8"))
    evaluate_test = bool(args.get("evaluate_test", True))

    root = Path(args["workspace_root"]).resolve()
    repo = root / "upstreams" / "ConsisGAD"
    sys.path.insert(0, str(repo))
    import main as upstream  # noqa: E402

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    args["device"] = device
    set_seed(int(args["seed"]))
    upstream.args = args  # compatibility for the upstream SoftAttentionDrop epsilon reference

    mat_path = Path(args["dataset_path"])
    split_path = Path(args["split_path"])
    graph = load_tsocial_graph(mat_path, normalize_for_lex=False)
    split = load_tsocial_split(split_path, graph.num_nodes())
    graph = graph.to(device)

    train_ids = split["train"]
    unlabeled_ids = np.concatenate(
        [split["train"], split["validation"], split["test"], split["unused"]]
    )
    label_loader = loader(
        train_ids,
        int(args["batch-size"]),
        True,
        int(args["seed"]),
        drop_last=True,
    )
    unlabel_loader = loader(
        unlabeled_ids,
        int(args["batch-size"]) * int(args["unlabel-ratio"]),
        True,
        int(args["seed"]) + 1,
        drop_last=True,
    )
    validation_loader = loader(split["validation"], 2 ** 16, False, int(args["seed"]))
    test_loader = loader(split["test"], 2 ** 16, False, int(args["seed"]))

    args["node-in-dim"] = int(graph.ndata["feature"].shape[1])
    args["node-out-dim"] = 2
    model = upstream.create_model(args, graph.etypes)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args["lr"]), weight_decay=0.0)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(int(args["num-layers"]))
    attention_drop = upstream.SoftAttentionDrop(args).to(device)
    attention_optimizer = torch.optim.Adam(
        attention_drop.parameters(),
        lr=float(args["trainable-lr"]),
        weight_decay=0.0,
    )
    augmentor = (sampler, attention_drop, attention_optimizer)

    run_id = f"{utc_now()}_tsocial_consisgad_{args['run_mode']}_seed{args['seed']}"
    result_dir = Path(args["results_root"]) / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    serialisable_args = {key: str(value) if isinstance(value, torch.device) else value for key, value in args.items()}
    write_json(result_dir / "run_config.json", serialisable_args)
    write_json(result_dir / "hardware.json", hardware_snapshot())
    write_json(result_dir / "split_summary.json", split_summary(split, graph.ndata["label"]))
    write_json(
        result_dir / "protocol_identity.json",
        {
            "dataset_sha256": "8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0",
            "split_sha256": sha256_file(split_path),
            "source_commit": args["source_commit"],
            "selection_metric": "validation AUROC",
            "threshold_selection": "validation macro-F1 over thresholds 0.01..0.99",
            "test_evaluations_during_training": 0,
            "test_evaluation_after_training": evaluate_test,
            "architecture_changed": False,
            "transductive_unlabeled_use": "all eligible Elliptic nodes; labels exposed only for training IDs",
            "compatibility_note": "sets upstream module global args so its trainable-eps branch behaves as authored",
        },
    )

    epoch_rows = []
    validation_rows = []
    best_auc = float("-inf")
    best_epoch = None
    best_model_state = None
    best_attention_state = None
    torch.cuda.reset_peak_memory_stats(device)
    complete_start = time.perf_counter()

    for epoch in range(int(args["epochs"])):
        torch.cuda.synchronize(device)
        training_start = time.perf_counter()
        upstream.UDA_train_epoch(
            epoch,
            model,
            upstream.nll_loss,
            graph,
            label_loader,
            unlabel_loader,
            optimizer,
            augmentor,
            args,
        )
        torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - training_start

        torch.cuda.synchronize(device)
        validation_start = time.perf_counter()
        validation_scores_tensor, validation_labels_tensor = upstream.get_model_pred(
            model, graph, validation_loader, sampler, args
        )
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        validation_scores = validation_scores_tensor.detach().cpu().numpy()
        validation_labels = validation_labels_tensor.detach().cpu().numpy()
        threshold = select_validation_threshold(validation_labels, validation_scores)
        validation_metrics = evaluate_scores(validation_labels, validation_scores, threshold)

        improved = validation_metrics["auroc"] > best_auc
        if improved:
            best_auc = validation_metrics["auroc"]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_attention_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in attention_drop.state_dict().items()}
            )

        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "total_epoch_seconds": training_seconds + validation_seconds,
                "validation_auroc": validation_metrics["auroc"],
                "validation_auprc": validation_metrics["auprc"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_threshold": threshold,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                "best_so_far": improved,
            }
        )
        validation_rows.append({"epoch": epoch + 1, **validation_metrics})
        print(
            f"{progress_text(epoch + 1, int(args['epochs']))} | "
            f"train={training_seconds:.3f}s | "
            f"val_AUROC={validation_metrics['auroc']:.6f} | "
            f"val_AUPRC={validation_metrics['auprc']:.6f} | "
            f"val_MacroF1={validation_metrics['macro_f1']:.6f} | "
            f"val_FraudF1={validation_metrics['fraud_f1']:.6f} | "
            f"val_GMean={validation_metrics['g_mean']:.6f} | "
            f"best={improved}",
            flush=True,
        )

    if best_model_state is None:
        raise RuntimeError("No ConsisGAD validation checkpoint was selected")
    model.load_state_dict(best_model_state)
    model.to(device)

    torch.cuda.synchronize(device)
    validation_start = time.perf_counter()
    validation_scores_tensor, validation_labels_tensor = upstream.get_model_pred(
        model, graph, validation_loader, sampler, args
    )
    torch.cuda.synchronize(device)
    final_validation_seconds = time.perf_counter() - validation_start
    validation_scores = validation_scores_tensor.detach().cpu().numpy()
    validation_labels = validation_labels_tensor.detach().cpu().numpy()
    selected_threshold = select_validation_threshold(validation_labels, validation_scores)
    final_validation = evaluate_scores(validation_labels, validation_scores, selected_threshold)

    test_seconds = 0.0
    test_metrics = None
    prediction_payload = {
        "validation_labels": validation_labels,
        "validation_scores": validation_scores,
    }
    if evaluate_test:
        torch.cuda.synchronize(device)
        test_start = time.perf_counter()
        test_scores_tensor, test_labels_tensor = upstream.get_model_pred(
            model, graph, test_loader, sampler, args
        )
        torch.cuda.synchronize(device)
        test_seconds = time.perf_counter() - test_start
        test_scores = test_scores_tensor.detach().cpu().numpy()
        test_labels = test_labels_tensor.detach().cpu().numpy()
        test_metrics = evaluate_scores(test_labels, test_scores, selected_threshold)
        prediction_payload.update(
            {"test_labels": test_labels, "test_scores": test_scores}
        )
    complete_seconds = time.perf_counter() - complete_start

    torch.save(
        {
            "model_state_dict": best_model_state,
            "attention_state_dict": best_attention_state,
            "best_epoch": best_epoch,
            "config": serialisable_args,
        },
        result_dir / "checkpoint.pt",
    )
    np.savez_compressed(result_dir / "predictions.npz", **prediction_payload)
    write_csv(result_dir / "epoch_times.csv", epoch_rows)
    write_csv(result_dir / "validation_metrics.csv", validation_rows)
    write_json(result_dir / "validation_final.json", final_validation)
    if evaluate_test:
        write_json(result_dir / "test_metrics.json", test_metrics)
    summary = {
        "run_id": run_id,
        "status": "PASS",
        "model": "ConsisGAD",
        "dataset": "T-Social",
        "run_mode": args["run_mode"],
        "seed": args["seed"],
        "epochs_completed": len(epoch_rows),
        "best_validation_epoch": best_epoch,
        "best_validation_auroc": best_auc,
        "selected_threshold": selected_threshold,
        "training_seconds_total": float(sum(row["training_seconds"] for row in epoch_rows)),
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

