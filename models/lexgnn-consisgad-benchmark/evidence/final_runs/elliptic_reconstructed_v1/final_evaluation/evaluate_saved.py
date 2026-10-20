import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import torch

WORK = Path("/workspace/comp8851_elliptic_work")
BATCH = WORK / "elliptic_reconstructed_v1"
OUT = BATCH / "final_evaluation"
sys.path.insert(0, str(WORK / "adapters"))

from benchmark_common import evaluate_scores, set_seed
from elliptic_data import load_elliptic_graph, load_elliptic_split


def load_adapter(model):
    path = WORK / "adapters" / model / "train_elliptic_compat.py"
    spec = importlib.util.spec_from_file_location(
        f"elliptic_eval_{model}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lexgnn", "consisgad"],
                        required=True)
    parser.add_argument("--ratio", type=int,
                        choices=[10, 20, 30, 40], required=True)
    parser.add_argument("--seed", type=int,
                        choices=[2, 42, 72], required=True)
    args = parser.parse_args()

    name = f"{args.model}_tr{args.ratio}_seed{args.seed}"
    marker = BATCH / "logs" / f"{name}.PASS"
    summary_path = Path(marker.read_text().strip())
    run_dir = summary_path.parent
    checkpoint_path = run_dir / "checkpoint.pt"
    config_path = run_dir / "run_config.json"
    prediction_path = run_dir / "predictions.npz"
    split_path = BATCH / "splits" / (
        f"elliptic_TR{args.ratio}_seed2.npz"
    )
    manifest_path = BATCH / "splits" / "manifest.json"

    result_path = OUT / f"{name}_test.json"
    lock_path = OUT / f"{name}.IN_PROGRESS"

    if result_path.exists():
        raise RuntimeError(f"Test already evaluated: {result_path}")
    if lock_path.exists():
        raise RuntimeError(
            f"Previous evaluation requires review: {lock_path}"
        )

    summary = json.loads(summary_path.read_text())
    cfg = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    expected_mode = (
        f"reconstructed_v1_tr{args.ratio}_"
        f"seed{args.seed}_validation_only"
    )

    assert summary["status"] == "PASS"
    assert summary["model"].lower().replace("-", "") == args.model
    assert summary["run_mode"] == expected_mode
    assert summary["epochs_completed"] == 100
    assert summary["test_evaluated"] is False
    assert cfg["evaluate_test"] is False
    assert cfg["seed"] == args.seed
    assert Path(cfg["split_path"]).resolve() == split_path.resolve()
    assert manifest["original_team_split"] is False
    assert sha256(split_path) == (
        manifest["splits"][f"TR{args.ratio}"]["sha256"]
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu"
    )

    assert checkpoint["best_epoch"] == summary["best_validation_epoch"]
    assert checkpoint["config"]["seed"] == args.seed
    assert checkpoint["config"]["run_mode"] == expected_mode

    threshold = float(summary["selected_threshold"])
    assert np.isfinite(threshold)
    assert 0 <= threshold <= 1

    # Confirm threshold was selected from saved validation data,
    # without accessing the test predictions.
    from benchmark_common import select_validation_threshold
    with np.load(prediction_path, allow_pickle=False) as pred:
        assert "test_scores" not in pred.files
        validation_threshold = select_validation_threshold(
            pred["validation_labels"],
            pred["validation_scores"],
        )
    assert abs(threshold - validation_threshold) < 1e-10, (
        "Validation threshold mismatch"
    )

    print(f"VERIFIED: {name}", flush=True)
    print(f"Best epoch: {checkpoint['best_epoch']}", flush=True)
    print(f"Frozen threshold: {threshold}", flush=True)

    # Created before test inference to prevent accidental repeat.
    lock_path.write_text(
        json.dumps({
            "run": name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "split_sha256": sha256(split_path),
        }, indent=2)
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    adapter = load_adapter(args.model)
    start = time.perf_counter()

    if args.model == "lexgnn":
        sys.path.insert(0, "/workspace/upstreams/LEX-GNN")
        from model import LEXGNN

        graph = load_elliptic_graph(
            WORK / "data", normalize_for_lex=True
        )
        split = load_elliptic_split(
            split_path, graph.num_nodes()
        )

        model = LEXGNN(
            graph.ndata["feature"].shape[1],
            2,
            int(cfg["hidden_dimension"]),
            int(cfg["layers"]),
            int(cfg["heads"]),
            float(cfg["dropout"]),
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )

        _, _, test_loader = adapter.make_loaders(
            graph, split,
            int(cfg["layers"]),
            int(cfg["batch_size"]),
        )
        test_labels, test_scores = adapter.collect_scores(
            model, test_loader, device
        )

    else:
        sys.path.insert(0, "/workspace/upstreams/ConsisGAD")
        upstream = importlib.import_module("main")

        graph = load_elliptic_graph(
            WORK / "data", normalize_for_lex=False
        )
        split = load_elliptic_split(
            split_path, graph.num_nodes()
        )
        graph = graph.to(device)

        model_args = dict(cfg)
        model_args["device"] = device
        model_args["node-in-dim"] = int(
            graph.ndata["feature"].shape[1]
        )
        model_args["node-out-dim"] = 2
        upstream.args = model_args

        model = upstream.create_model(
            model_args, graph.etypes
        )
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )

        # Attention weights are retained in the checkpoint.
        # The author's get_model_pred uses the trained backbone.
        assert "attention_state_dict" in checkpoint

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            int(model_args["num-layers"])
        )
        test_loader = adapter.loader(
            split["test"], 2**16, False, args.seed
        )

        model.eval()
        with torch.no_grad():
            score_tensor, label_tensor = upstream.get_model_pred(
                model, graph, test_loader, sampler, model_args
            )

        test_scores = score_tensor.detach().cpu().numpy()
        test_labels = label_tensor.detach().cpu().numpy()

    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start

    assert len(test_labels) == len(test_scores) == 17626
    assert set(np.unique(test_labels)).issubset({0, 1})
    assert np.isfinite(test_scores).all()

    metrics = evaluate_scores(
        test_labels, test_scores, threshold
    )

    result = {
        "run": name,
        "protocol": "elliptic_reconstructed_v1",
        "original_team_split": False,
        "model": args.model,
        "ratio": args.ratio,
        "seed": args.seed,
        "selected_validation_epoch": checkpoint["best_epoch"],
        "frozen_validation_threshold": threshold,
        "checkpoint_sha256": sha256(checkpoint_path),
        "split_sha256": sha256(split_path),
        "test_nodes": int(len(test_labels)),
        "test_inference_seconds": inference_seconds,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "test_metrics": metrics,
        "test_evaluated": True,
    }

    temp_path = OUT / f"{name}_test.json.tmp"
    temp_path.write_text(json.dumps(result, indent=2))
    os.replace(temp_path, result_path)
    lock_path.unlink()

    print(json.dumps(result, indent=2), flush=True)
    print(f"PASS: {name} TEST EVALUATED", flush=True)


if __name__ == "__main__":
    main()
