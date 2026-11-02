#!/usr/bin/env python3
import os, sys, gc, json, time, random, hashlib, subprocess, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dgl.data.utils import load_graphs
from sklearn.metrics import average_precision_score, roc_auc_score

WORK = Path("/workspace/bwgnn_vast")
RAW_ROOT = Path("/workspace/dataset_audit/raw/tsocial")
SPLIT_PATH = WORK / "splits/tsocial_seed2_nested_splits.npz"
EVIDENCE_DIR = WORK / "evidence/tsocial"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = EVIDENCE_DIR / "a6000_hidden64_csc_2epoch_smoke.json"

sys.path.insert(0, str(WORK / "adapters"))
from BWGNN_gpu import BWGNN

EXPECTED_COMMIT = "de0631f039bbd19c1890b483cc01f1007f596af7"
EXPECTED_DATA_SHA = "8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0"
EXPECTED_SPLIT_SHA = "b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519"

HIDDEN = 64
ORDER = 2
LR = 0.01
WEIGHT_DECAY = 0.0
SPLIT_SEED = 2
TRAIN_SEED = 2
DEVICE = torch.device("cuda:0")

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def gib_value(path):
    try:
        v = Path(path).read_text().strip()
    except Exception:
        return None, None
    if v == "max":
        return v, None
    try:
        n = int(v)
        return v, n / (1024 ** 3)
    except Exception:
        return v, None

def gpu_mem():
    return {
        "allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
        "reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
    }

def show_gpu(label):
    m = gpu_mem()
    print(
        f"{label} | allocated={m['allocated_mb']:.1f} MB | "
        f"reserved={m['reserved_mb']:.1f} MB | "
        f"peak={m['peak_allocated_mb']:.1f} MB",
        flush=True,
    )

def save_result(obj):
    RESULT_PATH.write_text(json.dumps(obj, indent=2))
    print("Evidence JSON:", RESULT_PATH, flush=True)

print("===== BWGNN VAST — T-SOCIAL A6000 HARDWARE SMOKE =====", flush=True)
print("Purpose: heaviest full-graph compatibility/timing probe only", flush=True)
print("Test set accessed: NO", flush=True)
print("Full graph: YES", flush=True)
print("Sampling/pruning: NO", flush=True)
print("Sparse storage: CSC", flush=True)
print("Config:", {
    "hidden": HIDDEN, "order": ORDER, "lr": LR,
    "weight_decay": WEIGHT_DECAY, "epochs": 2,
}, flush=True)

stage = "preflight"
base_result = {
    "model": "BWGNN",
    "dataset": "T-Social",
    "purpose": "A6000 heaviest full-graph hardware compatibility/timing smoke",
    "full_graph": True,
    "sampling_used": False,
    "graph_reduced": False,
    "test_accessed": False,
    "hidden": HIDDEN,
    "order": ORDER,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "sparse_format": "csc",
    "split_seed": SPLIT_SEED,
    "train_seed": TRAIN_SEED,
    "requested_epochs": 2,
}

try:
    # ------------------------------------------------------------------
    # 0. Host / environment gate
    # ------------------------------------------------------------------
    print("\n===== 0. HOST / ENVIRONMENT GATE =====", flush=True)

    mem_max_raw, mem_max_gib = gib_value("/sys/fs/cgroup/memory.max")
    mem_cur_raw, mem_cur_gib = gib_value("/sys/fs/cgroup/memory.current")
    print("cgroup memory.max:", mem_max_raw,
          "" if mem_max_gib is None else f"({mem_max_gib:.2f} GiB)", flush=True)
    print("cgroup memory.current:", mem_cur_raw,
          "" if mem_cur_gib is None else f"({mem_cur_gib:.2f} GiB)", flush=True)

    if mem_max_gib is not None and mem_max_gib < 128:
        raise RuntimeError(
            f"HOST_RAM_GATE_FAIL: cgroup limit {mem_max_gib:.2f} GiB < protocol minimum 128 GiB"
        )

    commit = subprocess.check_output(
        ["git", "-C", str(WORK / "repo/Rethinking-Anomaly-Detection"),
         "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    gpu_name = torch.cuda.get_device_name(0)
    assert "A6000" in gpu_name.upper(), gpu_name
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print("Repository commit:", commit, flush=True)
    print("GPU:", gpu_name, flush=True)
    print(f"GPU VRAM: {total_vram_mb:.1f} MB", flush=True)
    print("HOST_ENV_GATE=PASS", flush=True)

    # ------------------------------------------------------------------
    # 1. Canonical source hash
    # ------------------------------------------------------------------
    stage = "canonical_source_hash"
    print("\n===== 1. CANONICAL T-SOCIAL SOURCE =====", flush=True)

    preferred = RAW_ROOT / "tsocial"
    data_path = None
    checked = []

    candidates = []
    if preferred.exists() and preferred.is_file():
        candidates.append(preferred)
    candidates += [
        p for p in sorted(RAW_ROOT.rglob("*"))
        if p.is_file() and p != preferred
    ]

    for p in candidates:
        # Prefer files large enough to plausibly be the graph.
        if p.stat().st_size < 100 * 1024 * 1024:
            continue
        h = sha256_file(p)
        checked.append((str(p), h, p.stat().st_size))
        print("Candidate:", p, flush=True)
        print("SHA256:", h, flush=True)
        if h == EXPECTED_DATA_SHA:
            data_path = p
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Canonical T-Social file not found by SHA256 under {RAW_ROOT}; "
            f"expected {EXPECTED_DATA_SHA}; checked={checked[:10]}"
        )

    print("T-Social canonical path:", data_path, flush=True)
    print("T-Social SHA256:", EXPECTED_DATA_SHA, flush=True)
    print("TSOCIAL_SOURCE_GATE=PASS", flush=True)

    # ------------------------------------------------------------------
    # 2. Exact frozen split gate
    # ------------------------------------------------------------------
    stage = "split_gate"
    print("\n===== 2. FROZEN SPLIT GATE =====", flush=True)
    assert SPLIT_PATH.exists(), f"Missing exact frozen split: {SPLIT_PATH}"
    split_sha = sha256_file(SPLIT_PATH)
    print("Split SHA256:", split_sha, flush=True)
    print("Expected    :", EXPECTED_SPLIT_SHA, flush=True)
    assert split_sha == EXPECTED_SPLIT_SHA, (split_sha, EXPECTED_SPLIT_SHA)

    s = np.load(SPLIT_PATH, allow_pickle=False)
    expected_sizes = {
        "TR40": 2312426,
        "TR30": 1734319,
        "TR20": 1156212,
        "TR10": 578106,
        "val": 1156213,
        "test": 2312426,
    }
    for k, n in expected_sizes.items():
        assert k in s.files
        assert len(s[k]) == n, (k, len(s[k]), n)

    # Do not read/use test IDs for model selection/evaluation.
    train_ids_cpu = torch.tensor(s["TR40"], dtype=torch.long)
    val_ids_cpu = torch.tensor(s["val"], dtype=torch.long)
    print("TR40:", len(train_ids_cpu), flush=True)
    print("VAL :", len(val_ids_cpu), flush=True)
    print("TSOCIAL_SPLIT_GATE=PASS", flush=True)

    # ------------------------------------------------------------------
    # 3. CPU load and canonical structure
    # ------------------------------------------------------------------
    stage = "cpu_graph_load"
    print("\n===== 3. CPU GRAPH LOAD =====", flush=True)
    t0 = time.perf_counter()
    graphs, _ = load_graphs(str(data_path))
    graph = graphs[0]
    cpu_load_seconds = time.perf_counter() - t0

    assert graph.num_nodes() == 5781065, graph.num_nodes()
    assert graph.num_edges() == 146211016, graph.num_edges()
    assert "feature" in graph.ndata
    assert "label" in graph.ndata
    assert tuple(graph.ndata["feature"].shape) == (5781065, 10)

    raw_labels_cpu = graph.ndata["label"]
    labels_cpu = (
        raw_labels_cpu.argmax(1).long()
        if raw_labels_cpu.ndim == 2
        else raw_labels_cpu.reshape(-1).long()
    )
    normal = int((labels_cpu == 0).sum().item())
    anomaly = int((labels_cpu == 1).sum().item())
    assert (normal, anomaly) == (5606785, 174280), (normal, anomaly)

    print("Nodes:", graph.num_nodes(), flush=True)
    print("Stored edges:", graph.num_edges(), flush=True)
    print("Features:", tuple(graph.ndata["feature"].shape), flush=True)
    print("Normal/Anomaly:", normal, anomaly, flush=True)
    print(f"CPU load seconds: {cpu_load_seconds:.3f}", flush=True)
    print("TSOCIAL_STRUCTURE_GATE=PASS", flush=True)

    # ------------------------------------------------------------------
    # 4. Materialize CSC on CPU (allowed sparse-format conversion)
    # ------------------------------------------------------------------
    stage = "cpu_csc_conversion"
    print("\n===== 4. CPU CSC CONVERSION =====", flush=True)
    t0 = time.perf_counter()
    graph = graph.formats("csc")
    csc_seconds = time.perf_counter() - t0
    print("Graph formats:", graph.formats(), flush=True)
    print(f"CSC conversion seconds: {csc_seconds:.3f}", flush=True)

    # ------------------------------------------------------------------
    # 5. GPU transfer
    # ------------------------------------------------------------------
    stage = "gpu_transfer"
    print("\n===== 5. GPU TRANSFER =====", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    graph = graph.to(DEVICE)
    torch.cuda.synchronize()
    gpu_transfer_seconds = time.perf_counter() - t0

    print(f"GPU transfer seconds: {gpu_transfer_seconds:.3f}", flush=True)
    print("GPU sparse formats:", graph.formats(), flush=True)
    show_gpu("After graph transfer")

    # ------------------------------------------------------------------
    # 6. Features, labels, TR40/val IDs, class weight
    # ------------------------------------------------------------------
    stage = "gpu_data_prepare"
    print("\n===== 6. GPU DATA PREPARATION =====", flush=True)
    features = graph.ndata["feature"].float()
    raw_labels = graph.ndata["label"]
    labels = (
        raw_labels.argmax(1).long()
        if raw_labels.ndim == 2
        else raw_labels.reshape(-1).long()
    )
    train_ids = train_ids_cpu.to(DEVICE)
    val_ids = val_ids_cpu.to(DEVICE)

    train_labels = labels[train_ids]
    train_normal = int((train_labels == 0).sum().item())
    train_anomaly = int((train_labels == 1).sum().item())
    assert (train_normal, train_anomaly) == (2242714, 69712), (
        train_normal, train_anomaly
    )

    class_weight = torch.tensor(
        [1.0, train_normal / train_anomaly],
        dtype=torch.float32, device=DEVICE
    )

    print("TR40 normal/anomaly:", train_normal, train_anomaly, flush=True)
    print("Anomaly class weight:", float(train_normal / train_anomaly), flush=True)
    show_gpu("After features/labels/indices")

    # ------------------------------------------------------------------
    # 7. Model
    # ------------------------------------------------------------------
    stage = "model_construction"
    print("\n===== 7. MODEL CONSTRUCTION =====", flush=True)
    seed_all = lambda: None
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = BWGNN(
        in_feats=10,
        h_feats=HIDDEN,
        num_classes=2,
        graph=graph,
        d=ORDER,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    parameter_count = int(sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))
    print("Parameters:", parameter_count, flush=True)
    show_gpu("After model construction")

    # ------------------------------------------------------------------
    # 8. Two full training epochs + validation, test untouched
    # ------------------------------------------------------------------
    stage = "training"
    print("\n===== 8. TWO-EPOCH FULL-GRAPH SMOKE =====", flush=True)
    epochs = []

    for epoch in [1, 2]:
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(features)
        print(f"epoch={epoch} forward: PASS", flush=True)
        show_gpu(f"epoch={epoch} after forward")

        loss = F.cross_entropy(
            logits[train_ids], labels[train_ids], weight=class_weight
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")

        optimizer.zero_grad()
        loss.backward()
        print(f"epoch={epoch} backward: PASS", flush=True)
        optimizer.step()
        torch.cuda.synchronize()

        train_seconds = time.perf_counter() - t0
        train_loss = float(loss.item())

        del logits, loss
        gc.collect()
        torch.cuda.empty_cache()

        model.eval()
        torch.cuda.synchronize()
        v0 = time.perf_counter()
        with torch.no_grad():
            val_logits = model(features)
            val_prob = (
                torch.softmax(val_logits, dim=1)[val_ids, 1]
                .detach().cpu().numpy()
            )
        torch.cuda.synchronize()
        val_seconds = time.perf_counter() - v0
        val_y = labels[val_ids].detach().cpu().numpy()

        val_auprc = float(average_precision_score(val_y, val_prob))
        val_auroc = float(roc_auc_score(val_y, val_prob))

        row = {
            "epoch": epoch,
            "train_seconds": float(train_seconds),
            "validation_seconds": float(val_seconds),
            "training_loss": train_loss,
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "peak_gpu_memory_mb": float(
                torch.cuda.max_memory_allocated() / 1024**2
            ),
        }
        epochs.append(row)

        print(
            f"SMOKE epoch={epoch} | train={train_seconds:.3f}s | "
            f"val={val_seconds:.3f}s | loss={train_loss:.6f} | "
            f"val_AUPRC={val_auprc:.6f} | val_AUROC={val_auroc:.6f}",
            flush=True,
        )
        show_gpu(f"epoch={epoch} completed")

        del val_logits, val_prob, val_y
        gc.collect()
        torch.cuda.empty_cache()

    mean_train = float(np.mean([x["train_seconds"] for x in epochs]))
    median_train = float(np.median([x["train_seconds"] for x in epochs]))
    peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)

    result = {
        **base_result,
        "status": "PASS",
        "classification": "A6000_FULL_GRAPH_H64_CSC_COMPATIBLE",
        "repository_commit": commit,
        "dataset_path": str(data_path),
        "dataset_sha256": EXPECTED_DATA_SHA,
        "split_sha256": split_sha,
        "nodes": 5781065,
        "stored_edges": 146211016,
        "features": 10,
        "normal": 5606785,
        "anomaly": 174280,
        "gpu": gpu_name,
        "gpu_vram_mb": float(total_vram_mb),
        "cgroup_memory_max_raw": mem_max_raw,
        "cgroup_memory_max_gib": mem_max_gib,
        "cpu_graph_load_seconds": float(cpu_load_seconds),
        "csc_conversion_seconds": float(csc_seconds),
        "gpu_transfer_seconds": float(gpu_transfer_seconds),
        "parameter_count": parameter_count,
        "epochs_completed": 2,
        "epochs": epochs,
        "mean_train_seconds_per_epoch": mean_train,
        "median_train_seconds_per_epoch": median_train,
        "peak_gpu_memory_mb": peak_mb,
    }
    save_result(result)

    print("\n===== T-SOCIAL A6000 SMOKE GATE =====", flush=True)
    print("TSOCIAL_A6000_SMOKE=PASS", flush=True)
    print(f"MEAN_TRAIN_SECONDS_PER_EPOCH={mean_train:.6f}", flush=True)
    print(f"MEDIAN_TRAIN_SECONDS_PER_EPOCH={median_train:.6f}", flush=True)
    print(f"PEAK_GPU_MEMORY_MB={peak_mb:.2f}", flush=True)
    print("TEST_ACCESSED=NO", flush=True)
    print("FULL_FINAL_GRID_STARTED=NO", flush=True)

except Exception as error:
    err = str(error)
    is_oom = (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "out of memory" in err.lower()
        or "cudaoutofmemory" in type(error).__name__.lower()
    )

    rec = {
        **base_result,
        "status": "OOM" if is_oom else "FAIL",
        "classification": (
            "A6000_FULL_GRAPH_H64_CSC_OOM"
            if is_oom else "NON_OOM_FAILURE"
        ),
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error": err,
        "gpu_memory_at_failure": (
            gpu_mem() if torch.cuda.is_available() else None
        ),
    }
    save_result(rec)

    print("\n===== T-SOCIAL A6000 SMOKE GATE =====", flush=True)
    if is_oom:
        print("TSOCIAL_A6000_SMOKE=OOM", flush=True)
        print("Classification: CUDA OUT OF MEMORY", flush=True)
        print("FULL_FINAL_GRID_STARTED=NO", flush=True)
        print("TEST_ACCESSED=NO", flush=True)
        # OOM is a valid controlled feasibility outcome.
        sys.exit(0)

    print("TSOCIAL_A6000_SMOKE=FAIL", flush=True)
    traceback.print_exc()
    sys.exit(1)
