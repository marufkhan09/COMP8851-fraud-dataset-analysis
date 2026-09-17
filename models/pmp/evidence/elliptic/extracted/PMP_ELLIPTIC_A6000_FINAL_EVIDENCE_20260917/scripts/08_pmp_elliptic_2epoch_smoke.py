import os
import sys
import json
import time
import random
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import dgl
import yaml

WORK = Path("/workspace/pmp_vast")
REPO = WORK / "repo/PMP"
RAW = WORK / "shared/elliptic/raw"
SPLIT = WORK / "shared/elliptic/elliptic_chronological_nested_splits.npz"
EVID = WORK / "evidence/elliptic"
EVID.mkdir(parents=True, exist_ok=True)

EXPECTED_COMMIT = "3f7629f6c180891a0bc1bba3c66d94d288a1ddae"

RAW_HASHES = {
    "elliptic_txs_features.csv":
        "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
    "elliptic_txs_classes.csv":
        "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",
    "elliptic_txs_edgelist.csv":
        "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
}

EXPECTED_SPLIT_SHA = \
    "1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args],
        text=True
    ).strip()

print("===== PMP × ELLIPTIC — 2-EPOCH COMPATIBILITY SMOKE =====",
      flush=True)

assert git("rev-parse", "HEAD") == EXPECTED_COMMIT
assert git("status", "--porcelain", "--untracked-files=no") == ""

assert torch.cuda.is_available()
assert "A6000" in torch.cuda.get_device_name(0).upper()

torch.cuda.set_device(0)
torch.set_num_threads(8)

print("IDENTITY_GPU_GATE=PASS", flush=True)

for name, expected in RAW_HASHES.items():
    got = sha256_file(RAW / name)
    assert got == expected, (name, got, expected)

print("ELLIPTIC_RAW_HASH_GATE=PASS", flush=True)

feat_df = pd.read_csv(RAW / "elliptic_txs_features.csv", header=None)
class_df = pd.read_csv(RAW / "elliptic_txs_classes.csv")
edge_df = pd.read_csv(RAW / "elliptic_txs_edgelist.csv")

assert feat_df.shape == (203769, 167), feat_df.shape
assert len(edge_df) == 234355, len(edge_df)

tx_ids = feat_df.iloc[:, 0].astype(np.int64).to_numpy()
timesteps = feat_df.iloc[:, 1].astype(np.int64).to_numpy()

# v4.4: remove transaction ID only.
# Time step remains feature column 0 of the 166-dimensional model matrix.
x_np = feat_df.iloc[:, 1:].to_numpy(dtype=np.float32)

assert x_np.shape == (203769, 166)
assert np.array_equal(x_np[:, 0].astype(np.int64), timesteps)

id_to_idx = {int(tx): i for i, tx in enumerate(tx_ids)}

labels_np = np.full(len(tx_ids), -1, dtype=np.int64)
class_id_col, class_col = class_df.columns[:2]

for tx, raw_label in zip(class_df[class_id_col], class_df[class_col]):
    idx = id_to_idx[int(tx)]
    value = str(raw_label).strip()

    if value == "1":
        labels_np[idx] = 1
    elif value == "2":
        labels_np[idx] = 0

assert int((labels_np >= 0).sum()) == 46564
assert int((labels_np == 1).sum()) == 4545
assert int((labels_np == 0).sum()) == 42019
assert int((labels_np < 0).sum()) == 157205

print("ELLIPTIC_166_FEATURE_LABEL_GATE=PASS", flush=True)
print("TIME_STEP_RETAINED_AS_FEATURE=YES", flush=True)
print("UNKNOWN_NODES_RETAINED_STRUCTURALLY=157205", flush=True)

known = labels_np >= 0

def known_between(a, b):
    return np.where(
        known & (timesteps >= a) & (timesteps <= b)
    )[0].astype(np.int64)

splits = {
    "TR10": known_between(1, 3),
    "TR20": known_between(1, 7),
    "TR30": known_between(1, 12),
    "TR40": known_between(1, 20),
    "val":  known_between(21, 31),
    "test": known_between(32, 49),
}

expected_sizes = {
    "TR10": 4543,
    "TR20": 9553,
    "TR30": 13670,
    "TR40": 18889,
    "val": 8726,
    "test": 18949,
}

for key, n in expected_sizes.items():
    assert len(splits[key]) == n, (key, len(splits[key]), n)

assert set(splits["TR10"]) <= set(splits["TR20"])
assert set(splits["TR20"]) <= set(splits["TR30"])
assert set(splits["TR30"]) <= set(splits["TR40"])
assert not (set(splits["TR40"]) & set(splits["val"]))
assert not (set(splits["TR40"]) & set(splits["test"]))
assert not (set(splits["val"]) & set(splits["test"]))

# Recreate only if the registered split is absent.
# Candidate is promoted only if its byte SHA exactly matches the frozen registry.
if not SPLIT.exists():
    candidate = SPLIT.with_name(SPLIT.stem + ".candidate.npz")

    np.savez_compressed(
        candidate,
        **splits,
        split_seed=np.array([2], dtype=np.int64),
        tr10_max_timestep=np.array([3]),
        tr20_max_timestep=np.array([7]),
        tr30_max_timestep=np.array([12]),
        tr40_max_timestep=np.array([20]),
        val_start_timestep=np.array([21]),
        val_end_timestep=np.array([31]),
        test_start_timestep=np.array([32]),
    )

    candidate_sha = sha256_file(candidate)

    print("NUMPY_VERSION =", np.__version__, flush=True)
    print("RECREATED_SPLIT_SHA256 =", candidate_sha, flush=True)

    assert candidate_sha == EXPECTED_SPLIT_SHA, (
        "Frozen split byte hash mismatch; STOP before PMP smoke.",
        candidate_sha,
        EXPECTED_SPLIT_SHA,
        np.__version__,
    )

    candidate.replace(SPLIT)

assert sha256_file(SPLIT) == EXPECTED_SPLIT_SHA

registered = np.load(SPLIT, allow_pickle=False)

for key, n in expected_sizes.items():
    assert len(registered[key]) == n

print("ELLIPTIC_SPLIT_GATE=PASS", flush=True)
print("ELLIPTIC_SPLIT_SHA256=" + EXPECTED_SPLIT_SHA, flush=True)

# Preserve the canonical directed payment-flow graph.
# Do not symmetrise it and do not inject self-loops.
src_tx = edge_df.iloc[:, 0].astype(np.int64).to_numpy()
dst_tx = edge_df.iloc[:, 1].astype(np.int64).to_numpy()

src = np.fromiter(
    (id_to_idx[int(x)] for x in src_tx),
    dtype=np.int64,
    count=len(src_tx),
)

dst = np.fromiter(
    (id_to_idx[int(x)] for x in dst_tx),
    dtype=np.int64,
    count=len(dst_tx),
)

graph = dgl.graph(
    (torch.from_numpy(src), torch.from_numpy(dst)),
    num_nodes=len(tx_ids),
)

assert graph.num_nodes() == 203769
assert graph.num_edges() == 234355

print("DIRECTED_RAW_EDGES_PRESERVED=234355", flush=True)
print("REVERSE_EDGES_ADDED=NO", flush=True)
print("SELF_LOOPS_ADDED=NO", flush=True)

sys.path.insert(0, str(REPO))
os.chdir(REPO)

from DataHelper.datasetHelper import DatasetHelper
from training_procedure import Trainer
from utils.logger import Logger
from utils.random_seeder import set_random_seed
from dgl.dataloading import MultiLayerFullNeighborSampler

# Non-native configuration precedence:
# closest official PMP setting = native homogeneous financial T-Finance.
raw_cfg = yaml.safe_load((REPO / "config/tfinance.yml").read_text())
base = dict(raw_cfg["LA-SAGE-S"])

author_expected = {
    "add_self_loop": False,
    "agg": "mean",
    "batch_size": 256,
    "dataset_seed": 717,
    "dropout": 0.4,
    "full_neighbors": True,
    "hid_dim": 64,
    "homo": True,
    "lr": 0.005,
    "n_layer": 1,
    "norm_feat": True,
    "num_trans": 1,
    "optimizer": "Adam",
    "proj": True,
    "reduction": "mean",
    "relation_agg": "cat",
    "resi": 0.2,
    "weight_decay": 0.0,
    "weighted_loss": False,
}

for key, value in author_expected.items():
    assert base.get(key) == value, (key, base.get(key), value)

config = dict(base)

# COMP8851 compatibility-only override.
# Official PMP Amazon configuration supplies hid_dim=256.
# Required here because Elliptic has 166 model features and
# PMP LA-SAGE-S fails internally when in_dim > out_dim.
config["hid_dim"] = 256

# Frozen Elliptic v4.4 representation must not receive the
# T-Finance row-normalisation preprocessing.
config["norm_feat"] = True

config.update({
    "dataset": "elliptic",
    "model": "LA-SAGE-S",
    "model_name": "LA-SAGE-S",
    "gpu_id": 0,
    "seed": 2,
    "train_size": 0.4,
    "val_size": 0.2,
    "multirun": 1,
    "run_best": False,
    "no_dev": False,
    "num_workers": 8,
    "best_model_path":
        str(WORK / "checkpoints/elliptic_smoke_unused.pth"),
    "epochs": 2,
    "patience": 0,
    "eval_interval": 1,
    "test_each_epoch": False,
    "monitor": "ap_gnn",
})

args = SimpleNamespace(
    dataset="elliptic",
    num_workers=8,
    seed=2,
    data_dir=str(RAW),
    hyper_file=str(REPO / "config"),
    log_dir=str(WORK / "logs"),
    best_model_path=str(WORK / "checkpoints"),
    train_size=0.4,
    val_size=0.2,
    no_dev=False,
    gpu_id=0,
    multirun=1,
    model="LA-SAGE-S",
    run_best=False,
)

print("TRANSFER_CONFIG_SOURCE=PMP_NATIVE_TFINANCE_LA-SAGE-S",
      flush=True)
print(
    "HID=256 DROPOUT=0.4 LR=0.005 LAYERS=1 "
    "BATCH=256 NORM_FEAT=TRUE",
    flush=True,
)
print(
    "HID_DIM_COMPATIBILITY_SOURCE="
    "PMP_OFFICIAL_AMAZON_CONFIG_256",
    flush=True,
)
print(
    "HID_DIM_REASON="
    "ELLIPTIC_166_GT_TFINANCE_HID64_TRIGGERS_AUTHOR_DIMENSION_BUG",
    flush=True,
)
print("PERFORMANCE_SPECIFIC_TUNING=NO", flush=True)
print("MODEL_SIDE_FEATURE_NORMALIZATION=PMP_AUTHOR_CONFIG",
      flush=True)

helper = DatasetHelper(
    config=config,
    dName="elliptic",
    dDescription="COMP8851 Elliptic 166-feature compatibility smoke",
)

# Preserve source file unchanged; this is PMP's documented model-side
# feature-normalisation setting inherited from its native T-Finance config.
assert config["norm_feat"] is True
features = helper.row_normalize(x_np, dtype=np.float32)
assert features.shape == (203769, 166)
print("PMP_AUTHOR_MODEL_SIDE_NORMALIZATION=YES", flush=True)
print("PROJECT_SPECIFIC_EXTRA_NORMALIZATION=NO", flush=True)

graph.ndata["feature"] = torch.from_numpy(
    np.asarray(features, dtype=np.float32)
)

graph.ndata["label"] = torch.from_numpy(labels_np).long()

N = graph.num_nodes()

train_ids = torch.as_tensor(registered["TR40"], dtype=torch.long)
val_ids = torch.as_tensor(registered["val"], dtype=torch.long)
test_ids = torch.as_tensor(registered["test"], dtype=torch.long)

train_mask = torch.zeros(N, dtype=torch.bool)
val_mask = torch.zeros(N, dtype=torch.bool)
test_mask = torch.zeros(N, dtype=torch.bool)

train_mask[train_ids] = True
val_mask[val_ids] = True
test_mask[test_ids] = True

graph.ndata["train_mask"] = train_mask
graph.ndata["val_mask"] = val_mask
graph.ndata["test_mask"] = test_mask

# PMP's fraud/benign/unknown partition signal:
# ONLY TR40 labels are exposed. Every validation, test and genuinely
# unknown transaction receives label_unk=2.
label_unk = torch.full((N,), 2, dtype=torch.long)
label_unk[train_ids] = graph.ndata["label"][train_ids]
graph.ndata["label_unk"] = label_unk

assert int((label_unk != 2).sum()) == len(train_ids)
assert torch.all(label_unk[val_ids] == 2)
assert torch.all(label_unk[test_ids] == 2)

helper.data = graph
helper.dataset = None
helper.train_mask = train_mask
helper.val_mask = val_mask
helper.test_mask = test_mask
helper.train_nid = train_ids
helper.val_nid = val_ids
helper.test_nid = test_ids
helper.num_classes = 2
helper.relations = list(graph.etypes)
helper.num_relations = len(helper.relations)
helper.feat = graph.ndata["feature"]
helper.feat_dim = helper.feat.shape[1]
helper.labels = graph.ndata["label"]
helper.num_nodes = N

assert helper.feat_dim == 166
assert helper.num_relations == 1

print("PMP_TRAIN_LABEL_EXPOSURE_GATE=PASS", flush=True)
print("VAL_TEST_PARTITION_LABELS_HIDDEN=PASS", flush=True)

sampler = MultiLayerFullNeighborSampler(
    num_layers=config["n_layer"]
)

train_loader, val_loader, _test_loader_unused = \
    helper.get_DGLloader(graph, sampler)

set_random_seed(2)
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

logger = Logger(mode=[print])
logger.add_line = lambda: logger.log("-" * 50)

trainer = Trainer(
    args=args,
    config=config,
    logger=logger,
)

model, optimizer, loss_func, scheduler = trainer.init(helper)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

rows = []

for epoch in range(2):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model, loss = trainer.train(
        epoch,
        model,
        loss_func,
        optimizer,
        train_loader,
        helper,
    )

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t0

    torch.cuda.synchronize()
    v0 = time.perf_counter()

    labels, fraud_probs, preds = trainer.evaluation(
        helper,
        val_loader,
        model,
        threshold_moving=config["threshold_moving"],
        thres=config["thres"],
    )

    torch.cuda.synchronize()
    val_seconds = time.perf_counter() - v0

    metrics = trainer.eval_model(
        labels,
        fraud_probs,
        preds,
    )

    row = {
        "epoch": epoch + 1,
        "train_seconds": float(train_seconds),
        "validation_seconds": float(val_seconds),
        "val_auprc": float(metrics.ap_gnn),
        "val_auroc": float(metrics.auc_gnn),
        "val_macro_f1_at_author_smoke_threshold":
            float(metrics.f1_macro),
        "peak_gpu_memory_mb":
            float(torch.cuda.max_memory_allocated() / 1024**2),
    }

    rows.append(row)

    print(
        f"SMOKE epoch={epoch+1} "
        f"train={train_seconds:.3f}s "
        f"val={val_seconds:.3f}s "
        f"val_AUPRC={metrics.ap_gnn:.6f} "
        f"val_AUROC={metrics.auc_gnn:.6f}",
        flush=True,
    )

# No test-loader evaluation occurs in this smoke.
assert git("status", "--porcelain", "--untracked-files=no") == ""

result = {
    "model": "PMP / LA-SAGE-S",
    "dataset": "Elliptic",
    "purpose": "2-epoch compatibility smoke only",
    "compatibility_candidate": "INPUT_ADAPTER",
    "transfer_config_source": (
        "native PMP T-Finance LA-SAGE-S base; "
        "hid_dim=256 from official PMP Amazon config as "
        "dimension-compatibility fallback"
    ),
    "hid_dim": 256,
    "hid_dim_selection_reason": (
        "repository-supported compatibility value; "
        "not selected from validation/test performance"
    ),
    "author_model_side_normalization": True,
    "project_specific_extra_normalization": False,
    "repository_commit": EXPECTED_COMMIT,
    "raw_hashes": RAW_HASHES,
    "split_sha256": EXPECTED_SPLIT_SHA,
    "features": 166,
    "time_step_retained": True,
    "directed_edges": 234355,
    "reverse_edges_added": False,
    "self_loops_added": False,
    "unknown_nodes_retained": 157205,
    "train_partition_labels_only": True,
    "test_loader_iterated": False,
    "rows": rows,
}

out = EVID / "PMP_ELLIPTIC_2EPOCH_SMOKE.json"
out.write_text(json.dumps(result, indent=2))

print("PMP_ELLIPTIC_2EPOCH_SMOKE=PASS", flush=True)
print("TEST_EVALUATED=NO", flush=True)
print("PMP_AUTHOR_REPO_CLEAN=PASS", flush=True)
print("NEXT_ACTION=REVIEW_SMOKE_BEFORE_TUNING_OR_FINAL",
      flush=True)
