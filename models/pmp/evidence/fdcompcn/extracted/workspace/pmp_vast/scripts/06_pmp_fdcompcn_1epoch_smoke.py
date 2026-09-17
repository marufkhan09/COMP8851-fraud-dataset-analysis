#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
import dgl

from sklearn.metrics import average_precision_score, roc_auc_score


WORK = Path("/workspace/pmp_vast")
REPO = WORK / "repo/PMP"

GRAPH_PATH = WORK / "shared/fdcompcn/comp.dgl"
SPLIT_PATH = WORK / "shared/fdcompcn/fdcompcn_seed2_nested_splits.npz"

OUT_DIR = WORK / "unified/fdcompcn/smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUT_DIR / "PMP_FDCOMPCN_1EPOCH_SMOKE.json"

EXPECTED_COMMIT = "3f7629f6c180891a0bc1bba3c66d94d288a1ddae"

EXPECTED_DATA_SHA = (
    "e252b9a6b619b28a7bf6d9f5b16aacc"
    "232d43207d87ac26cd5160fb0d8a98baa"
)

EXPECTED_SPLIT_SHA = (
    "5cd7084342f4af48e9b82828dcac0e97"
    "879314cd5c9c010642b30c89e7c95168"
)

SEMANTIC_RELATIONS = [
    "invest_bc2bc",
    "provide_bc2bc",
    "sale_bc2bc",
]

TRAIN_SEED = 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args],
        text=True
    ).strip()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


print("===== PMP × FDCOMPCN — EXACT 1-EPOCH SMOKE =====", flush=True)

# ============================================================
# 1. HARD IDENTITY GATES
# ============================================================

assert git("rev-parse", "HEAD") == EXPECTED_COMMIT
assert git("status", "--porcelain", "--untracked-files=no") == ""

assert sha256_file(GRAPH_PATH) == EXPECTED_DATA_SHA
assert sha256_file(SPLIT_PATH) == EXPECTED_SPLIT_SHA

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert "A6000" in torch.cuda.get_device_name(0).upper()

torch.cuda.set_device(0)
torch.set_num_threads(8)

print("IDENTITY_GATE=PASS", flush=True)
print("GPU =", torch.cuda.get_device_name(0), flush=True)


# ============================================================
# 2. LOAD SOURCE GRAPH
# ============================================================

source_graphs, _ = dgl.load_graphs(str(GRAPH_PATH))
source = source_graphs[0]

assert source.num_nodes("company") == 5317

available_relations = set(source.etypes)

for rel in SEMANTIC_RELATIONS:
    assert rel in available_relations

assert "homo" in available_relations

features = source.nodes["company"].data["feature"].float().clone()
labels = source.nodes["company"].data["label"].view(-1).long().clone()

assert features.shape == (5317, 57)
assert labels.shape == (5317,)
assert int((labels == 1).sum()) == 559
assert int((labels == 0).sum()) == 4758


# ============================================================
# 3. MINIMUM RELATION-PRESERVING INPUT ADAPTER
#
# Use ONLY the three semantic relations.
# Do NOT pass the pre-combined `homo` view as a fourth relation.
# ============================================================

graph_data = {}

for rel in SEMANTIC_RELATIONS:
    u, v = source.edges(etype=rel)
    graph_data[("company", rel, "company")] = (
        u.clone(),
        v.clone(),
    )

graph = dgl.heterograph(
    graph_data,
    num_nodes_dict={"company": 5317},
)

graph.nodes["company"].data["feature"] = features
graph.nodes["company"].data["label"] = labels

assert graph.num_nodes("company") == 5317
assert set(graph.etypes) == set(SEMANTIC_RELATIONS)
assert "homo" not in graph.etypes

for rel in SEMANTIC_RELATIONS:
    assert (
        graph.num_edges(rel)
        == source.num_edges(rel)
    )

print(
    "Semantic relations:",
    list(graph.etypes),
    flush=True,
)

print(
    "Semantic stored edges:",
    graph.num_edges(),
    flush=True,
)

print("PMP_FDCOMPCN_RELATION_ADAPTER=PASS", flush=True)


# ============================================================
# 4. EXACT FROZEN SPLIT
# ============================================================

splits = np.load(
    SPLIT_PATH,
    allow_pickle=False,
)

train_ids = torch.as_tensor(
    splits["TR40"],
    dtype=torch.long,
)

val_ids = torch.as_tensor(
    splits["val"],
    dtype=torch.long,
)

test_ids = torch.as_tensor(
    splits["test"],
    dtype=torch.long,
)

assert len(train_ids) == 2126
assert len(val_ids) == 1064
assert len(test_ids) == 2127

assert int(labels[train_ids].sum()) == 223
assert int(labels[val_ids].sum()) == 112
assert int(labels[test_ids].sum()) == 224

train_mask = torch.zeros(5317, dtype=torch.bool)
val_mask = torch.zeros(5317, dtype=torch.bool)
test_mask = torch.zeros(5317, dtype=torch.bool)

train_mask[train_ids] = True
val_mask[val_ids] = True
test_mask[test_ids] = True

graph.nodes["company"].data["train_mask"] = train_mask
graph.nodes["company"].data["val_mask"] = val_mask
graph.nodes["company"].data["test_mask"] = test_mask

# PMP partition labels:
# ONLY train labels are visible.
label_unk = torch.full(
    (5317,),
    2,
    dtype=torch.long,
)

label_unk[train_ids] = labels[train_ids]

graph.nodes["company"].data["label_unk"] = label_unk

assert int((label_unk != 2).sum()) == len(train_ids)
assert torch.all(label_unk[val_ids] == 2)
assert torch.all(label_unk[test_ids] == 2)

print("FROZEN_SPLIT_GATE=PASS", flush=True)
print("TRAIN_LABELS_EXPOSED_ONLY=PASS", flush=True)
print("VAL_LABELS_HIDDEN_FROM_PMP=PASS", flush=True)
print("TEST_LABELS_HIDDEN_FROM_PMP=PASS", flush=True)


# ============================================================
# 5. AUTHOR PMP IMPORTS — OFFICIAL REPO UNCHANGED
# ============================================================

sys.path.insert(0, str(REPO))
os.chdir(REPO)

from DataHelper.datasetHelper import DatasetHelper
from training_procedure import Trainer
from utils.logger import Logger
from utils.random_seeder import set_random_seed
from dgl.dataloading import MultiLayerFullNeighborSampler


# ============================================================
# 6. SMOKE CONFIG
#
# There is no native FDCompCN PMP config.
# For compatibility smoke only, retain the already-verified
# author Yelp LA-SAGE-S architecture.
# This is NOT the frozen FDCompCN final configuration.
# ============================================================

raw_cfg = yaml.safe_load(
    (REPO / "config/yelp.yml").read_text()
)

config = dict(
    raw_cfg["LA-SAGE-S"]
)

config["hid_dim"] = 64

config.update({
    "dataset": "fdcompcn",
    "model": "LA-SAGE-S",
    "model_name": "LA-SAGE-S",
    "gpu_id": 0,
    "seed": TRAIN_SEED,
    "train_size": 0.4,
    "val_size": 0.2,
    "epochs": 1,
    "patience": 20,
    "eval_interval": 1,
    "test_each_epoch": False,
    "monitor": "ap_gnn",
    "multirun": 1,
    "run_best": False,
    "no_dev": False,
    "num_workers": 8,
    "best_model_path": str(
        OUT_DIR / "unused_smoke_checkpoint.pth"
    ),
})

# Preserve author architectural choices.
assert config["homo"] is False
assert config["n_layer"] == 1
assert config["relation_agg"] == "cat"
assert config["full_neighbors"] is True

print(
    "Smoke architecture:",
    {
        "hid_dim": config["hid_dim"],
        "dropout": config["dropout"],
        "n_layer": config["n_layer"],
        "relation_agg": config["relation_agg"],
        "full_neighbors": config["full_neighbors"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
    },
    flush=True,
)


# ============================================================
# 7. EXTERNAL DATASETHELPER ADAPTER
# ============================================================

helper = DatasetHelper(
    config=config,
    dName="fdcompcn",
    dDescription=(
        "COMP8851 PMP FDCompCN relation-preserving input adapter"
    ),
)

helper.data = graph
helper.dataset = SimpleNamespace(num_classes=2)

helper.train_mask = train_mask
helper.val_mask = val_mask
helper.test_mask = test_mask

helper.train_nid = train_ids
helper.val_nid = val_ids
helper.test_nid = test_ids

helper.num_classes = 2
helper.relations = list(graph.etypes)
helper.num_relations = len(helper.relations)

helper.feat = graph.nodes["company"].data["feature"]
helper.labels = graph.nodes["company"].data["label"].view(-1).long()

helper.feat_dim = int(helper.feat.shape[1])
helper.num_nodes = int(helper.labels.shape[0])

assert helper.num_nodes == 5317
assert helper.feat_dim == 57
assert helper.num_relations == 3
assert set(helper.relations) == set(SEMANTIC_RELATIONS)


# ============================================================
# 8. AUTHOR DGL LOADER
# ============================================================

sampler = MultiLayerFullNeighborSampler(
    num_layers=config["n_layer"]
)

train_loader, val_loader, test_loader = helper.get_DGLloader(
    helper.data,
    sampler,
)

# IMPORTANT:
# test_loader is constructed for structural parity with author
# pipeline but NEVER ITERATED in this smoke.


# ============================================================
# 9. EXACT ONE TRAINING EPOCH
# ============================================================

seed_all(TRAIN_SEED)
set_random_seed(TRAIN_SEED)

logger = Logger(mode=[print])

args = SimpleNamespace(
    dataset="fdcompcn",
    num_workers=8,
    seed=TRAIN_SEED,
    data_dir=str(WORK / "shared/fdcompcn"),
    hyper_file=str(REPO / "config"),
    log_dir=str(WORK / "logs"),
    best_model_path=str(OUT_DIR),
    train_size=0.4,
    val_size=0.2,
    no_dev=False,
    gpu_id=0,
    multirun=1,
    model="LA-SAGE-S",
    run_best=False,
)

trainer = Trainer(
    args=args,
    config=config,
    logger=logger,
)

model, optimizer, loss_func, scheduler = trainer.init(
    helper
)

for pg in optimizer.param_groups:
    assert tuple(pg.get("betas", (0.9, 0.999))) == (
        0.9,
        0.999,
    )
    assert abs(
        float(pg.get("eps", 1e-8)) - 1e-8
    ) < 1e-20

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

torch.cuda.synchronize()
train_start = time.perf_counter()

model, loss = trainer.train(
    0,
    model,
    loss_func,
    optimizer,
    train_loader,
    helper,
)

torch.cuda.synchronize()
train_seconds = time.perf_counter() - train_start

avg_loss = (
    float(loss.detach().item())
    / max(1, len(train_loader))
)


# ============================================================
# 10. VALIDATION ONLY — NO TEST
# ============================================================

torch.cuda.synchronize()
val_start = time.perf_counter()

val_y, val_prob, _ = trainer.evaluation(
    helper,
    val_loader,
    model,
    threshold_moving=False,
    thres=0.5,
)

torch.cuda.synchronize()
val_seconds = time.perf_counter() - val_start

val_auprc = float(
    average_precision_score(
        val_y,
        val_prob,
    )
)

val_auroc = float(
    roc_auc_score(
        val_y,
        val_prob,
    )
)

peak_mb = float(
    torch.cuda.max_memory_allocated()
    / 1024**2
)


# ============================================================
# 11. SAVE SMOKE EVIDENCE
# ============================================================

result = {
    "status": "PASS",
    "model": "PMP / LA-SAGE-S",
    "dataset": "FDCompCN",
    "compatibility": "INPUT ADAPTER",
    "epochs": 1,
    "training_seed": TRAIN_SEED,
    "train_ratio": "TR40",
    "train_nodes": len(train_ids),
    "validation_nodes": len(val_ids),
    "test_nodes": len(test_ids),
    "semantic_relations": helper.relations,
    "homo_relation_used": False,
    "nodes": helper.num_nodes,
    "features": helper.feat_dim,
    "fraud": int((helper.labels == 1).sum()),
    "normal": int((helper.labels == 0).sum()),
    "repository_commit": EXPECTED_COMMIT,
    "dataset_sha256": EXPECTED_DATA_SHA,
    "split_sha256": EXPECTED_SPLIT_SHA,
    "avg_train_loss": avg_loss,
    "validation_auprc": val_auprc,
    "validation_auroc": val_auroc,
    "train_seconds": train_seconds,
    "validation_seconds": val_seconds,
    "peak_gpu_memory_mb": peak_mb,
    "train_labels_visible_to_partition": True,
    "validation_labels_visible_to_partition": False,
    "test_labels_visible_to_partition": False,
    "test_evaluated": False,
}

RESULT_JSON.write_text(
    json.dumps(
        result,
        indent=2,
    )
)

assert git("status", "--porcelain", "--untracked-files=no") == ""

print()
print("===== PMP FDCOMPCN 1-EPOCH SMOKE RESULT =====", flush=True)

print(
    f"loss={avg_loss:.6f} "
    f"val_AUPRC={val_auprc:.6f} "
    f"val_AUROC={val_auroc:.6f}",
    flush=True,
)

print(
    f"train={train_seconds:.3f}s "
    f"val={val_seconds:.3f}s "
    f"peak_gpu={peak_mb:.2f}MB",
    flush=True,
)

print()
print("===== SMOKE GATE =====", flush=True)
print("PMP_FDCOMPCN_RELATION_ADAPTER=PASS", flush=True)
print("PMP_FDCOMPCN_1EPOCH_SMOKE=PASS", flush=True)
print("SEMANTIC_RELATIONS=3", flush=True)
print("HOMO_AS_FOURTH_RELATION=NO", flush=True)
print("TRAIN_LABELS_ONLY_FOR_PMP_PARTITION=PASS", flush=True)
print("TEST_ACCESSED=NO", flush=True)
print("PMP_AUTHOR_REPO_CLEAN=PASS", flush=True)
print("RESULT_JSON=" + str(RESULT_JSON), flush=True)
