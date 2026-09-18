#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import os, sys, json, time, random, hashlib, shutil, subprocess

import numpy as np
import torch
import yaml

WORK = Path("/workspace/pmp_vast")
REPO = WORK / "repo/PMP"
CANONICAL = Path("/workspace/dataset_audit/raw/yelpchi/YelpChi.mat")
SPLIT = WORK / "shared/yelp_seed2_nested_splits.npz"
EVID_DIR = WORK / "evidence/yelp"
EVID_DIR.mkdir(parents=True, exist_ok=True)
OUT = EVID_DIR / "yelp_tr40_seed2_2epoch_smoke.json"

EXPECTED_COMMIT = "3f7629f6c180891a0bc1bba3c66d94d288a1ddae"
EXPECTED_DATA_SHA = "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42"
EXPECTED_SPLIT_SHA = "0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()

print("===== PMP VAST — YELPCHI UNIFIED 2-EPOCH SMOKE =====", flush=True)
print("Author source modified: NO", flush=True)
print("Final test evaluated: NO", flush=True)
print("Training split: TR40", flush=True)
print("Training seed: 2", flush=True)

# ------------------------------------------------------------------
# 0. Immutable source/data/split gates.
# ------------------------------------------------------------------
print("\n===== 0. IDENTITY GATES =====", flush=True)
assert REPO.exists()
commit = git("rev-parse", "HEAD")
assert commit == EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)
status_before = git("status", "--porcelain")
assert status_before == "", f"Official PMP repo is dirty before smoke:\n{status_before}"

assert CANONICAL.exists()
data_sha = sha256_file(CANONICAL)
assert data_sha == EXPECTED_DATA_SHA, (data_sha, EXPECTED_DATA_SHA)

assert SPLIT.exists()
split_sha = sha256_file(SPLIT)
assert split_sha == EXPECTED_SPLIT_SHA, (split_sha, EXPECTED_SPLIT_SHA)

s = np.load(SPLIT, allow_pickle=False)
expected_sizes = {
    "TR40": 18381, "TR30": 13785, "TR20": 9190,
    "TR10": 4595, "val": 9191, "test": 18382,
}
for k, n in expected_sizes.items():
    assert len(s[k]) == n, (k, len(s[k]), n)
assert int(s["seed"][0]) == 2
assert int(s["source_nodes"][0]) == 45954

print("PMP commit       :", commit, flush=True)
print("YelpChi SHA256   :", data_sha, flush=True)
print("Frozen split SHA :", split_sha, flush=True)
print("IDENTITY_GATES=PASS", flush=True)

# ------------------------------------------------------------------
# 1. Import exact frozen author package.
# ------------------------------------------------------------------
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from DataHelper.datasetHelper import DatasetHelper
from training_procedure import Trainer
from utils.logger import Logger
from utils.random_seeder import set_random_seed
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.data.utils import _get_dgl_url

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert "A6000" in torch.cuda.get_device_name(0).upper()
torch.cuda.set_device(0)
torch.set_num_threads(8)

# ------------------------------------------------------------------
# 2. Load author Yelp configuration; override ONLY smoke/unified controls.
# ------------------------------------------------------------------
print("\n===== 1. AUTHOR CONFIG =====", flush=True)
raw_cfg = yaml.safe_load((REPO / "config/yelp.yml").read_text())
assert "LA-SAGE-S" in raw_cfg
config = dict(raw_cfg["LA-SAGE-S"])

# Preserve author architecture/data-path settings.
author_expected = {
    "batch_size": 512,
    "dataset_seed": 717,
    "dropout": 0.0,
    "full_neighbors": True,
    "hid_dim": 48,
    "homo": False,
    "lr": 0.01,
    "n_layer": 1,
    "relation_agg": "cat",
    "resi": 0.2,
    "weight_decay": 0.0,
}
for k, v in author_expected.items():
    assert config.get(k) == v, (k, config.get(k), v)

# Fields normally merged by main.py / args2config.
config.update({
    "dataset": "yelp",
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
    "best_model_path": str(WORK / "checkpoints/yelp_smoke_unused.pth"),
    # Smoke only: exactly two epochs; no test path.
    "epochs": 2,
    "patience": 0,
    "eval_interval": 1,
    "test_each_epoch": False,
    # Unified selection metric for subsequent tuning/final work.
    "monitor": "ap_gnn",
})

args = SimpleNamespace(
    dataset="yelp",
    num_workers=8,
    seed=2,
    data_dir=str(WORK / "canonical_raw"),
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

print("Author model      : LA-SAGE-S", flush=True)
print("Hidden / layers   :", config["hid_dim"], config["n_layer"], flush=True)
print("LR / weight decay :", config["lr"], config["weight_decay"], flush=True)
print("Batch size        :", config["batch_size"], flush=True)
print("Full neighbors    :", config["full_neighbors"], flush=True)
print("SMOKE_CONFIG_GATE=PASS", flush=True)

# ------------------------------------------------------------------
# 3. Prepare canonical Yelp file in DGL's expected raw_path WITHOUT download.
#    DGL's raw_path includes an URL SHA1 suffix in this version.
# ------------------------------------------------------------------
print("\n===== 2. CANONICAL AUTHOR DATA LOAD =====", flush=True)
raw_root = Path("/dev/shm/pmp_yelp_author_raw")
shutil.rmtree(raw_root, ignore_errors=True)
raw_root.mkdir(parents=True, exist_ok=True)

dgl_url = _get_dgl_url("dataset/FraudYelp.zip")
url_suffix = hashlib.sha1(dgl_url.encode("utf-8")).hexdigest()[:8]

# Current DGL raw_path convention.
for sub in [f"yelp_{url_suffix}", "yelp"]:
    d = raw_root / sub
    d.mkdir(parents=True, exist_ok=True)
    link = d / "YelpChi.mat"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(CANONICAL)

helper = DatasetHelper(config=config, dName="yelp", dDescription="COMP8851 unified YelpChi")
assert helper.dataset_name == "yelp"
helper.dataset_source_folder_path = str(raw_root)

# This calls the official PMP DatasetHelper.load -> DGL FraudDataset path.
helper.load()

assert helper.num_nodes == 45954, helper.num_nodes
assert helper.feat_dim == 32, helper.feat_dim
assert int((helper.labels == 1).sum().item()) == 6677
assert int((helper.labels == 0).sum().item()) == 39277
assert helper.data.number_of_edges() == 8051348, helper.data.number_of_edges()
assert set(helper.relations) == {"net_rsr", "net_rtr", "net_rur"}, helper.relations

print("Nodes             :", helper.num_nodes, flush=True)
print("Stored rel edges  :", helper.data.number_of_edges(), flush=True)
print("Features          :", helper.feat_dim, flush=True)
print("Normal / anomaly  : 39277 / 6677", flush=True)
print("Relations         :", helper.relations, flush=True)
print("AUTHOR_DATA_LOAD_GATE=PASS", flush=True)

# ------------------------------------------------------------------
# 4. Replace only stock random masks with exact frozen COMP8851 IDs.
#    Preserve author graph, features, relations and loader semantics.
# ------------------------------------------------------------------
print("\n===== 3. APPLY EXACT FROZEN TR40 / VAL / TEST IDS =====", flush=True)
N = helper.num_nodes
train_ids = torch.as_tensor(s["TR40"], dtype=torch.long)
val_ids = torch.as_tensor(s["val"], dtype=torch.long)
test_ids = torch.as_tensor(s["test"], dtype=torch.long)

train_mask = torch.zeros(N, dtype=torch.bool)
val_mask = torch.zeros(N, dtype=torch.bool)
test_mask = torch.zeros(N, dtype=torch.bool)
train_mask[train_ids] = True
val_mask[val_ids] = True
test_mask[test_ids] = True

helper.data.ndata["train_mask"] = train_mask
helper.data.ndata["val_mask"] = val_mask
helper.data.ndata["test_mask"] = test_mask

helper.train_mask = train_mask
helper.val_mask = val_mask
helper.test_mask = test_mask
helper.train_nid = train_ids
helper.val_nid = val_ids
helper.test_nid = test_ids

# PMP's partition signal must reveal labels for TRAIN nodes only.
label_unk = torch.full((N,), 2, dtype=torch.long)
label_unk[train_ids] = helper.labels[train_ids]
helper.data.ndata["label_unk"] = label_unk

assert int((label_unk != 2).sum()) == len(train_ids)
assert torch.all(label_unk[val_ids] == 2)
assert torch.all(label_unk[test_ids] == 2)

train_fraud = int(helper.labels[train_ids].sum().item())
val_fraud = int(helper.labels[val_ids].sum().item())
test_fraud = int(helper.labels[test_ids].sum().item())

sampler = MultiLayerFullNeighborSampler(num_layers=config["n_layer"])
train_loader, val_loader, _test_loader_unused = helper.get_DGLloader(helper.data, sampler)

print("TR40              :", len(train_ids), "fraud=", train_fraud, flush=True)
print("Validation        :", len(val_ids), "fraud=", val_fraud, flush=True)
print("Test IDs registered:", len(test_ids), "fraud count audit=", test_fraud, flush=True)
print("Partition labels  : TRAIN ONLY", flush=True)
print("Test loader iterated: NO", flush=True)
print("FROZEN_SPLIT_ADAPTER_GATE=PASS", flush=True)

# ------------------------------------------------------------------
# 5. Exact author Trainer/model API. Two epochs; validation only.
# ------------------------------------------------------------------
print("\n===== 4. TWO-EPOCH PMP SMOKE =====", flush=True)
set_random_seed(2)
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

logger = Logger(mode=[print])
logger.add_line = lambda: logger.log("-" * 50)

T = Trainer(args=args, config=config, logger=logger)
assert callable(T.init)
assert callable(T.train)
assert callable(T.evaluation)
assert callable(T.eval_model)

model, optimizer, loss_func, scheduler = T.init(helper)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
rows = []

for epoch in range(2):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model, loss = T.train(epoch, model, loss_func, optimizer, train_loader, helper)
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    avg_loss = float(loss.detach().item()) / max(1, len(train_loader))

    torch.cuda.synchronize()
    v0 = time.perf_counter()
    labels, fraud_probs, preds = T.evaluation(
        helper,
        val_loader,
        model,
        threshold_moving=config["threshold_moving"],
        thres=config["thres"],
    )
    torch.cuda.synchronize()
    val_s = time.perf_counter() - v0
    dev = T.eval_model(labels, fraud_probs, preds)

    row = {
        "epoch": epoch + 1,
        "avg_train_loss": avg_loss,
        "train_seconds": float(train_s),
        "validation_seconds": float(val_s),
        "val_auprc": float(dev.ap_gnn),
        "val_auroc": float(dev.auc_gnn),
        "val_macro_f1_at_author_smoke_threshold": float(dev.f1_macro),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
    }
    rows.append(row)

    print(
        f"SMOKE epoch={epoch+1} | train={train_s:.3f}s | "
        f"val={val_s:.3f}s | loss={avg_loss:.6f} | "
        f"val_AUPRC={dev.ap_gnn:.6f} | val_AUROC={dev.auc_gnn:.6f}",
        flush=True,
    )

# No T.evaluation(... test_loader ...) call exists in this script.
status_after = git("status", "--porcelain")
assert status_after == "", f"Official PMP repo became dirty:\n{status_after}"

result = {
    "model": "PMP / LA-SAGE-S",
    "dataset": "YelpChi",
    "purpose": "COMP8851 unified two-epoch compatibility smoke",
    "repository_commit": commit,
    "official_repo_clean_before": True,
    "official_repo_clean_after": True,
    "canonical_dataset_path": str(CANONICAL),
    "canonical_dataset_sha256": data_sha,
    "frozen_split_path": str(SPLIT),
    "frozen_split_sha256": split_sha,
    "split": "TR40",
    "split_seed": 2,
    "training_seed": 2,
    "train_size": len(train_ids),
    "validation_size": len(val_ids),
    "test_size_registered_only": len(test_ids),
    "test_evaluated": False,
    "test_loader_iterated": False,
    "author_model": "LA-SAGE-S",
    "author_architecture_preserved": True,
    "author_data_graph_preserved": True,
    "adapter_change": "replace stock random masks/IDs with exact frozen COMP8851 split IDs; rebuild author DGL loaders; TRAIN-only label_unk",
    "config": {
        "hid_dim": config["hid_dim"],
        "n_layer": config["n_layer"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "dropout": config["dropout"],
        "full_neighbors": config["full_neighbors"],
        "relation_agg": config["relation_agg"],
        "resi": config["resi"],
    },
    "epochs_completed": 2,
    "epochs": rows,
    "mean_train_seconds_per_epoch": float(np.mean([r["train_seconds"] for r in rows])),
    "mean_validation_seconds": float(np.mean([r["validation_seconds"] for r in rows])),
    "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
    "gpu": torch.cuda.get_device_name(0),
    "status": "PASS",
}
OUT.write_text(json.dumps(result, indent=2))

print("\n===== PMP YELPCHI SMOKE GATE =====", flush=True)
print("PMP_YELP_SMOKE=PASS", flush=True)
print(f"MEAN_TRAIN_SECONDS_PER_EPOCH={result['mean_train_seconds_per_epoch']:.6f}", flush=True)
print(f"PEAK_GPU_MEMORY_MB={result['peak_gpu_memory_mb']:.2f}", flush=True)
print("TEST_EVALUATED=NO", flush=True)
print("AUTHOR_REPO_MODIFIED=NO", flush=True)
print("EVIDENCE_JSON=", OUT, flush=True)
