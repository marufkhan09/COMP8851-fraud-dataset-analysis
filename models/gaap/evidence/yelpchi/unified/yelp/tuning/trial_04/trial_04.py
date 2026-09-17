#!/usr/bin/env python3

import os
import sys
import time
import json
import hashlib
import random
import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

import torch
import torch.nn.functional as F
import dgl
import yaml

from sklearn.metrics import average_precision_score, roc_auc_score
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


ROOT = Path("/workspace/gaap_vast")
REPO = ROOT / "repo/GAAP"
CANON = ROOT / "canonical_yelp/YelpChi.mat"
SPLIT = ROOT / "shared/yelp_seed2_nested_splits.npz"
OUT = Path("/workspace/gaap_vast/unified/yelp/tuning/trial_04")
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_DATA = "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42"
EXPECTED_SPLIT = "0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b"
EXPECTED_COMMIT = "6a7dbb0447c4897504525de49e41a0526ee777f8"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


assert sha256(CANON) == EXPECTED_DATA
assert sha256(SPLIT) == EXPECTED_SPLIT

os.chdir(REPO)
sys.path.insert(0, str(REPO / "mycode"))

spec = importlib.util.spec_from_file_location(
    "gaap_author_101",
    REPO / "mycode/exp/101_retrain.py"
)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)

import utils.dataloader as gaap_dl


# ============================================================
# CONFIG — AUTHOR YELP CONFIG, ONLY PROTOCOL SMOKE OVERRIDES
# ============================================================

with open(REPO / "config/SAGE_MiniF_DyPLE_MHA/yelp.yaml") as f:
    cfg = yaml.safe_load(f)

assert cfg["model_name"] == "SAGE"
assert cfg["loader_type"] == "MiniF"
assert cfg["data_name"] == "yelp"

assert cfg["d_hidden"] == 64
assert cfg["gnn_n_layers"] == 3
assert abs(float(cfg["gnn_dropout"]) - 0.1) < 1e-12
assert cfg["gnn_agg"] == "max_pool"
assert cfg["use_dyple"] is True
assert cfg["use_mha"] is True
assert cfg["bs"] == 128
assert abs(float(cfg["lr"]) - 0.001) < 1e-15

cfg["seed"] = 2
cfg["max_epochs"] = 100
cfg["patience"] = 20
cfg["device"] = "cuda"
cfg["device_id"] = 0

cfg["nowandb"] = True

# ------------------------------------------------------------
# PREREGISTERED TRIAL 04/12
# ------------------------------------------------------------
cfg["d_hidden"] = 40
cfg["n_bins"] = 32
cfg["lr"] = 0.002

# Everything comparison-critical below stays frozen.
assert cfg["gnn_n_layers"] == 3
assert abs(float(cfg["gnn_dropout"]) - 0.1) < 1e-12
assert cfg["gnn_agg"] == "max_pool"
assert cfg["use_dyple"] is True
assert cfg["use_mha"] is True
assert cfg["bs"] == 128
assert cfg["val_bs"] == 1280
assert float(cfg["weight_decay"]) == 0.0

print(
    "TRIAL_CONFIG | "
    "trial=04/12 | "
    "hidden=40 | "
    "bins=32 | "
    "lr=0.002",
    flush=True
)


print("===== GAAP × YELPCHI — UNIFIED 2-EPOCH SMOKE =====", flush=True)
print("RATIO=TR40 | TRAIN_SEED=2 | SPLIT_SEED=2", flush=True)
print("AUTHOR_ARCHITECTURE_CHANGED=NO", flush=True)
print("TEST_USED_FOR_SELECTION=NO", flush=True)
print("MAX_EPOCHS=100 | PATIENCE=20", flush=True)


# ============================================================
# EXACT SHARED YELP HOMOGENEOUS VIEW
# ============================================================

m = loadmat(CANON)

features = m["features"]
if sp.issparse(features):
    features = features.toarray()

features = np.asarray(features, dtype=np.float32)
labels = np.asarray(m["label"]).reshape(-1).astype(np.int64)

assert features.shape == (45954, 32)
assert labels.shape == (45954,)
assert int((labels == 1).sum()) == 6677
assert int((labels == 0).sum()) == 39277

relations = {}
edge_total = 0

for rel, key in [
    ("rsr", "net_rsr"),
    ("rtr", "net_rtr"),
    ("rur", "net_rur"),
]:
    a = sp.coo_matrix(m[key])

    src = torch.from_numpy(a.row.astype(np.int64))
    dst = torch.from_numpy(a.col.astype(np.int64))

    edge_total += int(a.nnz)

    relations[("review", rel, "review")] = (src, dst)

assert edge_total == 8051348

hg = dgl.heterograph(
    relations,
    num_nodes_dict={"review": 45954}
)

hg.nodes["review"].data["feature"] = torch.from_numpy(features)
hg.nodes["review"].data["label"] = torch.from_numpy(labels)

g = dgl.to_homogeneous(
    hg,
    ndata=["feature", "label"]
)

g = dgl.add_self_loop(g)

assert g.num_nodes() == 45954
assert g.num_edges() == 8097302

print("CANONICAL_NODES=45954", flush=True)
print("CANONICAL_FEATURES=32", flush=True)
print("RELATION_EDGE_ENTRIES=8051348", flush=True)
print("HOMOGENEOUS_WITH_SELF_LOOPS=8097302", flush=True)
print("SHARED_YELP_VIEW_GATE=PASS", flush=True)


# ============================================================
# EXACT FROZEN SPLIT
# ============================================================

s = np.load(SPLIT, allow_pickle=False)

expected = {
    "TR40": 18381,
    "TR30": 13785,
    "TR20": 9190,
    "TR10": 4595,
    "val": 9191,
    "test": 18382,
}

for k, n in expected.items():
    assert len(s[k]) == n

assert int(s["seed"][0]) == 2
assert int(s["source_nodes"][0]) == 45954

tr = torch.tensor(s["TR40"], dtype=torch.long)
va = torch.tensor(s["val"], dtype=torch.long)
te = torch.tensor(s["test"], dtype=torch.long)

def mask(ids):
    x = torch.zeros(45954, dtype=torch.bool)
    x[ids] = True
    return x

g.ndata["train_mask"] = mask(tr)
g.ndata["val_mask"] = mask(va)
g.ndata["test_mask"] = mask(te)

assert int(g.ndata["train_mask"].sum()) == 18381
assert int(g.ndata["val_mask"].sum()) == 9191
assert int(g.ndata["test_mask"].sum()) == 18382

print("EXACT_FROZEN_SPLIT_GATE=PASS", flush=True)


# ============================================================
# USE AUTHOR MINI-F DATALOADER ON SHARED GRAPH
# ============================================================

# External input adapter only; author repository remains untouched.
gaap_dl.read_dataset = lambda _: g

dm_kwargs = dict(cfg)
dm_kwargs.pop("data_name", None)

dm = gaap_dl.MiniFDataModule(
    "yelp",
    **dm_kwargs
)

assert dm.g.num_nodes() == 45954
assert dm.g.num_edges() == 8097302
assert dm.d_in == 32
assert dm.n_classes == 2

assert torch.equal(dm.trn_idx.cpu(), tr)
assert torch.equal(dm.val_idx.cpu(), va)
assert torch.equal(dm.tst_idx.cpu(), te)

assert torch.isfinite(dm.g.ndata["feature"]).all()

print("GAAP_NATIVE_NORM_TYPE=" + str(cfg["norm_type"]), flush=True)
print("GAAP_NATIVE_PREPROCESS=" + str(cfg["preprocess"]), flush=True)
print("GAAP_AUTHOR_DATALOADER_GATE=PASS", flush=True)


# ============================================================
# PHYSICAL TEST-LABEL ISOLATION
# ============================================================

assert int((dm.g.ndata["label"] == 1).sum()) == 6677

dm.g.ndata["label"][te] = -1
dm.y[te] = -1

assert torch.all(dm.g.ndata["label"][te] == -1)
assert torch.all(dm.y[te] == -1)
assert torch.all(dm.g.ndata["label"][tr] >= 0)
assert torch.all(dm.g.ndata["label"][va] >= 0)

print("TEST_LABELS_HIDDEN_DURING_FIT=PASS", flush=True)


# ============================================================
# VALIDATION-ONLY GAAP EVALUATOR
# ============================================================

class UnifiedLitSAGE(author.LitSAGE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ub_best = -1.0
        self.ub_best_epoch = 0
        self.ub_wait = 0

    def on_validation_epoch_end(self):

        if self.trainer.sanity_checking:
            self.outs.clear()
            return

        assert self.outs

        y = torch.cat([x[0] for x in self.outs])
        logit = torch.cat([x[1] for x in self.outs])
        nid = torch.cat([x[2] for x in self.outs])
        his = torch.cat([x[3] for x in self.outs])

        order = torch.argsort(nid)

        nid = nid[order]
        y = y[order]
        logit = logit[order]
        his = his[order]

        expected_nid = torch.arange(
            dm.g.num_nodes(),
            device=nid.device,
            dtype=nid.dtype
        )

        assert torch.equal(nid, expected_nid)

        # Preserve GAAP historical-embedding update.
        self.his_emb = his

        vi = self.val_idx.to(logit.device)
        ti = self.tst_idx.to(y.device)

        # True test labels are unavailable to this evaluator.
        assert torch.all(y[ti] == -1)

        val_y_t = y[vi]
        val_logit = logit[vi]

        val_loss = F.cross_entropy(
            val_logit,
            val_y_t
        )

        val_prob = (
            val_logit.softmax(-1)[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

        val_y = (
            val_y_t
            .detach()
            .cpu()
            .numpy()
        )

        aps = float(
            average_precision_score(
                val_y,
                val_prob
            )
        )

        auc = float(
            roc_auc_score(
                val_y,
                val_prob
            )
        )

        self.log(
            "val_aps",
            aps,
            prog_bar=False,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "val_auc",
            auc,
            prog_bar=False,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "valoss",
            val_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True
        )

        ep = int(self.current_epoch) + 1

        if aps > self.ub_best:
            self.ub_best = aps
            self.ub_best_epoch = ep
            self.ub_wait = 0
        else:
            self.ub_wait += 1

        train_loss = self.trainer.callback_metrics.get(
            "trloss_epoch",
            self.trainer.callback_metrics.get("trloss")
        )

        if train_loss is None:
            loss_txt = "NA"
        else:
            try:
                loss_txt = f"{float(train_loss.detach().cpu()):.6f}"
            except Exception:
                loss_txt = f"{float(train_loss):.6f}"

        print(
            f"trial=04/12 | TR40 seed=2 | "
            f"epoch={ep:03d} | "
            f"loss={loss_txt} | "
            f"valAUPRC={aps:.6f} | "
            f"valAUROC={auc:.6f} | "
            f"best={self.ub_best:.6f}@{self.ub_best_epoch} | "
            f"patience={self.ub_wait}/20",
            flush=True
        )

        self.outs.clear()


# ============================================================
# MODEL + TRAINER
# ============================================================

cfg["d_in"] = dm.d_in
cfg["n_classes"] = dm.n_classes
cfg["n_nodes"] = dm.g.number_of_nodes()

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
author.fix_seed(2)

torch.set_num_threads(8)

model = UnifiedLitSAGE(**cfg)

ckpt_dir = OUT / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

checkpoint = ModelCheckpoint(
    dirpath=str(ckpt_dir),
    filename="trial-04-best-{epoch:03d}",
    monitor="val_aps",
    mode="max",
    save_top_k=1,
    save_last=False,
    auto_insert_metric_name=False
)

early = EarlyStopping(
    monitor="val_aps",
    mode="max",
    patience=20
)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    logger=False,
    callbacks=[checkpoint, early],
    gradient_clip_val=10,
    enable_progress_bar=False,
    enable_model_summary=False,
    num_sanity_val_steps=0,
    default_root_dir=str(OUT)
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()

trainer.fit(
    model,
    datamodule=dm
)

elapsed = time.perf_counter() - start

peak_mb = (
    torch.cuda.max_memory_allocated()
    / 1024**2
)

best = float(
    checkpoint.best_model_score
    .detach()
    .cpu()
)

record = {
    "model": "GAAP",
    "dataset": "YelpChi",
    "mode": "unified_tuning",
    "ratio": "TR40",
    "train_seed": 2,
    "split_seed": 2,
    "dataset_sha256": EXPECTED_DATA,
    "split_sha256": EXPECTED_SPLIT,
    "repository_commit": EXPECTED_COMMIT,
    "nodes": 45954,
    "features": 32,
    "relation_edge_entries": 8051348,
    "homogeneous_edges_with_self_loops": 8097302,
    "best_val_auprc": best,
    "best_checkpoint": checkpoint.best_model_path,
    "training_wall_seconds": elapsed,
    "peak_gpu_memory_mb": peak_mb,
    "test_evaluated": False,
    "test_labels_hidden_during_fit": True,
    "author_architecture_changed": False,
}

with open(OUT / "trial_result.json", "w") as f:
    json.dump(record, f, indent=2)

print("", flush=True)
print("===== TRIAL 04/12 FINAL GATE =====", flush=True)
print(f"BEST_VAL_AUPRC={best:.6f}", flush=True)
print(f"TRAINING_WALL_SECONDS={elapsed:.3f}", flush=True)
print(f"PEAK_GPU_MEMORY_MB={peak_mb:.2f}", flush=True)
print("TEST_EVALUATED=NO", flush=True)
print("TEST_LABEL_ISOLATION=PASS", flush=True)
print("GAAP_AUTHOR_ARCHITECTURE_CHANGED=NO", flush=True)
print("GAAP_YELP_TUNING_TRIAL_04=PASS", flush=True)
