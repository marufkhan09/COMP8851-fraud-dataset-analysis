#!/usr/bin/env python3

import os
import sys
import time
import json
import yaml
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import dgl

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

ROOT = Path("/workspace/gaap_vast")
REPO = ROOT / "repo" / "GAAP"
SPLIT = ROOT / "shared" / "yelp_seed2_nested_splits.npz"

EXPECTED_SPLIT_SHA = (
    "0ea0af36dfc5a3a1f381ea2e3168377e"
    "45826498b6063190185c673e9ec8c22b"
)

AUTHOR = REPO / "mycode" / "exp" / "101_retrain.py"
CFG_PATH = (
    REPO /
    "config" /
    "SAGE_MiniF_DyPLE_MHA" /
    "yelp.yaml"
)

OUT = ROOT / "evidence" / "gaap_yelp_unified_smoke"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mycode"))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)
    return h.hexdigest()

assert sha256(SPLIT) == EXPECTED_SPLIT_SHA

spec = importlib.util.spec_from_file_location(
    "gaap101_author",
    AUTHOR
)

gaap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gaap)

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

assert cfg["model_name"] == "SAGE"
assert cfg["loader_type"] == "MiniF"
assert cfg["data_name"] == "yelp"

assert cfg.get("preprocess") in [
    None, "None", "none", "NONE"
]

# ----------------------------------------------------------
# UNIFIED SMOKE OVERRIDES ONLY
# ----------------------------------------------------------

cfg = copy.deepcopy(cfg)

cfg["device"] = "cuda"
cfg["device_id"] = 0
cfg["seed"] = 2

# Protocol smoke only.
cfg["max_epochs"] = 2
cfg["patience"] = 20

print("============================================================")
print("GAAP × YELPCHI — UNIFIED 2-EPOCH SMOKE")
print("============================================================")
print("MODEL=GAAP")
print("DATASET=YelpChi")
print("RATIO=TR40")
print("TRAIN_SEED=2")
print("SPLIT_SEED=2")
print("MAX_EPOCHS=2")
print("PATIENCE=20")
print("CHECKPOINT_SELECTION=VALIDATION_AUPRC")
print("TEST_USED_FOR_SELECTION=NO")
print(
    "SPLIT_SHA256="
    + sha256(SPLIT)
)

gaap.fix_seed(2)
torch.set_num_threads(20)

# ----------------------------------------------------------
# CREATE AUTHOR DATAMODULE
# ----------------------------------------------------------

dm = gaap.tag_dm_map[cfg["loader_type"]](
    **cfg
)

g = dm.g

n = g.num_nodes()
d = g.ndata["feature"].shape[1]

labels_original = (
    g.ndata["label"]
    .clone()
    .detach()
)

fraud = int(
    (labels_original == 1)
    .sum()
    .item()
)

normal = int(
    (labels_original == 0)
    .sum()
    .item()
)

print("GAAP_YELP_NODES =", n)
print("GAAP_YELP_FEATURES =", d)
print("GAAP_YELP_EDGES =", g.num_edges())
print("GAAP_YELP_FRAUD =", fraud)
print("GAAP_YELP_NORMAL =", normal)

assert n == 45954
assert d == 32
assert fraud == 6677
assert normal == 39277

print("YELP_GRAPH_LABEL_GATE=PASS")

# ----------------------------------------------------------
# INJECT EXACT FROZEN UNIFIED SPLIT
# ----------------------------------------------------------

sp = np.load(SPLIT)

train_idx = np.asarray(
    sp["TR40"],
    dtype=np.int64
)

val_idx = np.asarray(
    sp["val"],
    dtype=np.int64
)

test_idx = np.asarray(
    sp["test"],
    dtype=np.int64
)

def make_mask(idx):
    m = torch.zeros(
        n,
        dtype=torch.bool
    )
    m[
        torch.as_tensor(
            idx,
            dtype=torch.long
        )
    ] = True
    return m

g.ndata["train_mask"] = make_mask(train_idx)
g.ndata["val_mask"] = make_mask(val_idx)
g.ndata["test_mask"] = make_mask(test_idx)

dm.trn_idx = torch.as_tensor(
    train_idx,
    dtype=torch.long
)

dm.val_idx = torch.as_tensor(
    val_idx,
    dtype=torch.long
)

dm.tst_idx = torch.as_tensor(
    test_idx,
    dtype=torch.long
)

assert int(g.ndata["train_mask"].sum()) == 18381
assert int(g.ndata["val_mask"].sum()) == 9191
assert int(g.ndata["test_mask"].sum()) == 18382

print("UNIFIED_TR40_MASK_GATE=PASS")

# ----------------------------------------------------------
# PHYSICALLY HIDE TEST LABELS DURING FIT
#
# n_classes was already computed by the author datamodule.
# GAAP model receives features/topology, not labels.
# Test labels become -1 and are never indexed by validation.
# ----------------------------------------------------------

g.ndata["label"][dm.tst_idx] = -1

assert bool(
    (g.ndata["label"][dm.tst_idx] == -1)
    .all()
)

assert torch.equal(
    g.ndata["label"][dm.trn_idx],
    labels_original[dm.trn_idx]
)

assert torch.equal(
    g.ndata["label"][dm.val_idx],
    labels_original[dm.val_idx]
)

print(
    "TEST_LABELS_PHYSICALLY_MASKED_DURING_FIT=YES"
)
print(
    "TRAIN_VAL_LABELS_UNCHANGED=YES"
)

# ----------------------------------------------------------
# TEST-ISOLATED GAAP LITSAGE
#
# Keeps GAAP's actual SAGE/DyPLE/MHA architecture and
# training_step. Only validation aggregation is replaced so
# no tst_auc/tst_aps is computed during training.
# ----------------------------------------------------------

class UnifiedLitSAGE(gaap.LitSAGE):

    def __init__(
        self,
        d_in,
        n_classes,
        lr,
        **kwargs
    ):
        super().__init__(
            d_in,
            n_classes,
            lr,
            **kwargs
        )

        self.ub_best = -float("inf")
        self.ub_best_epoch = 0
        self.ub_wait = 0

    def on_validation_epoch_end(self):

        if self.trainer.sanity_checking:
            self.outs.clear()
            return

        outs = self.outs

        assert len(outs) > 0

        y = torch.cat(
            [x[0] for x in outs]
        )

        logit = torch.cat(
            [x[1] for x in outs]
        )

        nid = torch.cat(
            [x[2] for x in outs]
        )

        his_emb = torch.cat(
            [x[3] for x in outs]
        )

        order = torch.argsort(nid)

        y = y[order]
        logit = logit[order]
        his_emb = his_emb[order]

        # Preserve official GAAP historical-embedding update.
        self.his_emb = his_emb

        vi = self.val_idx.to(
            logit.device
        )

        val_loss = F.cross_entropy(
            logit[vi],
            y[vi]
        )

        probs = (
            logit
            .softmax(-1)[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

        y_np = (
            y
            .detach()
            .cpu()
            .numpy()
        )

        val_nodes = (
            self.val_idx
            .detach()
            .cpu()
            .numpy()
        )

        val_y = y_np[val_nodes]
        val_p = probs[val_nodes]

        # Strong test-isolation check.
        assert np.all(
            y_np[
                self.tst_idx
                .detach()
                .cpu()
                .numpy()
            ] == -1
        )

        aps = float(
            average_precision_score(
                val_y,
                val_p
            )
        )

        auc = float(
            roc_auc_score(
                val_y,
                val_p
            )
        )

        self.log(
            "valoss",
            val_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True
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

        ep = int(
            self.current_epoch + 1
        )

        if aps > self.ub_best:
            self.ub_best = aps
            self.ub_best_epoch = ep
            self.ub_wait = 0
        else:
            self.ub_wait += 1

        print(
            f"TR40 seed=2 | "
            f"epoch={ep:03d} | "
            f"valAUPRC={aps:.6f} | "
            f"valAUROC={auc:.6f} | "
            f"best={self.ub_best:.6f}"
            f"@{self.ub_best_epoch} | "
            f"patience={self.ub_wait}/20",
            flush=True
        )

        self.outs.clear()


# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------

cfg["d_in"] = dm.d_in
cfg["n_classes"] = dm.n_classes
cfg["n_nodes"] = dm.g.number_of_nodes()

model = UnifiedLitSAGE(
    **cfg
)

ckpt_dir = OUT / "checkpoints"
ckpt_dir.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint = ModelCheckpoint(
    dirpath=str(ckpt_dir),
    filename="gaap-yelp-tr40-seed2-smoke-{epoch:03d}",
    monitor="val_aps",
    mode="max",
    save_top_k=1,
    save_last=False
)

early = EarlyStopping(
    monitor="val_aps",
    mode="max",
    patience=20,
    verbose=False
)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=2,
    logger=False,
    callbacks=[
        checkpoint,
        early
    ],
    gradient_clip_val=10,
    enable_progress_bar=False,
    enable_model_summary=False,
    num_sanity_val_steps=0,
    default_root_dir=str(OUT)
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

t0 = time.perf_counter()

trainer.fit(
    model,
    datamodule=dm
)

elapsed = time.perf_counter() - t0

peak_mb = (
    torch.cuda.max_memory_allocated()
    / 1024**2
)

print()
print("===== SMOKE RESULT =====")
print(
    "BEST_VAL_AUPRC="
    f"{float(checkpoint.best_model_score):.6f}"
)
print(
    "BEST_CHECKPOINT="
    + checkpoint.best_model_path
)
print(
    "TRAINING_WALL_SECONDS="
    f"{elapsed:.6f}"
)
print(
    "PEAK_GPU_MEMORY_MB="
    f"{peak_mb:.2f}"
)

# No test labels restored/evaluated in smoke.
print("TEST_EVALUATED=NO")
print(
    "VAL_TEST_PARTITION_LABELS_HIDDEN=PASS"
)
print(
    "GAAP_YELP_UNIFIED_2EPOCH_SMOKE=PASS"
)

record = {
    "model": "GAAP",
    "dataset": "YelpChi",
    "mode": "unified_smoke",
    "ratio": "TR40",
    "split_seed": 2,
    "train_seed": 2,
    "split_sha256": sha256(SPLIT),
    "epochs": 2,
    "patience": 20,
    "best_val_auprc": float(
        checkpoint.best_model_score
    ),
    "best_checkpoint": (
        checkpoint.best_model_path
    ),
    "test_evaluated": False,
    "peak_gpu_memory_mb": peak_mb,
    "training_wall_seconds": elapsed,
    "source_commit": (
        "6a7dbb0447c4897504525de49e41a0526ee777f8"
    ),
}

with open(
    OUT / "smoke_result.json",
    "w"
) as f:
    json.dump(
        record,
        f,
        indent=2
    )
