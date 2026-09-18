#!/usr/bin/env python3
import os, sys, json, time, gc, shutil, hashlib, importlib.util
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from omegaconf import OmegaConf
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

ROOT = Path("/workspace/gaap_vast")
REPO = ROOT / "repo/GAAP"
AUTHOR_ENTRY = REPO / "mycode/exp/101_retrain.py"
CFG_PATH = REPO / "config/SAGE_MiniF_DyPLE_MHA/tfinance.yaml"
ADAPTER = ROOT / "shared/tfinance/tfinance_gaap_input_adapter"
CANON = ROOT / "shared/tfinance/tfinance"
SPLIT = ROOT / "shared/tfinance/tfinance_seed2_nested_splits.npz"
OUTDIR = ROOT / "unified/tfinance/tuning"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COMMIT = "6a7dbb0447c4897504525de49e41a0526ee777f8"
EXPECTED_CANON = "b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700"
EXPECTED_SPLIT = "de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1"
EXPECTED_ADAPTER = "7db7e48617038b35dbac041a6b90426f25237886e157d2f54776bbbd756d69ae"

MAX_EPOCHS = 100
PATIENCE = 20
SEED = 2

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def git_output(*args):
    import subprocess
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()

# -------------------------- frozen gates --------------------------
assert git_output("rev-parse", "HEAD") == EXPECTED_COMMIT
assert git_output("status", "--porcelain", "--untracked-files=no") == ""
assert sha256(CANON) == EXPECTED_CANON
assert sha256(SPLIT) == EXPECTED_SPLIT
assert sha256(ADAPTER) == EXPECTED_ADAPTER

s = np.load(SPLIT, allow_pickle=False)
expected_sizes = {
    "TR40": 15742, "TR30": 11806, "TR20": 7870,
    "TR10": 3935, "val": 7872, "test": 15743,
}
for k, n in expected_sizes.items():
    assert len(s[k]) == n, (k, len(s[k]), n)
assert int(s["seed"][0]) == 2
assert int(s["source_nodes"][0]) == 39357
S = {k: set(map(int, s[k])) for k in expected_sizes}
assert S["TR10"] <= S["TR20"] <= S["TR30"] <= S["TR40"]
assert S["val"].isdisjoint(S["test"])
for r in ("TR40", "TR30", "TR20", "TR10"):
    assert S[r].isdisjoint(S["val"])
    assert S[r].isdisjoint(S["test"])

print("TFINANCE_CANONICAL_DATA_GATE=PASS", flush=True)
print("TFINANCE_FROZEN_SPLIT_GATE=PASS", flush=True)
print("GAAP_AUTHOR_REPO_CLEAN=PASS", flush=True)
print("TUNING_PROTOCOL=TR40_SEED2_MAX12_MAX100_PATIENCE20_VAL_AUPRC_ONLY", flush=True)
print("TEST_EVALUATED=NO", flush=True)

# -------------------------- author import -------------------------
spec = importlib.util.spec_from_file_location("gaap_author_101_retrain", AUTHOR_ENTRY)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

native = OmegaConf.to_container(OmegaConf.load(CFG_PATH), resolve=True)
native = dict(native)

# Exact native T-Finance mechanism/config family.
assert native["model_name"] == "SAGE"
assert native["loader_type"] == "MiniF"
assert native["data_name"] == "tfinance"
assert native["norm_type"] == "norm01"
assert str(native["preprocess"]) == "None"
assert int(native["n_bins"]) == 40
assert int(native["d_hidden"]) == 128
assert int(native["gnn_n_layers"]) == 3
assert float(native["gnn_dropout"]) == 0.0
assert native["gnn_agg"] == "max_pool"
assert bool(native["use_dyple"]) is True
assert bool(native["use_mha"]) is True
assert int(native["bs"]) == 64
assert int(native["val_bs"]) == 1280
assert float(native["lr"]) == 0.001
assert float(native["weight_decay"]) == 0.0

print("GAAP_NATIVE_CONFIG_GATE=PASS", flush=True)

# 2 hidden widths x 3 author-family bin counts x 2 benchmark LR candidates = 12.
# Native T-Finance (128,40,0.001) is trial 01.
grid = []
for h in (128, 64):
    for b in (40, 32, 16):
        for lr in (0.001, 0.002):
            grid.append({"d_hidden": h, "n_bins": b, "lr": lr})
assert len(grid) == 12
assert grid[0] == {"d_hidden": 128, "n_bins": 40, "lr": 0.001}

print("TUNING_GRID=" + json.dumps(grid, separators=(",", ":")), flush=True)

# Resolve the same runtime dataset directory used by GAAP's MiniF loader.
loader_cls = A.tag_dm_map[native["loader_type"]]
loader_globals = loader_cls.__init__.__globals__
assert "DIR_FRAUD_DATASET" in loader_globals
runtime_dir = Path(str(loader_globals["DIR_FRAUD_DATASET"]))
runtime_path = runtime_dir / "tfinance"
runtime_dir.mkdir(parents=True, exist_ok=True)
print("GAAP_RUNTIME_DATA_PATH=" + str(runtime_path), flush=True)

test_idx_np = np.asarray(s["test"], dtype=np.int64)

class ControlledLitSAGE(A.LitSAGE):
    """Author LitSAGE with validation-only metrics; never reads test labels."""
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.outs.clear()
            return
        outs = self.outs
        if not outs:
            return
        y = torch.cat([x[0] for x in outs])
        logit = torch.cat([x[1] for x in outs])
        nid = torch.cat([x[2] for x in outs])
        his_emb = torch.cat([x[3] for x in outs])

        order = torch.argsort(nid)
        y = y[order]
        logit = logit[order]
        self.his_emb = his_emb[order]

        val_idx = self.val_idx
        if not torch.is_tensor(val_idx):
            val_idx = torch.as_tensor(val_idx, dtype=torch.long, device=y.device)
        else:
            val_idx = val_idx.to(y.device)

        val_loss = F.cross_entropy(logit[val_idx], y[val_idx])
        prob = logit.softmax(-1)[:, 1].detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        vi = val_idx.detach().cpu().numpy()

        val_auprc = float(average_precision_score(y_np[vi], prob[vi]))
        val_auroc = float(roc_auc_score(y_np[vi], prob[vi]))

        self.log("valoss", val_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_aps", val_auprc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_auc", val_auroc, prog_bar=False, on_step=False, on_epoch=True)
        self.outs.clear()

class LiveEpoch(Callback):
    def __init__(self, trial_no):
        super().__init__()
        self.trial_no = trial_no
        self.best = -float("inf")
        self.best_epoch = 0
        self.bad = 0
        self.last_auc = float("nan")

    @staticmethod
    def scalar(x):
        if x is None:
            return float("nan")
        if torch.is_tensor(x):
            return float(x.detach().cpu())
        return float(x)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        val = self.scalar(m.get("val_aps"))
        auc = self.scalar(m.get("val_auc"))
        loss = self.scalar(m.get("trloss_epoch", m.get("trloss")))
        epoch = int(trainer.current_epoch) + 1

        if np.isfinite(val) and val > self.best:
            self.best = val
            self.best_epoch = epoch
            self.bad = 0
        else:
            self.bad += 1
        self.last_auc = auc

        print(
            f"trial={self.trial_no:02d}/12 | TR40 seed=2 | epoch={epoch:03d} | "
            f"loss={loss:.6f} | valAUPRC={val:.6f} | valAUROC={auc:.6f} | "
            f"best={self.best:.6f}@{self.best_epoch} | patience={self.bad}/{PATIENCE}",
            flush=True,
        )

def prepare_dm(cfg):
    # Exact verified input adapter -> exact runtime path used by author loader.
    shutil.copyfile(ADAPTER, runtime_path)
    assert sha256(runtime_path) == EXPECTED_ADAPTER

    dm = A.tag_dm_map[cfg["loader_type"]](**cfg)

    assert dm.g.number_of_nodes() == 39357
    assert int(dm.g.ndata["train_mask"].sum()) == 15742
    assert int(dm.g.ndata["val_mask"].sum()) == 7872
    assert int(dm.g.ndata["test_mask"].sum()) == 15743

    ti = torch.as_tensor(test_idx_np, dtype=torch.long)
    labels = dm.g.ndata["label"].clone()
    labels[ti] = 0
    dm.g.ndata["label"] = labels
    assert torch.all(dm.g.ndata["label"][ti] == 0)
    assert not torch.any(dm.g.ndata["train_mask"][ti])
    assert not torch.any(dm.g.ndata["val_mask"][ti])

    casted = []
    for k in list(dm.g.ndata.keys()):
        v = dm.g.ndata[k]
        if torch.is_tensor(v) and torch.is_floating_point(v) and v.dtype != torch.float32:
            dm.g.ndata[k] = v.float().contiguous()
            casted.append("g.ndata[" + k + "]")
    for attr in ("feature", "features", "feat", "x", "bin_feat", "bin_feature", "feat_bin", "feature_bin"):
        if hasattr(dm, attr):
            v = getattr(dm, attr)
            if torch.is_tensor(v) and torch.is_floating_point(v) and v.dtype != torch.float32:
                setattr(dm, attr, v.float().contiguous())
                casted.append("dm." + attr)

    for k in dm.g.ndata.keys():
        v = dm.g.ndata[k]
        if torch.is_tensor(v) and torch.is_floating_point(v):
            assert v.dtype == torch.float32
    if hasattr(dm, "feat") and torch.is_tensor(dm.feat) and torch.is_floating_point(dm.feat):
        assert dm.feat.dtype == torch.float32

    return dm, casted

results = []
winner = None

for i, hp in enumerate(grid, start=1):
    trial_dir = OUTDIR / f"trial_{i:02d}"
    ckpt_dir = trial_dir / "checkpoints"
    trial_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_path = trial_dir / "result.json"

    # Resume only a previously completed PASS trial with matching protocol/config.
    if result_path.exists():
        try:
            old = json.loads(result_path.read_text())
            if (
                old.get("status") == "PASS"
                and old.get("trial") == i
                and old.get("hyperparams") == hp
                and old.get("seed") == SEED
                and old.get("max_epochs") == MAX_EPOCHS
                and old.get("patience") == PATIENCE
                and old.get("selection") == "val_AUPRC"
                and old.get("test_evaluated") is False
            ):
                print(f"trial={i:02d}/12 | RESUME=PASS | best={old['best_val_auprc']:.6f}@{old['best_epoch']}", flush=True)
                results.append(old)
                if winner is None or old["best_val_auprc"] > winner["best_val_auprc"]:
                    winner = old
                continue
        except Exception:
            pass

    cfg = deepcopy(native)
    cfg.update(hp)
    cfg["seed"] = SEED
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["patience"] = PATIENCE
    cfg["device"] = "cuda"
    cfg["device_id"] = 0
    cfg["nowandb"] = True

    A.fix_seed(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        f"\n===== TRIAL {i:02d}/12 | d_hidden={hp['d_hidden']} | "
        f"n_bins={hp['n_bins']} | lr={hp['lr']} =====",
        flush=True,
    )

    t0 = time.perf_counter()
    dm, casted = prepare_dm(cfg)
    print("RUNTIME_FLOAT32_CAST_FIELDS=" + repr(casted), flush=True)
    print("TEST_LABELS_HIDDEN_DURING_FIT=PASS", flush=True)

    cfg["d_in"] = dm.d_in
    cfg["n_classes"] = dm.n_classes
    cfg["n_nodes"] = dm.g.number_of_nodes()

    model = ControlledLitSAGE(**cfg)
    live = LiveEpoch(i)
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:03d}-{val_aps:.6f}",
        monitor="val_aps",
        mode="max",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
    )
    early = EarlyStopping(
        monitor="val_aps",
        mode="max",
        patience=PATIENCE,
        min_delta=0.0,
        verbose=False,
        check_finite=True,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[live, checkpoint, early],
        gradient_clip_val=10,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    trainer.fit(model, dm)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    ti = torch.as_tensor(test_idx_np, dtype=torch.long)
    assert torch.all(dm.g.ndata["label"][ti] == 0)

    best_score = float(checkpoint.best_model_score.detach().cpu())
    best_epoch = int(live.best_epoch)
    epochs_ran = int(trainer.current_epoch) + 1

    rec = {
        "status": "PASS",
        "trial": i,
        "hyperparams": hp,
        "seed": SEED,
        "ratio": "TR40",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection": "val_AUPRC",
        "best_val_auprc": best_score,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "wall_seconds": wall,
        "peak_gpu_mb": peak,
        "checkpoint": checkpoint.best_model_path,
        "test_evaluated": False,
        "test_label_isolation": True,
        "adapter_sha256": EXPECTED_ADAPTER,
        "canonical_sha256": EXPECTED_CANON,
        "split_sha256": EXPECTED_SPLIT,
        "repo_commit": EXPECTED_COMMIT,
        "native_architecture_changed": False,
        "compatibility": "INPUT_ADAPTER",
    }
    result_path.write_text(json.dumps(rec, indent=2))
    results.append(rec)

    print(
        f"TRIAL_COMPLETE={i:02d}/12 | best_valAUPRC={best_score:.6f}@{best_epoch} | "
        f"epochs={epochs_ran} | wall={wall:.3f}s | peakGPU={peak:.2f}MB | TEST_EVALUATED=NO",
        flush=True,
    )

    if winner is None or best_score > winner["best_val_auprc"]:
        winner = rec

    del trainer, model, dm
    gc.collect()
    torch.cuda.empty_cache()

assert len(results) == 12
assert winner is not None

summary = {
    "dataset": "T-Finance",
    "model": "GAAP",
    "ratio": "TR40",
    "training_seed": SEED,
    "trial_budget": 12,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "selection": "validation AUPRC only",
    "test_evaluated": False,
    "grid": grid,
    "winner": winner,
    "all_trials": sorted(results, key=lambda x: x["trial"]),
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
(OUTDIR / "GAAP_TFINANCE_TUNING_SUMMARY.json").write_text(json.dumps(summary, indent=2))

frozen = {
    "dataset": "T-Finance",
    "model": "GAAP",
    "frozen_from": "TR40 seed2 validation-AUPRC tuning",
    "winner_trial": winner["trial"],
    "d_hidden": winner["hyperparams"]["d_hidden"],
    "n_bins": winner["hyperparams"]["n_bins"],
    "lr": winner["hyperparams"]["lr"],
    "weight_decay": float(native["weight_decay"]),
    "gnn_n_layers": int(native["gnn_n_layers"]),
    "gnn_dropout": float(native["gnn_dropout"]),
    "gnn_agg": native["gnn_agg"],
    "gnn_use_bn": bool(native["gnn_use_bn"]),
    "gnn_use_res": bool(native["gnn_use_res"]),
    "use_dyple": bool(native["use_dyple"]),
    "use_mha": bool(native["use_mha"]),
    "mha_n_layers": int(native["mha_n_layers"]),
    "mha_n_heads": int(native["mha_n_heads"]),
    "mha_alpha": float(native["mha_alpha"]),
    "d_feat_emb": int(native["d_feat_emb"]),
    "bs": int(native["bs"]),
    "val_bs": int(native["val_bs"]),
    "norm_type": native["norm_type"],
    "preprocess": str(native["preprocess"]),
    "best_val_auprc": winner["best_val_auprc"],
    "best_epoch": winner["best_epoch"],
    "training_seed": SEED,
    "canonical_sha256": EXPECTED_CANON,
    "split_sha256": EXPECTED_SPLIT,
    "adapter_sha256": EXPECTED_ADAPTER,
    "repo_commit": EXPECTED_COMMIT,
    "test_evaluated": False,
    "compatibility": "INPUT_ADAPTER",
}
winner_path = OUTDIR / "GAAP_TFINANCE_FROZEN_WINNER.json"
winner_path.write_text(json.dumps(frozen, indent=2))

assert git_output("status", "--porcelain", "--untracked-files=no") == ""

print("\n===== TUNING COMPLETE =====", flush=True)
print("WINNER_TRIAL=" + str(winner["trial"]), flush=True)
print("WINNER_CONFIG=" + json.dumps(winner["hyperparams"], sort_keys=True), flush=True)
print(f"WINNER_VAL_AUPRC={winner['best_val_auprc']:.6f}@{winner['best_epoch']}", flush=True)
print("TEST_EVALUATED=NO", flush=True)
print("TEST_LABEL_ISOLATION=PASS", flush=True)
print("GAAP_AUTHOR_ARCHITECTURE_CHANGED=NO", flush=True)
print("GAAP_AUTHOR_REPO_CLEAN=PASS", flush=True)
print("GAAP_TFINANCE_TUNING=PASS", flush=True)
print("CONFIGURATION_FROZEN=YES", flush=True)
print("FROZEN_WINNER=" + str(winner_path), flush=True)
print("NEXT=GAAP_TFINANCE_FINAL_12_RUNS", flush=True)
