#!/usr/bin/env python3
"""FDCompCN RGTAN validation-only TR40 tuning."""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from dgl.dataloading import (
    MultiLayerFullNeighborSampler,
    NodeDataLoader
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

ROOT = Path("/workspace/comp8851")
SRC = ROOT / "repos/antifraud"
sys.path.insert(0, str(SRC))

from methods.rgtan.rgtan_model import RGTAN

DATA = Path("/workspace/uploads/rgtan_fdcompcn_i2_incoming")
ASSET = ROOT / "assets/rgtan/fdcompcn"
OUT = ROOT / "results/rgtan/fdcompcn_tr40_tuning_v1"

GRAPH = DATA / "comp.dgl"
SPLIT = DATA / "fdcompcn_seed2_nested_splits.npz"
ADAPTER = ASSET / "fdcompcn_tr40_crossfit.npz"
MANIFEST = ASSET / "fdcompcn_tr40_crossfit_manifest.json"

PIN = "65e583f03ac5c5983e49cc35179fe7a65908b9c1"
SOURCE_SHA = "e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa"
SPLIT_SHA = "5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168"
ADAPTER_SHA = "79c88a3421795dc4c420781567f7bdb9288b1ff1aee9ec551ba4b3f669878744"

N = 5317
SEED = 2
DEVICE = "cuda:0"

# Accelerated dataset-specific batch configuration.
BATCH = 256
MAX_EPOCHS = 100
PATIENCE = 20

NAMES = [
    "degree",
    "riskstat",
    "1hop_degree",
    "2hop_degree",
    "1hop_riskstat",
    "2hop_riskstat"
]

GRID = [
    dict(
        hidden_dim=h,
        drop=[d, d / 2],
        base_lr=lr
    )
    for h in (32, 64)
    for d in (0.2, 0.3)
    for lr in (0.001, 0.003, 0.005)
]

assert len(GRID) == 12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(4 * 1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")

    with temp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)


def atomic_pt(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    torch.save(obj, temp)
    os.replace(temp, path)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)


def load_data():
    assert torch.cuda.is_available()
    assert torch.__version__ == "1.12.1+cu113"
    assert dgl.__version__ == "0.8.1"

    commit = subprocess.check_output(
        ["git", "-C", str(SRC), "rev-parse", "HEAD"],
        text=True
    ).strip()
    assert commit == PIN

    assert sha(GRAPH) == SOURCE_SHA
    assert sha(SPLIT) == SPLIT_SHA
    assert sha(ADAPTER) == ADAPTER_SHA

    meta = json.loads(MANIFEST.read_text())

    assert meta["status"] == "RGTAN_CROSSFIT_ADAPTER_PASS"
    assert meta["artifact_sha256"] == ADAPTER_SHA
    assert meta["source_sha256"] == SOURCE_SHA
    assert meta["split_sha256"] == SPLIT_SHA
    assert meta["ratio"] == "TR40"
    assert meta["risk_attention_heads"] == 3

    with np.load(SPLIT, allow_pickle=False) as z:
        train_expected = z["TR40"].copy()
        val_expected = z["val"].copy()
        test_ids = z["test"].copy()
        assert int(z["seed"].item()) == 2

    with np.load(ADAPTER, allow_pickle=False) as z:
        train = z["train_ids"].astype(np.int64).copy()
        y = z["train_labels"].astype(np.int64).copy()
        val = z["val_ids"].astype(np.int64).copy()
        fold = z["fold_of_node"].astype(np.int64).copy()
        x = z["features"].astype(np.float32).copy()
        names = z["risk_feature_names"].tolist()

        views = [
            z[f"risk_view_{j}"].astype(np.float32).copy()
            for j in range(6)
        ]
        contexts = [
            z[f"label_context_{j}"].astype(np.int64).copy()
            for j in range(6)
        ]

    assert np.array_equal(train, train_expected)
    assert np.array_equal(val, val_expected)
    assert x.shape == (N, 57)
    assert names == NAMES
    assert len(train) == 2126
    assert len(val) == 1064
    assert len(test_ids) == 2127
    assert int((y == 1).sum()) == 223
    assert set(np.unique(y)) == {0, 1}

    assert len(np.unique(train)) == len(train)
    assert len(np.unique(val)) == len(val)
    assert not np.intersect1d(train, val).size
    assert not np.intersect1d(train, test_ids).size
    assert not np.intersect1d(val, test_ids).size

    assert set(fold[train]) == set(range(5))
    assert np.all(fold[val] == -1)
    assert np.all(fold[test_ids] == -1)
    assert np.isfinite(x).all()

    for j in range(6):
        assert views[j].shape == (N, 6)
        assert np.isfinite(views[j]).all()
        assert contexts[j].shape == (N,)
        assert np.all(contexts[j][val] == 2)
        assert np.all(contexts[j][test_ids] == 2)

        expected = np.full(N, 2, dtype=np.int64)

        if j < 5:
            keep = fold[train] != j
        else:
            keep = np.ones(len(train), dtype=bool)

        expected[train[keep]] = y[keep]

        assert np.array_equal(contexts[j], expected)

    gs, _ = dgl.load_graphs(str(GRAPH))
    assert len(gs) == 1

    base = gs[0]

    assert base.num_nodes() == N
    assert base.num_edges() == 30752
    assert tuple(
        base.nodes["company"].data["feature"].shape
    ) == (N, 57)

    assert np.array_equal(
        base.nodes["company"].data["feature"].numpy(),
        x
    )

    # Canonical homo topology only.
    # Never copy source edge labels or split masks.
    u, v = base.edges(
        etype=("company", "homo", "company")
    )

    adj = sp.csr_matrix(
        (
            np.ones(len(u), dtype=np.float32),
            (u.numpy(), v.numpy())
        ),
        shape=(N, N)
    )

    adj = (adj + adj.T).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0
    adj.setdiag(1)
    adj.eliminate_zeros()
    adj.sort_indices()

    assert adj.nnz == 12585
    assert (adj != adj.T).nnz == 0

    graph = dgl.from_scipy(adj)

    assert len(graph.ndata) == 0
    assert len(graph.edata) == 0

    # Validation truth only. Test truth is never read.
    val_y = (
        base.nodes["company"].data["label"][val]
        .numpy()
        .astype(np.int64)
        .copy()
    )

    assert set(val_y) == {0, 1}

    del base, gs

    print("I2_INPUT_AND_CONTEXT_AUDIT=PASS", flush=True)
    print("TRAIN_N=2126 VAL_N=1064", flush=True)
    print("TEST_LABELS_NOT_READ=YES", flush=True)
    print("BATCH_SIZE=256", flush=True)

    return (
        graph, x, train, y, val,
        val_y, fold, views, contexts
    )


class Experiment:
    def __init__(self, data, config):
        (
            self.graph, self.x, self.train,
            self.y, self.val, self.val_y,
            self.fold, self.views, self.contexts
        ) = data

        self.config = config

        self.model = RGTAN(
            in_feats=57,
            hidden_dim=config["hidden_dim"],
            n_layers=2,
            n_classes=2,
            heads=[4, 4],
            activation=torch.nn.PReLU(),
            drop=config["drop"],
            device=DEVICE,
            gated=True,
            ref_df=pd.DataFrame(self.x),
            cat_features={},
            neigh_features={
                name: torch.zeros(N, device=DEVICE)
                for name in NAMES
            },
            nei_att_head=3
        ).to(DEVICE)

        self.lr = (
            config["base_lr"]
            * (BATCH / 1024.0) ** 0.5
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.0001
        )

        self.supervision = np.full(
            N, -1, dtype=np.int64
        )
        self.supervision[self.train] = self.y

    def loader(self, ids, shuffle):
        return NodeDataLoader(
            self.graph,
            torch.from_numpy(
                np.asarray(ids, dtype=np.int64)
            ),
            MultiLayerFullNeighborSampler(2),
            batch_size=BATCH,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0
        )

    def predict(self, batch, view):
        input_nodes, targets, blocks = batch

        input_ids = input_nodes.numpy()
        target_ids = targets.numpy()

        assert np.all(
            self.contexts[view][target_ids] == 2
        )

        labels = self.contexts[view][input_ids].copy()
        labels[np.isin(input_ids, target_ids)] = 2

        risk = torch.from_numpy(
            self.views[view][input_ids]
        ).to(DEVICE)

        risk_dict = {
            name: risk[:, j]
            for j, name in enumerate(NAMES)
        }

        logits = self.model(
            [block.to(DEVICE) for block in blocks],
            torch.from_numpy(
                self.x[input_ids]
            ).to(DEVICE),
            torch.from_numpy(
                labels
            ).long().to(DEVICE),
            {},
            risk_dict
        )

        assert logits.shape == (len(target_ids), 2)
        assert torch.isfinite(logits).all().item()

        return target_ids, logits

    def epoch(self):
        self.model.train()

        losses = []
        seen = []

        for j in range(5):
            ids = self.train[
                self.fold[self.train] == j
            ]

            for batch in self.loader(ids, True):
                self.optimizer.zero_grad(set_to_none=True)

                target_ids, logits = self.predict(
                    batch, j
                )

                truth = self.supervision[target_ids]
                assert np.isin(truth, (0, 1)).all()

                loss = F.cross_entropy(
                    logits,
                    torch.from_numpy(
                        truth
                    ).long().to(DEVICE)
                )

                assert torch.isfinite(loss).item()

                loss.backward()
                self.optimizer.step()

                losses.append(
                    float(loss.detach().item())
                )
                seen.extend(target_ids.tolist())

        assert len(seen) == len(self.train)
        assert len(set(seen)) == len(self.train)

        return float(np.mean(losses)), len(losses)

    def validate(self):
        self.model.eval()

        predictions = np.full(
            N, np.nan, dtype=np.float32
        )
        seen = []

        with torch.no_grad():
            for batch in self.loader(
                self.val, False
            ):
                ids, logits = self.predict(
                    batch, 5
                )

                predictions[ids] = (
                    torch.softmax(logits, dim=1)[:, 1]
                    .cpu().numpy()
                )
                seen.extend(ids.tolist())

        assert len(seen) == len(self.val)
        assert len(set(seen)) == len(self.val)
        assert np.isfinite(
            predictions[self.val]
        ).all()

        ap = float(
            average_precision_score(
                self.val_y,
                predictions[self.val]
            )
        )
        auc = float(
            roc_auc_score(
                self.val_y,
                predictions[self.val]
            )
        )

        assert np.isfinite(ap)
        assert np.isfinite(auc)

        return ap, auc


def pilot(data):
    seed_all()

    exp = Experiment(data, GRID[0])

    ids = exp.train[
        exp.fold[exp.train] == 0
    ]

    batch = next(
        iter(exp.loader(ids, False))
    )

    target_ids, logits = exp.predict(batch, 0)

    loss = F.cross_entropy(
        logits,
        torch.from_numpy(
            exp.supervision[target_ids]
        ).long().to(DEVICE)
    )

    loss.backward()

    assert torch.isfinite(loss).item()

    exp.optimizer.step()

    ap, auc = exp.validate()

    print(
        "I2_REAL_RISK_BACKWARD_PILOT=PASS",
        flush=True
    )
    print(
        f"PILOT_VALIDATION_AUPRC={ap:.6f}",
        flush=True
    )
    print(
        f"PILOT_VALIDATION_AUROC={auc:.6f}",
        flush=True
    )
    print(
        "PILOT_METRICS_NOT_TUNING_RESULTS=YES",
        flush=True
    )
    print("TEST_EVALUATED=NO", flush=True)

    del exp
    torch.cuda.empty_cache()


def run_trial(data, idx):
    config = GRID[idx - 1]
    directory = OUT / "tuning" / f"trial_{idx:02d}"

    directory.mkdir(
        parents=True, exist_ok=True
    )

    result_path = directory / "tuning.json"
    latest = directory / "latest.pt"
    best = directory / "best_val_auprc.pt"

    if result_path.exists():
        result = json.loads(
            result_path.read_text()
        )

        assert result["trial"] == idx
        assert result["config"] == config
        assert result["adapter_sha256"] == ADAPTER_SHA
        assert sha(best) == result["checkpoint_sha256"]

        print(
            f"TRIAL_ALREADY_COMPLETE={idx:02d}/12",
            flush=True
        )
        return result

    seed_all()

    exp = Experiment(data, config)

    start_epoch = 1
    best_ap = -1.0
    best_auc = -1.0
    best_epoch = 0
    wait = 0
    history = []

    started = time.perf_counter()

    if latest.exists():
        state = torch.load(
            latest, map_location=DEVICE
        )

        assert state["trial"] == idx
        assert state["config"] == config
        assert state["adapter_sha256"] == ADAPTER_SHA
        assert sha(best) == state["best_sha256"]

        exp.model.load_state_dict(
            state["model"]
        )
        exp.optimizer.load_state_dict(
            state["optim"]
        )

        best_ap = state["best_ap"]
        best_auc = state["best_auc"]
        best_epoch = state["best_ep"]
        wait = state["wait"]
        history = state["history"]
        start_epoch = state["epoch"] + 1

        torch.set_rng_state(
            state["torch_rng"]
        )
        torch.cuda.set_rng_state(
            state["cuda_rng"], device=DEVICE
        )
        np.random.set_state(
            state["numpy_rng"]
        )
        random.setstate(
            state["python_rng"]
        )

        print(
            f"RESUMING_TRIAL={idx:02d} "
            f"EPOCH={start_epoch}",
            flush=True
        )
    else:
        assert not best.exists(), (
            "Orphan checkpoint; manual audit required"
        )

        print(
            f"STARTING_TRIAL={idx:02d}/12 "
            f"CONFIG={json.dumps(config, sort_keys=True)}",
            flush=True
        )

    torch.cuda.reset_peak_memory_stats()

    for epoch in range(
        start_epoch, MAX_EPOCHS + 1
    ):
        tick = time.perf_counter()

        loss, steps = exp.epoch()
        ap, auc = exp.validate()

        if ap > best_ap + 1e-12:
            best_ap = ap
            best_auc = auc
            best_epoch = epoch
            wait = 0

            atomic_pt(
                best,
                {
                    "model_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in exp.model.state_dict().items()
                    },
                    "epoch": epoch,
                    "trial": idx,
                    "config": config,
                    "best_val_auprc": ap,
                    "adapter_sha256": ADAPTER_SHA
                }
            )
        else:
            wait += 1

        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "steps": steps,
                "val_auprc": ap,
                "val_auroc": auc,
                "seconds": round(
                    time.perf_counter() - tick, 3
                )
            }
        )

        atomic_pt(
            latest,
            {
                "trial": idx,
                "config": config,
                "adapter_sha256": ADAPTER_SHA,
                "epoch": epoch,
                "model": exp.model.state_dict(),
                "optim": exp.optimizer.state_dict(),
                "best_ap": best_ap,
                "best_auc": best_auc,
                "best_ep": best_epoch,
                "wait": wait,
                "history": history,
                "best_sha256": sha(best),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(DEVICE),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate()
            }
        )

        print(
            f"TRIAL={idx:02d}/12 "
            f"EPOCH={epoch:03d}/{MAX_EPOCHS} "
            f"LOSS={loss:.5f} "
            f"valAUPRC={ap:.6f} "
            f"BEST={best_ap:.6f}@{best_epoch} "
            f"WAIT={wait}/{PATIENCE} "
            f"SECONDS={history[-1]['seconds']}",
            flush=True
        )

        if wait >= PATIENCE:
            break

    assert best_epoch >= 1
    assert best.exists()

    result = {
        "status": "TUNING_COMPLETE",
        "model": "RGTAN",
        "dataset": "FDCompCN",
        "ratio": "TR40",
        "trial": idx,
        "config": config,
        "split_seed": 2,
        "seed": 2,
        "batch_size": BATCH,
        "base_learning_rate": config["base_lr"],
        "effective_lr": exp.lr,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_auprc": best_ap,
        "best_val_auroc": best_auc,
        "source_sha256": SOURCE_SHA,
        "split_sha256": SPLIT_SHA,
        "adapter_sha256": ADAPTER_SHA,
        "author_commit": PIN,
        "checkpoint_sha256": sha(best),
        "peak_gpu_allocated_gib": (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
        ),
        "elapsed_seconds": round(
            time.perf_counter() - started, 3
        ),
        "protocol": (
            "Single model, five OOF contexts, "
            "accelerated batch size 256"
        ),
        "test_scoring_performed": False,
        "test_used_for_selection": False
    }

    atomic_json(
        directory / "history.json",
        history
    )
    atomic_json(result_path, result)

    latest.unlink(missing_ok=True)

    print(
        f"TRIAL_FINISHED={idx:02d}/12 "
        f"best_val_AUPRC={best_ap:.6f}",
        flush=True
    )

    return result


def main():
    assert len(sys.argv) == 2
    assert sys.argv[1] in (
        "pilot", "tune-all", "status"
    )

    mode = sys.argv[1]

    if mode == "status":
        for idx in range(1, 13):
            path = (
                OUT / "tuning"
                / f"trial_{idx:02d}"
                / "tuning.json"
            )

            print(
                f"TRIAL={idx:02d}/12 "
                + (
                    "COMPLETE"
                    if path.exists()
                    else "PENDING"
                )
            )

        print(
            "WINNER_FROZEN=",
            (OUT / "winner.json").exists()
        )
        return

    data = load_data()

    if mode == "pilot":
        pilot(data)
        return

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    rows = [
        run_trial(data, idx)
        for idx in range(1, 13)
    ]

    winner = max(
        rows,
        key=lambda row: (
            row["best_val_auprc"],
            -row["trial"]
        )
    )

    record = {
        "status": "WINNER_FROZEN",
        "model": "RGTAN",
        "dataset": "FDCompCN",
        "ratio": "TR40",
        "winner_trial": winner["trial"],
        "config": winner["config"],
        "batch_size": BATCH,
        "best_val_auprc": winner["best_val_auprc"],
        "adapter_sha256": ADAPTER_SHA,
        "source_sha256": SOURCE_SHA,
        "split_sha256": SPLIT_SHA,
        "test_scoring_performed": False
    }

    path = OUT / "winner.json"

    if path.exists():
        assert json.loads(
            path.read_text()
        ) == record
    else:
        atomic_json(path, record)

    print(
        "I2_RGTAN_FDCOMPCN_TR40_TUNING=12/12_COMPLETE",
        flush=True
    )
    print(
        "WINNER_FROZEN="
        + json.dumps(record, sort_keys=True),
        flush=True
    )
    print("TEST_EVALUATED=NO", flush=True)


if __name__ == "__main__":
    main()
