#!/usr/bin/env python3
import os, sys, gc, csv, json, time, math, random, hashlib, tarfile, subprocess, traceback, contextlib, io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dgl.data.utils import load_graphs
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

WORK = Path("/workspace/bwgnn_vast")
RAW_ROOT = Path("/workspace/dataset_audit/raw/tsocial")
SPLIT_PATH = WORK / "splits/tsocial_seed2_nested_splits.npz"
SMOKE_JSON = WORK / "evidence/tsocial/a6000_hidden64_csc_2epoch_smoke.json"

BASE = WORK / "results/bwgnn/tsocial"
TUNING = BASE / "tuning"
FINAL = BASE / "final"
RUN_DIR = FINAL / "runs"
TIME_DIR = FINAL / "epoch_times"
CKPT_DIR = WORK / "checkpoints/tsocial"
EVIDENCE = WORK / "evidence/tsocial"
ARCHIVES = WORK / "archives"
for d in [TUNING, RUN_DIR, TIME_DIR, CKPT_DIR, EVIDENCE, ARCHIVES, SPLIT_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORK / "adapters"))
from BWGNN_gpu import BWGNN

EXPECTED_COMMIT = "de0631f039bbd19c1890b483cc01f1007f596af7"
EXPECTED_DATA_SHA = "8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0"
EXPECTED_SPLIT_SHA = "b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519"

# Fixed 12-trial TR40 budget. Trial 1 is the author-compatible H64/O2/LR0.01
# path that already passed the A6000 two-epoch smoke.
TRIALS = [
    {"hidden":64, "order":2, "lr":1e-2, "weight_decay":0.0},
    {"hidden":64, "order":2, "lr":5e-3, "weight_decay":0.0},
    {"hidden":64, "order":2, "lr":1e-3, "weight_decay":0.0},
    {"hidden":64, "order":2, "lr":5e-4, "weight_decay":0.0},
    {"hidden":32, "order":2, "lr":1e-2, "weight_decay":0.0},
    {"hidden":32, "order":2, "lr":5e-3, "weight_decay":0.0},
    {"hidden":32, "order":2, "lr":1e-3, "weight_decay":0.0},
    {"hidden":64, "order":1, "lr":5e-3, "weight_decay":0.0},
    {"hidden":64, "order":3, "lr":5e-3, "weight_decay":0.0},
    {"hidden":32, "order":3, "lr":5e-3, "weight_decay":0.0},
    {"hidden":64, "order":2, "lr":5e-3, "weight_decay":1e-4},
    {"hidden":32, "order":2, "lr":5e-3, "weight_decay":1e-4},
]

MAX_EPOCHS = 100
PATIENCE = 20
SEEDS = [2, 42, 72]
RATIOS = ["TR40", "TR30", "TR20", "TR10"]
WARMUPS = 3
REPEATS = 10
DEVICE = torch.device("cuda:0")

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def choose_threshold(y, p):
    candidates = []
    for t in np.arange(0.01, 1.00, 0.01):
        pred = (p >= t).astype(np.int64)
        candidates.append((
            f1_score(y, pred, average="macro", zero_division=0),
            recall_score(y, pred, pos_label=1, zero_division=0),
            -abs(float(t) - 0.5),
            float(t),
        ))
    return max(candidates)[3]

def calc_metrics(y, p, t):
    pred = (p >= t).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "fraud_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "fraud_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "gmean": float(math.sqrt(sens * spec)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def is_oom(exc):
    s = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in s
        or "cudaoutofmemory" in type(exc).__name__.lower()
    )

print("===== BWGNN VAST — T-SOCIAL FINAL PIPELINE =====", flush=True)
print("A6000 smoke has already passed; this script tunes TR40 then runs the final 12-run grid.", flush=True)

# ------------------------------------------------------------------
# 0. Verify successful A6000 smoke.
# ------------------------------------------------------------------
assert SMOKE_JSON.exists(), f"Missing smoke evidence: {SMOKE_JSON}"
smoke = json.loads(SMOKE_JSON.read_text())
assert smoke.get("status") == "PASS", smoke
assert smoke.get("full_graph") is True
assert smoke.get("sampling_used") is False
assert smoke.get("test_accessed") is False
print("A6000 smoke evidence: PASS", flush=True)
print("Measured smoke mean train s/epoch:", smoke.get("mean_train_seconds_per_epoch"), flush=True)
print("Measured smoke peak GPU MB:", smoke.get("peak_gpu_memory_mb"), flush=True)

# ------------------------------------------------------------------
# 1. Source / environment gates.
# ------------------------------------------------------------------
commit = subprocess.check_output(
    ["git", "-C", str(WORK / "repo/Rethinking-Anomaly-Detection"), "rev-parse", "HEAD"],
    text=True,
).strip()
assert commit == EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
gpu_name = torch.cuda.get_device_name(0)
assert "A6000" in gpu_name.upper(), gpu_name

preferred = RAW_ROOT / "tsocial"
candidates = []
if preferred.exists() and preferred.is_file():
    candidates.append(preferred)
candidates += [p for p in sorted(RAW_ROOT.rglob("*")) if p.is_file() and p != preferred]

data_path = None
for p in candidates:
    if p.stat().st_size < 100 * 1024 * 1024:
        continue
    h = sha256_file(p)
    if h == EXPECTED_DATA_SHA:
        data_path = p
        break
assert data_path is not None, f"Canonical T-Social SHA not found under {RAW_ROOT}"

assert SPLIT_PATH.exists()
split_sha = sha256_file(SPLIT_PATH)
assert split_sha == EXPECTED_SPLIT_SHA, (split_sha, EXPECTED_SPLIT_SHA)

s = np.load(SPLIT_PATH, allow_pickle=False)
expected_sizes = {
    "TR40": 2312426, "TR30": 1734319, "TR20": 1156212,
    "TR10": 578106, "val": 1156213, "test": 2312426,
}
for k, n in expected_sizes.items():
    assert len(s[k]) == n, (k, len(s[k]), n)
assert int(s["seed"][0]) == 2

print("Dataset SHA256:", EXPECTED_DATA_SHA, flush=True)
print("Split SHA256:", split_sha, flush=True)
print("GPU:", gpu_name, flush=True)
print("SOURCE_SPLIT_HARDWARE_GATE=PASS", flush=True)

# ------------------------------------------------------------------
# 2. Load full graph once and keep CSC representation.
# ------------------------------------------------------------------
load_t0 = time.perf_counter()
graphs, _ = load_graphs(str(data_path))
graph = graphs[0]
assert graph.num_nodes() == 5781065
assert graph.num_edges() == 146211016
assert tuple(graph.ndata["feature"].shape) == (5781065, 10)

raw_labels_cpu = graph.ndata["label"]
labels_cpu = (
    raw_labels_cpu.argmax(1).long()
    if raw_labels_cpu.ndim == 2
    else raw_labels_cpu.reshape(-1).long()
)
normal = int((labels_cpu == 0).sum().item())
fraud = int((labels_cpu == 1).sum().item())
assert (normal, fraud) == (5606785, 174280), (normal, fraud)

csc_t0 = time.perf_counter()
graph = graph.formats("csc")
csc_seconds = time.perf_counter() - csc_t0

graph = graph.to(DEVICE)
torch.cuda.synchronize()
features = graph.ndata["feature"].float()
raw_labels = graph.ndata["label"]
labels = (
    raw_labels.argmax(1).long()
    if raw_labels.ndim == 2
    else raw_labels.reshape(-1).long()
)
val_ids = torch.tensor(s["val"], dtype=torch.long, device=DEVICE)
test_ids = torch.tensor(s["test"], dtype=torch.long, device=DEVICE)
load_seconds = time.perf_counter() - load_t0

print("Nodes/edges/features:", graph.num_nodes(), graph.num_edges(), tuple(features.shape), flush=True)
print("Normal/Fraud:", normal, fraud, flush=True)
print(f"CSC conversion seconds: {csc_seconds:.3f}", flush=True)
print(f"Load/preprocess seconds: {load_seconds:.3f}", flush=True)

# ------------------------------------------------------------------
# 3. TR40 12-trial tuning, validation AUPRC only. Test untouched.
# ------------------------------------------------------------------
print("\n===== T-SOCIAL TR40 TUNING — 12 FIXED TRIALS =====", flush=True)
train_ids_tr40 = torch.tensor(s["TR40"], dtype=torch.long, device=DEVICE)
tuning_rows = []

for trial_no, cfg in enumerate(TRIALS, 1):
    trial_file = TUNING / f"trial_{trial_no:02d}.json"
    if trial_file.exists():
        row = json.loads(trial_file.read_text())
        tuning_rows.append(row)
        print(
            f"SKIP trial {trial_no:02d} status={row['status']} "
            f"best_val_AUPRC={row.get('best_val_auprc')}",
            flush=True,
        )
        continue

    print(f"\nTRIAL {trial_no:02d}/12 {cfg}", flush=True)
    seed_all(2)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = optimizer = logits = vp = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = BWGNN(
                in_feats=10,
                h_feats=cfg["hidden"],
                num_classes=2,
                graph=graph,
                d=cfg["order"],
            ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        ytr = labels[train_ids_tr40]
        n0 = int((ytr == 0).sum())
        n1 = int((ytr == 1).sum())
        class_weight = torch.tensor([1.0, n0 / n1], device=DEVICE)

        best = -1.0
        best_epoch = -1
        stale = 0
        start = time.perf_counter()
        train_seconds_sum = 0.0
        val_seconds_sum = 0.0

        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(features)
            loss = F.cross_entropy(
                logits[train_ids_tr40],
                labels[train_ids_tr40],
                weight=class_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite tuning loss at trial {trial_no}, epoch {ep}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            tr = time.perf_counter() - t0
            train_seconds_sum += tr

            model.eval()
            torch.cuda.synchronize()
            v0 = time.perf_counter()
            with torch.no_grad():
                vp = torch.softmax(model(features), 1)[val_ids, 1]
            torch.cuda.synchronize()
            val_seconds_sum += time.perf_counter() - v0

            val_ap = float(
                average_precision_score(
                    labels[val_ids].detach().cpu().numpy(),
                    vp.detach().cpu().numpy(),
                )
            )
            if val_ap > best:
                best = val_ap
                best_epoch = ep
                stale = 0
            else:
                stale += 1

            if ep == 1 or ep % 10 == 0 or stale >= PATIENCE:
                print(
                    f"trial={trial_no:02d} epoch={ep:03d} "
                    f"val_AP={val_ap:.6f} best={best:.6f}@{best_epoch} "
                    f"train={tr:.3f}s stale={stale}",
                    flush=True,
                )

            del logits, loss, vp
            logits = vp = None

            if stale >= PATIENCE:
                break

        row = {
            "trial": trial_no,
            **cfg,
            "status": "PASS",
            "train_seed": 2,
            "split": "TR40",
            "best_val_auprc": float(best),
            "best_epoch": int(best_epoch),
            "completed_epochs": int(ep),
            "train_seconds_sum": float(train_seconds_sum),
            "validation_seconds_sum": float(val_seconds_sum),
            "wall_seconds": float(time.perf_counter() - start),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
            "test_accessed": False,
        }

    except Exception as exc:
        if is_oom(exc):
            row = {
                "trial": trial_no,
                **cfg,
                "status": "OOM",
                "best_val_auprc": None,
                "best_epoch": None,
                "completed_epochs": 0,
                "error": str(exc),
                "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
                "test_accessed": False,
            }
            print(f"trial={trial_no:02d} OOM — counted in 12-trial budget", flush=True)
        else:
            raise
    finally:
        try:
            del model, optimizer, logits, vp
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    trial_file.write_text(json.dumps(row, indent=2))
    tuning_rows.append(row)

successful = [r for r in tuning_rows if r["status"] == "PASS"]
assert successful, "No successful T-Social tuning trial."
winner = max(successful, key=lambda r: (r["best_val_auprc"], -r["trial"]))

(TUNING / "tsocial_tuning_trials.json").write_text(json.dumps(tuning_rows, indent=2))
with (TUNING / "tsocial_tuning_trials.csv").open("w", newline="") as f:
    fields = sorted(set().union(*(r.keys() for r in tuning_rows)))
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(tuning_rows)

frozen = {
    "model": "BWGNN",
    "dataset": "T-Social",
    "selection_metric": "validation AUPRC",
    "test_accessed": False,
    "trials_budget": 12,
    "successful_trials": len(successful),
    "winner": winner,
}
(TUNING / "tsocial_frozen_config.json").write_text(json.dumps(frozen, indent=2))

print("\nTUNING_GATE=PASS", flush=True)
print("WINNER:", winner, flush=True)

HIDDEN = int(winner["hidden"])
ORDER = int(winner["order"])
LR = float(winner["lr"])
WEIGHT_DECAY = float(winner["weight_decay"])

# ------------------------------------------------------------------
# 4. Final controlled 12-run grid.
# ------------------------------------------------------------------
print("\n===== T-SOCIAL FINAL GRID =====", flush=True)
for ratio in RATIOS:
    train_ids = torch.tensor(s[ratio], dtype=torch.long, device=DEVICE)

    for seed in SEEDS:
        key = f"{ratio}_seed{seed}"
        result_file = RUN_DIR / f"{key}.json"
        epoch_file = TIME_DIR / f"{key}_epoch_times.csv"
        ckpt = CKPT_DIR / f"{key}.pt"

        if result_file.exists():
            print(f"SKIP complete {key}", flush=True)
            continue

        epoch_file.unlink(missing_ok=True)
        ckpt.unlink(missing_ok=True)
        print(f"\nSTART {key}", flush=True)

        seed_all(seed)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with contextlib.redirect_stdout(io.StringIO()):
            model = BWGNN(
                in_feats=10,
                h_feats=HIDDEN,
                num_classes=2,
                graph=graph,
                d=ORDER,
            ).to(DEVICE)

        parameter_count = int(sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        ytr = labels[train_ids]
        n0 = int((ytr == 0).sum())
        n1 = int((ytr == 1).sum())
        class_weight = torch.tensor([1.0, n0 / n1], device=DEVICE)

        best = -1.0
        best_epoch = -1
        stale = 0
        train_times = []
        val_times = []

        with epoch_file.open("w", newline="") as ef:
            ew = csv.DictWriter(
                ef,
                fieldnames=[
                    "epoch","loss","train_seconds","validation_seconds",
                    "cumulative_train_seconds","val_auprc",
                    "best_val_auprc","best_epoch",
                ],
            )
            ew.writeheader()
            cumulative = 0.0

            for ep in range(1, MAX_EPOCHS + 1):
                model.train()
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                logits = model(features)
                loss = F.cross_entropy(
                    logits[train_ids], labels[train_ids], weight=class_weight
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite final loss {key} epoch {ep}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()

                tr = time.perf_counter() - t0
                train_times.append(tr)
                cumulative += tr

                model.eval()
                torch.cuda.synchronize()
                v0 = time.perf_counter()
                with torch.no_grad():
                    val_logits = model(features)
                    vp = torch.softmax(val_logits, 1)[val_ids, 1]
                torch.cuda.synchronize()
                vs = time.perf_counter() - v0
                val_times.append(vs)

                val_ap = float(
                    average_precision_score(
                        labels[val_ids].detach().cpu().numpy(),
                        vp.detach().cpu().numpy(),
                    )
                )

                if val_ap > best:
                    best = val_ap
                    best_epoch = ep
                    stale = 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    stale += 1

                ew.writerow({
                    "epoch": ep,
                    "loss": float(loss.item()),
                    "train_seconds": tr,
                    "validation_seconds": vs,
                    "cumulative_train_seconds": cumulative,
                    "val_auprc": val_ap,
                    "best_val_auprc": best,
                    "best_epoch": best_epoch,
                })
                ef.flush()

                if ep == 1 or ep % 10 == 0 or stale >= PATIENCE:
                    print(
                        f"{key} epoch={ep:03d} loss={loss.item():.5f} "
                        f"val_AP={val_ap:.6f} best={best:.6f}@{best_epoch} "
                        f"train={tr:.3f}s val={vs:.3f}s stale={stale}",
                        flush=True,
                    )

                del logits, loss, val_logits, vp

                if stale >= PATIENCE:
                    break

        train_times = np.asarray(train_times, dtype=float)
        val_times = np.asarray(val_times, dtype=float)

        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        # Separate inference latency.
        with torch.no_grad():
            for _ in range(WARMUPS):
                _ = model(features)
                torch.cuda.synchronize()

            latency = []
            for _ in range(REPEATS):
                torch.cuda.synchronize()
                q = time.perf_counter()
                _ = model(features)
                torch.cuda.synchronize()
                latency.append((time.perf_counter() - q) * 1000.0)

            prob = torch.softmax(model(features), 1)[:, 1]

        # Validation-only threshold.
        vy = labels[val_ids].detach().cpu().numpy()
        vp = prob[val_ids].detach().cpu().numpy()
        threshold = choose_threshold(vy, vp)

        # Test used only now, after all choices are frozen.
        ty = labels[test_ids].detach().cpu().numpy()
        tp = prob[test_ids].detach().cpu().numpy()
        m = calc_metrics(ty, tp, threshold)

        result = {
            "model": "BWGNN",
            "dataset": "T-Social",
            "full_graph": True,
            "sampling_used": False,
            "graph_reduced": False,
            "sparse_format": "csc",
            "hardware_profile_id": "vast-a6000-instance-51035671",
            "repository_commit": commit,
            "dataset_sha256": EXPECTED_DATA_SHA,
            "split_sha256": split_sha,
            "split": ratio,
            "split_seed": 2,
            "train_seed": seed,
            "hidden": HIDDEN,
            "order": ORDER,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "completed_epochs": int(len(train_times)),
            "best_epoch": int(best_epoch),
            "best_val_auprc": float(best),
            "parameter_count": parameter_count,
            "load_preprocess_seconds": float(load_seconds),
            "total_train_seconds": float(train_times.sum()),
            "mean_epoch_train_seconds": float(train_times.mean()),
            "median_epoch_train_seconds": float(np.median(train_times)),
            "std_epoch_train_seconds": float(
                train_times.std(ddof=1) if len(train_times) > 1 else 0.0
            ),
            "total_validation_seconds": float(val_times.sum()),
            "mean_validation_seconds": float(val_times.mean()),
            "inference_latency_mean_ms": float(np.mean(latency)),
            "inference_latency_std_ms": float(np.std(latency, ddof=1)),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
            "validation_selected_threshold": float(threshold),
            "test_evaluated_once_after_freeze": True,
            **m,
        }
        result_file.write_text(json.dumps(result, indent=2))
        ckpt.unlink(missing_ok=True)

        print(
            f"DONE {key} | test_AUPRC={m['auprc']:.6f} "
            f"AUROC={m['auroc']:.6f} MacroF1={m['macro_f1']:.6f} "
            f"| {result['mean_epoch_train_seconds']:.3f}s/epoch",
            flush=True,
        )

        del model, optimizer, prob
        gc.collect()
        torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 5. Aggregate + manifest + archive.
# ------------------------------------------------------------------
results = []
for ratio in RATIOS:
    for seed in SEEDS:
        p = RUN_DIR / f"{ratio}_seed{seed}.json"
        assert p.exists(), f"Missing {p}"
        results.append(json.loads(p.read_text()))

(FINAL / "all_runs.json").write_text(json.dumps(results, indent=2))
with (FINAL / "all_runs.csv").open("w", newline="") as f:
    fields = list(results[0].keys())
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(results)

summary = {
    "model": "BWGNN",
    "dataset": "T-Social",
    "status": "COMPLETE",
    "full_graph": True,
    "sampling_used": False,
    "frozen_winner": winner,
    "completed_runs": len(results),
    "ratios": {},
}

metric_keys = [
    "auprc","auroc","macro_f1","fraud_precision","fraud_recall",
    "fraud_f1","gmean","mean_epoch_train_seconds",
    "median_epoch_train_seconds","total_train_seconds",
    "inference_latency_mean_ms","peak_gpu_memory_mb",
]

for ratio in RATIOS:
    rows = [r for r in results if r["split"] == ratio]
    summary["ratios"][ratio] = {}
    for k in metric_keys:
        x = np.asarray([r[k] for r in rows], dtype=float)
        summary["ratios"][ratio][k] = {
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
        }

(FINAL / "summary.json").write_text(json.dumps(summary, indent=2))

manifest = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model": "BWGNN",
    "dataset": "T-Social",
    "repository_commit": commit,
    "dataset_path": str(data_path),
    "dataset_sha256": EXPECTED_DATA_SHA,
    "split_sha256": split_sha,
    "nodes": 5781065,
    "stored_edges": 146211016,
    "features": 10,
    "normal": 5606785,
    "fraud_or_anomaly": 174280,
    "full_graph": True,
    "sampling_used": False,
    "graph_reduced": False,
    "sparse_format": "csc",
    "a6000_smoke": smoke,
    "tuning_trials": tuning_rows,
    "frozen_configuration": winner,
    "final_training_seeds": SEEDS,
    "final_ratios": RATIOS,
    "checkpoint_metric": "validation AUPRC",
    "threshold_rule": "validation Macro-F1; ties fraud recall then closest to 0.5",
    "test_isolation": True,
    "summary": summary,
}
(EVIDENCE / "final_manifest.json").write_text(json.dumps(manifest, indent=2))

archive = ARCHIVES / "bwgnn_tsocial_vast_a6000_final.tar.gz"
with tarfile.open(archive, "w:gz") as t:
    for p, arc in [
        (TUNING, "results/tsocial/tuning"),
        (FINAL, "results/tsocial/final"),
        (EVIDENCE, "evidence/tsocial"),
        (SPLIT_PATH, "splits/tsocial_seed2_nested_splits.npz"),
        (WORK / "adapters/BWGNN_gpu.py", "adapters/BWGNN_gpu.py"),
        (Path(__file__), "runner/08_bwgnn_tsocial_final.py"),
    ]:
        t.add(p, arcname=str(arc))

digest = sha256_file(archive)
(archive.with_suffix(archive.suffix + ".sha256")).write_text(
    f"{digest}  {archive.name}\n"
)

print("\n===== T-SOCIAL FINAL COMPLETE =====", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("ARCHIVE:", archive, flush=True)
print("SHA256:", digest, flush=True)
