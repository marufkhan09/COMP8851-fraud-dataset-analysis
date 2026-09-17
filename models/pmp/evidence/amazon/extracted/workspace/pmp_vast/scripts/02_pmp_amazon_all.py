#!/usr/bin/env python3
"""
COMP8851 PMP × Amazon unified benchmark: one process after environment bootstrap.

Protocol:
- exact frozen PMP commit
- exact canonical Amazon.mat SHA256
- exact frozen Amazon split SHA256
- exact 1-epoch unified smoke
- TR40 tuning only, train seed 2, max 12 trials, max 100 epochs, patience 20
- tune LR × weight decay only; preserve official Amazon architecture/dropout
- winner by validation AUPRC
- fresh final grid: TR40/TR30/TR20/TR10 × seeds 2,42,72
- threshold selected on validation Macro-F1 over 0.01..0.99,
  tie fraud recall then closeness to 0.5
- test evaluated exactly once per final run after choices are frozen
- raw epoch timing, inference timing, peak GPU memory, checkpoints and hashes saved

This is external COMP8851 orchestration. It does not edit the official PMP repository.
"""

from pathlib import Path
from types import SimpleNamespace
import copy, csv, gc, hashlib, json, math, os, random, shutil, subprocess, sys, time

import dgl
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.data.utils import _get_dgl_url

WORK = Path("/workspace/pmp_vast")
REPO = WORK / "repo/PMP"
CANONICAL = Path("/workspace/dataset_audit/raw/amazon/Amazon.mat")
SPLIT = WORK / "shared/amazon_seed2_nested_splits.npz"

ROOT = WORK / "unified/amazon"
SMOKE_DIR = ROOT / "smoke"
TUNING_DIR = ROOT / "tuning"
TRIALS_DIR = TUNING_DIR / "trials"
FINAL_DIR = ROOT / "final"
RUNS_DIR = FINAL_DIR / "runs"
CKPT_DIR = FINAL_DIR / "checkpoints"

for p in [SMOKE_DIR, TUNING_DIR, TRIALS_DIR, FINAL_DIR, RUNS_DIR, CKPT_DIR, WORK/"logs"]:
    p.mkdir(parents=True, exist_ok=True)

EXPECTED_COMMIT = "3f7629f6c180891a0bc1bba3c66d94d288a1ddae"
EXPECTED_DATA_SHA = "4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63"
EXPECTED_SPLIT_SHA = "0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd"

RATIOS = ["TR40", "TR30", "TR20", "TR10"]
FINAL_SEEDS = [2, 42, 72]
MAX_EPOCHS = 100
PATIENCE = 20
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

# Team-controlled 12-trial LR × WD sensitivity budget.
# Official Amazon architecture/dropout/batch/mechanism remain frozen.
TRIALS = []
tid = 1
for lr in [1e-2, 1e-3, 1e-4]:
    for wd in [0.0, 1e-5, 1e-4, 1e-3]:
        TRIALS.append({"trial_id": tid, "lr": lr, "weight_decay": wd})
        tid += 1
assert len(TRIALS) == 12

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    try:
        dgl.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def save_state_cpu(model, path, meta):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)

def load_state(model, path):
    obj = torch.load(path, map_location="cpu")
    model.load_state_dict(obj["state_dict"])
    return obj.get("meta", {})

def mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

def choose_threshold(y_true, probs):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    rows = []
    for t in THRESHOLDS:
        pred = (probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "fraud_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
            "distance_to_0_5": float(abs(float(t) - 0.5)),
        })
    rows.sort(key=lambda r: (-r["macro_f1"], -r["fraud_recall"], r["distance_to_0_5"], r["threshold"]))
    return rows[0], rows

def metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    tpr = tp/(tp+fn) if tp+fn else 0.0
    tnr = tn/(tn+fp) if tn+fp else 0.0
    return {
        "auprc": float(average_precision_score(y_true, probs)),
        "auroc": float(roc_auc_score(y_true, probs)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "fraud_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "fraud_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "gmean": float(math.sqrt(tpr * tnr)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

class NullLogger:
    def __init__(self):
        self.append = ""
    def log(self, *args, **kwargs):
        pass
    def add_line(self, *args, **kwargs):
        pass

print("===== PMP × AMAZON — ONE-CHUNK UNIFIED PIPELINE =====", flush=True)

# ---------------------------------------------------------------------
# HARD IDENTITY GATES
# ---------------------------------------------------------------------
assert REPO.exists()
assert git("rev-parse", "HEAD") == EXPECTED_COMMIT
assert git("status", "--porcelain") == ""
assert CANONICAL.exists() and sha256_file(CANONICAL) == EXPECTED_DATA_SHA
assert SPLIT.exists() and sha256_file(SPLIT) == EXPECTED_SPLIT_SHA
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
GPU = torch.cuda.get_device_name(0)
assert "A6000" in GPU.upper(), GPU
torch.cuda.set_device(0)
torch.set_num_threads(8)

s = np.load(SPLIT, allow_pickle=False)
required = ["TR40","TR30","TR20","TR10","val","test","seed","source_nodes"]
for k in required:
    assert k in s.files, (k, s.files)
sets = {k:set(map(int,s[k])) for k in ["TR40","TR30","TR20","TR10","val","test"]}
assert sets["TR10"] <= sets["TR20"] <= sets["TR30"] <= sets["TR40"]
for r in RATIOS:
    assert sets[r].isdisjoint(sets["val"])
    assert sets[r].isdisjoint(sets["test"])
assert sets["val"].isdisjoint(sets["test"])
assert int(s["seed"][0]) == 2
assert int(s["source_nodes"][0]) == 11944
assert len(sets["TR40"] | sets["val"] | sets["test"]) == 8639
for k,v in sets.items():
    assert min(v) >= 3305 and max(v) < 11944, (k, min(v), max(v))

print("GPU:", GPU, flush=True)
print("PMP_COMMIT=PASS", flush=True)
print("AMAZON_CANONICAL_SHA=PASS", flush=True)
print("AMAZON_FROZEN_SPLIT_SHA=PASS", flush=True)
print("Frozen split sizes:", {k:len(s[k]) for k in ["TR40","TR30","TR20","TR10","val","test"]}, flush=True)

# ---------------------------------------------------------------------
# IMPORT OFFICIAL PMP
# ---------------------------------------------------------------------
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from DataHelper.datasetHelper import DatasetHelper
from training_procedure import Trainer

official = yaml.safe_load((REPO/"config/amazon.yml").read_text())["LA-SAGE-S"]
base = copy.deepcopy(official)

# Exact official Amazon model-specific configuration from frozen repo.
expected_author = {
    "add_self_loop": False,
    "agg": "mean",
    "batch_size": 128,
    "dataset_seed": 717,
    "dropout": 0.6,
    "full_neighbors": True,
    "hid_dim": 256,
    "homo": True,
    "lr": 0.01,
    "n_layer": 1,
    "relation_agg": "cat",
    "resi": 0.2,
    "weight_decay": 0.0,
}
for k,v in expected_author.items():
    assert base.get(k) == v, (k, base.get(k), v)

base.update({
    "model_name":"LA-SAGE-S",
    "model":"LA-SAGE-S",
    "dataset":"amazon",
    "gpu_id":0,
    "seed":2,
    "train_size":0.4,
    "val_size":0.2,
    "epochs":MAX_EPOCHS,
    "patience":PATIENCE,
    "eval_interval":1,
    "test_each_epoch":False,
    "monitor":"ap_gnn",
    "threshold_moving":True,
    "thres":0.5,
    "multirun":1,
    "run_best":False,
})

# DGL raw-dir identity bridge to canonical Amazon.mat.
raw_root = Path("/dev/shm/pmp_amazon_raw")
shutil.rmtree(raw_root, ignore_errors=True)
raw_root.mkdir(parents=True, exist_ok=True)
url = _get_dgl_url("dataset/FraudAmazon.zip")
suffix = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
for sub in [f"amazon_{suffix}", "amazon"]:
    d = raw_root/sub
    d.mkdir(parents=True, exist_ok=True)
    link = d/"Amazon.mat"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(CANONICAL)

helper = DatasetHelper(base, dName="amazon", dDescription="COMP8851 unified Amazon")
helper.dataset_source_folder_path = str(raw_root)
helper.load()

assert helper.num_nodes == 11944
assert helper.feat_dim == 25
eligible = torch.arange(3305, 11944, dtype=torch.long)
fraud = int(helper.labels[eligible].sum().item())
assert fraud == 821, fraud
assert len(eligible)-fraud == 7818
assert len(helper.relations) == 3, helper.relations

print("Amazon nodes/features:", helper.num_nodes, helper.feat_dim, flush=True)
print("Amazon stored relation edges:", helper.data.number_of_edges(), flush=True)
print("Amazon relations:", helper.relations, flush=True)
print("Eligible fraud/normal: 821 / 7818", flush=True)
print("OFFICIAL_AMAZON_CONFIG=PASS", flush=True)

def apply_split(ratio):
    N = helper.num_nodes
    tr = torch.as_tensor(s[ratio], dtype=torch.long)
    va = torch.as_tensor(s["val"], dtype=torch.long)
    te = torch.as_tensor(s["test"], dtype=torch.long)

    tm = torch.zeros(N, dtype=torch.bool)
    vm = torch.zeros(N, dtype=torch.bool)
    xm = torch.zeros(N, dtype=torch.bool)
    tm[tr]=True; vm[va]=True; xm[te]=True
    helper.data.ndata["train_mask"]=tm
    helper.data.ndata["val_mask"]=vm
    helper.data.ndata["test_mask"]=xm
    helper.train_mask=tm; helper.val_mask=vm; helper.test_mask=xm
    helper.train_nid=tr; helper.val_nid=va; helper.test_nid=te

    label_unk = torch.full((N,), 2, dtype=torch.long)
    label_unk[tr] = helper.labels[tr]
    helper.data.ndata["label_unk"] = label_unk
    assert torch.equal(label_unk[tr], helper.labels[tr])
    outside = torch.ones(N, dtype=torch.bool); outside[tr]=False
    assert torch.all(label_unk[outside] == 2)

    sampler = MultiLayerFullNeighborSampler(num_layers=base["n_layer"])
    train_loader, val_loader, test_loader = helper.get_DGLloader(helper.data, sampler)
    return tr, va, te, train_loader, val_loader, test_loader

def make_args(seed):
    return SimpleNamespace(
        gpu_id=0, seed=int(seed), dataset="amazon", num_workers=8,
        train_size=0.4, val_size=0.2, multirun=1, run_best=False,
        data_dir=str(raw_root), best_model_path=str(ROOT/"checkpoints"),
    )

def train_one(config, seed, train_loader, val_loader, checkpoint_path, epoch_csv, prefix):
    seed_all(seed)
    args = make_args(seed)
    T = Trainer(config=config, args=args, logger=NullLogger())
    model, optimizer, loss_func, scheduler = T.init(helper)

    pg = optimizer.param_groups[0]
    assert tuple(pg["betas"]) == (0.9,0.999)
    assert abs(float(pg["eps"])-1e-8) < 1e-15
    assert abs(float(pg["lr"])-float(config["lr"])) < 1e-15
    assert abs(float(pg.get("weight_decay",0.0))-float(config["weight_decay"])) < 1e-20

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    best=-float("inf")
    best_epoch=-1
    patience=0
    wall0=time.perf_counter()

    for epoch in range(1, MAX_EPOCHS+1):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        model, loss = T.train(epoch-1, model, loss_func, optimizer, train_loader, helper)
        torch.cuda.synchronize()
        train_s=time.perf_counter()-t0
        avg_loss=float(loss.detach().item())/max(1,len(train_loader))

        torch.cuda.synchronize()
        v0=time.perf_counter()
        y,p,_=T.evaluation(helper, val_loader, model, threshold_moving=True, thres=0.5)
        torch.cuda.synchronize()
        val_s=time.perf_counter()-v0
        ap=float(average_precision_score(y,p))
        auc=float(roc_auc_score(y,p))

        improved = ap > best + 1e-12
        if improved:
            best=ap; best_epoch=epoch; patience=0
            save_state_cpu(model, checkpoint_path, {
                "epoch":epoch, "val_auprc":ap, "val_auroc":auc,
                "seed":int(seed), "lr":float(config["lr"]),
                "weight_decay":float(config["weight_decay"]),
            })
        else:
            patience += 1

        rows.append({
            "epoch":epoch, "avg_train_loss":avg_loss,
            "train_seconds":train_s, "validation_seconds":val_s,
            "val_auprc":ap, "val_auroc":auc,
            "best_val_auprc_so_far":best,
            "best_epoch_so_far":best_epoch,
            "patience_count":patience,
            "peak_gpu_memory_mb_so_far":float(torch.cuda.max_memory_allocated()/1024**2),
        })
        write_csv(epoch_csv, rows)

        if epoch == 1 or epoch % 10 == 0 or improved or patience >= PATIENCE:
            print(
                f"{prefix} epoch={epoch:03d} loss={avg_loss:.6f} "
                f"val_AUPRC={ap:.6f} best={best:.6f}({best_epoch}) "
                f"patience={patience}/{PATIENCE} train={train_s:.3f}s val={val_s:.3f}s",
                flush=True,
            )
        if patience >= PATIENCE:
            break

    wall=time.perf_counter()-wall0
    peak=float(torch.cuda.max_memory_allocated()/1024**2)
    load_state(model, checkpoint_path)
    return T, model, optimizer, loss_func, scheduler, rows, best, best_epoch, wall, peak

# ---------------------------------------------------------------------
# EXACT ONE-EPOCH UNIFIED SMOKE
# ---------------------------------------------------------------------
smoke_json = SMOKE_DIR/"amazon_tr40_seed2_1epoch_smoke.json"
if not smoke_json.exists():
    print("\n===== EXACT 1-EPOCH AMAZON SMOKE =====", flush=True)
    tr,va,te,train_loader,val_loader,test_loader = apply_split("TR40")
    seed_all(2)
    cfg=copy.deepcopy(base)
    cfg.update({"lr":0.01,"weight_decay":0.0,"epochs":1,"patience":0,"seed":2})
    T=Trainer(config=cfg,args=make_args(2),logger=NullLogger())
    model,opt,loss_func,scheduler=T.init(helper)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize(); t0=time.perf_counter()
    model,loss=T.train(0,model,loss_func,opt,train_loader,helper)
    torch.cuda.synchronize(); train_s=time.perf_counter()-t0

    torch.cuda.synchronize(); v0=time.perf_counter()
    y,p,_=T.evaluation(helper,val_loader,model,threshold_moving=True,thres=0.5)
    torch.cuda.synchronize(); val_s=time.perf_counter()-v0

    obj={
        "status":"PASS","dataset":"Amazon","ratio":"TR40","train_seed":2,
        "epochs":1,"train_nodes":len(tr),"val_nodes":len(va),"test_nodes_registered":len(te),
        "val_auprc":float(average_precision_score(y,p)),
        "val_auroc":float(roc_auc_score(y,p)),
        "train_seconds":train_s,"validation_seconds":val_s,
        "peak_gpu_memory_mb":float(torch.cuda.max_memory_allocated()/1024**2),
        "test_evaluated":False,"train_only_partition_labels":True,
        "repository_commit":EXPECTED_COMMIT,"dataset_sha256":EXPECTED_DATA_SHA,
        "split_sha256":EXPECTED_SPLIT_SHA,
    }
    smoke_json.write_text(json.dumps(obj,indent=2))
    print("PMP_AMAZON_1EPOCH_SMOKE=PASS", flush=True)
    print("Smoke val AUPRC:",obj["val_auprc"],"AUROC:",obj["val_auroc"],flush=True)
    print("TEST_EVALUATED=NO",flush=True)
    del model,opt,loss_func,scheduler,T,train_loader,val_loader,test_loader
    gc.collect(); torch.cuda.empty_cache()
else:
    old=json.loads(smoke_json.read_text())
    assert old.get("status")=="PASS" and old.get("test_evaluated") is False
    print("\nPMP_AMAZON_1EPOCH_SMOKE=PASS (existing evidence reused)",flush=True)

# ---------------------------------------------------------------------
# TR40 TUNING — 12 LR × WD TRIALS
# ---------------------------------------------------------------------
print("\n===== AMAZON TR40 TUNING — 12 TRIALS =====", flush=True)
print("Frozen author hidden/dropout: 256 / 0.6", flush=True)
print("Test evaluation during tuning: FORBIDDEN", flush=True)

tr,va,te,tune_train_loader,tune_val_loader,tune_test_loader = apply_split("TR40")
trial_summaries=[]

for trial in TRIALS:
    tid=trial["trial_id"]
    td=TRIALS_DIR/f"trial_{tid:02d}"
    td.mkdir(parents=True,exist_ok=True)
    summary_path=td/"summary.json"
    epoch_csv=td/"epochs.csv"
    ckpt=td/"best_validation_auprc.pth"

    if summary_path.exists():
        old=json.loads(summary_path.read_text())
        if (
            old.get("status")=="COMPLETE"
            and int(old.get("trial_id",-1))==tid
            and abs(float(old["lr"])-trial["lr"])<1e-15
            and abs(float(old["weight_decay"])-trial["weight_decay"])<1e-20
            and old.get("test_evaluated") is False
            and ckpt.exists()
        ):
            trial_summaries.append(old)
            print(f"trial={tid:02d} RESUME_COMPLETE val_AUPRC={old['best_val_auprc']:.6f}",flush=True)
            continue

    for pth in [summary_path,epoch_csv,ckpt]:
        if pth.exists(): pth.unlink()

    cfg=copy.deepcopy(base)
    cfg.update({
        "lr":float(trial["lr"]),
        "weight_decay":float(trial["weight_decay"]),
        "hid_dim":256,"dropout":0.6,
        "epochs":MAX_EPOCHS,"patience":PATIENCE,"seed":2,
    })
    prefix=f"trial={tid:02d}"
    T,model,opt,loss_func,scheduler,rows,best,best_epoch,wall,peak = train_one(
        cfg,2,tune_train_loader,tune_val_loader,ckpt,epoch_csv,prefix
    )
    obj={
        "status":"COMPLETE","trial_id":tid,
        "lr":float(trial["lr"]),"weight_decay":float(trial["weight_decay"]),
        "hid_dim":256,"dropout":0.6,
        "best_epoch":int(best_epoch),"best_val_auprc":float(best),
        "completed_epochs":len(rows),"training_wall_seconds":wall,
        "total_train_seconds":float(sum(x["train_seconds"] for x in rows)),
        "total_validation_seconds":float(sum(x["validation_seconds"] for x in rows)),
        "peak_gpu_memory_mb":peak,
        "checkpoint":str(ckpt),"checkpoint_sha256":sha256_file(ckpt),
        "epoch_csv":str(epoch_csv),"epoch_csv_sha256":sha256_file(epoch_csv),
        "test_evaluated":False,
        "repository_commit":EXPECTED_COMMIT,"dataset_sha256":EXPECTED_DATA_SHA,
        "split_sha256":EXPECTED_SPLIT_SHA,
    }
    summary_path.write_text(json.dumps(obj,indent=2))
    trial_summaries.append(obj)
    print(
        f"TRIAL {tid:02d} DONE | lr={trial['lr']} wd={trial['weight_decay']} "
        f"| epoch={best_epoch} | val AUPRC={best:.6f}",flush=True
    )
    del model,opt,loss_func,scheduler,T
    gc.collect(); torch.cuda.empty_cache()

assert len(trial_summaries)==12
winner=sorted(trial_summaries,key=lambda x:(-x["best_val_auprc"],x["trial_id"]))[0]
frozen_path=TUNING_DIR/"PMP_AMAZON_FROZEN_WINNER.json"
frozen={
    "status":"FROZEN","dataset":"Amazon","winner_trial":int(winner["trial_id"]),
    "lr":float(winner["lr"]),"weight_decay":float(winner["weight_decay"]),
    "hid_dim":256,"dropout":0.6,"batch_size":128,"n_layer":1,
    "homo":True,"full_neighbors":True,"relation_agg":"cat","resi":0.2,
    "best_epoch":int(winner["best_epoch"]),
    "best_val_auprc":float(winner["best_val_auprc"]),
    "repository_commit":EXPECTED_COMMIT,"dataset_sha256":EXPECTED_DATA_SHA,
    "split_sha256":EXPECTED_SPLIT_SHA,
    "tuning_policy":"TR40 seed2; 12 LR×WD trials; author Amazon architecture/dropout fixed",
    "test_accessed_during_tuning":False,
}
frozen_path.write_text(json.dumps(frozen,indent=2))

tuning_rows=sorted(trial_summaries,key=lambda x:-x["best_val_auprc"])
write_csv(TUNING_DIR/"PMP_AMAZON_TR40_tuning_summary.csv",[
    {
        "trial":r["trial_id"],"lr":r["lr"],"weight_decay":r["weight_decay"],
        "best_epoch":r["best_epoch"],"best_val_auprc":r["best_val_auprc"],
        "completed_epochs":r["completed_epochs"],
    } for r in tuning_rows
])

print("\n===== PMP AMAZON TR40 TUNING RESULTS =====",flush=True)
for r in tuning_rows:
    print(
        f"trial={r['trial_id']:02d} | lr={r['lr']:.4g} | wd={r['weight_decay']:.4g} "
        f"| epoch={r['best_epoch']:3d} | val AUPRC={r['best_val_auprc']:.6f}",
        flush=True,
    )
print("===== FROZEN WINNER =====",flush=True)
print("winner trial   :",frozen["winner_trial"],flush=True)
print("learning rate  :",frozen["lr"],flush=True)
print("weight decay   :",frozen["weight_decay"],flush=True)
print("hidden/dropout : 256 / 0.6 (official Amazon settings)",flush=True)
print("best epoch     :",frozen["best_epoch"],flush=True)
print("best val AUPRC :",frozen["best_val_auprc"],flush=True)
print("TEST_SET_ACCESSED_DURING_TUNING=NO",flush=True)
print("PMP_AMAZON_TR40_TUNING=PASS",flush=True)

# Tuning loaders no longer needed.
del tune_train_loader,tune_val_loader,tune_test_loader
gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------
# FINAL 12 FRESH RUNS
# ---------------------------------------------------------------------
winner_sha=sha256_file(frozen_path)
all_runs=[]

for ratio in RATIOS:
    for seed in FINAL_SEEDS:
        run_id=f"amazon_{ratio.lower()}_seed{seed}"
        sp=RUNS_DIR/f"{run_id}_summary.json"
        epcsv=RUNS_DIR/f"{run_id}_epochs.csv"
        thcsv=RUNS_DIR/f"{run_id}_val_threshold_grid.csv"
        ckpt=CKPT_DIR/f"{run_id}_best_val_auprc.pth"

        if sp.exists():
            old=json.loads(sp.read_text())
            if (
                old.get("status")=="PASS"
                and old.get("run_id")==run_id
                and old.get("winner_sha256")==winner_sha
                and old.get("test_evaluated_once") is True
                and ckpt.exists()
            ):
                print(f"\n{run_id} RESUME_PASS",flush=True)
                all_runs.append(old)
                continue

        for pth in [sp,epcsv,thcsv,ckpt]:
            if pth.exists(): pth.unlink()

        print(f"\n===== FINAL {run_id} =====",flush=True)
        seed_all(seed)
        tr,va,te,train_loader,val_loader,test_loader = apply_split(ratio)

        cfg=copy.deepcopy(base)
        cfg.update({
            "seed":int(seed),
            "lr":float(frozen["lr"]),
            "weight_decay":float(frozen["weight_decay"]),
            "hid_dim":256,"dropout":0.6,
            "epochs":MAX_EPOCHS,"patience":PATIENCE,
        })

        T,model,opt,loss_func,scheduler,rows,best,best_epoch,wall,peak_trainval = train_one(
            cfg,seed,train_loader,val_loader,ckpt,epcsv,run_id
        )

        # Frozen checkpoint => threshold selection on validation.
        torch.cuda.synchronize(); v0=time.perf_counter()
        yv,pv,_=T.evaluation(helper,val_loader,model,threshold_moving=True,thres=0.5)
        torch.cuda.synchronize(); frozen_val_s=time.perf_counter()-v0
        choice,grid=choose_threshold(yv,pv)
        write_csv(thcsv,grid)
        threshold=float(choice["threshold"])
        val_m=metrics(yv,pv,threshold)

        # Exactly one final test evaluation for this run.
        torch.cuda.synchronize(); t0=time.perf_counter()
        yt,pt,_=T.evaluation(helper,test_loader,model,threshold_moving=True,thres=0.5)
        torch.cuda.synchronize(); test_s=time.perf_counter()-t0
        test_m=metrics(yt,pt,threshold)
        peak=float(torch.cuda.max_memory_allocated()/1024**2)

        train_times=[x["train_seconds"] for x in rows]
        val_times=[x["validation_seconds"] for x in rows]
        tmean,tsd=mean_sd(train_times)
        vmean,vsd=mean_sd(val_times)

        obj={
            "status":"PASS","run_id":run_id,"dataset":"Amazon","model":"PMP / LA-SAGE-S",
            "ratio":ratio,"train_seed":int(seed),
            "train_nodes":len(tr),"validation_nodes":len(va),"test_nodes":len(te),
            "train_fraud":int(helper.labels[tr].sum().item()),
            "validation_fraud":int(helper.labels[va].sum().item()),
            "test_fraud":int(helper.labels[te].sum().item()),
            "repository_commit":EXPECTED_COMMIT,"dataset_sha256":EXPECTED_DATA_SHA,
            "split_sha256":EXPECTED_SPLIT_SHA,"winner_sha256":winner_sha,
            "winner_trial":int(frozen["winner_trial"]),
            "lr":float(frozen["lr"]),"weight_decay":float(frozen["weight_decay"]),
            "hid_dim":256,"dropout":0.6,"batch_size":128,"n_layer":1,
            "homo":True,"full_neighbors":True,
            "max_epochs":MAX_EPOCHS,"patience":PATIENCE,
            "epochs_completed":len(rows),"best_epoch":int(best_epoch),
            "best_validation_auprc":float(best),
            "selected_threshold":threshold,
            "threshold_rule":"validation Macro-F1 max; tie fraud recall; then closeness to 0.5",
            "validation_metrics_at_selected_threshold":val_m,
            "test_metrics":test_m,
            "test_evaluated_once":True,"test_used_for_selection":False,
            "train_seconds_total":float(sum(train_times)),
            "train_seconds_mean":tmean,"train_seconds_sd":tsd,
            "validation_seconds_total_during_training":float(sum(val_times)),
            "validation_seconds_mean":vmean,"validation_seconds_sd":vsd,
            "frozen_validation_inference_seconds":frozen_val_s,
            "test_inference_seconds":test_s,
            "training_wall_seconds":wall,
            "peak_trainval_gpu_memory_mb":peak_trainval,
            "peak_overall_gpu_memory_mb":peak,
            "checkpoint_path":str(ckpt),"checkpoint_sha256":sha256_file(ckpt),
            "epoch_csv":str(epcsv),"epoch_csv_sha256":sha256_file(epcsv),
            "threshold_csv":str(thcsv),"threshold_csv_sha256":sha256_file(thcsv),
        }
        sp.write_text(json.dumps(obj,indent=2))
        all_runs.append(obj)

        print(
            f"FINAL {run_id} PASS | AUPRC={test_m['auprc']:.6f} "
            f"AUROC={test_m['auroc']:.6f} MacroF1={test_m['macro_f1']:.6f} "
            f"Recall={test_m['fraud_recall']:.6f}",flush=True
        )
        del model,opt,loss_func,scheduler,T,train_loader,val_loader,test_loader
        gc.collect(); torch.cuda.empty_cache()

assert len(all_runs)==12
by={(r["ratio"],int(r["train_seed"])):r for r in all_runs}
assert len(by)==12

flat=[]
for ratio in RATIOS:
    for seed in FINAL_SEEDS:
        r=by[(ratio,seed)]
        m=r["test_metrics"]
        flat.append({
            "dataset":"Amazon","model":"PMP","ratio":ratio,"seed":seed,
            "train_nodes":r["train_nodes"],"epochs_completed":r["epochs_completed"],
            "best_epoch":r["best_epoch"],"best_validation_auprc":r["best_validation_auprc"],
            "threshold":r["selected_threshold"],
            "auprc":m["auprc"],"auroc":m["auroc"],"macro_f1":m["macro_f1"],
            "fraud_precision":m["fraud_precision"],"fraud_recall":m["fraud_recall"],
            "fraud_f1":m["fraud_f1"],"gmean":m["gmean"],
            "train_seconds_total":r["train_seconds_total"],
            "train_seconds_mean":r["train_seconds_mean"],
            "test_inference_seconds":r["test_inference_seconds"],
            "peak_gpu_memory_mb":r["peak_overall_gpu_memory_mb"],
        })
write_csv(FINAL_DIR/"PMP_AMAZON_FINAL_12_RUNS.csv",flat)

metric_names=["auprc","auroc","macro_f1","fraud_precision","fraud_recall","fraud_f1","gmean"]
ratio_rows=[]
ratio_json={}
for ratio in RATIOS:
    rr=[x for x in flat if x["ratio"]==ratio]
    row={"ratio":ratio,"n_seeds":3}
    rj={}
    for metric in metric_names:
        mean,sd=mean_sd([x[metric] for x in rr])
        row[f"{metric}_mean"]=mean; row[f"{metric}_sd"]=sd
        rj[metric]={"mean":mean,"sd":sd}
    tm,ts=mean_sd([x["train_seconds_total"] for x in rr])
    im,is_=mean_sd([x["test_inference_seconds"] for x in rr])
    mm,ms=mean_sd([x["peak_gpu_memory_mb"] for x in rr])
    row["total_train_seconds_mean"]=tm; row["total_train_seconds_sd"]=ts
    row["test_inference_seconds_mean"]=im; row["test_inference_seconds_sd"]=is_
    row["peak_gpu_memory_mb_mean"]=mm; row["peak_gpu_memory_mb_sd"]=ms
    ratio_rows.append(row)
    ratio_json[ratio]={"metrics":rj,"total_train_seconds":{"mean":tm,"sd":ts},
                       "test_inference_seconds":{"mean":im,"sd":is_},
                       "peak_gpu_memory_mb":{"mean":mm,"sd":ms}}
write_csv(FINAL_DIR/"PMP_AMAZON_FINAL_RATIO_SUMMARY.csv",ratio_rows)

final_obj={
    "status":"PASS","dataset":"Amazon","model":"PMP / LA-SAGE-S",
    "final_runs_complete":12,"expected_final_runs":12,
    "ratios":RATIOS,"seeds":FINAL_SEEDS,
    "repository_commit":EXPECTED_COMMIT,"dataset_sha256":EXPECTED_DATA_SHA,
    "split_sha256":EXPECTED_SPLIT_SHA,"winner_sha256":winner_sha,
    "frozen_config":frozen,"ratio_summary":ratio_json,"gpu":GPU,
    "timing_note":"Cross-host runtime comparison remains subject to the project hardware-calibration gate.",
}
(FINAL_DIR/"PMP_AMAZON_FINAL_SUMMARY.json").write_text(json.dumps(final_obj,indent=2))
(FINAL_DIR/"PMP_AMAZON_FINAL_STATUS.txt").write_text(
    "PMP_AMAZON_FINAL=PASS\n"
    "PMP_AMAZON_FINAL_RUNS=12/12\n"
    "TEST_ISOLATION=PASS\n"
    f"WINNER_TRIAL={frozen['winner_trial']}\n"
    f"LR={frozen['lr']}\n"
    f"WEIGHT_DECAY={frozen['weight_decay']}\n"
)

assert git("status","--porcelain") == ""

print("\n===== PMP AMAZON FINAL RESULTS — MEAN ± SD =====",flush=True)
for row in ratio_rows:
    print(
        f"{row['ratio']} | "
        f"AUPRC={row['auprc_mean']:.6f}±{row['auprc_sd']:.6f} | "
        f"AUROC={row['auroc_mean']:.6f}±{row['auroc_sd']:.6f} | "
        f"MacroF1={row['macro_f1_mean']:.6f}±{row['macro_f1_sd']:.6f} | "
        f"Prec={row['fraud_precision_mean']:.6f}±{row['fraud_precision_sd']:.6f} | "
        f"Recall={row['fraud_recall_mean']:.6f}±{row['fraud_recall_sd']:.6f} | "
        f"FraudF1={row['fraud_f1_mean']:.6f}±{row['fraud_f1_sd']:.6f} | "
        f"GMean={row['gmean_mean']:.6f}±{row['gmean_sd']:.6f}",
        flush=True
    )

print("\n===== FINAL GATE =====",flush=True)
print("PMP_AMAZON_1EPOCH_SMOKE=PASS",flush=True)
print("PMP_AMAZON_TR40_TUNING=PASS",flush=True)
print("PMP_AMAZON_FINAL=PASS",flush=True)
print("PMP_AMAZON_FINAL_RUNS=12/12",flush=True)
print("TEST_ISOLATION=PASS",flush=True)
print("PMP_AUTHOR_REPO_CLEAN=PASS",flush=True)
