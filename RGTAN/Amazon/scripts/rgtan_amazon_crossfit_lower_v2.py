import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ARGP = argparse.ArgumentParser()
ARGP.add_argument("--ratio", choices=["TR30", "TR20", "TR10"], required=True)
RATIO = ARGP.parse_args().ratio
ROOT = Path("/workspace/comp8851")
OUT = ROOT / "assets/rgtan/amazon"
MAT = Path("/root/.dgl/amazon_1e446145/Amazon.mat")
SPLIT = Path(
    "/workspace/uploads/ass2_split_candidates/"
    "Amazon/amazon_seed2_nested_splits.npz"
)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

MAT_SHA = "4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63"
SPLIT_SHA = "0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd"

assert sha256(MAT) == MAT_SHA
assert sha256(SPLIT) == SPLIT_SHA

torch.backends.cuda.matmul.allow_tf32 = False

print("SOURCE_HASHES=PASS", flush=True)
start = time.perf_counter()

with np.load(SPLIT, allow_pickle=False) as z:
    train = np.asarray(z[RATIO], dtype=np.int64).copy()
    canonical_tr40 = np.asarray(z["TR40"], dtype=np.int64).copy()
    val = np.asarray(z["val"], dtype=np.int64).copy()
    test = np.asarray(z["test"], dtype=np.int64).copy()
    assert int(z["seed"].item()) == 2

assert len(train) == {"TR30":2591,"TR20":1727,"TR10":863}[RATIO], (RATIO,len(train))
assert (len(val),len(test)) == (1728,3456)
assert len(np.unique(np.concatenate([train,val,test]))) == len(train)+len(val)+len(test)
assert np.isin(train,canonical_tr40).all()
assert np.intersect1d(train,val).size==0 and np.intersect1d(train,test).size==0 and np.intersect1d(val,test).size==0
assert min(train.min(),val.min(),test.min()) >= 3305

print(f"EXACT_ASS2_{RATIO}_SPLIT=PASS", flush=True)

print("LOADING ORIGINAL AMAZON FEATURES AND GRAPH", flush=True)

data = loadmat(MAT, variable_names=["homo","features","label"])
n = 11944

adj = sp.csr_matrix(data["homo"], dtype=np.float32)
adj = (adj + adj.T).tocsr()
adj.sum_duplicates()
adj.data[:] = 1.0
adj.setdiag(1)
adj.eliminate_zeros()
adj.sort_indices()

assert adj.shape == (n,n)
assert adj.nnz == 8808728
assert (adj != adj.T).nnz == 0

raw_features = data["features"]
if sp.issparse(raw_features):
    features = raw_features.toarray().astype(np.float32)
else:
    features = np.asarray(raw_features, dtype=np.float32)

assert features.shape == (n,25)
assert np.isfinite(features).all()

# Read only the permitted training labels into the adapter.
train_y = np.asarray(data["label"]).reshape(-1)[train].astype(np.int8).copy()
assert set(np.unique(train_y).tolist()) == {0,1}
assert int((train_y == 1).sum()) >= 5

del data, raw_features

print("AUTHOR_GRAPH_RECONSTRUCTION=PASS", flush=True)
print("TRAIN_LABELS_ONLY=PASS", flush=True)

# Five stratified out-of-fold training contexts.
# View j excludes ALL labels belonging to fold j.
fold_of_train = np.full(n, -1, dtype=np.int8)

skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=2
)

for j, (_, heldout_positions) in enumerate(
    skf.split(np.zeros(len(train)), train_y)
):
    fold_of_train[train[heldout_positions]] = j

assert np.all(fold_of_train[train] >= 0)

# Five training contexts + one full-current-ratio validation context.
num_views = 6
positive = np.zeros((n,num_views), dtype=np.float32)
context = np.full((num_views,n), 2, dtype=np.int8)

for j in range(5):
    usable = fold_of_train[train] != j
    ids = train[usable]
    y = train_y[usable]
    context[j,ids] = y
    positive[ids,j] = (y == 1).astype(np.float32)

context[5,train] = train_y
positive[train,5] = (train_y == 1).astype(np.float32)

for j in range(num_views):
    assert np.all(context[j,val] == 2)
    assert np.all(context[j,test] == 2)
    assert np.all(context[j,:3305] == 2)

for j in range(5):
    heldout = train[fold_of_train[train] == j]
    assert np.all(context[j,heldout] == 2)
    assert np.all(positive[heldout,j] == 0)

print("FIVE_OUT_OF_FOLD_CONTEXTS=PASS", flush=True)
print("VALIDATION_TEST_EXCLUDED_FROM_RISK_SOURCES=PASS", flush=True)

# Exact structural neighbourhoods:
# 1-hop = adjacent nodes, including the author's self-loops.
# 2-hop = reachable by two steps, excluding 1-hop nodes.
#
# FP16 is used only for positive/zero graph reachability.
# All feature sums use FP32. Maximum path multiplicity is
# bounded by the graph's 11,944 nodes, below FP16 overflow.

print("COMPUTING EXACT TWO-HOP REACHABILITY ON GPU", flush=True)

dense_half = torch.from_numpy(
    adj.toarray().astype(np.float16)
).to("cuda:0")

twohop = (dense_half @ dense_half) > 0
twohop &= (dense_half == 0)

assert not torch.any(torch.diagonal(twohop))

degrees = np.diff(adj.indptr).astype(np.float32)
degree_gpu = torch.from_numpy(degrees).to("cuda:0")

onehop_f = dense_half.float()
twohop_f = twohop.float()

positive_gpu = torch.from_numpy(positive).to("cuda:0")

risk = onehop_f @ positive_gpu

onehop_degree = (onehop_f @ degree_gpu).cpu().numpy()
twohop_degree = (twohop_f @ degree_gpu).cpu().numpy()

onehop_risk_aggregate = (
    onehop_f @ risk
).cpu().numpy()

twohop_risk_aggregate = (
    twohop_f @ risk
).cpu().numpy()

risk_np = risk.cpu().numpy()
del dense_half, onehop_f, twohop, twohop_f, positive_gpu, risk
torch.cuda.empty_cache()

print("EXACT_ONE_AND_TWO_HOP_FEATURES_COMPUTED", flush=True)

# Independently check exact neighbourhood semantics
# against direct CSR traversal for three fixed nodes.
for node in (3305, 6000, 11943):
    row = adj.getrow(node)
    first = set(row.indices.tolist())
    second = set()
    for neighbor in first:
        second.update(
            adj.indices[
                adj.indptr[neighbor]:adj.indptr[neighbor+1]
            ].tolist()
        )
    second.difference_update(first)

    assert np.isclose(
        onehop_degree[node],
        degrees[list(first)].sum(),
        rtol=1e-5, atol=1e-2
    )
    assert np.isclose(
        twohop_degree[node],
        degrees[list(second)].sum(),
        rtol=1e-5, atol=1e-2
    )

print("INDEPENDENT_TWO_HOP_REFERENCE_CHECK=PASS", flush=True)

columns = np.array([
    "degree",
    "riskstat",
    "1hop_degree",
    "2hop_degree",
    "1hop_riskstat",
    "2hop_riskstat",
])

raw_views = []
for j in range(num_views):
    view = np.column_stack([
        degrees,
        risk_np[:,j],
        onehop_degree,
        twohop_degree,
        onehop_risk_aggregate[:,j],
        twohop_risk_aggregate[:,j],
    ]).astype(np.float32)
    assert np.isfinite(view).all()
    raw_views.append(view)

# Fit normalization on OOF training features only.
# Validation/test feature distributions are not used.
oof = np.zeros((len(train),6), dtype=np.float32)

for j in range(5):
    selected = fold_of_train[train] == j
    oof[selected] = raw_views[j][train[selected]]

scaler = StandardScaler().fit(oof)

views = [
    scaler.transform(view).astype(np.float32)
    for view in raw_views
]

assert all(np.isfinite(v).all() for v in views)

# Every training target's own fold is excluded from
# the complete risk-statistic source set, not just
# from its node-level risk count.
for j in range(5):
    heldout = train[fold_of_train[train] == j]
    assert np.all(context[j,heldout] == 2)

print("TRAIN_ONLY_SCALER=PASS", flush=True)
print("TARGET_FOLD_LABEL_EXCLUSION=PASS", flush=True)

artifact = OUT / f"amazon_{RATIO.lower()}_crossfit.npz"

payload = {
    "train_ids": train,
    "val_ids": val,
    "train_labels": train_y,
    "fold_of_node": fold_of_train,
    "features": features,
    "risk_feature_names": columns,
    "scaler_mean": scaler.mean_.astype(np.float64),
    "scaler_scale": scaler.scale_.astype(np.float64),
}

for j in range(num_views):
    payload[f"risk_view_{j}"] = views[j]
    payload[f"label_context_{j}"] = context[j]

assert not artifact.exists()
np.savez_compressed(artifact, **payload)

elapsed = time.perf_counter()-start

record = {
    "status": "RGTAN_CROSSFIT_ADAPTER_PASS",
    "model": "RGTAN",
    "dataset": "Amazon",
    "ratio": RATIO,
    "split_seed": 2,
    "source_commit": "65e583f03ac5c5983e49cc35179fe7a65908b9c1",
    "source_sha256": MAT_SHA,
    "split_sha256": SPLIT_SHA,
    "nodes": n,
    "edges": int(adj.nnz),
    "train_count": len(train),
    "train_positive": int((train_y==1).sum()),
    "fold_seed": 2,
    "derived_from": "I2 TR40 working crossfit builder; unchanged graph/risk/scaler mathematics",
    "val_count": len(val),
    "test_count": len(test),
    "risk_feature_count": 6,
    "training_contexts": 5,
    "validation_contexts": 1,
    "fold_labels_masked_from_their_own_risk_views": True,
    "validation_test_labels_used_in_features": False,
    "normalization_fit_on_oof_training_only": True,
    "full_model_forward_tested": False,
    "training_started": False,
    "official_benchmark_result": False,
    "adapter_seconds": round(elapsed,3),
    "artifact_sha256": sha256(artifact),
}

report = OUT / f"amazon_{RATIO.lower()}_crossfit_manifest.json"
assert not report.exists()
report.write_text(json.dumps(record,indent=2)+"\n")

print("\n========== FINAL ADAPTER STATUS ==========", flush=True)
print(json.dumps(record,indent=2), flush=True)
print("RGTAN_CROSSFIT_ADAPTER_PASS", flush=True)
print("TRAINING_STARTED=NO", flush=True)
print("TEST_EVALUATED=NO", flush=True)
