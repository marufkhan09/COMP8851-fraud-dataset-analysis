#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ============================================================
# FIXED PATHS / PROTOCOL
# ============================================================

WORK = Path("/workspace/pmp_vast")
REPO = WORK / "repo/PMP"

SMOKE = (
    WORK /
    "scripts/08_pmp_elliptic_2epoch_smoke.py"
)

RAW = WORK / "shared/elliptic/raw"

FEATURES = RAW / "elliptic_txs_features.csv"
CLASSES = RAW / "elliptic_txs_classes.csv"
EDGES = RAW / "elliptic_txs_edgelist.csv"

ROOT = WORK / "unified/elliptic/final"
RUN_DIR = ROOT / "runs"
CKPT_DIR = ROOT / "checkpoints"

for p in (
    ROOT,
    RUN_DIR,
    CKPT_DIR,
):
    p.mkdir(
        parents=True,
        exist_ok=True,
    )

FROZEN_CONFIG = (
    ROOT /
    "PMP_ELLIPTIC_FROZEN_CONFIG.json"
)

FINAL_RUNS_CSV = (
    ROOT /
    "PMP_ELLIPTIC_FINAL_12_RUNS.csv"
)

RATIO_SUMMARY_CSV = (
    ROOT /
    "PMP_ELLIPTIC_FINAL_RATIO_SUMMARY.csv"
)

FINAL_SUMMARY_JSON = (
    ROOT /
    "PMP_ELLIPTIC_FINAL_SUMMARY.json"
)

STATUS_TXT = (
    ROOT /
    "PMP_ELLIPTIC_FINAL_STATUS.txt"
)

ARCHIVE = Path(
    "/workspace/"
    "PMP_ELLIPTIC_A6000_FINAL_EVIDENCE_20260917.tar.gz"
)

ARCHIVE_SHA_TXT = (
    ROOT /
    "PMP_ELLIPTIC_FINAL_ARCHIVE_SHA256.txt"
)

EXPECTED_COMMIT = (
    "3f7629f6c180891a0bc1bba3c66d94d288a1ddae"
)

EXPECTED_FEATURES_SHA = (
    "fd7f83573443c9e302e371d3f110e3b"
    "6224160f5d1ed8a287757936127800ff0"
)

EXPECTED_CLASSES_SHA = (
    "93e2e7b2405c735ba752bf6ba06b9475"
    "61deddd1f5a8fc91e46f6a4c0e439493"
)

EXPECTED_EDGES_SHA = (
    "a35053ba68a98e4382cae2ba65b9d9e3"
    "6b23b6439e02dff084971b1b72a5156e"
)

EXPECTED_SPLIT_SHA = (
    "1e7963e9d09935fb33e0df74cfb786ca"
    "b826d1b95ae10b9592dbc240b48ffdbc"
)

RATIOS = [
    "TR40",
    "TR30",
    "TR20",
    "TR10",
]

SEEDS = [
    2,
    42,
    72,
]

EXPECTED_SIZES = {
    "TR40": 18889,
    "TR30": 13670,
    "TR20": 9553,
    "TR10": 4543,
    "val": 8726,
    "test": 18949,
}

MAX_EPOCHS = 100
PATIENCE = 20

THRESHOLDS = np.round(
    np.arange(
        0.01,
        1.00,
        0.01,
    ),
    2,
)

# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def git(*args):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            *args,
        ],
        text=True,
    ).strip()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(rows)


def save_state_cpu(
    model,
    path,
    meta,
):
    torch.save(
        {
            "state_dict": {
                k: v.detach().cpu()
                for k, v
                in model.state_dict().items()
            },
            "meta": meta,
        },
        path,
    )


def load_state(
    model,
    path,
):
    obj = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        obj["state_dict"]
    )

    return obj.get(
        "meta",
        {},
    )


def mean_sd(vals):
    a = np.asarray(
        vals,
        dtype=float,
    )

    mean = float(
        a.mean()
    )

    sd = (
        float(
            a.std(
                ddof=1
            )
        )
        if len(a) > 1
        else 0.0
    )

    return mean, sd


def choose_threshold(
    y,
    prob,
):
    y = np.asarray(
        y
    ).astype(int)

    prob = np.asarray(
        prob,
        dtype=float,
    )

    rows = []

    for threshold in THRESHOLDS:

        pred = (
            prob >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold":
                    float(threshold),

                "macro_f1":
                    float(
                        f1_score(
                            y,
                            pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),

                "fraud_recall":
                    float(
                        recall_score(
                            y,
                            pred,
                            pos_label=1,
                            zero_division=0,
                        )
                    ),

                "distance_to_0_5":
                    float(
                        abs(
                            float(
                                threshold
                            ) - 0.5
                        )
                    ),
            }
        )

    rows.sort(
        key=lambda r: (
            -r["macro_f1"],
            -r["fraud_recall"],
            r["distance_to_0_5"],
            r["threshold"],
        )
    )

    return (
        rows[0],
        rows,
    )


def metrics(
    y,
    prob,
    threshold,
):
    y = np.asarray(
        y
    ).astype(int)

    prob = np.asarray(
        prob,
        dtype=float,
    )

    pred = (
        prob >= float(
            threshold
        )
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[
                0,
                1,
            ],
        ).ravel()
    )

    tpr = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    tnr = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    return {
        "auprc":
            float(
                average_precision_score(
                    y,
                    prob,
                )
            ),

        "auroc":
            float(
                roc_auc_score(
                    y,
                    prob,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "fraud_precision":
            float(
                precision_score(
                    y,
                    pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "fraud_recall":
            float(
                recall_score(
                    y,
                    pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "fraud_f1":
            float(
                f1_score(
                    y,
                    pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "gmean":
            float(
                math.sqrt(
                    tpr * tnr
                )
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# START
# ============================================================

print(
    "============================================================",
    flush=True,
)

print(
    " PMP × ELLIPTIC — FINAL 12-RUN UNIFIED GRID",
    flush=True,
)

print(
    "============================================================",
    flush=True,
)

print(
    "Ratios=TR40/TR30/TR20/TR10",
    flush=True,
)

print(
    "Seeds=2/42/72",
    flush=True,
)

print(
    "Generic tuning before final=NO",
    flush=True,
)

print(
    "Checkpoint selection=validation AUPRC",
    flush=True,
)

print(
    "Threshold selection=validation Macro-F1",
    flush=True,
)

print(
    "Test used for selection=NO",
    flush=True,
)

# ============================================================
# SOURCE / GPU / RAW HASH GATES
# ============================================================

assert git(
    "rev-parse",
    "HEAD",
) == EXPECTED_COMMIT

assert git(
    "status",
    "--porcelain",
    "--untracked-files=no",
) == ""

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1

GPU = torch.cuda.get_device_name(
    0
)

assert "A6000" in GPU.upper()

torch.cuda.set_device(
    0
)

torch.set_num_threads(
    8
)

assert sha256_file(
    FEATURES
) == EXPECTED_FEATURES_SHA

assert sha256_file(
    CLASSES
) == EXPECTED_CLASSES_SHA

assert sha256_file(
    EDGES
) == EXPECTED_EDGES_SHA

print(
    "SOURCE_GPU_RAW_HASH_GATE=PASS",
    flush=True,
)

print(
    "GPU=" + GPU,
    flush=True,
)

# ============================================================
# FIND THE EXACT FROZEN SPLIT ARTIFACT BY HASH
# Do not guess based on filename.
# ============================================================

split_matches = []

for root, dirs, files in os.walk(
    WORK
):
    dirs[:] = [
        d
        for d in dirs
        if d not in {
            ".git",
            "envs",
            "repo",
            "__pycache__",
        }
    ]

    for name in files:

        if not name.endswith(
            ".npz"
        ):
            continue

        path = (
            Path(root) /
            name
        )

        try:
            if (
                sha256_file(path)
                == EXPECTED_SPLIT_SHA
            ):
                split_matches.append(
                    path
                )
        except Exception:
            pass


assert len(
    split_matches
) >= 1, (
    "Exact frozen Elliptic split artifact "
    "with required SHA256 was not found."
)

SPLIT = sorted(
    split_matches,
    key=lambda p: str(p),
)[0]

assert (
    sha256_file(
        SPLIT
    )
    == EXPECTED_SPLIT_SHA
)

print(
    "EXACT_FROZEN_SPLIT_FOUND=" +
    str(SPLIT),
    flush=True,
)

print(
    "ELLIPTIC_SPLIT_SHA256=" +
    EXPECTED_SPLIT_SHA,
    flush=True,
)

# ============================================================
# LOAD SPLIT
# Identify partitions by exact required lengths so this does
# not depend on arbitrary NPZ key naming.
# ============================================================

z = np.load(
    SPLIT,
    allow_pickle=False,
)

arrays_by_size = {}

for key in z.files:
    arr = np.asarray(
        z[key]
    )

    if (
        arr.ndim == 1
        and arr.size
        in set(
            EXPECTED_SIZES.values()
        )
    ):
        arrays_by_size.setdefault(
            int(arr.size),
            [],
        ).append(
            (
                key,
                arr.astype(
                    np.int64
                ),
            )
        )


split = {}

for name, size in (
    EXPECTED_SIZES.items()
):

    candidates = (
        arrays_by_size.get(
            size,
            [],
        )
    )

    assert (
        len(candidates) == 1
    ), (
        name,
        size,
        [
            c[0]
            for c in candidates
        ],
        z.files,
    )

    split[name] = (
        candidates[0][1]
    )


# ============================================================
# CHRONOLOGY VALIDATION FROM RAW FEATURE FILE
# ============================================================

raw_feat = pd.read_csv(
    FEATURES,
    header=None,
)

assert raw_feat.shape == (
    203769,
    167,
)

time_steps = (
    raw_feat.iloc[:, 1]
    .astype(
        np.int64
    )
    .to_numpy()
)

for key in split:
    assert (
        split[key].min()
        >= 0
    )

    assert (
        split[key].max()
        < 203769
    )


assert (
    time_steps[
        split["TR10"]
    ].min()
    >= 1
)

assert (
    time_steps[
        split["TR10"]
    ].max()
    <= 3
)

assert (
    time_steps[
        split["TR20"]
    ].min()
    >= 1
)

assert (
    time_steps[
        split["TR20"]
    ].max()
    <= 7
)

assert (
    time_steps[
        split["TR30"]
    ].min()
    >= 1
)

assert (
    time_steps[
        split["TR30"]
    ].max()
    <= 12
)

assert (
    time_steps[
        split["TR40"]
    ].min()
    >= 1
)

assert (
    time_steps[
        split["TR40"]
    ].max()
    <= 20
)

assert (
    time_steps[
        split["val"]
    ].min()
    >= 21
)

assert (
    time_steps[
        split["val"]
    ].max()
    <= 31
)

assert (
    time_steps[
        split["test"]
    ].min()
    >= 32
)

assert (
    time_steps[
        split["test"]
    ].max()
    <= 49
)


for small, large in (
    ("TR10", "TR20"),
    ("TR20", "TR30"),
    ("TR30", "TR40"),
):
    assert set(
        map(
            int,
            split[small],
        )
    ) <= set(
        map(
            int,
            split[large],
        )
    )


for ratio in RATIOS:

    assert not (
        set(
            map(
                int,
                split[ratio],
            )
        )
        &
        set(
            map(
                int,
                split["val"],
            )
        )
    )

    assert not (
        set(
            map(
                int,
                split[ratio],
            )
        )
        &
        set(
            map(
                int,
                split["test"],
            )
        )
    )


assert not (
    set(
        map(
            int,
            split["val"],
        )
    )
    &
    set(
        map(
            int,
            split["test"],
        )
    )
)

print(
    "ELLIPTIC_CHRONOLOGY_GATE=PASS",
    flush=True,
)

# ============================================================
# REUSE THE EXACT VERIFIED SMOKE ADAPTER SETUP
#
# Parse the already-passed smoke script and execute everything
# only up to (but not including) its first runtime training
# statement. This prevents us from rebuilding a different
# private Elliptic representation.
# ============================================================

smoke_source = (
    SMOKE.read_text()
)

smoke_tree = ast.parse(
    smoke_source,
    filename=str(
        SMOKE
    ),
)

cut_line = None

for stmt in smoke_tree.body:

    if isinstance(
        stmt,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    ):
        continue

    has_train_call = False

    for node in ast.walk(
        stmt
    ):
        if (
            isinstance(
                node,
                ast.Call,
            )
            and isinstance(
                node.func,
                ast.Attribute,
            )
            and node.func.attr
            == "train"
        ):
            has_train_call = True
            break

    if has_train_call:
        cut_line = stmt.lineno
        break


assert cut_line is not None, (
    "Could not locate smoke training boundary."
)

smoke_lines = (
    smoke_source.splitlines(
        keepends=True
    )
)

setup_source = "".join(
    smoke_lines[
        : cut_line - 1
    ]
)

ns = {
    "__name__":
        "__pmp_elliptic_verified_setup__",

    "__file__":
        str(SMOKE),
}

exec(
    compile(
        setup_source,
        str(SMOKE),
        "exec",
    ),
    ns,
    ns,
)

print(
    "VERIFIED_SMOKE_SETUP_REUSED=PASS",
    flush=True,
)

# ============================================================
# DISCOVER THE VERIFIED HELPER OBJECT
# ============================================================

helper_candidates = []

seen_ids = set()

for value in ns.values():

    if id(value) in seen_ids:
        continue

    if (
        hasattr(
            value,
            "data",
        )
        and hasattr(
            value,
            "labels",
        )
        and callable(
            getattr(
                value,
                "get_DGLloader",
                None,
            )
        )
    ):
        helper_candidates.append(
            value
        )

        seen_ids.add(
            id(value)
        )


assert (
    len(helper_candidates)
    == 1
), (
    "Expected exactly one verified PMP dataset helper; "
    f"found {len(helper_candidates)}"
)

helper = (
    helper_candidates[0]
)

base_config = ns.get(
    "config"
)

assert isinstance(
    base_config,
    dict,
)

base_config = copy.deepcopy(
    base_config
)

# ============================================================
# EXACT COMPATIBILITY CONFIG GATES
# ============================================================

assert int(
    base_config[
        "hid_dim"
    ]
) == 256

assert abs(
    float(
        base_config[
            "dropout"
        ]
    ) - 0.4
) < 1e-15

assert abs(
    float(
        base_config[
            "lr"
        ]
    ) - 0.005
) < 1e-15

assert int(
    base_config[
        "n_layer"
    ]
) == 1

assert int(
    base_config[
        "batch_size"
    ]
) == 256

assert (
    base_config[
        "norm_feat"
    ]
    is True
)

assert (
    base_config[
        "full_neighbors"
    ]
    is True
)

assert abs(
    float(
        base_config.get(
            "weight_decay",
            0.0,
        )
    )
) < 1e-15

assert (
    base_config.get(
        "add_self_loop",
        False,
    )
    is False
)

assert (
    helper.data.num_nodes()
    == 203769
)

assert (
    helper.data.number_of_edges()
    == 234355
)

if hasattr(
    helper,
    "feat_dim",
):
    assert int(
        helper.feat_dim
    ) == 166


labels_np = (
    helper.labels
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.int64
    )
)

assert len(
    labels_np
) == 203769

assert int(
    np.sum(
        labels_np == 1
    )
) == 4545

assert int(
    np.sum(
        labels_np == 0
    )
) == 42019

# Unknown-node identity must come from the canonical raw Elliptic
# class CSV, not from PMP helper.labels.  helper.labels is only
# required to provide valid binary labels for supervised nodes.
class_df = pd.read_csv(CLASSES)

assert class_df.shape[0] == 203769
assert class_df.shape[1] >= 2

feature_txids = (
    raw_feat.iloc[:, 0]
    .astype(np.int64)
    .to_numpy()
)

class_txids = (
    class_df.iloc[:, 0]
    .astype(np.int64)
    .to_numpy()
)

class_values = (
    class_df.iloc[:, 1]
    .astype(str)
    .str.strip()
    .str.lower()
    .to_numpy()
)

assert len(np.unique(class_txids)) == 203769

raw_class_map = {
    int(txid): cls
    for txid, cls
    in zip(class_txids, class_values)
}

raw_classes_aligned = np.asarray(
    [
        raw_class_map[int(txid)]
        for txid in feature_txids
    ],
    dtype=object,
)

RAW_UNKNOWN_IDX = np.where(
    raw_classes_aligned == "unknown"
)[0].astype(np.int64)

RAW_ILLICIT_IDX = np.where(
    raw_classes_aligned == "1"
)[0].astype(np.int64)

RAW_LICIT_IDX = np.where(
    raw_classes_aligned == "2"
)[0].astype(np.int64)

assert len(RAW_UNKNOWN_IDX) == 157205
assert len(RAW_ILLICIT_IDX) == 4545
assert len(RAW_LICIT_IDX) == 42019

assert (
    len(RAW_UNKNOWN_IDX)
    + len(RAW_ILLICIT_IDX)
    + len(RAW_LICIT_IDX)
    == 203769
)

print(
    "RAW_CLASS_ALIGNMENT_GATE=PASS",
    flush=True,
)

print(
    "RAW_UNKNOWN_NODES=157205",
    flush=True,
)


for key in (
    "TR40",
    "TR30",
    "TR20",
    "TR10",
    "val",
    "test",
):
    assert np.all(
        np.isin(
            labels_np[
                split[key]
            ],
            [
                0,
                1,
            ],
        )
    )


print(
    "ELLIPTIC_166_FEATURE_FINAL_GATE=PASS",
    flush=True,
)

print(
    "UNKNOWN_NODES_RETAINED_STRUCTURALLY=157205",
    flush=True,
)

print(
    "DIRECTED_RAW_EDGES_PRESERVED=234355",
    flush=True,
)

print(
    "HID_DIM=256",
    flush=True,
)

print(
    "DROPOUT=0.4",
    flush=True,
)

print(
    "LR=0.005",
    flush=True,
)

print(
    "WEIGHT_DECAY=0.0",
    flush=True,
)

print(
    "NORM_FEAT=TRUE",
    flush=True,
)

print(
    "PERFORMANCE_SPECIFIC_TUNING=NO",
    flush=True,
)

# ============================================================
# CLEAR ANY UNUSED MODEL CREATED BY SMOKE PRELUDE
# ============================================================

for name in (
    "model",
    "trainer",
    "optimizer",
    "opt",
    "loss",
    "loss_func",
    "scheduler",
    "train_loader",
    "val_loader",
    "test_loader",
):

    if name in ns:
        ns[name] = None


gc.collect()
torch.cuda.empty_cache()

# ============================================================
# PMP IMPORTS
# ============================================================

if str(
    REPO
) not in sys.path:
    sys.path.insert(
        0,
        str(REPO),
    )

os.chdir(
    REPO
)

from training_procedure import Trainer
from utils.logger import Logger
from utils.random_seeder import set_random_seed
from dgl.dataloading import MultiLayerFullNeighborSampler

logger = Logger(
    mode=[
        print
    ]
)

# ============================================================
# FREEZE EXACT FINAL CONFIG BEFORE TESTING
# ============================================================

frozen = {
    "status":
        "FROZEN",

    "model":
        "PMP / LA-SAGE-S",

    "dataset":
        "Elliptic",

    "compatibility":
        "PASS_WITH_SHARED_ADAPTER",

    "configuration_source":
        (
            "PMP native T-Finance LA-SAGE-S base; "
            "hid_dim=256 from official PMP Amazon "
            "configuration as deterministic "
            "dimension-compatibility fallback"
        ),

    "performance_specific_tuning":
        False,

    "repository_commit":
        EXPECTED_COMMIT,

    "split_sha256":
        EXPECTED_SPLIT_SHA,

    "raw_hashes": {
        "features":
            EXPECTED_FEATURES_SHA,

        "classes":
            EXPECTED_CLASSES_SHA,

        "edges":
            EXPECTED_EDGES_SHA,
    },

    "hid_dim":
        256,

    "dropout":
        0.4,

    "lr":
        0.005,

    "weight_decay":
        0.0,

    "n_layer":
        1,

    "batch_size":
        256,

    "norm_feat":
        True,

    "full_neighbors":
        True,

    "add_self_loop":
        False,

    "reverse_edges_added":
        False,

    "model_features":
        166,

    "time_step_retained_as_feature":
        True,

    "unknown_nodes_structural":
        157205,

    "max_epochs":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "checkpoint_selection":
        "validation AUPRC",

    "threshold_selection":
        (
            "validation Macro-F1 over 0.01..0.99; "
            "tie fraud recall, then closeness to 0.5"
        ),

    "test_selection":
        False,

    "ratios":
        RATIOS,

    "training_seeds":
        SEEDS,
}

frozen_text = json.dumps(
    frozen,
    indent=2,
    sort_keys=True,
)

if FROZEN_CONFIG.exists():

    existing = json.loads(
        FROZEN_CONFIG.read_text()
    )

    assert existing == frozen, (
        "Existing frozen final config differs."
    )

else:

    FROZEN_CONFIG.write_text(
        frozen_text
    )


FROZEN_SHA = sha256_file(
    FROZEN_CONFIG
)

print(
    "FINAL_CONFIG_FROZEN=PASS",
    flush=True,
)

print(
    "FROZEN_CONFIG_SHA256=" +
    FROZEN_SHA,
    flush=True,
)

# ============================================================
# APPLY CHRONOLOGICAL SPLIT
# ============================================================

def apply_split(
    ratio
):
    n = 203769

    tr = torch.as_tensor(
        split[ratio],
        dtype=torch.long,
    )

    va = torch.as_tensor(
        split["val"],
        dtype=torch.long,
    )

    te = torch.as_tensor(
        split["test"],
        dtype=torch.long,
    )

    train_mask = torch.zeros(
        n,
        dtype=torch.bool,
    )

    val_mask = torch.zeros(
        n,
        dtype=torch.bool,
    )

    test_mask = torch.zeros(
        n,
        dtype=torch.bool,
    )

    train_mask[tr] = True
    val_mask[va] = True
    test_mask[te] = True

    helper.data.ndata[
        "train_mask"
    ] = train_mask

    helper.data.ndata[
        "val_mask"
    ] = val_mask

    helper.data.ndata[
        "test_mask"
    ] = test_mask

    helper.train_mask = (
        train_mask
    )

    helper.val_mask = (
        val_mask
    )

    helper.test_mask = (
        test_mask
    )

    helper.train_nid = tr
    helper.val_nid = va
    helper.test_nid = te

    # PMP label-aware partition:
    # only current training labels are visible.
    label_unk = torch.full(
        (
            n,
        ),
        2,
        dtype=torch.long,
    )

    label_unk[
        tr
    ] = helper.labels[
        tr
    ]

    helper.data.ndata[
        "label_unk"
    ] = label_unk

    assert int(
        (
            label_unk
            != 2
        ).sum()
    ) == len(
        tr
    )

    assert torch.all(
        label_unk[
            va
        ] == 2
    )

    assert torch.all(
        label_unk[
            te
        ] == 2
    )

    unknown_idx = torch.as_tensor(
        RAW_UNKNOWN_IDX,
        dtype=torch.long,
    )

    assert torch.all(
        label_unk[
            unknown_idx
        ] == 2
    )

    return (
        tr,
        va,
        te,
    )


# ============================================================
# REUSE VERIFIED SMOKE ARGUMENT SHAPE
# ============================================================

base_args = ns.get(
    "args"
)


def args_for(
    seed,
    ratio,
):

    ratio_fraction = {
        "TR40": 0.4,
        "TR30": 0.3,
        "TR20": 0.2,
        "TR10": 0.1,
    }[
        ratio
    ]

    if (
        base_args is not None
        and hasattr(
            base_args,
            "__dict__",
        )
    ):

        values = dict(
            vars(
                base_args
            )
        )

        values[
            "seed"
        ] = int(seed)

        values[
            "train_size"
        ] = ratio_fraction

        values[
            "val_size"
        ] = 0.2

        values[
            "gpu_id"
        ] = 0

        values[
            "multirun"
        ] = 1

        values[
            "run_best"
        ] = False

        values[
            "no_dev"
        ] = False

        return SimpleNamespace(
            **values
        )


    return SimpleNamespace(
        dataset=base_config.get(
            "dataset",
            "tfinance",
        ),
        num_workers=8,
        seed=int(seed),
        data_dir=str(
            RAW
        ),
        hyper_file=str(
            REPO /
            "config"
        ),
        log_dir=str(
            WORK /
            "logs"
        ),
        best_model_path=str(
            CKPT_DIR
        ),
        train_size=
            ratio_fraction,
        val_size=0.2,
        no_dev=False,
        gpu_id=0,
        multirun=1,
        model="LA-SAGE-S",
        run_best=False,
    )


# ============================================================
# FINAL 12 RUNS
# ============================================================

summaries = []

for ratio in RATIOS:

    for seed in SEEDS:

        run_id = (
            "elliptic_" +
            ratio.lower() +
            "_seed" +
            str(seed)
        )

        summary_path = (
            RUN_DIR /
            f"{run_id}_summary.json"
        )

        epoch_csv = (
            RUN_DIR /
            f"{run_id}_epochs.csv"
        )

        threshold_csv = (
            RUN_DIR /
            f"{run_id}_val_threshold_grid.csv"
        )

        checkpoint = (
            CKPT_DIR /
            f"{run_id}_best_val_auprc.pth"
        )

        # ----------------------------------------
        # Resume completed valid run.
        # ----------------------------------------

        if summary_path.exists():

            old = json.loads(
                summary_path.read_text()
            )

            if (
                old.get(
                    "status"
                ) == "PASS"

                and old.get(
                    "run_id"
                ) == run_id

                and old.get(
                    "ratio"
                ) == ratio

                and int(
                    old.get(
                        "train_seed",
                        -1,
                    )
                ) == seed

                and old.get(
                    "repository_commit"
                ) == EXPECTED_COMMIT

                and old.get(
                    "split_sha256"
                ) == EXPECTED_SPLIT_SHA

                and old.get(
                    "frozen_config_sha256"
                ) == FROZEN_SHA

                and old.get(
                    "test_evaluated_once"
                ) is True

                and old.get(
                    "test_used_for_selection"
                ) is False

                and checkpoint.exists()
            ):

                print(
                    "===== " +
                    run_id +
                    ": RESUME PASS =====",
                    flush=True,
                )

                summaries.append(
                    old
                )

                continue


        for path in (
            summary_path,
            epoch_csv,
            threshold_csv,
            checkpoint,
        ):
            if path.exists():
                path.unlink()


        print(
            "\n===== FINAL RUN " +
            run_id +
            " =====",
            flush=True,
        )


        tr, va, te = apply_split(
            ratio
        )

        assert len(
            tr
        ) == EXPECTED_SIZES[
            ratio
        ]

        assert len(
            va
        ) == EXPECTED_SIZES[
            "val"
        ]

        assert len(
            te
        ) == EXPECTED_SIZES[
            "test"
        ]


        cfg = copy.deepcopy(
            base_config
        )

        cfg.update(
            {
                "seed":
                    int(seed),

                "train_size":
                    {
                        "TR40": 0.4,
                        "TR30": 0.3,
                        "TR20": 0.2,
                        "TR10": 0.1,
                    }[
                        ratio
                    ],

                "val_size":
                    0.2,

                "epochs":
                    MAX_EPOCHS,

                "patience":
                    PATIENCE,

                "eval_interval":
                    1,

                "test_each_epoch":
                    False,

                "monitor":
                    "ap_gnn",

                "best_model_path":
                    str(
                        checkpoint
                    ),
            }
        )

        # Frozen compatibility config.
        assert int(
            cfg["hid_dim"]
        ) == 256

        assert abs(
            float(
                cfg["dropout"]
            ) - 0.4
        ) < 1e-15

        assert abs(
            float(
                cfg["lr"]
            ) - 0.005
        ) < 1e-15

        assert abs(
            float(
                cfg.get(
                    "weight_decay",
                    0.0,
                )
            )
        ) < 1e-15

        assert int(
            cfg["n_layer"]
        ) == 1

        assert int(
            cfg["batch_size"]
        ) == 256

        assert (
            cfg["norm_feat"]
            is True
        )

        seed_all(
            seed
        )

        set_random_seed(
            seed
        )

        sampler = (
            MultiLayerFullNeighborSampler(
                num_layers=
                    cfg[
                        "n_layer"
                    ]
            )
        )

        # PMP_ELLIPTIC_FULL_TEST_EVAL_ADAPTER
        #
        # Author LA-SAGE-S has no guard for an entirely edge-empty
        # DGL evaluation minibatch.  The chronological Elliptic test
        # partition can produce such a minibatch when test evaluation
        # inherits the training batch size (256).
        #
        # This changes evaluation batching ONLY:
        #   - graph unchanged
        #   - direction unchanged
        #   - no reverse edges
        #   - no self-loops
        #   - model unchanged
        #   - checkpoint unchanged
        #   - threshold rule unchanged
        #
        # Evaluate the complete frozen test partition in one batch.
        helper.config["test_batch_size"] = len(te)
        cfg["test_batch_size"] = len(te)

        print(
            f"TEST_EVAL_BATCH_ADAPTER=FULL_FIXED_TEST_SINGLE_BATCH "
            f"nodes={len(te)}",
            flush=True,
        )

        (
            train_loader,
            val_loader,
            test_loader,
        ) = helper.get_DGLloader(
            helper.data,
            sampler,
        )

        assert len(test_loader) == 1

        T = Trainer(
            args=args_for(
                seed,
                ratio,
            ),
            config=cfg,
            logger=logger,
        )

        (
            model,
            optimizer,
            loss_func,
            scheduler,
        ) = T.init(
            helper
        )

        # Optimizer identity gate.
        for group in (
            optimizer.param_groups
        ):
            assert abs(
                float(
                    group["lr"]
                ) - 0.005
            ) < 1e-15

            assert abs(
                float(
                    group.get(
                        "weight_decay",
                        0.0,
                    )
                )
            ) < 1e-15

            assert tuple(
                group.get(
                    "betas",
                    (
                        0.9,
                        0.999,
                    ),
                )
            ) == (
                0.9,
                0.999,
            )

            assert abs(
                float(
                    group.get(
                        "eps",
                        1e-8,
                    )
                ) - 1e-8
            ) < 1e-20


        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        epoch_rows = []

        best_ap = -1.0
        best_epoch = -1
        patience_count = 0

        wall_start = (
            time.perf_counter()
        )


        # ========================================
        # TRAIN + VALIDATION ONLY
        # ========================================

        for epoch in range(
            1,
            MAX_EPOCHS + 1,
        ):

            torch.cuda.synchronize()

            t0 = (
                time.perf_counter()
            )

            model, loss = T.train(
                epoch - 1,
                model,
                loss_func,
                optimizer,
                train_loader,
                helper,
            )

            torch.cuda.synchronize()

            train_seconds = (
                time.perf_counter()
                - t0
            )

            if hasattr(
                loss,
                "detach",
            ):
                loss_value = float(
                    loss.detach()
                    .item()
                )
            else:
                loss_value = float(
                    loss
                )

            avg_loss = (
                loss_value
                /
                max(
                    1,
                    len(
                        train_loader
                    ),
                )
            )


            torch.cuda.synchronize()

            v0 = (
                time.perf_counter()
            )

            (
                val_y,
                val_prob,
                _,
            ) = T.evaluation(
                helper,
                val_loader,
                model,
                threshold_moving=False,
                thres=0.5,
            )

            torch.cuda.synchronize()

            validation_seconds = (
                time.perf_counter()
                - v0
            )


            val_ap = float(
                average_precision_score(
                    val_y,
                    val_prob,
                )
            )

            val_auc = float(
                roc_auc_score(
                    val_y,
                    val_prob,
                )
            )


            improved = (
                val_ap
                > best_ap + 1e-12
            )


            if improved:

                best_ap = val_ap
                best_epoch = epoch
                patience_count = 0

                save_state_cpu(
                    model,
                    checkpoint,
                    {
                        "run_id":
                            run_id,

                        "ratio":
                            ratio,

                        "train_seed":
                            seed,

                        "epoch":
                            epoch,

                        "val_auprc":
                            val_ap,

                        "frozen_config_sha256":
                            FROZEN_SHA,

                        "repository_commit":
                            EXPECTED_COMMIT,

                        "split_sha256":
                            EXPECTED_SPLIT_SHA,
                    },
                )

            else:

                patience_count += 1


            epoch_rows.append(
                {
                    "run_id":
                        run_id,

                    "ratio":
                        ratio,

                    "train_seed":
                        seed,

                    "epoch":
                        epoch,

                    "avg_train_loss":
                        avg_loss,

                    "train_seconds":
                        float(
                            train_seconds
                        ),

                    "validation_seconds":
                        float(
                            validation_seconds
                        ),

                    "val_auprc":
                        val_ap,

                    "val_auroc":
                        val_auc,

                    "best_val_auprc_so_far":
                        best_ap,

                    "best_epoch_so_far":
                        best_epoch,

                    "patience_count":
                        patience_count,

                    "peak_gpu_memory_mb_so_far":
                        float(
                            torch.cuda
                            .max_memory_allocated()
                            / 1024**2
                        ),
                }
            )

            write_csv(
                epoch_csv,
                epoch_rows,
            )


            if (
                epoch == 1
                or epoch % 10 == 0
                or improved
                or patience_count
                >= PATIENCE
            ):

                print(
                    f"{run_id} "
                    f"epoch={epoch:03d} "
                    f"loss={avg_loss:.6f} "
                    f"val_AUPRC={val_ap:.6f} "
                    f"best={best_ap:.6f}@{best_epoch} "
                    f"patience={patience_count}/{PATIENCE} "
                    f"train={train_seconds:.3f}s "
                    f"val={validation_seconds:.3f}s",
                    flush=True,
                )


            if (
                patience_count
                >= PATIENCE
            ):

                print(
                    f"EARLY_STOP {run_id} "
                    f"epoch={epoch} "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )

                break


        training_wall_seconds = (
            time.perf_counter()
            - wall_start
        )

        peak_trainval_mb = float(
            torch.cuda
            .max_memory_allocated()
            / 1024**2
        )

        assert checkpoint.exists()

        checkpoint_meta = load_state(
            model,
            checkpoint,
        )

        assert int(
            checkpoint_meta[
                "epoch"
            ]
        ) == best_epoch

        assert abs(
            float(
                checkpoint_meta[
                    "val_auprc"
                ]
            ) - best_ap
        ) < 1e-12


        # ========================================
        # FROZEN VALIDATION THRESHOLD
        # No test access yet.
        # ========================================

        torch.cuda.synchronize()

        v0 = (
            time.perf_counter()
        )

        (
            val_y,
            val_prob,
            _,
        ) = T.evaluation(
            helper,
            val_loader,
            model,
            threshold_moving=False,
            thres=0.5,
        )

        torch.cuda.synchronize()

        frozen_val_inference_seconds = (
            time.perf_counter()
            - v0
        )

        (
            threshold_choice,
            threshold_rows,
        ) = choose_threshold(
            val_y,
            val_prob,
        )

        write_csv(
            threshold_csv,
            threshold_rows,
        )

        selected_threshold = float(
            threshold_choice[
                "threshold"
            ]
        )

        validation_metrics = metrics(
            val_y,
            val_prob,
            selected_threshold,
        )

        print(
            f"{run_id} FROZEN "
            f"checkpoint_epoch={best_epoch} "
            f"val_AUPRC={best_ap:.6f} "
            f"threshold={selected_threshold:.2f}",
            flush=True,
        )


        # ========================================
        # EXACTLY ONE TEST EVALUATION
        # Only after config/checkpoint/threshold freeze.
        # ========================================

        torch.cuda.synchronize()

        test_start = (
            time.perf_counter()
        )

        (
            test_y,
            test_prob,
            _,
        ) = T.evaluation(
            helper,
            test_loader,
            model,
            threshold_moving=False,
            thres=0.5,
        )

        torch.cuda.synchronize()

        test_inference_seconds = (
            time.perf_counter()
            - test_start
        )

        test_metrics = metrics(
            test_y,
            test_prob,
            selected_threshold,
        )

        peak_overall_mb = float(
            torch.cuda
            .max_memory_allocated()
            / 1024**2
        )


        train_mean, train_sd = mean_sd(
            [
                r[
                    "train_seconds"
                ]
                for r
                in epoch_rows
            ]
        )

        val_mean, val_sd = mean_sd(
            [
                r[
                    "validation_seconds"
                ]
                for r
                in epoch_rows
            ]
        )


        result = {
            "status":
                "PASS",

            "run_id":
                run_id,

            "model":
                "PMP / LA-SAGE-S",

            "dataset":
                "Elliptic",

            "compatibility":
                "PASS_WITH_SHARED_ADAPTER",

            "ratio":
                ratio,

            "train_seed":
                seed,

            "train_nodes":
                len(
                    tr
                ),

            "validation_nodes":
                len(
                    va
                ),

            "test_nodes":
                len(
                    te
                ),

            "train_fraud":
                int(
                    np.sum(
                        labels_np[
                            split[
                                ratio
                            ]
                        ]
                        == 1
                    )
                ),

            "validation_fraud":
                int(
                    np.sum(
                        labels_np[
                            split[
                                "val"
                            ]
                        ]
                        == 1
                    )
                ),

            "test_fraud":
                int(
                    np.sum(
                        labels_np[
                            split[
                                "test"
                            ]
                        ]
                        == 1
                    )
                ),

            "repository_commit":
                EXPECTED_COMMIT,

            "split_sha256":
                EXPECTED_SPLIT_SHA,

            "frozen_config_sha256":
                FROZEN_SHA,

            "hid_dim":
                256,

            "dropout":
                0.4,

            "lr":
                0.005,

            "weight_decay":
                0.0,

            "n_layer":
                1,

            "batch_size":
                256,

            "norm_feat":
                True,

            "max_epochs":
                MAX_EPOCHS,

            "patience":
                PATIENCE,

            "epochs_completed":
                len(
                    epoch_rows
                ),

            "best_epoch":
                best_epoch,

            "best_validation_auprc":
                best_ap,

            "selected_threshold":
                selected_threshold,

            "threshold_rule":
                (
                    "validation Macro-F1 max; "
                    "tie fraud recall; "
                    "then closeness to 0.5"
                ),

            "validation_metrics_at_selected_threshold":
                validation_metrics,

            "test_metrics":
                test_metrics,

            "test_evaluated_once":
                True,

            "test_used_for_selection":
                False,

            "train_seconds_total":
                float(
                    sum(
                        r[
                            "train_seconds"
                        ]
                        for r
                        in epoch_rows
                    )
                ),

            "train_seconds_mean":
                train_mean,

            "train_seconds_sd":
                train_sd,

            "validation_seconds_total_during_training":
                float(
                    sum(
                        r[
                            "validation_seconds"
                        ]
                        for r
                        in epoch_rows
                    )
                ),

            "validation_seconds_mean":
                val_mean,

            "validation_seconds_sd":
                val_sd,

            "frozen_validation_inference_seconds":
                frozen_val_inference_seconds,

            "test_inference_seconds":
                test_inference_seconds,

            "training_wall_seconds":
                training_wall_seconds,

            "peak_trainval_gpu_memory_mb":
                peak_trainval_mb,

            "peak_overall_gpu_memory_mb":
                peak_overall_mb,

            "checkpoint_path":
                str(
                    checkpoint
                ),

            "checkpoint_sha256":
                sha256_file(
                    checkpoint
                ),

            "epoch_csv":
                str(
                    epoch_csv
                ),

            "epoch_csv_sha256":
                sha256_file(
                    epoch_csv
                ),

            "threshold_csv":
                str(
                    threshold_csv
                ),

            "threshold_csv_sha256":
                sha256_file(
                    threshold_csv
                ),
        }


        summary_path.write_text(
            json.dumps(
                result,
                indent=2,
            )
        )

        summaries.append(
            result
        )


        print(
            f"FINAL {run_id} PASS | "
            f"AUPRC={test_metrics['auprc']:.6f} "
            f"AUROC={test_metrics['auroc']:.6f} "
            f"MacroF1={test_metrics['macro_f1']:.6f} "
            f"Recall={test_metrics['fraud_recall']:.6f}",
            flush=True,
        )


        del (
            model,
            optimizer,
            loss_func,
            scheduler,
            train_loader,
            val_loader,
            test_loader,
            T,
        )

        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# FINAL GRID INTEGRITY
# ============================================================

assert len(
    summaries
) == 12

by_key = {
    (
        r[
            "ratio"
        ],
        int(
            r[
                "train_seed"
            ]
        ),
    ):
        r
    for r
    in summaries
}

assert len(
    by_key
) == 12


flat = []

for ratio in RATIOS:

    for seed in SEEDS:

        r = by_key[
            (
                ratio,
                seed,
            )
        ]

        assert (
            r[
                "status"
            ]
            == "PASS"
        )

        assert (
            r[
                "test_evaluated_once"
            ]
            is True
        )

        assert (
            r[
                "test_used_for_selection"
            ]
            is False
        )

        assert (
            r[
                "frozen_config_sha256"
            ]
            == FROZEN_SHA
        )

        m = r[
            "test_metrics"
        ]

        flat.append(
            {
                "dataset":
                    "Elliptic",

                "model":
                    "PMP",

                "ratio":
                    ratio,

                "seed":
                    seed,

                "train_nodes":
                    r[
                        "train_nodes"
                    ],

                "epochs_completed":
                    r[
                        "epochs_completed"
                    ],

                "best_epoch":
                    r[
                        "best_epoch"
                    ],

                "best_validation_auprc":
                    r[
                        "best_validation_auprc"
                    ],

                "threshold":
                    r[
                        "selected_threshold"
                    ],

                "auprc":
                    m[
                        "auprc"
                    ],

                "auroc":
                    m[
                        "auroc"
                    ],

                "macro_f1":
                    m[
                        "macro_f1"
                    ],

                "fraud_precision":
                    m[
                        "fraud_precision"
                    ],

                "fraud_recall":
                    m[
                        "fraud_recall"
                    ],

                "fraud_f1":
                    m[
                        "fraud_f1"
                    ],

                "gmean":
                    m[
                        "gmean"
                    ],

                "train_seconds_total":
                    r[
                        "train_seconds_total"
                    ],

                "train_seconds_mean":
                    r[
                        "train_seconds_mean"
                    ],

                "train_seconds_sd":
                    r[
                        "train_seconds_sd"
                    ],

                "test_inference_seconds":
                    r[
                        "test_inference_seconds"
                    ],

                "peak_gpu_memory_mb":
                    r[
                        "peak_overall_gpu_memory_mb"
                    ],
            }
        )


write_csv(
    FINAL_RUNS_CSV,
    flat,
)

# ============================================================
# MEAN ± SD BY RATIO
# ============================================================

metric_names = [
    "auprc",
    "auroc",
    "macro_f1",
    "fraud_precision",
    "fraud_recall",
    "fraud_f1",
    "gmean",
]

ratio_rows = []
ratio_json = {}

for ratio in RATIOS:

    ratio_runs = [
        x
        for x
        in flat
        if x[
            "ratio"
        ] == ratio
    ]

    assert len(
        ratio_runs
    ) == 3

    row = {
        "ratio":
            ratio,

        "n_seeds":
            3,
    }

    metric_json = {}

    for metric_name in (
        metric_names
    ):

        mean, sd = mean_sd(
            [
                x[
                    metric_name
                ]
                for x
                in ratio_runs
            ]
        )

        row[
            metric_name +
            "_mean"
        ] = mean

        row[
            metric_name +
            "_sd"
        ] = sd

        metric_json[
            metric_name
        ] = {
            "mean":
                mean,

            "sd":
                sd,
        }


    train_mean, train_sd = mean_sd(
        [
            x[
                "train_seconds_total"
            ]
            for x
            in ratio_runs
        ]
    )

    infer_mean, infer_sd = mean_sd(
        [
            x[
                "test_inference_seconds"
            ]
            for x
            in ratio_runs
        ]
    )

    mem_mean, mem_sd = mean_sd(
        [
            x[
                "peak_gpu_memory_mb"
            ]
            for x
            in ratio_runs
        ]
    )


    row[
        "total_train_seconds_mean"
    ] = train_mean

    row[
        "total_train_seconds_sd"
    ] = train_sd

    row[
        "test_inference_seconds_mean"
    ] = infer_mean

    row[
        "test_inference_seconds_sd"
    ] = infer_sd

    row[
        "peak_gpu_memory_mb_mean"
    ] = mem_mean

    row[
        "peak_gpu_memory_mb_sd"
    ] = mem_sd


    ratio_rows.append(
        row
    )

    ratio_json[
        ratio
    ] = {
        "metrics":
            metric_json,

        "total_train_seconds": {
            "mean":
                train_mean,

            "sd":
                train_sd,
        },

        "test_inference_seconds": {
            "mean":
                infer_mean,

            "sd":
                infer_sd,
        },

        "peak_gpu_memory_mb": {
            "mean":
                mem_mean,

            "sd":
                mem_sd,
        },
    }


write_csv(
    RATIO_SUMMARY_CSV,
    ratio_rows,
)


final_summary = {
    "status":
        "PASS",

    "model":
        "PMP / LA-SAGE-S",

    "dataset":
        "Elliptic",

    "compatibility":
        "PASS_WITH_SHARED_ADAPTER",

    "final_runs_complete":
        12,

    "expected_final_runs":
        12,

    "ratios":
        RATIOS,

    "seeds":
        SEEDS,

    "repository_commit":
        EXPECTED_COMMIT,

    "split_sha256":
        EXPECTED_SPLIT_SHA,

    "frozen_config":
        frozen,

    "frozen_config_sha256":
        FROZEN_SHA,

    "checkpoint_selection":
        "best validation AUPRC",

    "threshold_selection":
        (
            "validation Macro-F1 over 0.01..0.99; "
            "tie fraud recall, then closeness to 0.5"
        ),

    "test_isolation":
        (
            "test evaluated exactly once per final run "
            "after config, checkpoint and threshold frozen"
        ),

    "ratio_summary":
        ratio_json,

    "final_runs_csv":
        str(
            FINAL_RUNS_CSV
        ),

    "ratio_summary_csv":
        str(
            RATIO_SUMMARY_CSV
        ),

    "gpu":
        GPU,
}


FINAL_SUMMARY_JSON.write_text(
    json.dumps(
        final_summary,
        indent=2,
    )
)


STATUS_TXT.write_text(
    "PMP_ELLIPTIC_FINAL=PASS\n"
    "PMP_ELLIPTIC_FINAL_RUNS=12/12\n"
    "COMPATIBILITY=PASS_WITH_SHARED_ADAPTER\n"
    "TEST_ISOLATION=PASS\n"
    "PMP_AUTHOR_REPO_CLEAN=PASS\n"
    "FEATURE_COUNT=166\n"
    "UNKNOWN_STRUCTURAL_NODES=157205\n"
    "DIRECTED_EDGES=234355\n"
    "HID_DIM=256\n"
    "DROPOUT=0.4\n"
    "LR=0.005\n"
    "WEIGHT_DECAY=0.0\n"
    "NORM_FEAT=TRUE\n"
    "PERFORMANCE_SPECIFIC_TUNING=NO\n"
    f"SPLIT_SHA256={EXPECTED_SPLIT_SHA}\n"
    f"FROZEN_CONFIG_SHA256={FROZEN_SHA}\n"
    f"SUMMARY_JSON={FINAL_SUMMARY_JSON}\n"
)


assert git(
    "status",
    "--porcelain",
    "--untracked-files=no",
) == ""

# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n===== PMP ELLIPTIC FINAL RESULTS — MEAN ± SD =====",
    flush=True,
)

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
        flush=True,
    )

# ============================================================
# PACKAGE EVIDENCE
# Checkpoints are deliberately excluded from archive;
# their SHA256 values are retained in each run JSON.
# ============================================================

if ARCHIVE.exists():
    ARCHIVE.unlink()

with tarfile.open(
    ARCHIVE,
    "w:gz",
) as tar:

    tar.add(
        FROZEN_CONFIG,
        arcname=
            "final/"
            "PMP_ELLIPTIC_FROZEN_CONFIG.json",
    )

    tar.add(
        FINAL_RUNS_CSV,
        arcname=
            "final/"
            "PMP_ELLIPTIC_FINAL_12_RUNS.csv",
    )

    tar.add(
        RATIO_SUMMARY_CSV,
        arcname=
            "final/"
            "PMP_ELLIPTIC_FINAL_RATIO_SUMMARY.csv",
    )

    tar.add(
        FINAL_SUMMARY_JSON,
        arcname=
            "final/"
            "PMP_ELLIPTIC_FINAL_SUMMARY.json",
    )

    tar.add(
        STATUS_TXT,
        arcname=
            "final/"
            "PMP_ELLIPTIC_FINAL_STATUS.txt",
    )

    for path in sorted(
        RUN_DIR.glob(
            "*"
        )
    ):

        tar.add(
            path,
            arcname=
                "final/runs/" +
                path.name,
        )

    tar.add(
        Path(__file__),
        arcname=
            "scripts/"
            "09_pmp_elliptic_final_12_runs.py",
    )

    tar.add(
        SMOKE,
        arcname=
            "scripts/"
            "08_pmp_elliptic_2epoch_smoke.py",
    )

    compatibility_dir = (
        WORK /
        "unified/elliptic/"
        "compatibility_smoke"
    )

    if compatibility_dir.exists():

        tar.add(
            compatibility_dir,
            arcname=
                "compatibility_smoke",
        )

    tar.add(
        SPLIT,
        arcname=
            "splits/" +
            SPLIT.name,
    )


archive_sha = sha256_file(
    ARCHIVE
)

ARCHIVE_SHA_TXT.write_text(
    archive_sha +
    "  " +
    str(
        ARCHIVE
    ) +
    "\n"
)

print(
    "\n===== FINAL GATE =====",
    flush=True,
)

print(
    "PMP_ELLIPTIC_FINAL=PASS",
    flush=True,
)

print(
    "PMP_ELLIPTIC_FINAL_RUNS=12/12",
    flush=True,
)

print(
    "COMPATIBILITY=PASS_WITH_SHARED_ADAPTER",
    flush=True,
)

print(
    "TEST_ISOLATION=PASS",
    flush=True,
)

print(
    "PERFORMANCE_SPECIFIC_TUNING=NO",
    flush=True,
)

print(
    "PMP_AUTHOR_REPO_CLEAN=PASS",
    flush=True,
)

print(
    "FINAL_ARCHIVE=" +
    str(
        ARCHIVE
    ),
    flush=True,
)

print(
    "FINAL_ARCHIVE_SHA256=" +
    archive_sha,
    flush=True,
)

print(
    "FINAL_RUNS_CSV=" +
    str(
        FINAL_RUNS_CSV
    ),
    flush=True,
)

print(
    "RATIO_SUMMARY_CSV=" +
    str(
        RATIO_SUMMARY_CSV
    ),
    flush=True,
)

print(
    "FINAL_SUMMARY_JSON=" +
    str(
        FINAL_SUMMARY_JSON
    ),
    flush=True,
)
