#!/usr/bin/env python3

from pathlib import Path
import subprocess
import hashlib
import shutil
import json
import csv
import os
import sys

ROOT = Path("/workspace/gaap_vast")
REPO = ROOT / "repo/GAAP"

BASE = (
    ROOT /
    "scripts/02_gaap_yelp_unified_tr40_smoke.py"
)

LAUNCHER = ROOT / "gaap_python.sh"

TUNE = ROOT / "unified/yelp/tuning"
TUNE.mkdir(parents=True, exist_ok=True)

EXPECTED_COMMIT = (
    "6a7dbb0447c4897504525de49e41a0526ee777f8"
)

EXPECTED_DATA_SHA = (
    "fedb35a8fa539b27866244d3515a47a76b20080c"
    "dacb33112da3458fd2487b42"
)

EXPECTED_SPLIT_SHA = (
    "0ea0af36dfc5a3a1f381ea2e3168377e45826498"
    "b6063190185c673e9ec8c22b"
)

# ============================================================
# PREREGISTERED GRID
#
# Only these three numerical GAAP hyperparameters vary.
# All mechanism / architecture settings remain native Yelp.
# ============================================================

TRIALS = [
    {"trial": 1,  "d_hidden": 64,  "n_bins": 32, "lr": 0.001},
    {"trial": 2,  "d_hidden": 64,  "n_bins": 32, "lr": 0.002},

    {"trial": 3,  "d_hidden": 40,  "n_bins": 32, "lr": 0.001},
    {"trial": 4,  "d_hidden": 40,  "n_bins": 32, "lr": 0.002},

    {"trial": 5,  "d_hidden": 128, "n_bins": 32, "lr": 0.001},
    {"trial": 6,  "d_hidden": 128, "n_bins": 32, "lr": 0.002},

    {"trial": 7,  "d_hidden": 64,  "n_bins": 16, "lr": 0.001},
    {"trial": 8,  "d_hidden": 64,  "n_bins": 16, "lr": 0.002},

    {"trial": 9,  "d_hidden": 40,  "n_bins": 16, "lr": 0.001},
    {"trial": 10, "d_hidden": 40,  "n_bins": 16, "lr": 0.002},

    {"trial": 11, "d_hidden": 128, "n_bins": 16, "lr": 0.001},
    {"trial": 12, "d_hidden": 128, "n_bins": 16, "lr": 0.002},
]

assert len(TRIALS) == 12
assert len({
    (x["d_hidden"], x["n_bins"], x["lr"])
    for x in TRIALS
}) == 12


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda: f.read(8 * 1024 * 1024),
            b""
        ):
            h.update(b)

    return h.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args],
        text=True
    ).strip()


assert git("rev-parse", "HEAD") == EXPECTED_COMMIT
assert git(
    "status",
    "--porcelain",
    "--untracked-files=no"
) == ""

assert BASE.exists()
base_source = BASE.read_text()

split = (
    ROOT /
    "shared/yelp_seed2_nested_splits.npz"
)

data = (
    ROOT /
    "canonical_yelp/YelpChi.mat"
)

assert sha256(split) == EXPECTED_SPLIT_SHA
assert sha256(data) == EXPECTED_DATA_SHA


# ============================================================
# SAVE PREREGISTRATION BEFORE TUNING
# ============================================================

prereg = {
    "model": "GAAP",
    "dataset": "YelpChi",
    "mode": "unified",
    "stage": "TR40_tuning",
    "split": "TR40",
    "split_seed": 2,
    "training_seed": 2,
    "selection_metric": "validation_AUPRC",
    "max_trials": 12,
    "max_epochs": 100,
    "patience": 20,
    "test_accessed": False,

    "fixed": {
        "gnn_n_layers": 3,
        "gnn_dropout": 0.1,
        "gnn_agg": "max_pool",
        "use_dyple": True,
        "use_mha": True,
        "bs": 128,
        "val_bs": 1280,
        "weight_decay": 0,
        "norm_type": "norm01",
        "preprocess": None,
    },

    "varied": {
        "d_hidden": [40, 64, 128],
        "n_bins": [16, 32],
        "lr": [0.001, 0.002],
    },

    "trials": TRIALS,

    "repository_commit": EXPECTED_COMMIT,
    "dataset_sha256": EXPECTED_DATA_SHA,
    "split_sha256": EXPECTED_SPLIT_SHA,

    "architecture_or_mechanism_replacement": False,
}

prereg_path = (
    TUNE /
    "GAAP_YELP_TR40_TUNING_PREREGISTRATION.json"
)

if prereg_path.exists():
    old = json.loads(prereg_path.read_text())

    assert old == prereg, (
        "Existing preregistration differs. "
        "Refusing to change trial grid after tuning began."
    )
else:
    prereg_path.write_text(
        json.dumps(
            prereg,
            indent=2
        )
    )

print(
    "TUNING_PREREGISTRATION_GATE=PASS",
    flush=True
)

print(
    "TRIAL_COUNT=12",
    flush=True
)

print(
    "TEST_ACCESS_DURING_TUNING=NO",
    flush=True
)


# ============================================================
# SOURCE PATCHER
# ============================================================

def replace_once(src, old, new):

    count = src.count(old)

    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence:\n{old}\n"
            f"Found: {count}"
        )

    return src.replace(
        old,
        new,
        1
    )


def make_trial_source(t):

    num = t["trial"]

    trial_dir = (
        TUNE /
        f"trial_{num:02d}"
    )

    src = base_source

    src = replace_once(
        src,
        'OUT = ROOT / "evidence/gaap_yelp_smoke"',
        f'OUT = Path("{trial_dir}")'
    )

    src = replace_once(
        src,
        'cfg["max_epochs"] = 2',
        'cfg["max_epochs"] = 100'
    )

    src = replace_once(
        src,
        'print("MAX_EPOCHS=2 | PATIENCE=20", flush=True)',
        'print("MAX_EPOCHS=100 | PATIENCE=20", flush=True)'
    )

    marker = 'cfg["nowandb"] = True'

    assert src.count(marker) == 1

    override = f'''
{marker}

# ------------------------------------------------------------
# PREREGISTERED TRIAL {num:02d}/12
# ------------------------------------------------------------
cfg["d_hidden"] = {t["d_hidden"]}
cfg["n_bins"] = {t["n_bins"]}
cfg["lr"] = {t["lr"]!r}

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
    "trial={num:02d}/12 | "
    "hidden={t["d_hidden"]} | "
    "bins={t["n_bins"]} | "
    "lr={t["lr"]}",
    flush=True
)
'''

    src = src.replace(
        marker,
        override,
        1
    )

    src = replace_once(
        src,
        'f"run=SMOKE | TR40 seed=2 | "',
        f'f"trial={num:02d}/12 | TR40 seed=2 | "'
    )

    src = replace_once(
        src,
        'filename="tr40-seed2-smoke-{epoch:03d}",',
        f'filename="trial-{num:02d}-best-{{epoch:03d}}",'
    )

    src = replace_once(
        src,
        "max_epochs=2,",
        "max_epochs=100,"
    )

    src = replace_once(
        src,
        '"mode": "unified_2epoch_smoke",',
        '"mode": "unified_tuning",'
    )

    src = replace_once(
        src,
        'with open(OUT / "smoke_result.json", "w") as f:',
        'with open(OUT / "trial_result.json", "w") as f:'
    )

    src = replace_once(
        src,
        'print("===== SMOKE FINAL GATE =====", flush=True)',
        f'print("===== TRIAL {num:02d}/12 FINAL GATE =====", flush=True)'
    )

    src = replace_once(
        src,
        'print("GAAP_YELP_UNIFIED_2EPOCH_SMOKE=PASS", flush=True)',
        f'print("GAAP_YELP_TUNING_TRIAL_{num:02d}=PASS", flush=True)'
    )

    return src


# ============================================================
# RUN / RESUME 12 TRIALS
# ============================================================

all_results = []

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
env["PYTHONUNBUFFERED"] = "1"
env["OMP_NUM_THREADS"] = "8"
env["MKL_NUM_THREADS"] = "8"
env["OPENBLAS_NUM_THREADS"] = "8"


for t in TRIALS:

    num = t["trial"]

    trial_dir = (
        TUNE /
        f"trial_{num:02d}"
    )

    summary_path = (
        trial_dir /
        "trial_summary.json"
    )

    # --------------------------------------------------------
    # SAFE RESUME
    # --------------------------------------------------------

    if summary_path.exists():

        old = json.loads(
            summary_path.read_text()
        )

        if (
            old.get("status") == "PASS"
            and old.get("trial") == num
            and old.get("d_hidden") == t["d_hidden"]
            and old.get("n_bins") == t["n_bins"]
            and abs(
                float(old.get("lr"))
                - float(t["lr"])
            ) < 1e-15
            and old.get("split_sha256")
                == EXPECTED_SPLIT_SHA
            and old.get("repository_commit")
                == EXPECTED_COMMIT
            and old.get("test_evaluated") is False
        ):
            print(
                f"\nRESUME PASS | "
                f"trial={num:02d}/12 | "
                f"valAUPRC="
                f"{old['best_val_auprc']:.6f}",
                flush=True
            )

            all_results.append(old)
            continue

    if trial_dir.exists():
        shutil.rmtree(trial_dir)

    trial_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    script = (
        trial_dir /
        f"trial_{num:02d}.py"
    )

    log = (
        trial_dir /
        f"trial_{num:02d}.log"
    )

    script.write_text(
        make_trial_source(t)
    )

    # syntax gate
    subprocess.run(
        [
            str(LAUNCHER),
            "-m",
            "py_compile",
            str(script)
        ],
        check=True,
        env=env
    )

    print(
        "\n"
        + "=" * 72,
        flush=True
    )

    print(
        f"START TRIAL {num:02d}/12 | "
        f"TR40 seed=2 | "
        f"hidden={t['d_hidden']} | "
        f"bins={t['n_bins']} | "
        f"lr={t['lr']}",
        flush=True
    )

    print(
        "=" * 72,
        flush=True
    )

    # --------------------------------------------------------
    # LIVE STREAM
    # --------------------------------------------------------

    with log.open(
        "w",
        buffering=1
    ) as lf:

        p = subprocess.Popen(
            [
                str(LAUNCHER),
                str(script)
            ],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        assert p.stdout is not None

        for line in p.stdout:
            print(
                line,
                end="",
                flush=True
            )
            lf.write(line)

        rc = p.wait()

    if rc != 0:
        raise RuntimeError(
            f"TRIAL {num:02d} FAILED | "
            f"exit={rc} | log={log}"
        )

    raw_path = (
        trial_dir /
        "trial_result.json"
    )

    assert raw_path.exists()

    raw = json.loads(
        raw_path.read_text()
    )

    assert raw["test_evaluated"] is False
    assert raw["test_labels_hidden_during_fit"] is True
    assert raw["author_architecture_changed"] is False
    assert raw["dataset_sha256"] == EXPECTED_DATA_SHA
    assert raw["split_sha256"] == EXPECTED_SPLIT_SHA
    assert raw["repository_commit"] == EXPECTED_COMMIT

    result = {
        "status": "PASS",

        "trial": num,

        "d_hidden":
            t["d_hidden"],

        "n_bins":
            t["n_bins"],

        "lr":
            t["lr"],

        "best_val_auprc":
            float(
                raw["best_val_auprc"]
            ),

        "best_checkpoint":
            raw["best_checkpoint"],

        "training_wall_seconds":
            float(
                raw["training_wall_seconds"]
            ),

        "peak_gpu_memory_mb":
            float(
                raw["peak_gpu_memory_mb"]
            ),

        "test_evaluated":
            False,

        "test_labels_hidden_during_fit":
            True,

        "repository_commit":
            EXPECTED_COMMIT,

        "dataset_sha256":
            EXPECTED_DATA_SHA,

        "split_sha256":
            EXPECTED_SPLIT_SHA,
    }

    summary_path.write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    all_results.append(
        result
    )

    print(
        f"TRIAL {num:02d} DONE | "
        f"best val AUPRC="
        f"{result['best_val_auprc']:.6f}",
        flush=True
    )

    # Author source must remain untouched after every trial.
    assert git(
        "status",
        "--porcelain",
        "--untracked-files=no"
    ) == ""


# ============================================================
# COMPLETE RESULTS
# ============================================================

assert len(all_results) == 12

all_results = sorted(
    all_results,
    key=lambda x: x["trial"]
)

json_path = (
    TUNE /
    "GAAP_YELP_TR40_TUNING_TRIALS.json"
)

json_path.write_text(
    json.dumps(
        all_results,
        indent=2
    )
)

csv_path = (
    TUNE /
    "GAAP_YELP_TR40_TUNING_TRIALS.csv"
)

with csv_path.open(
    "w",
    newline=""
) as f:

    fields = [
        "trial",
        "d_hidden",
        "n_bins",
        "lr",
        "best_val_auprc",
        "training_wall_seconds",
        "peak_gpu_memory_mb",
        "status",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()

    for r in all_results:
        w.writerow({
            k: r[k]
            for k in fields
        })


# ============================================================
# WINNER
#
# Primary ordering = validation AUPRC.
# Exact ties = earlier preregistered trial number.
# ============================================================

winner = sorted(
    all_results,
    key=lambda x: (
        -x["best_val_auprc"],
        x["trial"]
    )
)[0]

frozen = {
    "model": "GAAP",
    "dataset": "YelpChi",
    "mode": "unified",
    "stage": "frozen_tuning_winner",

    "selection_metric":
        "validation_AUPRC",

    "tie_break":
        "lower preregistered trial number "
        "only on exact AUPRC equality",

    "ratio_used_for_tuning":
        "TR40",

    "split_seed":
        2,

    "training_seed":
        2,

    "max_trials":
        12,

    "max_epochs":
        100,

    "patience":
        20,

    "test_accessed":
        False,

    "winner": winner,

    "fixed": prereg["fixed"],

    "repository_commit":
        EXPECTED_COMMIT,

    "dataset_sha256":
        EXPECTED_DATA_SHA,

    "split_sha256":
        EXPECTED_SPLIT_SHA,
}

winner_path = (
    TUNE /
    "GAAP_YELP_FROZEN_WINNER.json"
)

winner_path.write_text(
    json.dumps(
        frozen,
        indent=2
    )
)


# ============================================================
# HUMAN-READABLE RESULT
# ============================================================

print(
    "\n"
    + "=" * 72,
    flush=True
)

print(
    "===== GAAP YELP TR40 TUNING RESULTS =====",
    flush=True
)

ranked = sorted(
    all_results,
    key=lambda x: (
        -x["best_val_auprc"],
        x["trial"]
    )
)

for r in ranked:

    print(
        f"trial={r['trial']:02d} | "
        f"hidden={r['d_hidden']:3d} | "
        f"bins={r['n_bins']:2d} | "
        f"lr={r['lr']:.4f} | "
        f"valAUPRC="
        f"{r['best_val_auprc']:.6f}",
        flush=True
    )


print(
    "\n===== FROZEN WINNER =====",
    flush=True
)

print(
    f"trial={winner['trial']:02d}",
    flush=True
)

print(
    f"d_hidden={winner['d_hidden']}",
    flush=True
)

print(
    f"n_bins={winner['n_bins']}",
    flush=True
)

print(
    f"lr={winner['lr']}",
    flush=True
)

print(
    "gnn_n_layers=3",
    flush=True
)

print(
    "gnn_dropout=0.1",
    flush=True
)

print(
    "gnn_agg=max_pool",
    flush=True
)

print(
    "use_dyple=True",
    flush=True
)

print(
    "use_mha=True",
    flush=True
)

print(
    f"BEST_VAL_AUPRC="
    f"{winner['best_val_auprc']:.6f}",
    flush=True
)

print(
    "TEST_ACCESSED_DURING_TUNING=NO",
    flush=True
)

print(
    "TUNING_TRIALS_COMPLETED=12/12",
    flush=True
)

print(
    "GAAP_YELP_TUNING=PASS",
    flush=True
)

print(
    "CONFIGURATION_FROZEN=YES",
    flush=True
)

print(
    "NEXT=GAAP_YELP_FINAL_12_RUNS",
    flush=True
)

assert git(
    "status",
    "--porcelain",
    "--untracked-files=no"
) == ""

print(
    "GAAP_AUTHOR_REPO_CLEAN=PASS",
    flush=True
)
