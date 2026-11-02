#!/usr/bin/env python3
from pathlib import Path
import hashlib, itertools
import numpy as np

SRC = Path("/workspace/bwgnn_vast/splits/yelp_seed2_nested_splits.npz")
OUTDIR = Path("/workspace/pmp_vast/shared")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "yelp_seed2_nested_splits.npz"
TARGET = "0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

assert SRC.exists(), SRC
print("Source candidate:", SRC)
print("Current SHA256 :", sha256(SRC))

s = np.load(SRC, allow_pickle=False)
print("Current keys   :", s.files)
for k in s.files:
    a = s[k]
    print(f"  {k}: shape={a.shape} dtype={a.dtype}")

sizes = {"TR40":18381,"TR30":13785,"TR20":9190,"TR10":4595,"val":9191,"test":18382}
for k,n in sizes.items():
    assert k in s.files, f"missing {k}"
    assert len(s[k]) == n, (k, len(s[k]), n)

base = {k: np.asarray(s[k]) for k in ["TR40","TR30","TR20","TR10","val","test"]}
sets = {k:set(map(int,v)) for k,v in base.items()}
assert sets["TR10"] <= sets["TR20"] <= sets["TR30"] <= sets["TR40"]
for tr in ["TR40","TR30","TR20","TR10"]:
    assert sets[tr].isdisjoint(sets["val"])
    assert sets[tr].isdisjoint(sets["test"])
assert sets["val"].isdisjoint(sets["test"])
print("Structural split gate: PASS")

base_orders = [
    ["TR40","TR30","TR20","TR10","val","test"],
    ["TR10","TR20","TR30","TR40","val","test"],
]
meta_defs = [
    ("seed",2),
    ("source_nodes",45954),
    ("source_normal",39277),
    ("source_fraud",6677),
]
name_variants = [
    [("seed",2),("source_nodes",45954)],
    [("seed",2),("source_nodes",45954),("source_normal",39277),("source_fraud",6677)],
    [("seed",2),("source_nodes",45954),("source_fraud",6677),("source_normal",39277)],
]
for perm in itertools.permutations(meta_defs):
    name_variants.append(list(perm))

tmp = OUTDIR / "._candidate_yelp_split.npz"
checked = 0
found = False

for order in base_orders:
    for arr_dtype in [None, np.int64, np.int32]:
        for meta_dtype in [np.int64, np.int32]:
            for mv in name_variants:
                names = [k for k,_ in mv]
                if len(names) != len(set(names)):
                    continue
                payload = {}
                for k in order:
                    arr = np.asarray(base[k])
                    if arr_dtype is not None:
                        arr = arr.astype(arr_dtype, copy=False)
                    payload[k] = arr
                for k,v in mv:
                    payload[k] = np.array([v], dtype=meta_dtype)

                if tmp.exists():
                    tmp.unlink()
                np.savez_compressed(tmp, **payload)
                checked += 1

                if sha256(tmp) == TARGET:
                    tmp.replace(OUT)
                    found = True
                    break
            if found: break
        if found: break
    if found: break

print("Candidate layouts checked:", checked)
if found:
    print("PMP_YELP_SPLIT_RECOVERY=PASS")
    print("Recovered exact SHA256:", sha256(OUT))
    z = np.load(OUT, allow_pickle=False)
    print("Recovered keys:", z.files)
else:
    if tmp.exists():
        tmp.unlink()
    print("PMP_YELP_SPLIT_RECOVERY=NO_EXACT_MATCH")
    print("The existing BWGNN split IDs passed structural checks,")
    print("but none of the tested deterministic packaging variants")
    print("reproduced the registered PMP split bytes.")
    print("Do NOT start PMP training until the exact split asset is restored.")
    raise SystemExit(30)
