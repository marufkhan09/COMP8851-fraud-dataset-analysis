#!/usr/bin/env python3
from pathlib import Path
import hashlib, zipfile, tarfile, shutil, sys, os

ROOT = Path("/workspace")
TARGET_SPLIT_SHA = "0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b"
TARGET_BUNDLE_SHA = "562a1d0a2b27942af6ffae31e30bf683f0696ce6a418c7e37343178ebee1e1cc"
OUT = Path("/workspace/pmp_vast/shared/yelp_seed2_nested_splits.npz")
OUT.parent.mkdir(parents=True, exist_ok=True)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha_file(path, block=8*1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(block), b""):
            h.update(b)
    return h.hexdigest()

def accept_bytes(data, source):
    h = sha_bytes(data)
    if h == TARGET_SPLIT_SHA:
        OUT.write_bytes(data)
        print(f"EXACT_SPLIT_FOUND={source}")
        print(f"EXACT_SPLIT_SHA256={h}")
        print(f"RESTORED_TO={OUT}")
        return True
    return False

print("===== SEARCH DIRECT SPLIT FILES =====", flush=True)

# Direct candidates: exact/similar names and all small NPZs under likely evidence paths.
direct = []
for base in [ROOT, ROOT/"dataset_audit", ROOT/"bwgnn_vast", ROOT/"pmp_vast"]:
    if not base.exists():
        continue
    for pattern in ["**/yelp_seed2_nested_splits.npz", "**/*yelp*split*.npz", "**/*.npz"]:
        try:
            direct.extend(base.glob(pattern))
        except Exception:
            pass

seen=set()
for path in direct:
    try:
        rp=path.resolve()
    except Exception:
        rp=path
    if rp in seen or not path.is_file():
        continue
    seen.add(rp)
    try:
        # Skip unexpectedly large NPZs; the split file should be small.
        if path.stat().st_size > 20*1024*1024:
            continue
        h=sha_file(path)
        if "yelp" in path.name.lower() or "split" in path.name.lower():
            print(f"{h}  {path}", flush=True)
        if h == TARGET_SPLIT_SHA:
            shutil.copy2(path, OUT)
            print(f"EXACT_SPLIT_FOUND={path}")
            print(f"EXACT_SPLIT_SHA256={h}")
            print(f"RESTORED_TO={OUT}")
            sys.exit(0)
    except Exception as e:
        pass

print("\n===== SEARCH ARCHIVES =====", flush=True)

archive_exts = (".zip",".tar.gz",".tgz",".tar")
archives=[]
for base in [ROOT, ROOT/"dataset_audit", ROOT/"bwgnn_vast"]:
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        low=path.name.lower()
        if low.endswith(archive_exts):
            archives.append(path)

seen=set()
for path in archives:
    try:
        rp=path.resolve()
    except Exception:
        rp=path
    if rp in seen:
        continue
    seen.add(rp)

    try:
        # First check if this is the expected frozen split bundle itself.
        if path.stat().st_size <= 100*1024*1024:
            h=sha_file(path)
            if h == TARGET_BUNDLE_SHA:
                print(f"EXACT_BUNDLE_FOUND={path}", flush=True)
                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as z:
                        names=z.namelist()
                        for name in names:
                            if name.endswith("yelp_seed2_nested_splits.npz"):
                                data=z.read(name)
                                if accept_bytes(data, f"{path}!{name}"):
                                    sys.exit(0)
                print("Expected bundle hash found, but exact Yelp split member was not recoverable.", flush=True)

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    low=name.lower()
                    if (
                        low.endswith("yelp_seed2_nested_splits.npz")
                        or ("yelp" in low and "split" in low and low.endswith(".npz"))
                    ):
                        try:
                            data=z.read(name)
                        except Exception:
                            continue
                        print(f"ARCHIVE_CANDIDATE={path}!{name} SHA256={sha_bytes(data)}", flush=True)
                        if accept_bytes(data, f"{path}!{name}"):
                            sys.exit(0)
            continue

        # tar / tar.gz / tgz
        if tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as t:
                for m in t.getmembers():
                    if not m.isfile():
                        continue
                    low=m.name.lower()
                    if (
                        low.endswith("yelp_seed2_nested_splits.npz")
                        or ("yelp" in low and "split" in low and low.endswith(".npz"))
                    ):
                        f=t.extractfile(m)
                        if f is None:
                            continue
                        data=f.read()
                        print(f"ARCHIVE_CANDIDATE={path}!{m.name} SHA256={sha_bytes(data)}", flush=True)
                        if accept_bytes(data, f"{path}!{m.name}"):
                            sys.exit(0)
    except Exception as e:
        print(f"ARCHIVE_SCAN_WARNING={path}: {type(e).__name__}: {e}", flush=True)

print("\nPMP_YELP_EXACT_SPLIT_SEARCH=NOT_FOUND", flush=True)
print("Expected split SHA256:", TARGET_SPLIT_SHA, flush=True)
print("Expected bundle SHA256:", TARGET_BUNDLE_SHA, flush=True)
sys.exit(31)
