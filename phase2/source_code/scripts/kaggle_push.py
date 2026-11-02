"""Upload the bundle to Kaggle as a private dataset, run the GPU test kernel,
wait for it and download its outputs. Uses the official Kaggle CLI only.

    python scripts/kaggle_push.py --user ahmadyasin1            # everything
    python scripts/kaggle_push.py --user ahmadyasin1 --status   # poll only
    python scripts/kaggle_push.py --user ahmadyasin1 --download # fetch outputs

Authentication: the Kaggle CLI reads the token from ~/.kaggle/access_token
(or kaggle.json / the KAGGLE_API_TOKEN environment variable) that YOU
configured. This script never reads, prints or stores a credential; it only
checks that the CLI can authenticate (`kaggle config view` exit code).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "dist" / "kaggle_stage"
INCLUDE = ["phase2", "models", "shared", "scripts", "configs", "tests", "env", "third_party/manifest.json",
           "kaggle", "README.md", "audit", "sitecustomize.py"]
CANONICAL = ["yelpchi", "amazon", "tfinance", "elliptic", "fdcompcn", "tsocial"]


def kaggle(*args, capture=True):
    exe = shutil.which("kaggle")
    cmd = [exe, *args] if exe else [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(cmd, capture_output=capture, text=True)


def check_auth():
    proc = kaggle("datasets", "list", "--mine", "--page-size", "1")
    if proc.returncode != 0:
        raise SystemExit("Kaggle CLI cannot authenticate. Configure your *new* token yourself "
                         "(see kaggle/README_KAGGLE.md); never paste it into a chat or a file in this bundle.")


def stage(user: str) -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    data_dir = STAGE / "data"
    data_dir.mkdir(parents=True)
    archive = data_dir / "comp8851_phase2.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
            for f in files:
                z.write(f, Path("comp8851_phase2") / f.relative_to(ROOT))
        for name in CANONICAL:
            f = ROOT / "data" / "canonical" / f"{name}_canonical.npz"
            if f.exists():
                z.write(f, Path("comp8851_phase2") / f.relative_to(ROOT))
        status = ROOT / "data" / "canonical" / "data_status.json"
        if status.exists():
            z.write(status, Path("comp8851_phase2") / status.relative_to(ROOT))
    (data_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "comp8851-phase2-bundle", "id": f"{user}/comp8851-phase2-bundle",
        "licenses": [{"name": "unknown"}]}, indent=2))
    kernel_dir = STAGE / "kernel"
    kernel_dir.mkdir()
    shutil.copy2(ROOT / "kaggle" / "kaggle_smoke.py", kernel_dir / "kaggle_smoke.py")
    meta = json.loads((ROOT / "kaggle" / "kernel-metadata.template.json").read_text())
    meta["id"] = meta["id"].replace("KAGGLE_USERNAME", user)
    meta["dataset_sources"] = [s.replace("KAGGLE_USERNAME", user) for s in meta["dataset_sources"]]
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"staged {archive} ({archive.stat().st_size / 2**20:.1f} MiB)")
    return STAGE


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sys.stdout.reconfigure(line_buffering=True)
    p.add_argument("--user", required=True)
    p.add_argument("--status", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--kernel-only", action="store_true")
    p.add_argument("--wait-minutes", type=float, default=600)
    args = p.parse_args(argv)
    kernel = f"{args.user}/comp8851-phase2-smoke"
    out = ROOT / "results" / "kaggle"
    check_auth()
    import tempfile
    (Path(tempfile.gettempdir()) / ".kaggle" / "uploads").mkdir(parents=True, exist_ok=True)
    if not (args.status or args.download):
        staged = stage(args.user)
        kernel_path = str((staged / "kernel").resolve())
        if not args.kernel_only:
            data_path = str((staged / "data").resolve())
            exists = kaggle("datasets", "status", f"{args.user}/comp8851-phase2-bundle").returncode == 0
            if exists:
                proc = kaggle("datasets", "version", "-p", data_path, "-m", "bundle update", "--dir-mode", "zip")
            else:
                proc = kaggle("datasets", "create", "-p", data_path, "--dir-mode", "zip")
            print(proc.stdout, proc.stderr, flush=True)
            if proc.returncode != 0:
                return 1
            print("waiting 120 s for Kaggle to process the dataset ...", flush=True)
            time.sleep(120)
        proc = kaggle("kernels", "push", "-p", kernel_path)
        print(proc.stdout, proc.stderr, flush=True)
        if proc.returncode != 0:
            return 1
        print("Kernel pushed. Waiting 30s for Kaggle to transition to queued/running ...", flush=True)
        time.sleep(30)
    deadline = time.time() + args.wait_minutes * 60
    while True:
        proc = kaggle("kernels", "status", kernel)
        text = (proc.stdout + proc.stderr).strip()
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        if args.status or any(s in text.lower() for s in ("complete", "error", "cancel")) or time.time() > deadline:
            break
        time.sleep(45)
    if args.download or "complete" in text.lower() or "error" in text.lower():
        out.mkdir(parents=True, exist_ok=True)
        proc = kaggle("kernels", "output", kernel, "-p", str(out))
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        print(f"Kaggle outputs saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
