"""Kaggle GPU integration test for the COMP8851 phase-2 bundle.

Runs on a Kaggle notebook/script kernel with GPU and Internet enabled and the
private dataset ``<user>/comp8851-phase2-bundle`` attached. It executes the
*same* production modules used on Vast.ai:

  1. install torch 2.4.0 + dgl 2.4.0 matching the session's CUDA
  2. verify the environment (GPU type, VRAM, versions, DGL GPU op)
  3. fetch the pinned official implementations (file-hash verified)
  4. prepare / verify all six datasets against the previous benchmark's hashes
  5. run the automated test suite (real data)
  6. GPU smoke matrix: 18 pairs x 2 epochs (T-Social on the documented
     300k-node subset, because a Kaggle session has ~30 GB RAM)
  7. GPU pilot: full protocol-budget TR40 seed-2 runs on FDCompCN for all
     three models (real early stopping, real checkpoints)
  8. aggregate + independent integrity verification
Everything is written to /kaggle/working/outputs. Smoke and pilot outputs are
feasibility evidence only - they are not benchmark results.
"""

import collections
import collections.abc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Python 3.10+ / 3.13 compatibility shim for DGL and older scientific libraries
for _name in ("Mapping", "MutableMapping", "Iterable", "Iterator", "Sequence",
              "MutableSequence", "Callable", "Container", "Set", "MutableSet"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

WORK = Path("/kaggle/working")
OUT = WORK / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "kaggle_run.log", "a", buffering=1)
STEPS = []


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


def run(name, cmd, cwd=None, check=False):
    log(f"$ {' '.join(map(str, cmd))}")
    t0 = time.time()
    env = dict(os.environ, DGLBACKEND="pytorch", VIRTUAL_ENV=str(WORK / "venv311"),
               PATH=f"{WORK}/venv311/bin:{os.environ.get('PATH', '')}")
    if cwd:
        root_dir = str(cwd)
        env["PYTHONPATH"] = f"{root_dir}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(list(map(str, cmd)), cwd=cwd, capture_output=True, text=True, env=env)
    (OUT / f"{name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    elapsed = time.time() - t0
    STEPS.append({"step": name, "returncode": proc.returncode, "seconds": round(elapsed, 1)})
    log(f"  -> exit {proc.returncode} in {elapsed:.0f}s (log: outputs/{name}.log)")
    tail = (proc.stdout or "").strip().splitlines()[-12:]
    for line in tail:
        log("     " + line)
    if check and proc.returncode != 0:
        raise SystemExit(f"step {name} failed; see outputs/{name}.log")
    return proc.returncode


def locate_bundle():
    for candidate in Path("/kaggle/input").rglob("phase2/harness.py"):
        return candidate.parents[1]
    import zipfile

    for archive in Path("/kaggle/input").rglob("comp8851_phase2.zip"):  # Kaggle did not auto-extract
        target = WORK / "bundle_extracted"
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        return target / "comp8851_phase2"
    raise SystemExit("bundle not found under /kaggle/input: attach the comp8851-phase2-bundle dataset")


def main():
    src = locate_bundle()
    root = WORK / "comp8851_phase2"
    if not root.exists():
        shutil.copytree(src, root)
    log(f"bundle copied from {src} to {root}")
    # Ensure Python 3.11 with official CUDA PyTorch 2.4.0 and DGL 2.4.0
    py311_bin = WORK / "venv311" / "bin" / "python"
    if not py311_bin.exists():
        log("Bootstrapping Python 3.11 with official PyTorch 2.4.0 + DGL 2.4.0 via uv...")
        run("install_uv", ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], check=False)
        uv_candidates = [
            Path(os.path.expanduser("~/.local/bin/uv")),
            Path("/root/.local/bin/uv"),
            Path("/usr/local/bin/uv"),
            Path(shutil.which("uv")) if shutil.which("uv") else None
        ]
        uv_exe = next((str(p) for p in uv_candidates if p and Path(p).exists()), None)
        if uv_exe:
            run("create_venv", [uv_exe, "venv", "--seed", "--python", "3.11", str(WORK / "venv311")])
            run("install_torch", [uv_exe, "pip", "install", "--python", str(py311_bin), "torch==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu121"])
            run("install_dgl", [uv_exe, "pip", "install", "--python", str(py311_bin), "dgl==2.4.0", "-f", "https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html"])
            run("install_common", [uv_exe, "pip", "install", "--python", str(py311_bin), "pyyaml", "tqdm", "psutil", "pytest", "pydantic", "requests", "networkx", "scikit-learn", "scipy"])
    
    if py311_bin.exists():
        py = str(py311_bin)
        log(f"Active runtime environment: {py} (Python 3.11 + PyTorch 2.4.0 + DGL 2.4.0)")
    else:
        py = sys.executable
        log(f"Falling back to container runtime: {py}")
    run("verify_env", [py, "scripts/verify_env.py"], cwd=root, check=False)
    run("fetch_official", [py, "scripts/fetch_official.py"], cwd=root, check=True)
    run("prepare_data", [py, "scripts/prepare_data.py"], cwd=root)
    run("eda", [py, "scripts/run_eda.py"], cwd=root)
    run("pytest", [py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning"], cwd=root)
    run("smoke_matrix", [py, "scripts/run_matrix.py", "smoke", "--smoke-subsets"], cwd=root)
    run("smoke_aggregate", [py, "scripts/aggregate_results.py", "--phase", "smoke"], cwd=root)
    run("smoke_verify", [py, "scripts/verify_results.py", "--phase", "smoke"], cwd=root)
    run("pilot_fdcompcn", [py, "scripts/run_matrix.py", "pilot", "--datasets", "fdcompcn"], cwd=root)
    run("pilot_aggregate", [py, "scripts/aggregate_results.py", "--phase", "pilot"], cwd=root)
    run("pilot_verify", [py, "scripts/verify_results.py", "--phase", "pilot", "--deep"], cwd=root)

    for item in ("results/aggregated", "results/ledger.csv", "results/env_report.json", "results/logs",
                 "results/eda", "data/canonical/data_status.json"):
        p = root / item
        if p.is_dir():
            shutil.copytree(p, OUT / p.name, dirs_exist_ok=True)
        elif p.exists():
            shutil.copy2(p, OUT / p.name)
    for phase in ("smoke", "pilot"):
        for summary in (root / "results" / phase).rglob("summary.json"):
            rel = summary.relative_to(root / "results")
            dest = OUT / "run_summaries" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary, dest)
            for extra in ("terminal.log", "predictions.npz", "test_metrics.json"):
                if (summary.parent / extra).exists():
                    shutil.copy2(summary.parent / extra, dest.parent / extra)
    (OUT / "kaggle_steps.json").write_text(json.dumps(STEPS, indent=2))
    # Clean up large canonical directory from root so Kaggle outputs only contain deliverables
    shutil.rmtree(root / "data" / "canonical", ignore_errors=True)
    log("finished; outputs in /kaggle/working/outputs")


if __name__ == "__main__":
    main()
