"""COMP8851 Vast.ai setup: bare instance to a verified environment, one command.

    python3 scripts/vastai_setup.py

Written in Python rather than shell deliberately: Python is guaranteed present
on the instance, and this file is covered by the local test suite, so what runs
on the rented GPU is code that has actually been exercised.

Order matters, and each step gates the next:

  1. Resource controls, set before anything heavy starts.
  2. GPU check. Stops if there is no usable GPU, because a controlled run on
     CPU is not comparable and would waste rental time.
  3. torch 2.4.0 + cu121. This is the key step, and the version is exact.
     DGL's wheels are compiled against one PyTorch ABI: DGL 2.4.0 needs torch
     2.4.0, and even 2.4.1 breaks `import dgl`. Pinning it here makes DGL work
     natively, which unblocks the three datasets stored as DGL binaries and
     lets GHRN run on the authors' real stack.
  4. DGL from the matching CUDA 12.1 index.
  5. The rest of the stack at protocol-fixed versions.
  6. Environment and hardware records for the manifest.
  7. Datasets.
  8. Self-test.

Safe to re-run: every step detects work already done and skips it.

Options:
    --skip-datasets     set up the environment only
    --check-only        report what is present, install nothing
    --no-torch-pin      leave torch alone (use if the image already suits you)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 2.4.0 exactly, not 2.4.1. DGL 2.4.0's wheels are compiled against the
# torch 2.4.0 ABI; on 2.4.1 `import dgl` raises and GHRN cannot run at all.
TORCH_VERSION = "2.4.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
DGL_INDEXES = ("https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html",
               "https://data.dgl.ai/wheels/torch-2.4/repo.html")

STACK = ("numpy==1.26.4", "scipy==1.12.0", "scikit-learn==1.4.2",
         "pandas==2.2.1", "sympy==1.12", "PyYAML==6.0.1", "psutil==7.0.0",
         "matplotlib==3.8.3", "gdown", "kagglehub")

THREAD_CAPS = {
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8",
}

failures: list[str] = []
warnings: list[str] = []


def step(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def ok(message: str) -> None:
    print(f"  [ok]   {message}")


def warn(message: str) -> None:
    warnings.append(message)
    print(f"  [warn] {message}")


def fail(message: str) -> None:
    failures.append(message)
    print(f"  [FAIL] {message}")


def run(command: list[str], label: str, timeout: int = 3600) -> bool:
    print(f"  ...    {label}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        fail(f"{label} timed out after {timeout}s")
        return False
    except OSError as exc:
        fail(f"{label} could not start: {exc}")
        return False
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        print(f"         {tail[-1][:200] if tail else 'no output'}")
        return False
    return True


def pip(*packages: str, extra: tuple[str, ...] = ()) -> bool:
    return run([sys.executable, "-m", "pip", "install", "-q", *packages, *extra],
               f"pip install {' '.join(packages)[:60]}")


def probe(code: str) -> tuple[bool, str]:
    """Run a snippet in this interpreter and return (success, output)."""
    try:
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def apply_thread_caps() -> None:
    for name, value in THREAD_CAPS.items():
        os.environ[name] = value
    bashrc = Path.home() / ".bashrc"
    try:
        existing = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
        additions = [f"export {name}={value}" for name, value in THREAD_CAPS.items()
                     if f"export {name}={value}" not in existing]
        if additions:
            with bashrc.open("a", encoding="utf-8") as handle:
                handle.write("\n# COMP8851 controlled-run resource limits\n")
                handle.write("\n".join(additions) + "\n")
        ok("resource controls set and persisted to ~/.bashrc")
    except OSError:
        warn("could not persist resource controls to ~/.bashrc; they are set for "
             "this process only")


def check_gpu() -> str | None:
    success, output = probe(
        "import torch;"
        "print('CUDA', torch.cuda.is_available());"
        "print('NAME', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none');"
        "print('VRAM', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1)"
        " if torch.cuda.is_available() else 0)"
    )
    if not success:
        # torch may not be installed yet; fall back to the driver.
        smi, smi_output = probe("import subprocess;"
                                "print(subprocess.run(['nvidia-smi','--query-gpu=name',"
                                "'--format=csv,noheader'],capture_output=True,text=True).stdout)")
        if smi and smi_output.strip():
            name = smi_output.strip().splitlines()[0]
            ok(f"GPU visible to the driver: {name}")
            return name
        fail("no usable GPU detected. A controlled run must not proceed on CPU.")
        return None

    values = dict(line.split(" ", 1) for line in output.splitlines() if " " in line)
    if values.get("CUDA") != "True":
        fail("torch reports CUDA unavailable. Stop; check the instance has a GPU "
             "attached and the image is a CUDA build.")
        return None
    name = values.get("NAME", "unknown")
    ok(f"GPU: {name} ({values.get('VRAM', '?')} GB)")
    if "t4" in name.lower():
        ok("Tesla T4 matches the controlled specification")
    else:
        warn(f"GPU is '{name}', not a Tesla T4. Predictive metrics stay valid, but "
             "runtime and memory are NOT comparable with T4 numbers. The manifest "
             "records the GPU actually used.")
    return name


def dgl_usable() -> bool:
    success, _ = probe("import dgl; dgl.graph(([0],[1])); print(dgl.__version__)")
    return success


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COMP8851 Vast.ai setup")
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--check-only", action="store_true",
                        help="Report what is present; install nothing")
    parser.add_argument("--no-torch-pin", action="store_true",
                        help="Leave the installed torch alone")
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    print(f"Repository: {REPO_ROOT}")
    print(f"Python    : {sys.version.split()[0]}")

    step("1. Resource controls")
    if args.check_only:
        print("  (check-only: not modifying the environment)")
    else:
        apply_thread_caps()

    step("2. GPU")
    gpu = check_gpu()
    if gpu is None and not args.check_only:
        print("\nStopping: without a GPU this rental would only waste money.")
        return 1

    step(f"3. torch {TORCH_VERSION} + CUDA 12.1")
    success, current = probe("import torch; print(torch.__version__)")
    current = current.strip() if success else "not installed"
    print(f"  currently: {current}")
    if args.check_only or args.no_torch_pin:
        print("  (skipping the pin)")
    elif current.startswith(TORCH_VERSION):
        ok("already at the pinned version")
    else:
        print(f"  installing torch {TORCH_VERSION}+cu121; this takes a few minutes")
        if pip(f"torch=={TORCH_VERSION}", extra=("--index-url", TORCH_INDEX)):
            ok(f"torch {TORCH_VERSION} installed")
        else:
            fail(f"could not install torch {TORCH_VERSION}. DGL will not work, so "
                 "three datasets will fall back to the isolated decoder.")

    step("4. DGL")
    if dgl_usable():
        _, version = probe("import dgl; print(dgl.__version__)")
        ok(f"DGL already usable: {version}")
    elif args.check_only:
        warn("DGL not usable (check-only, nothing installed)")
    else:
        pip("torchdata==0.7.1")
        for index in DGL_INDEXES:
            pip("dgl", extra=("-f", index))
            if dgl_usable():
                break
        if dgl_usable():
            _, version = probe("import dgl; print(dgl.__version__)")
            ok(f"DGL usable: {version}  (all six datasets reachable)")
        else:
            warn("DGL is not usable. FDCompCN, T-Finance and T-Social will be "
                 "decoded by the isolated environment instead, and GHRN will use "
                 "its torch.sparse backend. Everything still runs; record it.")

    step("5. Scientific stack")
    if args.check_only:
        for package in ("numpy", "scipy", "sklearn", "pandas", "sympy",
                        "yaml", "matplotlib", "psutil"):
            present, version = probe(f"import {package}; "
                                     f"print(getattr({package}, '__version__', 'present'))")
            print(f"  {'[ok]  ' if present else '[miss]'} {package}: "
                  f"{version if present else 'missing'}")
    elif pip(*STACK):
        ok("stack installed")
    else:
        warn("some packages failed to install; check the messages above")

    step("6. Environment record")
    docs = REPO_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    if not args.check_only:
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True)
        (docs / "vastai_environment.txt").write_text(frozen.stdout, encoding="utf-8")
        hardware = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv"], capture_output=True, text=True)
        (docs / "vastai_hardware.txt").write_text(hardware.stdout or "nvidia-smi unavailable",
                                                  encoding="utf-8")
        ok("wrote docs/vastai_environment.txt and docs/vastai_hardware.txt")

    step("7. Datasets")
    if args.skip_datasets or args.check_only:
        print("  (skipped)")
    else:
        if not os.environ.get("KAGGLE_USERNAME"):
            warn("KAGGLE_USERNAME is not set, so Elliptic will be skipped. Export "
                 "KAGGLE_USERNAME and KAGGLE_KEY, then re-run this script.")
        run([sys.executable, "shared/comp8851/fetch_datasets.py",
             "--data-root", "data", "--freeze"], "fetching and freezing datasets",
            timeout=10800)

    step("8. Self-test")
    if args.check_only:
        print("  (skipped)")
    else:
        result = subprocess.run([sys.executable, "shared/comp8851/selftest.py"],
                                capture_output=True, text=True, timeout=3600)
        for line in result.stdout.strip().splitlines()[-6:]:
            print(f"  {line}")
        if result.returncode != 0:
            fail("self-test failed. Do not start the benchmark on this machine.")

    step("Summary")
    if failures:
        print(f"  {len(failures)} blocking problem(s):")
        for item in failures:
            print(f"    FAIL {item}")
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for item in warnings:
            print(f"    warn {item}")
    if not failures and not warnings:
        print("  everything checks out")

    if failures:
        print("\nFix the failures before renting more time.")
        return 1

    print("\nNext, and this one is worth the ten minutes:")
    print("    python3 scripts/preflight.py")
    print("\nIt proves every model-dataset pair trains before you commit hours of GPU time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
