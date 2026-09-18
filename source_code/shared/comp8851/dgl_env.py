"""Build an isolated interpreter that can read DGL files, when the main one cannot.

Why this exists
---------------
DGL ships compiled wheels linked against a specific PyTorch ABI. The published
wheels currently top out at torch 2.4, while hosted notebook environments move
much faster (Kaggle ships torch 2.10 at the time of writing). On such a host
``pip install dgl`` appears to succeed but the import fails, so the three
datasets stored as DGL binaries become unreadable.

Rather than downgrade the training environment, which would change the very
hardware and software specification the benchmark is supposed to hold fixed,
this module creates a **separate, disposable virtual environment** whose only
job is to decode those files into the canonical ``.npz`` format.

The training environment is left untouched. After conversion:

* CARE-GNN reads the ``.npz`` and needs no graph library at all.
* GHRN reads the ``.npz`` and uses its ``torch.sparse`` backend.

The conversion is one-off, CPU-only, and its output is deterministic, so it can
be done once and the resulting files shared with the rest of the team.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Torch builds paired with the DGL wheel index compiled against them, newest
#: first. DGL links against a specific PyTorch ABI, so the pair must match.
#: Several are listed because no single torch release has wheels for every
#: Python version; the first pair that installs and imports wins.
#: Each entry is (torch version, preferred DGL version, wheel index).
#:
#: The torch patch version matters. A DGL wheel is compiled against one exact
#: PyTorch build: DGL 2.4.0 needs torch 2.4.0, and even 2.4.1 makes `import
#: dgl` fail. Pairing 2.4.1 with the torch-2.4 index cost two runs on rented
#: hardware before the cause was found.
#:
#: The DGL version is pinned for the same reason. An index can serve a newer
#: DGL than the torch it sits beside — the torch-2.4 index has carried a 2.5.0
#: wheel — and an unpinned install silently picks it, producing a package that
#: unpacks and then will not load.
TORCH_DGL_PAIRS = (
    ("2.4.0", "2.4.0", "https://data.dgl.ai/wheels/torch-2.4/repo.html"),
    ("2.3.1", "2.3.0", "https://data.dgl.ai/wheels/torch-2.3/repo.html"),
    ("2.2.1", "2.2.1", "https://data.dgl.ai/wheels/torch-2.2/repo.html"),
    ("2.1.2", "2.1.0", "https://data.dgl.ai/wheels/torch-2.1/repo.html"),
)

#: Packages the decoder needs beyond torch and DGL. Deliberately unpinned: DGL
#: pulls its own compatible NumPy, and pinning a major version here breaks on
#: any Python for which that pin has no wheel.
SUPPORT_PACKAGES = ("scipy",)


def dgl_importable(python: Optional[str] = None) -> bool:
    """True when the given interpreter can genuinely import and use DGL.

    An install that unpacks but cannot load its native library does not count,
    which is exactly the failure mode this module works around.
    """
    executable = python or sys.executable
    probe = "import dgl; g = dgl.graph(([0],[1])); print(dgl.__version__)"
    try:
        result = subprocess.run([executable, "-c", probe],
                                capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def create_environment(location: Path, verbose: bool = True) -> Tuple[Optional[str], str]:
    """Create the isolated environment and install torch plus DGL into it.

    Returns ``(python_executable, message)``. The executable is None when the
    environment could not be built, and the message explains why.
    """
    location = Path(location)
    python = str(location / ("Scripts" if sys.platform == "win32" else "bin") /
                 ("python.exe" if sys.platform == "win32" else "python"))

    if Path(python).exists() and dgl_importable(python):
        return python, "reusing the existing isolated environment"

    def run(command: List[str], label: str) -> bool:
        if verbose:
            print(f"    {label}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 and verbose:
            tail = (result.stderr or result.stdout).strip().splitlines()
            print(f"      failed: {tail[-1][:220] if tail else 'unknown error'}")
        return result.returncode == 0

    if verbose:
        print(f"  building an isolated DGL environment at {location}")
        print("  (this does not touch the training environment)")

    # A previous failed attempt can leave an interpreter behind with no pip in
    # it: venv builds the tree first and only then runs ensurepip, and a
    # failure there is not cleaned up. Reusing that shell makes every later pip
    # call fail with "No module named pip", which this module then reports as a
    # missing wheel — pointing at entirely the wrong thing. Treat a pip-less
    # environment as absent and rebuild it.
    if Path(python).exists():
        has_pip = subprocess.run([python, "-m", "pip", "--version"],
                                 capture_output=True, text=True)
        if has_pip.returncode != 0:
            if verbose:
                print("  discarding a previous environment that has no usable pip")
            shutil.rmtree(location, ignore_errors=True)

    if not Path(python).exists():
        # Deliberately NOT --system-site-packages. The whole purpose is to keep
        # the main environment's torch out of sight; letting it leak in would
        # put DGL back next to the very torch build it cannot link against.
        # Three ways, because `python -m venv` fails on images whose ensurepip
        # is stripped or broken — Kaggle's is, and it reports only a non-zero
        # exit from ensurepip, which reads like a permissions problem.
        created = run([sys.executable, "-m", "venv", str(location)],
                      "creating an isolated virtual environment")

        if not created:
            # virtualenv ships its own pip and never calls ensurepip.
            run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"],
                "installing virtualenv (venv's ensurepip is unusable here)")
            created = run([sys.executable, "-m", "virtualenv", "-q", str(location)],
                          "creating the environment with virtualenv")

        if not created:
            # Last resort: a venv with no pip, then bootstrap pip into it.
            if run([sys.executable, "-m", "venv", "--without-pip", str(location)],
                   "creating the environment without pip"):
                bootstrap = location / "get-pip.py"
                try:
                    import urllib.request
                    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py",
                                               str(bootstrap))
                    created = run([python, str(bootstrap), "-q"],
                                  "bootstrapping pip into the environment")
                except Exception as error:  # noqa: BLE001 - report, do not raise
                    if verbose:
                        print(f"      could not fetch get-pip.py: {error}")

        if not created:
            return None, ("could not create a virtual environment (venv, "
                          "virtualenv and get-pip all failed)")

    run([python, "-m", "pip", "install", "-q", "--upgrade", "pip"], "upgrading pip")

    attempts: List[str] = []
    for torch_version, dgl_version, dgl_index in TORCH_DGL_PAIRS:
        if not run([python, "-m", "pip", "install", "-q", f"torch=={torch_version}",
                    "--index-url", "https://download.pytorch.org/whl/cpu"],
                   f"trying torch {torch_version} (CPU only)"):
            # Distinguish "pip is broken" from "this wheel does not exist".
            # Reporting the second when the first is true wastes a lot of time.
            probe = subprocess.run([python, "-m", "pip", "--version"],
                                   capture_output=True, text=True)
            reason = ("pip is not usable in the decoder environment"
                      if probe.returncode != 0 else "no wheel for this Python")
            attempts.append(f"torch {torch_version}: {reason}")
            if probe.returncode != 0:
                break      # every later pair fails the same way
            continue

        # Pinned first, because an unpinned install can pull a DGL built for a
        # different torch. Fall back to unpinned only if the pin has no wheel
        # for this Python, and let the import probe below be the real judge.
        installed = run(
            [python, "-m", "pip", "install", "-q", f"dgl=={dgl_version}",
             "-f", dgl_index],
            f"installing DGL {dgl_version} (built against torch {torch_version})")
        if not installed:
            installed = run(
                [python, "-m", "pip", "install", "-q", "dgl", "-f", dgl_index],
                f"DGL {dgl_version} unavailable; trying the newest on this index")
        if not installed:
            attempts.append(f"torch {torch_version}: DGL would not install")
            continue

        # DGL pulls its own NumPy; add only what the decoder still needs, and do
        # not treat this as fatal since DGL may already have satisfied it.
        run([python, "-m", "pip", "install", "-q", *SUPPORT_PACKAGES],
            "installing scipy for the decoder")

        if dgl_importable(python):
            return python, f"isolated decoder ready (torch {torch_version} + DGL)"
        attempts.append(f"torch {torch_version}: DGL installed but would not import")

    detail = "; ".join(attempts) if attempts else "no torch/DGL pair was attempted"
    return None, f"no working torch and DGL pairing for this Python. Tried: {detail}"


def convert(repo_root: Path, sources: Dict[str, Path], output: Path,
            env_location: Optional[Path] = None,
            verbose: bool = True) -> Dict[str, object]:
    """Decode DGL-format datasets into canonical ``.npz`` files.

    Uses the main interpreter when it can already read DGL, and otherwise builds
    the isolated environment. Returns a report describing what happened, so a
    caller can record a conversion failure instead of silently losing datasets.
    """
    repo_root = Path(repo_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {"requested": {k: str(v) for k, v in sources.items()}}

    if not sources:
        report.update({"performed": False, "reason": "nothing to convert"})
        return report

    # Skip anything already converted.
    pending = {name: path for name, path in sources.items()
               if not (output / f"{name}_canonical.npz").exists()}
    already = sorted(set(sources) - set(pending))
    if already and verbose:
        print(f"  already converted: {', '.join(already)}")
    if not pending:
        report.update({"performed": False, "reason": "all datasets already converted",
                       "converted": already})
        return report

    if dgl_importable():
        python, message = sys.executable, "using the main interpreter, which can read DGL"
        if verbose:
            print(f"  {message}")
    else:
        if verbose:
            print("  the main interpreter cannot read DGL files")
        location = Path(env_location) if env_location else repo_root / ".dgl_env"
        python, message = create_environment(location, verbose=verbose)
        if python is None:
            if verbose:
                print(f"  CONVERSION UNAVAILABLE: {message}")
                print("  FDCompCN, T-Finance and T-Social cannot be decoded on this host.")
                print("  Convert them on any machine where DGL installs, then copy the")
                print("  resulting *_canonical.npz files across; they need no DGL to read.")
            report.update({"performed": False, "reason": message,
                           "converted": already, "failed": sorted(pending)})
            return report
        if verbose:
            print(f"  {message}")

    command = [python, str(repo_root / "shared" / "comp8851" / "convert_dgl.py"),
               "--repo-root", str(repo_root), "--output", str(output),
               "--report", str(output / "conversion_report.json")]
    for name, path in pending.items():
        command += ["--dataset", name, str(path)]

    if verbose:
        print(f"  decoding {', '.join(sorted(pending))}")
    result = subprocess.run(command, capture_output=True, text=True)
    if verbose:
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"    {result.stderr.strip().splitlines()[-1][:300]}")

    converted = [name for name in pending
                 if (output / f"{name}_canonical.npz").exists()]
    failed = sorted(set(pending) - set(converted))

    report.update({
        "performed": True,
        "interpreter": python,
        "converted": sorted(set(already) | set(converted)),
        "failed": failed,
        "returncode": result.returncode,
    })
    return report
