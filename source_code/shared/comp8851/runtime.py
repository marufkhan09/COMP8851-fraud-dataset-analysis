"""Runtime control: seeding, devices, thread caps, timing and environment capture.

These rules are identical for every model so that runtime and memory numbers
are comparable (master plan v4.4, sections 3.4 and 9.3).
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from .protocol import CPU_THREAD_CAP


def utc_now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    """Compact UTC stamp for run identifiers, e.g. ``20260912T093000Z``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def apply_thread_caps(threads: int = CPU_THREAD_CAP) -> None:
    """Cap CPU threads identically on every benchmark machine.

    Must be called before heavy numeric libraries spin up their pools, so call
    it at the very top of a script.
    """
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and pin cuDNN to deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(no_cuda: bool = False, require_cuda: bool = False):
    """Return the torch device, exposing GPU 0 only.

    ``require_cuda`` turns a missing GPU into an error rather than a silent CPU
    fallback, which matters for controlled benchmark runs: a run that quietly
    dropped to CPU would corrupt the runtime comparison.
    """
    import torch

    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if require_cuda:
        raise RuntimeError(
            "CUDA was required for this controlled run but torch.cuda.is_available() is False."
        )
    return torch.device("cpu")


def environment_metadata() -> Dict[str, Any]:
    """Capture the software environment for ``environment.txt``/``summary.json``."""
    metadata: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
    }
    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_version"] = torch.version.cuda
        metadata["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
    except ImportError:
        metadata["torch_version"] = None
    try:
        import dgl

        metadata["dgl_version"] = dgl.__version__
    except Exception:  # pragma: no cover - DGL is optional for CARE-GNN
        metadata["dgl_version"] = None
    try:
        import sklearn

        metadata["sklearn_version"] = sklearn.__version__
    except ImportError:
        metadata["sklearn_version"] = None
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES"):
        metadata[variable] = os.environ.get(variable)
    return metadata


def hardware_metadata(device=None) -> Dict[str, Any]:
    """Capture the hardware profile for ``hardware.json``."""
    info: Dict[str, Any] = {
        "hostname": platform.node(),
        "cpu_count_logical": os.cpu_count(),
        "platform": platform.platform(),
    }
    try:
        import psutil

        info["system_memory_total_bytes"] = int(psutil.virtual_memory().total)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["system_memory_total_bytes"] = None
        info["cpu_count_physical"] = None

    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_capability": f"{properties.major}.{properties.minor}",
                "gpu_multi_processor_count": properties.multi_processor_count,
                "visible_device_count": torch.cuda.device_count(),
            })
        else:
            info.update({"gpu_name": None, "gpu_total_memory_bytes": None})
    except ImportError:
        info.update({"gpu_name": None, "gpu_total_memory_bytes": None})

    info["nvidia_smi"] = _nvidia_smi()
    if device is not None:
        info["resolved_device"] = str(device)
    return info


def _nvidia_smi() -> Optional[str]:
    """Return a one-line nvidia-smi summary, or None when unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def gpu_profile_id(device=None) -> str:
    """Short hardware-profile identifier derived from the GPU name."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            for token in ("a6000", "a100", "4090", "3090", "l40s", "l40", "v100", "t4"):
                if token in name.replace(" ", ""):
                    return token
            return "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")[:24]
    except ImportError:
        pass
    return "cpu"


def synchronise(device=None) -> None:
    """Synchronise CUDA before/after a timed region so timings are truthful."""
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.synchronize(device)
    except ImportError:
        pass


def reset_peak_memory(device=None) -> None:
    """Reset CUDA peak-memory statistics immediately before a measured region."""
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.reset_peak_memory_stats(device)
    except ImportError:
        pass


def peak_memory_mb(device=None) -> float:
    """Peak CUDA memory in MiB since the last reset, or 0.0 on CPU."""
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    except ImportError:
        pass
    return 0.0


@dataclass
class Timer:
    """Elapsed wall-clock seconds for a CUDA-synchronised region."""

    device: Any = None
    seconds: float = 0.0

    def __enter__(self) -> "Timer":
        synchronise(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronise(self.device)
        self.seconds = time.perf_counter() - self._start


@contextmanager
def timed(device=None):
    """Context manager yielding a :class:`Timer`."""
    timer = Timer(device=device)
    with timer:
        yield timer


class Tee:
    """Duplicate stdout/stderr into ``terminal.log`` while still printing.

    Master plan section 9.4 requires the complete terminal output of every run,
    including warnings and tracebacks.
    """

    def __init__(self, path, stream):
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self._stream = stream

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


@contextmanager
def capture_terminal(path):
    """Tee stdout and stderr to ``path`` for the duration of the block."""
    path = str(path)
    out, err = sys.stdout, sys.stderr
    tee_out = Tee(path, out)
    tee_err = Tee(path, err)
    sys.stdout, sys.stderr = tee_out, tee_err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err
        tee_out.close()
        tee_err.close()


def count_parameters(model) -> Dict[str, int]:
    """Count total and trainable parameters of a torch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
