"""Verify the execution environment before any GPU money is spent.

    python scripts/verify_env.py                 # report
    python scripts/verify_env.py --require-gpu   # exit 1 without a working CUDA + DGL GPU stack

Checks GPU model/VRAM/driver/CUDA, system RAM, free disk, CPU count, library
versions, a real DGL message-passing op on the GPU, the pinned official code,
and the presence/hashes of the canonical datasets. Writes results/env_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")

# Per-dataset peak-memory guidance (GB) for an RTX A6000; revised by the pilot.
DISK_NEEDED_GB = 40


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    report = {"python": platform.python_version(), "platform": platform.platform(), "problems": [], "warnings": []}

    try:
        import psutil

        report["ram_total_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        report["ram_available_gb"] = round(psutil.virtual_memory().available / 2 ** 30, 1)
        report["cpu_count"] = psutil.cpu_count()
    except ImportError:
        report["warnings"].append("psutil missing")
    usage = shutil.disk_usage(ROOT)
    report["disk_free_gb"] = round(usage.free / 2 ** 30, 1)
    if usage.free / 2 ** 30 < DISK_NEEDED_GB:
        report["warnings"].append(f"less than {DISK_NEEDED_GB} GB free disk; T-Social runs write ~1 GB per run")

    for mod in ("numpy", "scipy", "sklearn", "pandas", "yaml", "torch", "dgl"):
        try:
            m = __import__(mod)
            report[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            report["problems"].append(f"cannot import {mod}: {type(exc).__name__}: {exc}")

    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_vram_gb"] = round(p.total_memory / 2 ** 30, 1)
            report["gpu_count"] = torch.cuda.device_count()
            try:
                smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                      "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
                report["nvidia_smi"] = smi.stdout.strip()
            except Exception:  # noqa: BLE001
                pass
            if "A6000" not in report["gpu_name"] and "A100" not in report["gpu_name"]:
                report["warnings"].append(
                    f"GPU {report['gpu_name']} is not the protocol's controlled RTX A6000 (fallback A100); "
                    "AUPRC remains valid but runtimes are not hardware-comparable with the previous benchmark")
            if report["gpu_vram_gb"] < 40:
                report["warnings"].append("under 40 GB VRAM: T-Social runs may not fit")
            import dgl

            g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]))).to("cuda")
            g.ndata["h"] = torch.ones(3, 4, device="cuda")
            g.update_all(dgl.function.copy_u("h", "m"), dgl.function.sum("m", "h"))
            report["dgl_gpu_message_passing"] = bool(torch.allclose(g.ndata["h"], torch.ones(3, 4, device="cuda")))
        elif args.require_gpu:
            report["problems"].append("CUDA not available")
        if not str(report.get("torch_version", "")).startswith("2.4.0"):
            report["warnings"].append("torch is not 2.4.0 (the pinned production version)")
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"GPU/DGL check failed: {type(exc).__name__}: {exc}")

    try:
        from phase2 import official

        for repo in ("sagad", "spacegnn", "croc"):
            official.verify(repo)
        report["official_code"] = "verified at pinned commits"
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"official code: {exc} (run scripts/fetch_official.py)")

    status_path = ROOT / "data" / "canonical" / "data_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        report["datasets"] = {k: v.get("status") for k, v in status.items()}
    else:
        report["datasets"] = "not prepared (run scripts/prepare_data.py)"

    out = ROOT / "results" / "env_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["problems"] and (args.require_gpu or any("cannot import" in p for p in report["problems"])):
        print("ENVIRONMENT NOT READY")
        return 1
    print("environment check finished" + (" with warnings" if report["warnings"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
