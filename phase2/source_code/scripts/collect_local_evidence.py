"""Copy locally executed evidence into evidence/local_cpu/ (read-only record).

Copies (never moves): smoke-run summaries, terminal logs, prediction files and
metrics; aggregated smoke tables and integrity report; the local end-to-end
pipeline validation (results_local_cpu/); test-suite logs; the data status
file. Production runs on Vast.ai write to results/, which this folder never
touches.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evidence" / "local_cpu"
KEEP = ("summary.json", "terminal.log", "test_metrics.json", "predictions.npz", "run_config.yml",
        "epoch_times.csv", "validation_metrics.csv", "split_summary.json", "protocol_identity.json",
        "hardware.json", "environment.txt")


def copy_runs(src_root: Path, dst_root: Path) -> int:
    n = 0
    for summary in src_root.rglob("summary.json"):
        if "_attempts" in summary.parts:
            continue
        rel = summary.parent.relative_to(src_root)
        for name in KEEP:
            f = summary.parent / name
            if f.exists():
                (dst_root / rel).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst_root / rel / name)
        n += 1
    return n


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    report = {"collected_utc": datetime.now(timezone.utc).isoformat(), "host_platform": platform.platform(),
              "python": platform.python_version(), "note": "Local CPU evidence (Windows laptop, no GPU). "
              "Feasibility and correctness evidence only - not benchmark results."}
    for name, src in (("smoke", ROOT / "results" / "smoke"),
                      ("pipe_full", ROOT / "results_local_cpu" / "raw"),
                      ("pipe_spacegnn_reduced", ROOT / "results_local_cpu_spacegnn_reduced" / "raw")):
        if src.exists():
            report[name] = copy_runs(src, DEST / name)
    for src, dst in ((ROOT / "results" / "aggregated" / "smoke", DEST / "smoke_tables"),
                     (ROOT / "results_local_cpu" / "aggregated", DEST / "pipe_full_tables"),
                     (ROOT / "results_local_cpu" / "tuning", DEST / "pipe_full_tuning"),
                     (ROOT / "results_local_cpu_spacegnn_reduced" / "aggregated",
                      DEST / "pipe_spacegnn_reduced_tables")):
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.pt", "checkpoints"))
    for f in (ROOT / "results" / "ledger.csv", ROOT / "results_local_cpu" / "ledger.csv",
              ROOT / "results_local_cpu_spacegnn_reduced" / "ledger.csv",
              ROOT / "data" / "canonical" / "data_status.json"):
        if f.exists():
            shutil.copy2(f, DEST / (f.parent.name + "_" + f.name))
    report["labels"] = {
        "smoke": "2-epoch feasibility runs; T-Social 300k-node subset; SpaceGNN T-Finance 8k-node subset",
        "pipe_full": "FDCompCN, full protocol (12-trial tuning, seeds 2/42/72, 100/20) on CPU - provisional",
        "pipe_spacegnn_reduced": "FDCompCN, REDUCED budget (2 trials, 10 epochs) - orchestration check only",
    }
    for log in (ROOT / "evidence").glob("pytest_*.log"):
        shutil.copy2(log, DEST / log.name)
    (DEST / "README.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
