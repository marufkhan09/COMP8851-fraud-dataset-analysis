"""Two-epoch CARE-GNN feasibility check.

The protocol requires a smoke test before any full training: it must exercise
loading, forward pass, backward pass, validation, checkpointing and structured
result writing. This is a thin wrapper over ``run_one.py --smoke`` so there is
exactly one training path, not a second one that could drift.

A smoke metric is feasibility evidence. It is never the model's benchmark
performance, and the artefacts it writes are tagged ``feasibility_only`` so the
aggregator excludes them from the result tables.

Usage:
    python models/care-gnn/scripts/smoke.py --dataset yelpchi --data-path data/yelpchi
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_one import main as run_one_main  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--smoke" not in argv:
        argv.append("--smoke")
    return run_one_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
