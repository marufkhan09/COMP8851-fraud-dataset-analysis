"""Two-epoch GHRN feasibility check.

Thin wrapper over ``run_one.py --smoke`` so there is exactly one training path.
With a non-zero ``--del-ratio`` the smoke test covers both stages, which is the
point: the refinement step is the part most likely to fail on an unfamiliar
dataset, so it must be exercised before a full run is launched.

A smoke metric is feasibility evidence, never benchmark performance.

Usage:
    python models/ghrn/scripts/smoke.py --dataset tfinance --data-path data/tfinance/tfinance
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
