"""CLI entry-point to execute a single SAGAD (WWW 2026) experiment under COMP8851 protocol vast-v4.4.

This script binds the SAGADAdapter (Multi-Resolution Quantile sampling + ChebNetII spectral
filtering + NodeDimGated fusion) to the unified COMP8851 Phase 2 execution harness.

Example usage:
    python models/sagad/run_one.py --dataset amazon --ratio TR40 --seed 2
    python models/sagad/run_one.py --dataset tsocial --ratio TR40 --seed 42 --require-cuda
"""

import sys
from pathlib import Path

# Resolve the repository root and current model directory on sys.path so both
# shared/phase2 modules and local adapter modules import cleanly regardless of CWD.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2.harness import main  # noqa: E402
from sagad_adapter import SAGADAdapter  # noqa: E402

if __name__ == "__main__":
    # Delegate CLI parsing, split verification, training, checkpointing, and evaluation to the harness
    raise SystemExit(main(SAGADAdapter))
