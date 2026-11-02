import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sub in ("models/sagad", "models/spacegnn", "models/croc"):
    sys.path.insert(0, str(ROOT / sub))
import phase2  # noqa: F401 - activates collections.Mapping shim
os.environ.setdefault("DGLBACKEND", "pytorch")


def canonical_available(name: str) -> bool:
    return (ROOT / "data" / "canonical" / f"{name}_canonical.npz").exists()


def needs_dataset(name: str):
    return pytest.mark.skipif(not canonical_available(name),
                              reason=f"{name} canonical cache not prepared (scripts/prepare_data.py)")
