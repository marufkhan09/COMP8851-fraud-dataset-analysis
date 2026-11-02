"""Import the pinned official implementations without name clashes.

The three repositories use generic top-level module names (``model``,
``utils``, ``name``, ``models``, ``trainers`` ...). ``activate`` verifies the
pinned files, evicts any cached module with one of those names, and puts the
requested repository first on ``sys.path`` so that the authors' own
intra-repository imports (``from name import *``, ``import utils``) resolve to
the authors' files and nothing else.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = BUNDLE_ROOT / "third_party"
MANIFEST = json.loads((THIRD_PARTY / "manifest.json").read_text(encoding="utf-8"))

GENERIC_NAMES = ("model", "models", "utils", "name", "main", "trainers", "dataloading",
                 "preprocess", "dataloader", "chebnet", "mrqsampler")

_active: Dict[str, str] = {}


def repo_dir(repo: str) -> Path:
    """Return the local directory path for the specified third-party repository."""
    return THIRD_PARTY / repo


def verify(repo: str) -> None:
    """Verify every file in the upstream repository byte-for-byte against its pinned SHA-256 digest."""
    spec = MANIFEST["repositories"][repo]
    for rel, digest in spec["files"].items():
        path = repo_dir(repo) / rel
        if not path.exists():
            raise FileNotFoundError(
                f"{repo}: {path} missing. Run `python scripts/fetch_official.py` first.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"{repo}: {rel} differs from the pinned commit {spec['commit']}")


def activate(repo: str) -> Path:
    """Make ``repo`` the active source of generic module names (`model`, `utils`, etc.). Idempotent."""
    verify(repo)
    path = str(repo_dir(repo))
    if _active.get("repo") != repo:
        # Evict cached generic modules so intra-repo imports resolve to the newly activated model
        for mod in GENERIC_NAMES:
            sys.modules.pop(mod, None)
        for other in MANIFEST["repositories"]:
            other_path = str(repo_dir(other))
            while other_path in sys.path:
                sys.path.remove(other_path)
        sys.path.insert(0, path)
        _active["repo"] = repo
    return repo_dir(repo)


def load(repo: str, module: str) -> ModuleType:
    """Verify, activate, and import a module from the pinned upstream repository."""
    activate(repo)
    return importlib.import_module(module)


def commit(repo: str) -> str:
    """Return the pinned git commit hash for the specified upstream repository."""
    return MANIFEST["repositories"][repo]["commit"]


def provenance(repo: str) -> Dict[str, object]:
    """Return a structured provenance dictionary (URL, commit, paper, verified file count) for summary.json."""
    spec = MANIFEST["repositories"][repo]
    return {"repository": spec["url"], "commit": spec["commit"], "paper": spec["paper"],
            "license_detected_by_github": spec["license_detected_by_github"],
            "files_verified": len(spec["files"])}
