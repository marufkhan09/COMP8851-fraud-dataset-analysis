"""Build the uploadable Vast.ai bundle.

    python scripts/build_bundle.py              # dist/comp8851_phase2_vastai.zip (code + verified data)
    python scripts/build_bundle.py --code-only  # dist/comp8851_phase2_code.zip (data fetched on the instance)

The bundle never contains: the authors' code (fetched at pinned commits on the
instance), raw downloads, local virtual environments, caches, credentials, or
an active results/ directory (local evidence is shipped read-only under
evidence/local_cpu/ so it can never be mistaken for, or skip, production runs).
A SHA256 manifest of every file is written into the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["README.md", "FINAL_TECHNICAL_REPORT.md", "FINAL_TECHNICAL_REPORT.pdf", "FINAL_TECHNICAL_REPORT.docx",
        ".gitignore", "phase2", "models", "shared", "configs", "scripts", "tests", "env", "kaggle",
        "audit", "third_party/manifest.json", "evidence", "results", "sitecustomize.py"]
DATA = ["data/canonical"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", "dist", ".venv"}
EXCLUDE_FILES = {"COST_ESTIMATE.md", "estimate_cost.py"}
SECRET_PATTERNS = [re.compile(rb"KGAT_[A-Za-z0-9]{8,}"), re.compile(rb'"key"\s*:\s*"[0-9a-f]{32}"'),
                   re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")]


def files_under(item: str):
    p = ROOT / item
    if p.is_file() and p.name not in EXCLUDE_FILES:
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if (f.is_file() and not (EXCLUDE_PARTS & set(f.parts))
                    and f.name not in EXCLUDE_FILES
                    and not f.name.endswith((".pyc", ".tmp", ".part"))):
                yield f


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args(argv)
    items = CODE + ([] if args.code_only else DATA)
    name = "comp8851_phase2_code.zip" if args.code_only else "comp8851_phase2_vastai.zip"
    out = ROOT / "dist" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    leaks = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for item in items:
            for f in files_under(item):
                rel = f.relative_to(ROOT).as_posix()
                data = f.read_bytes()
                if f.suffix in (".py", ".md", ".json", ".yaml", ".yml", ".txt", ".sh", ".csv", ".log", ".ipynb"):
                    if any(p.search(data) for p in SECRET_PATTERNS):
                        leaks.append(rel)
                manifest["files"][rel] = hashlib.sha256(data).hexdigest()
                z.writestr(f"comp8851_phase2/{rel}", data)
        z.writestr("comp8851_phase2/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
    if leaks:
        out.unlink()
        print(f"REFUSED: possible credential material in {leaks}")
        return 1
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix(".zip.sha256")).write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 2**20:.1f} MiB  {len(manifest['files'])} files  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
