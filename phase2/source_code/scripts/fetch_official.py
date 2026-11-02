"""Fetch the three official implementations at their pinned commits.

    python scripts/fetch_official.py              # download from GitHub codeload
    python scripts/fetch_official.py --from-dir D # copy from a local checkout/archive dir
    python scripts/fetch_official.py --verify     # only verify what is in third_party/

Every file is checked against third_party/manifest.json. Any difference stops
the script: a silently changed upstream file would invalidate the provenance
of every run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "third_party"
MANIFEST = json.loads((THIRD / "manifest.json").read_text(encoding="utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(name: str) -> list[str]:
    spec = MANIFEST["repositories"][name]
    problems = []
    for rel, digest in spec["files"].items():
        path = THIRD / name / rel
        if not path.exists():
            problems.append(f"{name}/{rel}: missing")
        elif sha256(path.read_bytes()) != digest:
            problems.append(f"{name}/{rel}: sha256 mismatch")
    return problems


def install_from_zip(name: str, payload: bytes) -> None:
    spec = MANIFEST["repositories"][name]
    target = THIRD / name
    staging = THIRD / f".{name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else info.filename
            if rel not in spec["files"]:
                continue
            data = archive.read(info)
            if sha256(data) != spec["files"][rel]:
                raise RuntimeError(f"{name}/{rel}: downloaded file does not match the pinned hash")
            out = staging / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    missing = [rel for rel in spec["files"] if not (staging / rel).exists()]
    if missing:
        raise RuntimeError(f"{name}: archive lacks pinned files {missing}")
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    (target / "PINNED_COMMIT").write_text(spec["commit"] + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-dir", default=None,
                        help="directory holding <repo>-<commit>.zip archives or extracted <repo>/ folders")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--only", nargs="*", default=list(MANIFEST["repositories"]))
    args = parser.parse_args(argv)

    failures = 0
    for name in args.only:
        spec = MANIFEST["repositories"][name]
        if not args.verify and verify(name):
            if args.from_dir:
                src = Path(args.from_dir)
                zips = list(src.glob(f"*{spec['commit']}*.zip"))
                if zips:
                    install_from_zip(name, zips[0].read_bytes())
                else:
                    folder = next((p for p in src.rglob("*") if p.is_dir() and spec["commit"] in p.name), None)
                    if folder is None:
                        raise FileNotFoundError(f"{name}: nothing for commit {spec['commit']} under {src}")
                    target = THIRD / name
                    target.mkdir(parents=True, exist_ok=True)
                    for rel in spec["files"]:
                        (target / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(folder / rel, target / rel)
                    (target / "PINNED_COMMIT").write_text(spec["commit"] + "\n", encoding="utf-8")
            else:
                owner_repo = spec["url"].split("github.com/")[1]
                url = f"https://codeload.github.com/{owner_repo}/zip/{spec['commit']}"
                print(f"[{name}] downloading {url}")
                with urllib.request.urlopen(url, timeout=120) as response:
                    install_from_zip(name, response.read())
        problems = verify(name)
        if problems:
            failures += 1
            print(f"[{name}] FAILED verification: {problems}")
        else:
            print(f"[{name}] ok  commit {spec['commit'][:12]}  ({len(spec['files'])} files verified)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
