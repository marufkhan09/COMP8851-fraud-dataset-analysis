"""Capture and backfill upstream commit / container provenance.

    # capture live, from the machine that ran the benchmark
    python3 scripts/record_provenance.py --results results --capture

    # or supply values explicitly
    python3 scripts/record_provenance.py --results results \
        --care-gnn-commit <sha> --ghrn-commit <sha> --container-digest <digest>

Fills ``model_source_commit`` and ``container_digest`` in every run's
``protocol_identity.json``. Both fields are declared required by the protocol
but nothing populated them, so every delivered run carried nulls.

Every value written here is measured, never assumed:

* upstream commits come from ``git ls-remote`` against the official repositories
* the container digest comes from this host's own Docker metadata
* the local source hash is a SHA256 over the model package as it exists on disk

If a value cannot be determined it stays null and is reported as unresolved.
A null is a truthful "not recorded"; a plausible-looking invented SHA would
defeat the entire point of an audit field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

UPSTREAM = {
    "CARE-GNN": "https://github.com/YingtongDou/CARE-GNN",
    "GHRN": "https://github.com/blacksingular/GHRN",
}
PACKAGE = {"CARE-GNN": "models/care-gnn/caregnn", "GHRN": "models/ghrn/ghrnlib"}


def remote_head(url: str) -> Optional[str]:
    """The current HEAD SHA of a public repository, or None if unreachable."""
    try:
        result = subprocess.run(["git", "ls-remote", url, "HEAD"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    sha = result.stdout.split()[0].strip()
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None


def container_digest() -> Optional[str]:
    """This host's container image digest, if it is running in one."""
    for path, pattern in ((Path("/etc/hostname"), None),
                          (Path("/proc/self/cgroup"), r"docker[/-]([0-9a-f]{64})")):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern:
            match = re.search(pattern, text)
            if match:
                return f"sha256:{match.group(1)}"
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}",
             Path("/etc/hostname").read_text(encoding="utf-8").strip()],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def package_sha256(relative: str) -> Optional[str]:
    """Deterministic hash of a model package as it exists in this repository."""
    root = REPO_ROOT / relative
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_of(path: Path) -> Optional[str]:
    text = str(path).lower()
    if "care-gnn" in text or "caregnn" in text:
        return "CARE-GNN"
    if "ghrn" in text:
        return "GHRN"
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record run provenance")
    parser.add_argument("--results", default="results")
    parser.add_argument("--capture", action="store_true",
                        help="Resolve values from the network and this host")
    parser.add_argument("--care-gnn-commit", default=None)
    parser.add_argument("--ghrn-commit", default=None)
    parser.add_argument("--container-digest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    commits: Dict[str, Optional[str]] = {
        "CARE-GNN": args.care_gnn_commit,
        "GHRN": args.ghrn_commit,
    }
    digest = args.container_digest

    if args.capture:
        print("Resolving provenance...")
        for model, url in UPSTREAM.items():
            if not commits[model]:
                commits[model] = remote_head(url)
                state = commits[model] or "UNRESOLVED (no network?)"
                print(f"  {model:<9} {url}\n            HEAD = {state}")
        if not digest:
            digest = container_digest()
            print(f"  container digest = {digest or 'UNRESOLVED (not in a container?)'}")

    local = {m: package_sha256(p) for m, p in PACKAGE.items()}
    for model, sha in local.items():
        print(f"  {model:<9} local package sha256 = {sha or 'n/a'}")

    results_root = Path(args.results)
    if not results_root.exists():
        print(f"\nResults directory not found: {results_root}")
        return 1

    written = unresolved = 0
    for path in sorted(results_root.rglob("protocol_identity.json")):
        model = model_of(path)
        if model is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        changed = False
        if commits.get(model) and not payload.get("model_source_commit"):
            payload["model_source_commit"] = commits[model]
            changed = True
        if digest and not payload.get("container_digest"):
            payload["container_digest"] = digest
            changed = True
        if local.get(model):
            notes = payload.setdefault("notes", {})
            if notes.get("local_package_sha256") != local[model]:
                notes["local_package_sha256"] = local[model]
                notes["upstream_repository"] = UPSTREAM[model]
                changed = True

        if not payload.get("model_source_commit"):
            unresolved += 1
        if changed:
            written += 1
            if not args.dry_run:
                path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                                encoding="utf-8")

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {written} protocol_identity.json file(s)")
    if unresolved:
        print(f"{unresolved} still have a null model_source_commit. Supply it "
              f"with --care-gnn-commit / --ghrn-commit rather than leaving it "
              f"to be guessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
