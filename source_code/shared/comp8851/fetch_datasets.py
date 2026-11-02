"""Acquire the six COMP8851 benchmark datasets and freeze canonical views.

Acquisition, representation conversion and splitting are three separate stages.
This script covers the first two: it downloads or locates each canonical source,
records a SHA256, builds the canonical view, and writes a frozen ``.npz`` plus a
manifest row. It never creates a split.

Three datasets download without credentials. Two require a human step, because
they sit behind Google Drive and Kaggle:

    yelpchi    automatic   data.dgl.ai
    amazon     automatic   data.dgl.ai
    fdcompcn   automatic   github codeload archive of the SplitGNN repository
    tfinance   MANUAL      BWGNN authors' Google Drive
    tsocial    MANUAL      BWGNN authors' Google Drive
    elliptic   MANUAL      Kaggle, needs an account

For the manual ones the script prints the exact source and destination and exits
without pretending to have the data.

Per the master plan, one team member (the dataset-registry owner) should run
this once, verify the hashes, and share the frozen copies. Every other machine
verifies the same hashes before training. A hash mismatch is a stop condition.

Usage:
    python shared/comp8851/fetch_datasets.py --data-root data
    python shared/comp8851/fetch_datasets.py --data-root data --only yelpchi amazon
    python shared/comp8851/fetch_datasets.py --data-root data --freeze
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.comp8851 import datasets as ds  # noqa: E402
from shared.comp8851.protocol import DATASETS, sha256_file  # noqa: E402

AUTOMATIC = {
    "yelpchi": {
        "url": "https://data.dgl.ai/dataset/FraudYelp.zip",
        "member": "YelpChi.mat",
    },
    "amazon": {
        "url": "https://data.dgl.ai/dataset/FraudAmazon.zip",
        "member": "Amazon.mat",
    },
    "fdcompcn": {
        # The raw.githubusercontent path is frequently rate-limited; the
        # codeload archive of the whole repository is the reliable route.
        "url": "https://codeload.github.com/Split-GNN/SplitGNN/zip/refs/heads/master",
        "member": "SplitGNN-master/data/FDCompCN.zip",
        "inner_member": "comp.dgl",
    },
}

MANUAL = {
    "tfinance": {
        "source": "https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr",
        "repository": "https://github.com/squareRoot3/Rethinking-Anomaly-Detection",
        "expected": "tfinance",
        "note": "BWGNN authors' release. Download the 'tfinance' DGL graph file.",
    },
    "tsocial": {
        "source": "https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr",
        "repository": "https://github.com/squareRoot3/Rethinking-Anomaly-Detection",
        "expected": "tsocial",
        "note": "BWGNN authors' release. Large file; allow disk and time for it.",
    },
    "elliptic": {
        "source": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/data",
        "repository": "",
        "expected": "elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv",
        "note": ("Kaggle requires an account. Either download in a browser and unzip "
                 "into this directory, or configure the Kaggle API and run: "
                 "kaggle datasets download -d ellipticco/elliptic-data-set"),
    },
}


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  already present: {destination.name} "
              f"({destination.stat().st_size:,} bytes)")
        return destination
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"  saved {destination.name} ({destination.stat().st_size:,} bytes)")
    return destination


def fetch_automatic(name: str, data_root: Path) -> Optional[Path]:
    """Download and extract one of the three credential-free datasets."""
    spec = AUTOMATIC[name]
    target_dir = data_root / name
    target_dir.mkdir(parents=True, exist_ok=True)

    archive = download(spec["url"], target_dir / f"{name}_source.zip")

    with zipfile.ZipFile(archive) as outer:
        member = spec["member"]
        if member not in outer.namelist():
            candidates = [n for n in outer.namelist() if n.endswith(Path(member).name)]
            if not candidates:
                raise FileNotFoundError(f"{name}: {member} not inside {archive.name}")
            member = candidates[0]
        outer.extract(member, target_dir)
        extracted = target_dir / member

    # FDCompCN nests a second archive containing the DGL graph.
    if "inner_member" in spec:
        with zipfile.ZipFile(extracted) as inner:
            inner.extractall(target_dir)
        extracted = next(target_dir.rglob(spec["inner_member"]))

    print(f"  extracted: {extracted}")
    print(f"  sha256: {sha256_file(extracted)}")
    return extracted


def report_manual(name: str, data_root: Path) -> None:
    spec = MANUAL[name]
    target = data_root / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"  MANUAL STEP REQUIRED for {name}")
    print(f"    source:      {spec['source']}")
    if spec["repository"]:
        print(f"    repository:  {spec['repository']}")
    print(f"    place into:  {target}")
    print(f"    expecting:   {spec['expected']}")
    print(f"    note:        {spec['note']}")


#: Datasets distributed as DGL binaries. Reading these needs a working DGL;
#: everything else in the pipeline reads the frozen .npz and needs none.
DGL_BACKED = ("tfinance", "tsocial", "fdcompcn")


def decode_via_isolated_env(name: str, target: Path) -> bool:
    """Decode a DGL-format dataset using the isolated decoder environment.

    Hosts with a recent PyTorch often have no matching DGL wheel, which would
    otherwise make T-Finance, T-Social and FDCompCN unreachable. The decoder
    builds a throwaway environment with a compatible torch plus DGL, uses it
    only to write the canonical .npz, and leaves training on the main
    interpreter. Returns True if a cache now exists.
    """
    # Absolute import, not relative: this file is executed as a script
    # (`python3 shared/comp8851/fetch_datasets.py`), so it has no parent package.
    from shared.comp8851 import dgl_env

    print(f"  {name}: no DGL here; trying the isolated decoder")
    try:
        report = dgl_env.convert(
            repo_root=REPO_ROOT,
            sources={name: target},
            output=target,
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001 - a decoder failure must not abort
        print(f"  {name}: isolated decoder failed: {type(exc).__name__}: {exc}")
        return False
    if not report.get("performed") and not (target / f"{name}_canonical.npz").exists():
        print(f"  {name}: isolated decoder could not run "
              f"({report.get('reason', 'no reason given')})")
        return False
    return (target / f"{name}_canonical.npz").exists()


def freeze(name: str, data_root: Path, strict: bool) -> Optional[Dict[str, Any]]:
    """Build the canonical view and write the frozen .npz plus statistics."""
    target = data_root / name
    if not target.exists():
        print(f"  {name}: nothing at {target}, skipping freeze")
        return None

    # A DGL-backed dataset on a host without DGL has to be decoded first, or
    # load_dataset below fails on the import and the dataset is lost.
    if name in DGL_BACKED and not (target / f"{name}_canonical.npz").exists():
        try:
            import dgl  # noqa: F401
            have_dgl = True
        except Exception:  # noqa: BLE001 - any import problem means "cannot read"
            have_dgl = False
        if not have_dgl and not decode_via_isolated_env(name, target):
            print(f"  {name}: could not decode without DGL; skipping")
            return None

    try:
        dataset = ds.load_dataset(name, target, strict=strict)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the whole run
        print(f"  {name}: could not build the canonical view: {type(exc).__name__}: {exc}")
        return None

    cache = target / f"{name}_canonical.npz"
    ds.save_canonical(dataset, cache)
    statistics = dataset.statistics()
    heterophily = dataset.heterophily()

    print(f"  froze {cache.name} ({cache.stat().st_size:,} bytes)")
    print(f"    nodes {statistics['nodes']:,} | features {statistics['feature_dimension']} | "
          f"fraud {statistics['fraud_nodes']:,} "
          f"({statistics['fraud_percentage']:.2f}%) | "
          f"edges {statistics['union_stored_directed_edges']:,}")
    if heterophily.get("global_heterophily") is not None:
        print(f"    global heterophily {heterophily['global_heterophily']:.4f}")

    report_path = target / f"{name}_dataset_report.json"
    ds.write_dataset_report(dataset, report_path)

    return {
        "dataset_id": name,
        "canonical_cache": str(cache),
        "canonical_cache_sha256": sha256_file(cache),
        "source_files": dataset.source_files,
        "source_sha256": dataset.source_sha256,
        "statistics": statistics,
        "heterophily": heterophily,
        "report": str(report_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Acquire COMP8851 datasets")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--only", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--freeze", action="store_true",
                        help="Build and cache the canonical view after acquisition")
    parser.add_argument("--no-strict", dest="strict", action="store_false", default=True,
                        help="Do not fail on a published-statistic mismatch (inspection only)")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("COMP8851 DATASET ACQUISITION")
    print("=" * 78)
    print(f"Destination: {data_root.resolve()}")

    manifest: List[Dict[str, Any]] = []
    manual_pending: List[str] = []

    for name in args.only:
        print(f"\n[{name}]")
        if name in AUTOMATIC:
            try:
                fetch_automatic(name, data_root)
            except Exception as exc:  # noqa: BLE001
                print(f"  download failed: {type(exc).__name__}: {exc}")
                manual_pending.append(name)
                continue
        else:
            report_manual(name, data_root)
            probe = data_root / name
            present = any(probe.rglob("*")) if probe.exists() else False
            if not present:
                manual_pending.append(name)
                continue
            print("  files already present, continuing")

        if args.freeze:
            row = freeze(name, data_root, args.strict)
            if row:
                manifest.append(row)

    if manifest:
        manifest_path = data_root / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str),
                                 encoding="utf-8")
        print(f"\n[manifest] wrote {manifest_path}")

        checksums = data_root / "checksums.sha256"
        lines = []
        for row in manifest:
            lines.append(f"{row['canonical_cache_sha256']}  {Path(row['canonical_cache']).name}")
            for key, digest in (row.get("source_sha256") or {}).items():
                source = row["source_files"].get(key, key)
                lines.append(f"{digest}  {Path(source).name}")
        checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[manifest] wrote {checksums}")

    print("\n" + "=" * 78)
    print(f"acquired and frozen: {len(manifest)} of {len(args.only)} requested")
    if manual_pending:
        print(f"still needing a manual download: {', '.join(manual_pending)}")
        print("Re-run with --freeze once those files are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
