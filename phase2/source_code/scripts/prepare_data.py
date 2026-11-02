"""Acquire, freeze and verify the six benchmark datasets and their splits.

    python scripts/prepare_data.py                       # everything available
    python scripts/prepare_data.py --only yelpchi amazon
    python scripts/prepare_data.py --tfinance-raw /path/to/tfinance --elliptic-raw /path/to/csvs

Sources (identical to shared/comp8851/fetch_datasets.py):
    yelpchi   https://data.dgl.ai/dataset/FraudYelp.zip            (automatic)
    amazon    https://data.dgl.ai/dataset/FraudAmazon.zip          (automatic)
    fdcompcn  SplitGNN repo @ 32fbdb8, data/FDCompCN.zip -> comp.dgl (automatic)
    tfinance  BWGNN authors' Google Drive, or Kaggle mirror         (manual / kaggle CLI)
    elliptic  Kaggle ellipticco/elliptic-data-set                   (manual / kaggle CLI)
    tsocial   canonical cache shipped in the bundle (from comp8851.zip; sha256 a30fb6a0...)

Nothing is ever synthesised: a missing source is reported with the exact manual
step and the dataset is marked unavailable. Writes data/canonical/data_status.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.comp8851 import datasets as cds  # noqa: E402
from shared.comp8851.protocol import DATASETS, sha256_file  # noqa: E402
from phase2 import data as pdata  # noqa: E402

RAW = ROOT / "data" / "raw"
CANON = ROOT / "data" / "canonical"

SOURCES = {
    "yelpchi": {"url": "https://data.dgl.ai/dataset/FraudYelp.zip", "file": "FraudYelp.zip",
                "sha256": "3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3"},
    "amazon": {"url": "https://data.dgl.ai/dataset/FraudAmazon.zip", "file": "FraudAmazon.zip",
               "sha256": "c1b7a1f8cd8621467d5121b690fef61a7d398e4f582232bb3e8a154424f7a5b9"},
    "fdcompcn": {"url": "https://codeload.github.com/Split-GNN/SplitGNN/zip/32fbdb836781d5c02523844a7079425f89be8831",
                 "file": "SplitGNN-32fbdb8.zip",
                 "sha256": "7bba89d4d3176e8fa9d3072854e8cd16f865a071292726dda2f666fd0b90755a",
                 "member": "SplitGNN-32fbdb836781d5c02523844a7079425f89be8831/data/FDCompCN.zip"},
}
#: Elliptic: PyG's dataset mirror of the EllipticCo Kaggle CSVs. The extracted
#: CSVs are byte-identical to the Kaggle files used by the previous benchmark
#: (sha256 recorded in evidence/caregnn_elliptic_seed2/dataset_manifest.json),
#: which is asserted below, so no Kaggle credentials are needed.
ELLIPTIC_MIRROR = "https://data.pyg.org/datasets/elliptic"
ELLIPTIC_CSV_SHA256 = {
    "elliptic_txs_classes.csv": "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",
    "elliptic_txs_edgelist.csv": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
    "elliptic_txs_features.csv": "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
}
ELLIPTIC_ZIP_SHA256 = {
    "elliptic_txs_classes.csv.zip": "4ca957f0ceffd5dd164e255c7d5ad9ee69a6fa64ae1dd94d6f113e5ebf3b07ba",
    "elliptic_txs_edgelist.csv.zip": "a2f9f6b67a39da2d8cf87fe77b9db89571ba6d880e5dd5b5991dc45c80fa34ec",
    "elliptic_txs_features.csv.zip": "d33d62159e64b5e889f1a7ea880227c612775b58d409598855e0c4400fa52b3e",
}

#: BWGNN-release T-Finance graph hash recorded in the T-Finance split manifest.
TFINANCE_SOURCE_SHA256 = "b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700"
#: The BWGNN authors' public Google Drive folder
#: (https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr)
#: holds dataset/tfinance.zip; its member 'tfinance' hashes to the value above.
TFINANCE_DRIVE_ID = "1gAMek_5Dsd4xPD0o7z-FFD8Jwxr0gJR5"
TFINANCE_ZIP_SHA256 = "0a0091978425a0e9f1cd0e001240517ecd786344bd580c2bd00ef32ad1c80475"
TSOCIAL_CANONICAL_SHA256 = "a30fb6a0561ed20c999d4feab6d23d7c3dd566e16d853d7dd79fed98cec527d5"
#: Raw T-Social graph hash from the owner's data/dataset_manifest.json (comp8851.zip).
TSOCIAL_SOURCE_SHA256 = "8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0"
TSOCIAL_DRIVE_ID = "10nY_IwxT32KdfpcTqoy8wgckXK2xayXd"  # BWGNN folder: dataset/tsocial.zip

MANUAL_HELP = {
    "tfinance": ("Download the 'tfinance' DGL graph from the BWGNN release "
                 "(https://github.com/squareRoot3/Rethinking-Anomaly-Detection -> Google Drive) "
                 "or, with Kaggle auth configured, `kaggle datasets download -d marufdkhan/tfinance`. "
                 f"Expected sha256 {TFINANCE_SOURCE_SHA256}. Then pass --tfinance-raw <file or dir>."),
    "elliptic": ("Download https://www.kaggle.com/datasets/ellipticco/elliptic-data-set "
                 "(`kaggle datasets download -d ellipticco/elliptic-data-set`), unzip, then pass "
                 "--elliptic-raw <dir containing elliptic_txs_*.csv>."),
    "tsocial": ("tsocial_canonical.npz ships in the bundle. If it is missing, copy "
                "data/tsocial/tsocial_canonical.npz out of comp8851.zip into data/canonical/."),
}


def download(url: str, dest: Path, expected_sha: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    digest = sha256_file(dest)
    if digest != expected_sha:
        raise RuntimeError(f"{dest.name}: sha256 {digest} != expected {expected_sha}. "
                           "The upstream file changed or the download is corrupt; stop.")
    print(f"  source ok: {dest.name} sha256 {digest[:16]}...")
    return dest


def build(name: str, args) -> cds.CanonicalDataset:
    if name in ("yelpchi", "amazon"):
        spec = SOURCES[name]
        archive = download(spec["url"], RAW / name / spec["file"], spec["sha256"])
        return cds.load_dataset(name, archive.parent, strict=True)
    if name == "fdcompcn":
        spec = SOURCES[name]
        archive = download(spec["url"], RAW / name / spec["file"], spec["sha256"])
        inner = RAW / name / "FDCompCN.zip"
        if not inner.exists():
            with zipfile.ZipFile(archive) as outer, outer.open(spec["member"]) as src, inner.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return cds.load_dataset(name, inner, strict=True)
    if name == "tfinance":
        if args.tfinance_raw:
            path = cds._resolve_bwgnn_path("tfinance", Path(args.tfinance_raw))
        else:
            target = RAW / "tfinance"
            target.mkdir(parents=True, exist_ok=True)
            archive = target / "tfinance.zip"
            if not archive.exists():
                try:
                    import gdown
                except ImportError as exc:
                    raise FileNotFoundError("gdown is not installed; " + MANUAL_HELP["tfinance"]) from exc
                print("  downloading tfinance.zip from the BWGNN authors' Google Drive")
                gdown.download(id=TFINANCE_DRIVE_ID, output=str(archive) + ".part", quiet=True)
                Path(str(archive) + ".part").replace(archive)
            if sha256_file(archive) != TFINANCE_ZIP_SHA256:
                raise RuntimeError("tfinance.zip hash differs from the verified BWGNN release")
            path = target / "tfinance"
            if not path.exists():
                with zipfile.ZipFile(archive) as z:
                    z.extract("tfinance", target)
        digest = sha256_file(path)
        if digest != TFINANCE_SOURCE_SHA256:
            raise RuntimeError(f"tfinance source sha256 {digest} != registry source {TFINANCE_SOURCE_SHA256}; "
                               "the frozen T-Finance split was built from a different file.")
        return cds.load_dataset("tfinance", path, strict=True)
    if name == "elliptic":
        if args.elliptic_raw:
            raw = Path(args.elliptic_raw)
            if raw.suffix.lower() == ".zip":
                target = RAW / "elliptic"
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(raw) as z:
                    z.extractall(target)
                raw = target
        else:
            raw = RAW / "elliptic_pyg" / "csv"
            raw.mkdir(parents=True, exist_ok=True)
            for zip_name, digest in ELLIPTIC_ZIP_SHA256.items():
                archive = download(f"{ELLIPTIC_MIRROR}/{zip_name}", RAW / "elliptic_pyg" / zip_name, digest)
                csv_name = zip_name[:-4]
                if not (raw / csv_name).exists():
                    with zipfile.ZipFile(archive) as z:
                        z.extract(csv_name, raw)
        for csv_name, digest in ELLIPTIC_CSV_SHA256.items():
            found = next(raw.rglob(csv_name), None)
            if found is None or sha256_file(found) != digest:
                raise RuntimeError(f"{csv_name}: missing or not byte-identical to the previous benchmark's source")
        return cds.load_dataset("elliptic", raw, strict=True)
    if name == "tsocial":
        path = pdata.canonical_path("tsocial")
        if not path.exists():
            # Fallback: BWGNN authors' Drive release, decoded with DGL, then frozen.
            target = RAW / "tsocial"
            target.mkdir(parents=True, exist_ok=True)
            graph_file = target / "tsocial"
            if not graph_file.exists():
                archive = target / "tsocial.zip"
                if not archive.exists():
                    try:
                        import gdown
                    except ImportError as exc:
                        raise FileNotFoundError(MANUAL_HELP["tsocial"]) from exc
                    print("  downloading tsocial.zip (~GBs) from the BWGNN authors' Google Drive")
                    gdown.download(id=TSOCIAL_DRIVE_ID, output=str(archive) + ".part", quiet=True)
                    Path(str(archive) + ".part").replace(archive)
                with zipfile.ZipFile(archive) as z:
                    z.extract("tsocial", target)
            if sha256_file(graph_file) != TSOCIAL_SOURCE_SHA256:
                raise RuntimeError("tsocial graph differs from the dataset owner's recorded source hash")
            dataset = cds.load_dataset("tsocial", graph_file, strict=True)
            cds.save_canonical(dataset, path)
            return dataset
        if sha256_file(path) != TSOCIAL_CANONICAL_SHA256:
            raise RuntimeError("tsocial_canonical.npz hash differs from the dataset owner's manifest")
        return cds.load_dataset("tsocial", path)
    raise ValueError(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--tfinance-raw", default=None)
    parser.add_argument("--elliptic-raw", default=None)
    parser.add_argument("--skip-heterophily", action="store_true",
                        help="skip the (slow on T-Social) heterophily report")
    args = parser.parse_args(argv)

    CANON.mkdir(parents=True, exist_ok=True)
    status_path = CANON / "data_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}

    for name in args.only:
        print(f"\n[{name}]")
        cache = pdata.canonical_path(name)
        entry = {"dataset": name}
        try:
            if cache.exists() and name != "tsocial":
                dataset = cds.load_dataset(name, cache)
                print(f"  canonical cache present: {cache.name}")
            else:
                dataset = build(name, args)
                if name != "tsocial":
                    cds.save_canonical(dataset, cache)
                    print(f"  froze {cache.name}")
            entry["canonical_cache"] = str(cache.relative_to(ROOT))
            entry["canonical_sha256"] = sha256_file(cache)
            entry["source_sha256"] = dataset.source_sha256
            entry["report_check"] = pdata.check_against_report(dataset)
            print(f"  Table-3 check: {'MATCH' if entry['report_check']['matches_report'] else 'MISMATCH'} "
                  f"{entry['report_check']['observed']}")
            if not entry["report_check"]["matches_report"]:
                raise RuntimeError(f"statistics differ from the report: {entry['report_check']['mismatches']}")
            entry["previous_benchmark_check"] = pdata.check_against_previous(dataset)
            print(f"  identity vs previous benchmark: {entry['previous_benchmark_check']}")
            if not entry["previous_benchmark_check"]["identical_to_previous_benchmark"]:
                raise RuntimeError("canonical data differ from the data used by the previous benchmark")
            feats = dataset.features
            entry["feature_checks"] = {
                "finite": bool(np.isfinite(feats).all()),
                "constant_columns": int((feats.max(0) == feats.min(0)).sum()),
                "duplicate_feature_rows_eligible": int(
                    dataset.eligible_idx.size - np.unique(feats[dataset.eligible_idx], axis=0).shape[0]),
            }
            print(f"  feature checks: {entry['feature_checks']}")
            entry["registry_bytes"] = pdata.verify_registry_bytes(name, pdata.DEFAULT_SPLIT_DIR, dataset.labels)
            if entry["registry_bytes"]:
                print(f"  registry files: {entry['registry_bytes']}")
                if any(not v.startswith("ok") for v in entry["registry_bytes"].values()):
                    raise RuntimeError("registry file hash mismatch")
            ratios = {}
            for ratio in ("TR40", "TR30", "TR20", "TR10"):
                arrays, path, digest = pdata.resolve_split(dataset, ratio)
                ratios[ratio] = {"file": Path(path).name, "sha256": digest,
                                 "counts": pdata._partition_counts(arrays, dataset.labels)}
            entry["splits"] = ratios
            print(f"  TR40 counts verified vs previous benchmark: {ratios['TR40']['counts']}")
            if not args.skip_heterophily and name != "tsocial":
                report = CANON / f"{name}_dataset_report.json"
                cds.write_dataset_report(dataset, report)
            entry["status"] = "AVAILABLE"
        except FileNotFoundError as exc:
            entry["status"] = "MISSING_SOURCE"
            entry["action_required"] = str(exc)
            print(f"  UNAVAILABLE: {exc}")
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            entry["status"] = "INVALID"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  INVALID: {entry['error']}")
        status[name] = entry
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print("\nSummary:")
    for name in DATASETS:
        print(f"  {name:9s} {status.get(name, {}).get('status', 'NOT_ATTEMPTED')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
