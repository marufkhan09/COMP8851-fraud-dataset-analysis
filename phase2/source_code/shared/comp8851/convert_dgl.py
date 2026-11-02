"""Convert DGL-format datasets into the library-free canonical ``.npz`` cache.

Three of the six benchmark datasets ship as DGL binaries: FDCompCN (`comp.dgl`),
T-Finance and T-Social. Reading them needs DGL, whose wheels are compiled
against a specific PyTorch ABI. On a host whose torch is newer than any
published DGL wheel (Kaggle currently ships torch 2.10), DGL cannot be imported
at all and those three datasets become unreachable.

The fix is to decode them once in an isolated environment and store the result
in the canonical ``.npz`` format, which needs only NumPy and SciPy. Training
then runs in the main GPU environment with no DGL at all:

* CARE-GNN never needed DGL.
* GHRN falls back to its ``torch.sparse`` backend.

This script is the decoder. It is meant to be executed **by the isolated
interpreter**, not the main one:

    <isolated python> shared/comp8851/convert_dgl.py \
        --repo-root . --output data/canonical \
        --dataset fdcompcn data/fdcompcn \
        --dataset tfinance data/bwgnn_drive/tfinance

Conversion is deterministic and preserves the graph exactly: the same nodes,
features, labels and edges, only in a different container. The resulting file
carries its own SHA256 so the converted view has its own identity in the
dataset manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode DGL-format datasets into the canonical .npz cache")
    parser.add_argument("--repo-root", required=True,
                        help="Repository root, so shared.comp8851 is importable")
    parser.add_argument("--output", required=True,
                        help="Directory to write <name>_canonical.npz into")
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "PATH"),
                        required=True, help="Dataset name and its source path")
    parser.add_argument("--report", default=None,
                        help="Optional JSON file summarising what was converted")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    # Fail loudly and early if this interpreter cannot actually use DGL, since
    # that is the entire reason this script exists.
    try:
        import dgl

        print(f"DGL {dgl.__version__} available in this interpreter")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: this interpreter cannot import DGL: {type(exc).__name__}: {exc}")
        print("Run this script with the isolated interpreter that has DGL installed.")
        return 2

    from shared.comp8851 import datasets as ds
    from shared.comp8851.protocol import sha256_file

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    results = []
    failures = 0

    for name, source in args.dataset:
        print(f"\n[{name}] decoding {source}")
        try:
            dataset = ds.load_dataset(name, source, strict=False)
            cache = output / f"{name}_canonical.npz"
            ds.save_canonical(dataset, cache)
            stats = dataset.statistics()
            digest = sha256_file(cache)

            print(f"    nodes {stats['nodes']:,} | features {stats['feature_dimension']} | "
                  f"relations {len(dataset.relations)}")
            print(f"    fraud {stats['fraud_nodes']:,} "
                  f"({stats['fraud_percentage']:.2f}%) | "
                  f"edges {stats['union_stored_directed_edges']:,}")
            print(f"    wrote {cache.name} ({cache.stat().st_size:,} bytes)")
            print(f"    sha256 {digest}")

            results.append({
                "dataset": name,
                "source": str(source),
                "canonical_cache": str(cache),
                "canonical_cache_sha256": digest,
                "statistics": stats,
                "status": "CONVERTED",
            })
        except Exception as exc:  # noqa: BLE001 - record, never hide
            failures += 1
            import traceback

            traceback.print_exc()
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            results.append({
                "dataset": name,
                "source": str(source),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if args.report:
        Path(args.report).write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")

    converted = sum(1 for r in results if r["status"] == "CONVERTED")
    print(f"\nconverted {converted}/{len(results)}; {failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
