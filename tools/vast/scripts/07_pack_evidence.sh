#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/workspace/comp8851/evidence}"; OUT="${2:-/workspace/comp8851/manifests}"
[[ -d "$ROOT" ]] || { echo "NO EVIDENCE PATH: $ROOT" >&2; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
find . -type f ! -name '*.tmp' -print0 | sort -z | xargs -0 -r sha256sum > "$OUT/evidence_SHA256SUMS.txt"
find . -type f | sort > "$OUT/evidence_filelist.txt"
printf 'EVIDENCE_INDEX_CREATED=%s\n' "$OUT/evidence_SHA256SUMS.txt"
printf 'WARNING: hash manifest proves file integrity only, not experimental validity.\n'
