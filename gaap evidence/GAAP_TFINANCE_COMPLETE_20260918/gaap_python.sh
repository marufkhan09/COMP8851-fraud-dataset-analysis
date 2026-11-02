#!/usr/bin/env bash
set -euo pipefail
PY="/workspace/gaap_vast/envs/gaap-author/bin/python"
LIBS=$(find "/workspace/gaap_vast/envs/gaap-author/lib" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${LIBS}:${LD_LIBRARY_PATH:-}"
exec "$PY" "$@"
