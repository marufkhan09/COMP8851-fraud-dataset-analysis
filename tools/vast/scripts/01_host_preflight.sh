#!/usr/bin/env bash
set -euo pipefail
printf '== HOST PREFLIGHT (read only) ==\n'
date -u '+utc=%FT%TZ'; hostname || true
id || true
printf '\n== GPU / DRIVER ==\n'
if ! command -v nvidia-smi >/dev/null; then echo 'BLOCKED: nvidia-smi absent'; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || nvidia-smi
printf '\n== CPU / RAM ==\n'; nproc; free -h || true
printf '\n== DISK ==\n'; df -h /workspace; df -h /tmp || true
printf '\n== TOOLS ==\n'
for x in git python3 sha256sum unzip tmux; do command -v "$x" || echo "MISSING_TOOL=$x"; done
printf '\n== NETWORK DNS ==\n'
getent hosts github.com | head -1 || echo 'WARNING: GitHub hostname resolution failed'
printf '\nPREFLIGHT FINISHED; no packages installed or tests trained.\n'
