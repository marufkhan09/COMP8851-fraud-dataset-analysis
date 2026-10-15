#!/usr/bin/env bash
# Run on any EXISTING accessible prior Vast instance to locate canonical files and evidence.
# This script changes NOTHING and does not rent or train.
set -euo pipefail
SEARCH_ROOT="${1:-/workspace}"
[[ -d "$SEARCH_ROOT" ]] || { echo "NOT_A_DIRECTORY: $SEARCH_ROOT" >&2; exit 2; }
printf 'READ_ONLY_RECOVERY_SEARCH: %s\n' "$SEARCH_ROOT"
find "$SEARCH_ROOT" -maxdepth 10 -type f \( -name 'tsocial' -o -name 'tsocial_seed2_nested_splits.npz' -o -name '*FROZEN_WINNER*.json' -o -name '*FINAL_12_RUNS*.csv' -o -name 'SHA256SUMS.txt' \) -print 2>/dev/null | sort
printf '\nHash ONLY canonical T-Social candidates (may take some time):\n'
while IFS= read -r -d '' file; do
  case "$file" in
   */shared/tsocial/tsocial|*/shared/tsocial/tsocial_seed2_nested_splits.npz)
       sha256sum "$file" ;;
  esac
done < <(find "$SEARCH_ROOT" -maxdepth 10 -type f \( -name 'tsocial' -o -name 'tsocial_seed2_nested_splits.npz' \) -print0 2>/dev/null)
printf '\nEXPECTED_SOURCE_SHA=8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0\n'
printf 'EXPECTED_SPLIT_SHA=b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519\n'
printf 'Search does not validate label/split semantics and makes no changes.\n'
