#!/usr/bin/env bash
set -euo pipefail
OFFER_ID="${1:-}"; DISK_GB="${2:-}"
if [[ ! "$OFFER_ID" =~ ^[0-9]+$ || ! "$DISK_GB" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/00_offer_dry_run.sh <VAST_OFFER_ID> <DISK_GB>" >&2; exit 2
fi
if (( DISK_GB < 40 )); then echo "WARNING: disk less than 40 GB risks failure." >&2; fi
printf 'PREVIEW ONLY — no instance will be created.\n\n'
printf "vastai create instance '%s' --image 'vastai/base-image:@vastai-automatic-tag' --disk '%s' --ssh --direct --label 'comp8851-protocol-locked'\n" "$OFFER_ID" "$DISK_GB"
printf '\nVast offer ID is NOT host ID or machine ID. Disk allocation is fixed.\n'
printf 'This kit intentionally does not execute rental commands.\n'
