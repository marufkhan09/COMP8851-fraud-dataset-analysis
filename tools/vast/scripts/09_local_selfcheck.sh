#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
printf '== PACKAGE INTEGRITY ==\n'
python3 - <<'PY_SHA'
from pathlib import Path
import hashlib, sys
bad=[]
for line in Path('SHA256SUMS.txt').read_text().splitlines():
    if not line.strip(): continue
    expected, name=line.split('  ',1)
    file=Path(name)
    found=hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else 'MISSING'
    if found != expected: bad.append(name)
    print(f'{name}: {"OK" if found==expected else "FAIL"}')
if bad: sys.exit('SHA256 INTEGRITY FAILED: '+', '.join(bad))
PY_SHA
printf '\n== SCRIPT SYNTAX ==\n'
for f in scripts/*.sh; do bash -n "$f"; done
python3 - <<'PY_VALIDATE'
import json, pathlib, ast, csv
for p in list(pathlib.Path('configs').glob('*.json'))+list(pathlib.Path('templates').glob('*.json')):
    json.loads(p.read_text())
for p in pathlib.Path('scripts').glob('*.py'):
    ast.parse(p.read_text(),filename=str(p))
with open('reports/blocked_job_plan.csv',newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==432,(len(rows),'expected 432 blocked planned fits')
assert all('BLOCKED' in row['status'] for row in rows)
print('JSON_SYNTAX=PASS PYTHON_SYNTAX=PASS PLAN_432_BLOCKED=PASS')
PY_VALIDATE
printf '\n== SOURCE PINS / UNKNOWN ASSETS ==\n'
python3 - <<'PY_LOCK'
import json
locks=json.load(open('configs/source_lock.json'))
for name,data in locks.items():
    print(name, data['commit'], data['url'])
assets=json.load(open('configs/assets_manifest.json'))
print('CANONICAL_ASSET_BYTES_BUNDLED=0')
print('ACTUAL_DATA_AND_SPLIT_HASH_GATE_REQUIRES_RECOVERY=YES')
PY_LOCK
printf '\nLOCAL PACKAGE CHECK PASS; THIS DOES NOT VERIFY GPU OR CANONICAL DATA.\n'
