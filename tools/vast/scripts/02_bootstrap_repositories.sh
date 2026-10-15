#!/usr/bin/env bash
set -euo pipefail
# Run only after host preflight, at the start of a NEW GPU instance.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${COMP8851_ROOT:-/workspace/comp8851}"
for command_name in git python3 sha256sum; do
  command -v "$command_name" >/dev/null || { echo "BLOCKED missing program: $command_name" >&2; exit 1; }
done
mkdir -p "$ROOT"/{repos,assets,cache,logs,evidence,manifests,envs}
# DO NOT overwrite preexisting manifests: retain record of earlier decisions.
for name in source_lock protocol assets_manifest; do
  source_file="$HERE/configs/${name}.json"
  destination="$ROOT/manifests/${name}_initial.json"
  if [[ ! -e "$destination" ]]; then cp -n "$source_file" "$destination"; fi
done
export ROOT HERE
python3 - <<'PY_REPOS'
import json,os,subprocess,pathlib,sys
root=pathlib.Path(os.environ['ROOT'])/'repos'
lock=json.loads((pathlib.Path(os.environ['HERE'])/'configs/source_lock.json').read_text())
def cmd(*args):
    subprocess.run(args,check=True)
def capture(*args):
    return subprocess.check_output(args,text=True).strip()
for label,spec in lock.items():
    path=root/spec['destination']
    target=spec['commit']
    if not (path/'.git').is_dir():
        if path.exists() and any(path.iterdir()):
            raise SystemExit(f'BLOCKED {label}: nonempty path lacks valid Git metadata: {path}')
        cmd('git','clone','--depth','1','--single-branch','--branch',spec['branch'],'--filter=blob:none',spec['url'],str(path))
    actual=capture('git','-C',str(path),'rev-parse','HEAD')
    if actual != target:
        # Retrieve exact verified revision even when upstream branch advanced.
        # Do not mutate any local checkout with uncommitted edits.
        changes=capture('git','-C',str(path),'status','--porcelain')
        if changes:
            raise SystemExit(f'BLOCKED {label}: checkout has local edits. Do not overwrite: {path}')
        cmd('git','-C',str(path),'fetch','--depth','1','origin',target)
        cmd('git','-C',str(path),'checkout','--detach',target)
    actual=capture('git','-C',str(path),'rev-parse','HEAD')
    if actual != target:
        raise SystemExit(f'BLOCKED {label}: expected {target} actual {actual}')
    if capture('git','-C',str(path),'status','--porcelain'):
        raise SystemExit(f'BLOCKED {label}: dirty checkout: {path}')
    print(f'SOURCE_PIN_PASS {label} {actual}')
print('SOURCE PINS ALL PASS; ENVIRONMENTS AND DATA NOT VERIFIED')
PY_REPOS
{
  date -u '+prepared_utc=%FT%TZ'
  echo "workspace=$ROOT"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  python3 -V
  df -h "$ROOT"
} > "$ROOT/manifests/bootstrap_host_state.txt"
echo 'BOOTSTRAP_DONE=YES TRAINING_STARTED=NO ASSETS_VERIFIED=NO'
