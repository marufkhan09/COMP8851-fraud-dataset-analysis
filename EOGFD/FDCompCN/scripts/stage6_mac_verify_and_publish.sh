#!/usr/bin/env bash
# On Mac: verify BOTH Vast ZIPs; publish exactly EOGFD/FDCompCN and EOGFD/Elliptic to Pathik-ass3.
set -Eeuo pipefail
export GIT_PAGER=cat PAGER=cat GIT_TERMINAL_PROMPT=0
D="$HOME/Downloads"
PUB="$D/EOGFD_57212396_GITHUB_PUBLICATION.zip"
PRIV="$D/EOGFD_57212396_PRIVATE_FINAL_CHECKPOINTS.zip"
BRANCH=Pathik-ass3
URL=https://github.com/marufkhan09/COMP8851-fraud-dataset-analysis.git

command -v git >/dev/null
command -v python3 >/dev/null
[[ -f "$PUB" && -f "$PRIV" ]] || { echo "STOP: Download BOTH completed export ZIPs from Vast to ~/Downloads first"; exit 2; }
STAGING="$(mktemp -d "${TMPDIR:-/tmp/}EOGFD_57212396_PUBLISH.XXXXXXXX")"
trap 'echo "PUBLICATION_WORKDIR=$STAGING"' EXIT
mkdir -p "$STAGING/public"

python3 - "$PUB" "$PRIV" "$STAGING/public" <<'PY'
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import sys, hashlib, json
public,private,out=map(Path,sys.argv[1:])
def sha(b):return hashlib.sha256(b).hexdigest()
with ZipFile(private) as z:
 assert z.testzip() is None,'STOP: private archive damaged'
 members=set(z.namelist());manifest=json.loads(z.read('PRIVATE/SHA256_MANIFEST.json'))
 assert set(manifest)==members-{'PRIVATE/SHA256_MANIFEST.json'}
 assert len([n for n in members if n.endswith('/best_validation.pt') and '/finals/' in n])==24
 assert len([n for n in members if n.endswith('/best_validation.pt') and '/tuning/' in n])==24
 assert len([n for n in members if n.endswith('_splits.npz')])==2
 for n,meta in manifest.items():
  b=z.read(n);assert sha(b)==meta['sha256'] and len(b)==meta['bytes'],('STOP: private SHA mismatch',n)
print('EOGFD_MAC_PRIVATE_CHECKPOINTS_48_AND_SPLITS_2=VERIFIED',flush=True)
with ZipFile(public) as z:
 assert z.testzip() is None,'STOP: publication archive damaged'
 names=z.namelist();assert len(names)==len(set(names)) and len(names)>240
 allowed={'.json','.md','.log','.txt','.csv','.py','.sh','.npy'}
 assert all(n.startswith(('EOGFD/FDCompCN/','EOGFD/Elliptic/')) and PurePosixPath(n).suffix in allowed and '..' not in PurePosixPath(n).parts and not PurePosixPath(n).is_absolute() for n in names)
 for ds in ('FDCompCN','Elliptic'):
  pref=f'EOGFD/{ds}/'
  mn=pref+f'evidence/EOGFD_{ds}_57212396_PublicationFileManifest.json'
  m=json.loads(z.read(mn))
  assert m['branch']=='Pathik-ass3' and m['destination']==pref
  assert set(m['files'])=={n for n in names if n.startswith(pref)}-{mn}
  for n,meta in m['files'].items():
   data=z.read(n);assert sha(data)==meta['sha256'] and len(data)==meta['bytes'],('STOP: public hash mismatch',n)
  audit=json.loads(z.read(pref+f'evidence/EOGFD_{ds}_57212396_IndependentPublicationAudit.json'))
  status=json.loads(z.read(pref+'status.json'))
  assert audit['status']==f'EOGFD_{ds}_57212396_PUBLIC_EVIDENCE_AUDIT_PASS'
  assert audit['heldout_test_once_completed']==12 and audit['additional_model_inference_in_export'] is False
  assert status['status']=='COMPLETED_CONDITIONAL_UNIFIED_BENCHMARK'
  assert status['completed_once_only_heldout_test_evaluations']==12 and status['venus_approval_recorded'] is False
  for ratio in ('TR40','TR30','TR20','TR10'):
   for seed in (2,42,72):
    b=pref+f'results/finals/{ratio}/seed_{seed}/'
    for fn in ('frozen_training.json','test_metrics.json','test_once_claim.json','test_scores.npy','epoch_times.csv'):
     assert b+fn in names,('STOP missing artifact',b+fn)
 for n in names:
  target=out/n;target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(z.read(n))
 print('EOGFD_MAC_PUBLICATION_ARCHIVE=VERIFIED files='+str(len(names)),flush=True)
print('PUBLICATION_ZIP_SHA256='+sha(public.read_bytes()))
print('PRIVATE_ZIP_SHA256='+sha(private.read_bytes()))
PY

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh auth setup-git >/dev/null 2>&1 || true
fi

echo '=== CLONE AUTHENTICATED Pathik-ass3 BRANCH ==='
git -c credential.interactive=never clone --quiet --depth 1 --single-branch --branch "$BRANCH" "$URL" "$STAGING/repo"
[[ "$(git -C "$STAGING/repo" branch --show-current)" == "$BRANCH" ]]
[[ -z "$(git -C "$STAGING/repo" status --porcelain)" ]]
OLD_SHA="$(git -C "$STAGING/repo" rev-parse HEAD)"

python3 - "$STAGING/repo" <<'PY'
from pathlib import Path
import sys,json
repo=Path(sys.argv[1]);target=('FDCompCN','Elliptic')
for ds in target:
 p=repo/'EOGFD'/ds
 assert p.is_dir(),('Missing existing destination',p)
 assert {f.name for f in p.iterdir()}=={'README.md','status.json'},('STOP: not a placeholder-only folder',str(p))
 s=json.loads((p/'status.json').read_text())
 assert s['status']=='BLOCKED_NOT_STARTED',('STOP: target changed or already published',ds)
print('GITHUB_DESTINATIONS_PLACEHOLDER_ONLY=PASS')
PY

for ds in FDCompCN Elliptic; do
  cp -R "$STAGING/public/EOGFD/$ds/." "$STAGING/repo/EOGFD/$ds/"
done

git -C "$STAGING/repo" add -f -- EOGFD/FDCompCN EOGFD/Elliptic
CHANGES="$(git -C "$STAGING/repo" diff --cached --name-only)"
[[ -n "$CHANGES" ]]
if printf '%s\n' "$CHANGES" | grep -Ev '^EOGFD/(FDCompCN|Elliptic)/' >/dev/null; then
 echo 'STOP: Out-of-scope changes staged';exit 2
fi

git -C "$STAGING/repo" -c core.whitespace=-blank-at-eof diff --cached --check
for ds in FDCompCN Elliptic; do
 disk=$(find "$STAGING/repo/EOGFD/$ds" -type f | wc -l | tr -d ' ')
 staged=$(git -C "$STAGING/repo" ls-files -- "EOGFD/$ds" | wc -l | tr -d ' ')
 [[ "$disk" == "$staged" ]] || { echo "STOP: count mismatch $ds $disk $staged"; exit 2; }
 echo "EOGFD_${ds}_STAGED_FILES=$staged"
done

git -C "$STAGING/repo" --no-pager diff --cached --stat | tail -n 14
REMOTE="$(git -C "$STAGING/repo" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')"
[[ "$REMOTE" == "$OLD_SHA" ]] || { echo "STOP: Remote branch changed; no push attempted";exit 2; }
[[ -n "$(git -C "$STAGING/repo" config user.name || true)" ]] || { echo 'STOP: Mac git user.name not configured'; exit 2; }
[[ -n "$(git -C "$STAGING/repo" config user.email || true)" ]] || { echo 'STOP: Mac git user.email not configured'; exit 2; }

git -C "$STAGING/repo" commit -m "EOGFD: publish audited FDCompCN and Elliptic conditional benchmark evidence" >/dev/null
NEW_SHA="$(git -C "$STAGING/repo" rev-parse HEAD)"
git -C "$STAGING/repo" push origin "HEAD:refs/heads/$BRANCH"
AFTER="$(git -C "$STAGING/repo" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')"
[[ "$AFTER" == "$NEW_SHA" ]] || { echo 'STOP: remote commit does not match';exit 2; }
echo "EOGFD_BOTH_DATASETS_GITHUB_REMOTE_VERIFIED=PASS"
echo "GITHUB_COMMIT=$NEW_SHA"
echo "FDCompCN=https://github.com/marufkhan09/COMP8851-fraud-dataset-analysis/tree/Pathik-ass3/EOGFD/FDCompCN"
echo "Elliptic=https://github.com/marufkhan09/COMP8851-fraud-dataset-analysis/tree/Pathik-ass3/EOGFD/Elliptic"
