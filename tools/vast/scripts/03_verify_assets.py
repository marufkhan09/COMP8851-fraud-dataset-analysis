#!/usr/bin/env python3
import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser(description='SHA256 gate; NO split recreation or downloads')
p.add_argument('--manifest',required=True);p.add_argument('--workspace',default='/workspace/comp8851');p.add_argument('--dataset',default=None)
a=p.parse_args();cfg=json.loads(pathlib.Path(a.manifest).read_text());base=pathlib.Path(a.workspace)
items=cfg['datasets']; selected={a.dataset:items[a.dataset]} if a.dataset else items
ok=True
for name,entry in selected.items():
  print(f'== {name} ==')
  for kind in ['source','split']:
    rel=entry.get(kind+'_path'); expected=entry.get(kind+'_sha256')
    if not rel or not expected:
      print(f'BLOCKED {kind}: missing verified path/hash');ok=False;continue
    path=pathlib.Path(rel)
    if not path.is_absolute():path=base/path
    if not path.is_file():
      print(f'BLOCKED {kind}: file missing: {path}');ok=False;continue
    h=hashlib.sha256()
    with path.open('rb') as f:
      for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    got=h.hexdigest()
    good=got.lower()==expected.lower()
    print(f'{"PASS" if good else "BLOCKED"} {kind} bytes={path.stat().st_size} sha256={got} path={path}')
    if not good:ok=False
  for note in entry.get('special_validation',[]):print(f'REQUIRES_MANUAL_STRUCTURAL_VALIDATION: {note}')
print('FILE_SHA_GATE='+('PASS' if ok else 'BLOCKED'))
sys.exit(0 if ok else 2)
