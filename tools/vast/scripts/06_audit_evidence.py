#!/usr/bin/env python3
import os,pathlib,argparse,csv,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',default='reports/evidence_file_index.csv');p.add_argument('--hash-small',action='store_true');a=p.parse_args()
root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
if not root.exists():raise SystemExit(f'NOT_FOUND {root}')
rows=[];needles=('final','tuning','sha256','provenance','manifest','split','smoke','winner','evidence','log','result')
for dirpath, dirs, files in os.walk(root):
  dirs[:]=[n for n in dirs if n not in ('.git','site-packages','__pycache__','.venv','envs','node_modules')]
  for file in files:
    path=pathlib.Path(dirpath)/file
    if not any(key in str(path.relative_to(root)).lower() for key in needles):continue
    size=path.stat().st_size;sha=''
    if a.hash_small and size<=50*1024*1024:
      h=hashlib.sha256()
      with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
      sha=h.hexdigest()
    rows.append({'relative_path':str(path.relative_to(root)),'bytes':size,'sha256_if_requested':sha,'completion':'UNVERIFIED_FILE_EXISTS_ONLY'})
with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256_if_requested','completion']);w.writeheader();w.writerows(rows)
print(f'INDEXED={len(rows)}; FILE_EXISTENCE_IS_NOT_FINAL_RUN_PROOF; OUTPUT={out}')
