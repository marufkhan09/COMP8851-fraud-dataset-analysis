#!/usr/bin/env python3
import argparse,json,pathlib,datetime
p=argparse.ArgumentParser(description='Append-only pre-run template; NEVER trains or evaluates test')
p.add_argument('--model',required=True,choices=['RGTAN','SAWGAD','EOGFD'])
p.add_argument('--dataset',required=True,choices=['YelpChi','Amazon','T-Finance','Elliptic','FDCompCN','T-Social'])
p.add_argument('--ratio',type=float,required=True,choices=[.4,.3,.2,.1]);p.add_argument('--seed',type=int,required=True,choices=[2,42,72]);p.add_argument('--phase',required=True,choices=['SMOKE','TUNING','FINAL']);p.add_argument('--output',required=True);a=p.parse_args()
if a.phase=='TUNING' and (a.ratio!=.4 or a.seed!=2):p.error('TUNING must be TR40 seed2')
out=pathlib.Path(a.output)
if out.exists():raise SystemExit('BLOCKED: manifest exists; no overwrite')
obj=dict(schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model=a.model,dataset=a.dataset,ratio=a.ratio,training_seed=a.seed,split_seed=2,phase=a.phase,
status='BLOCKED_PRE_RUN_GATES',source_sha256=None,split_sha256=None,repo_commit=None,frozen_tr40_winner_sha256=None,adapter_review='PENDING',supervisor_approval='PENDING',validation_only_selection=True,
test_access_during_smoke_or_tuning=False,metrics={},timing_seconds={'preprocessing':None,'training':None,'validation':None,'test_inference':None},peak_gpu_gib=None,notes='Evidence header only. Does not prove compliance until artifacts independently verified.')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,indent=2)+'\n');print(f'MANIFEST_CREATED: {out} (BLOCKED; do not train yet)')
