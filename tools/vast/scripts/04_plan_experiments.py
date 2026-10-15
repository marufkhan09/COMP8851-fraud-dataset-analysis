#!/usr/bin/env python3
import argparse,csv,pathlib,json
p=argparse.ArgumentParser();p.add_argument('--output',default='reports/blocked_job_plan.csv');a=p.parse_args()
cfg=json.loads((pathlib.Path(__file__).resolve().parent.parent/'configs/protocol.json').read_text())
models=cfg['models_new'];ds=cfg['datasets'];rows=[]
for model in models:
  for d in ds:
    for trial in range(1,13):
      rows.append(dict(model=model,dataset=d,phase='TUNING_TR40',trial=trial,ratio='0.40',seed=2,
      status='BLOCKED_REQUIRE_DATA_ENV_CANONICAL_ADAPTER_APPROVAL',run_id=f'{model}_{d}_T40_TRIAL{trial:02d}'))
    for ratio in cfg['train_ratios']:
      for seed in cfg['train_seeds']:
        r=int(round(100*ratio))
        rows.append(dict(model=model,dataset=d,phase='FINAL',trial='',ratio=f'{ratio:.2f}',seed=seed,
        status='BLOCKED_REQUIRE_FROZEN_TR40_WINNER_AND_DATA_HASHES',run_id=f'{model}_{d}_TR{r}_SEED{seed}'))
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'{len(rows)} planning rows (12 TR40 tuning trials + 12 final ratios/seeds per model x dataset), all BLOCKED until gates pass: {out}')
