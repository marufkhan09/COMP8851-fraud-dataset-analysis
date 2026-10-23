#!/usr/bin/env python3
"""SAWGAD T-Social, 12 TR40 val-only tuning jobs -> frozen winner -> 4x3 training.
No test labels or scores. Resumable from completed job records. Research adaptation.
"""
from __future__ import annotations
import argparse, datetime, fcntl, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path

ROOT=Path('/workspace/comp8851')
KIT=Path('/workspace/COMP8851_I2_SAWGAD_12TUNE_3x4_20261023')
PY=ROOT/'repos/sawgad_tsocial_author/.venv/bin/python'
WORKER=KIT/'sawgad_job.py'
ROOT_OUT=ROOT/'results/sawgad/tsocial/stage3_tune12_final3x4_adapted_v1'
SOURCE=ROOT/'assets/tsocial/tsocial'
SPLIT=ROOT/'assets/tsocial/tsocial_seed2_nested_splits.npz'
PREVIOUS=ROOT/'results/sawgad/tsocial/stage3_tr40_seed2_fullval_adapted_training_v1/pilot_completed.json'
SHA_SOURCE='8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'
SHA_SPLIT='b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519'
PIN='64d6d531a2f03549f0bfbe461092beedd6f70bd2'
SEEDS=(2,42,72)
RATIOS=('TR40','TR30','TR20','TR10')
GRID=[{'hidden_dim':h,'drop_rate':d,'full_lr':lr}
      for h in (32,64) for d in (0.0,0.25) for lr in (0.001,0.003,0.005)]
assert len(GRID)==12


def fail(message): raise RuntimeError('STOP: '+message)
def require(v,msg):
    if not v: fail(msg)
def say(*a): print(*a,flush=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic_json(p,obj):
    tmp=p.with_name(p.name+'.writing')
    with open(tmp,'w') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def grid_spec(i):
    return dict(schema='sawgad-tsocial-job-v1',stage='tune',ratio='TR40',seed=2,
      run_id='tune_%02d'%i,trial=i,config=GRID[i-1],
      warmup_steps=32,supervised_steps_per_epoch=1024,supervised_epochs=8,
      validation_batch_size=2048,fanout=10,
      source_sha256=SHA_SOURCE,split_sha256=SHA_SPLIT,
      frozen_sha256=None,frozen_from_trial=None)

def final_spec(ratio,seed,lock,locksha):
    return dict(schema='sawgad-tsocial-job-v1',stage='final',ratio=ratio,seed=seed,
      run_id=f'final_{ratio}_seed{seed}',trial=None,config=lock['winner_config'],
      warmup_steps=32,supervised_steps_per_epoch=1024,supervised_epochs=16,
      validation_batch_size=2048,fanout=10,
      source_sha256=SHA_SOURCE,split_sha256=SHA_SPLIT,
      frozen_path=str(ROOT_OUT/'frozen_tr40_winner.json'),
      frozen_sha256=locksha,frozen_from_trial=lock['winner_trial'])

def selftest():
    require(len(GRID)==12 and len({json.dumps(x,sort_keys=True) for x in GRID})==12,'grid invalid')
    require(len([(r,s) for r in RATIOS for s in SEEDS])==12,'final grid invalid')
    require(grid_spec(1)['config']!=grid_spec(12)['config'],'grid endpoints same')
    require(all(grid_spec(i)['run_id']==f'tune_{i:02d}' for i in range(1,13)),'ids invalid')
    require(all(grid_spec(i)['stage']=='tune' and grid_spec(i)['ratio']=='TR40' and grid_spec(i)['seed']==2 for i in range(1,13)),'tune labels invalid')
    say('I2_PIPELINE_STATIC_12TUNE_3x4_SELFTEST=PASS')
    say('I2_PRETEST_HOLDOUT_EVALUATIONS=0')
    say('I2_ADAPTED_NOT_AUTHOR_IDENTICAL=YES')

def preflight():
    require(socket.gethostname()=='a92ee6fa0813','wrong I2 host')
    require(KIT.is_dir() and WORKER.is_file() and PY.is_file(),'kit or isolated Python absent')
    require(sha(SOURCE)==SHA_SOURCE and sha(SPLIT)==SHA_SPLIT,'source or split SHA mismatch')
    commit=subprocess.check_output(['git','-C',str(ROOT/'repos/sawgad_tsocial_author'),'rev-parse','HEAD'],text=True).strip()
    require(commit==PIN,'SAWGAD author repo commit moved')
    dirty=subprocess.check_output(['git','-C',str(ROOT/'repos/sawgad_tsocial_author'),'status','--porcelain'],text=True).strip()
    require(not dirty,'original upstream working tree modified')
    require(PREVIOUS.is_file(),'sustained TR40 fullval pilot evidence missing')
    previous=json.loads(PREVIOUS.read_text())
    require(previous['status']=='ADAPTED_SINGLE_CONFIG_FULLVAL_TRAINING_COMPLETED_NOT_FINAL_BENCHMARK','previous pilot not complete')
    require(previous['heldout_test_labels_indexed'] is False and previous['heldout_test_metrics_computed'] is False,'previous pilot test guard')
    require(previous['all_validation_nodes_evaluated'] is True and previous['full_validation_examples']==1156213,'pilot validation incomplete')
    require(abs(previous['full_validation_AUPRC_best']-0.1596089366267632)<1e-9,'different previous pilot best')
    ck=ROOT/'results/sawgad/tsocial/stage3_tr40_seed2_fullval_adapted_training_v1/best_full_validation.pt'
    require(ck.exists(),'prior best full-val checkpoint missing')
    subprocess.run([str(PY),str(WORKER),'--selftest'],check=True)
    require(sha(WORKER)==sha(KIT/'sawgad_job.py'),'worker SHA check failed')
    say('I2_PIPELINE_AUTHOR_SPLIT_PREVIOUS_PILOT_VERIFIED=PASS')
    say('I2_GRID_TRIALS=12 FINAL_RUNS=12 TEST_SCORING=DISABLED')

def ensure_plan():
    ROOT_OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(schema='sawgad-tsocial-12tune-3x4-plan-v1',source_sha256=SHA_SOURCE,
              split_sha256=SHA_SPLIT,upstream_commit=PIN,worker_sha256=sha(WORKER),
              orchestrator_sha256=sha(Path(__file__)),grid=GRID,seeds=list(SEEDS),ratios=list(RATIOS),
              tuning_budget=dict(warmup_steps=32,rounds=8,updates_per_round=1024),
              finals_budget=dict(warmup_steps=32,rounds=16,updates_per_round=1024),
              full_validation=True,heldout_test_scoring=False,supervisor_approval_recorded=False,
              author_equivalent=False,experimental_status='adapted research evidence; not an approved canonical final benchmark')
    dest=ROOT_OUT/'plan.json'
    if dest.exists(): require(json.loads(dest.read_text())==plan,'plan/source changed after first launch')
    else:atomic_json(dest,plan)
    return plan

def verify_job_done(spec):
    p=ROOT_OUT/spec['run_id']; completed=p/'pilot_completed.json'
    if not completed.exists():return None
    r=json.loads(completed.read_text())
    cp=p/'best_full_validation.pt'
    require(cp.is_file(),'completed job without best checkpoint '+spec['run_id'])
    require(r['protocol']['run_id']==spec['run_id'],'run id mismatch')
    for key in ('ratio','seed','stage','run_id'):
        require(r['protocol'][key]==spec[key],'completed job protocol mismatch '+key)
    require({k:r['protocol'][k] for k in ('hidden_dim','drop_rate','full_lr')}==spec['config'],'job config drift')
    require(r['all_validation_nodes_evaluated'] is True and r['full_validation_examples']==1156213,'full val missing')
    require(r['heldout_test_labels_indexed'] is False and r['heldout_test_metrics_computed'] is False,'test contamination')
    require(r['total_supervised_optimizer_steps']==spec['supervised_steps_per_epoch']*spec['supervised_epochs'],'training budget truncated')
    require(r['total_warmup_optimizer_steps']==spec['warmup_steps'],'warmup truncated')
    require(0<=r['full_validation_AUPRC_best']<=1,'invalid AUPRC')
    hist_path=p/'pilot_history.json'
    require(hist_path.exists(),'history missing')
    hist=json.loads(hist_path.read_text())
    rows=hist['history']
    require(len(rows)==spec['supervised_epochs'],'history epoch count mismatch')
    require(all(x['full_validation_count']==1156213 for x in rows),'incomplete validation row')
    best=max(x['full_validation_AUPRC'] for x in rows)
    require(abs(best-r['full_validation_AUPRC_best'])<1e-12,'best AUPRC not supported by history')
    # Independently inspect locally generated best checkpoint metadata.
    import torch
    best_saved=torch.load(cp,map_location='cpu',weights_only=False)
    require(best_saved['protocol']==r['protocol'],'checkpoint protocol differs from completion record')
    require(abs(float(best_saved['full_validation_AUPRC'])-best)<1e-12,'best checkpoint AUPRC differs from history')
    require(best_saved['epoch'] in [x['epoch'] for x in rows if abs(x['full_validation_AUPRC']-best)<1e-12], 'checkpoint epoch not best')
    # The checkpoint file has a hash for the validation selection audit.
    return dict(run_id=spec['run_id'],best_val_AUPRC=float(r['full_validation_AUPRC_best']),
       checkpoint_sha256=sha(cp),history_sha256=sha(p/'pilot_history.json'),
       completed_sha256=sha(completed))

def run_one(spec):
    rid=spec['run_id'];dest=ROOT_OUT/rid;dest.mkdir(exist_ok=True,parents=True)
    sp=dest/'job_spec.json'
    if sp.exists():require(json.loads(sp.read_text())==spec,'job specification changed '+rid)
    else:atomic_json(sp,spec)
    saved=verify_job_done(spec)
    if saved:
        say('I2_JOB_REUSE_VERIFIED=PASS',rid,'best_fullval_AUPRC',round(saved['best_val_AUPRC'],6))
        return saved
    resume=(dest/'pilot_protocol.json').exists()
    cmd=[str(PY),'-u',str(WORKER),'--spec',str(sp)]
    if resume:cmd.append('--resume')
    say('I2_JOB_LAUNCH',rid,'mode',('resume' if resume else 'new'),'config',json.dumps(spec['config'],sort_keys=True))
    log=dest/'training.log'
    with open(log,'a',buffering=1) as f:
        f.write('\n=== LAUNCH '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' ===\n')
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout
        for line in p.stdout:
            print(line,end='',flush=True)
            f.write(line)
        rc=p.wait()
        require(rc==0,f'{rid} exited with returncode {rc}; inspect {log}; restart pipeline --run for safe resume')
    v=verify_job_done(spec)
    require(v is not None,'job returned without audited completion')
    say('I2_JOB_COMPLETE_VERIFIED=PASS',rid,'best_val_AUPRC',round(v['best_val_AUPRC'],6))
    return v

def winner_after_all_tuning():
    entries=[]
    for i in range(1,13):
        s=grid_spec(i); e=verify_job_done(s)
        require(e is not None,f'tune_{i:02d} missing, cannot freeze')
        entries.append(dict(trial=i,config=GRID[i-1],**e))
    selected=max(entries,key=lambda r:(r['best_val_AUPRC'],-r['trial']))
    record=dict(schema='sawgad-tsocial-winner-v1',source_sha256=SHA_SOURCE,split_sha256=SHA_SPLIT,
                winner_trial=selected['trial'],winner_config=selected['config'],
                winner_full_validation_AUPRC=selected['best_val_AUPRC'],
                winner_checkpoint_sha256=selected['checkpoint_sha256'],
                trials=entries,
                selection_rule='maximum full validation AUPRC, tie -> earlier trial',
                test_labels_indexed=False,test_scored=False,methodological_approval_recorded=False,
                final_test_authorized=False)
    p=ROOT_OUT/'frozen_tr40_winner.json'
    if p.exists():require(json.loads(p.read_text())==record,'frozen winner changed; refusing overwrite')
    else:atomic_json(p,record)
    say('I2_TR40_12TRIAL_FROZEN_WINNER_AUDIT=PASS','winner_trial',selected['trial'],
        'validation_AUPRC',round(selected['best_val_AUPRC'],6),'winner_sha',sha(p))
    return record,sha(p)

def run_all():
    preflight()
    ROOT_OUT.mkdir(parents=True,exist_ok=True)
    lockf=open(ROOT_OUT/'.single_writer.lock','a+')
    try:
        fcntl.flock(lockf.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:fail('another pipeline process holds the writer lock')
    ensure_plan()
    say('I2_SAWGAD_PIPELINE_BEGINS=YES','tuning=12','finals=12','heldout_scoring=NO')
    for i in range(1,13):run_one(grid_spec(i))
    winner,locksha=winner_after_all_tuning()
    finals=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            s=final_spec(ratio,seed,winner,locksha)
            finals.append(dict(ratio=ratio,seed=seed,**run_one(s)))
    require(len(finals)==12,'final grid count')
    summary=dict(schema='sawgad-tsocial-12tune-3x4-training-summary-v1',
                 status='ADAPTED_TRAINING_COMPLETE_PRETEST_AUDIT_REQUIRED',
                 winner_trial=winner['winner_trial'],winner_config=winner['winner_config'],
                 winner_sha256=locksha,finals=finals, all_12_finals_completed=True,
                 test_labels_indexed=False,heldout_test_metrics_computed=False,
                 final_heldout_test_authorized=False,supervisor_methodological_approval_recorded=False)
    summ=ROOT_OUT/'training_matrix_completed.json'
    if summ.exists():require(json.loads(summ.read_text())==summary,'training matrix completed file changed')
    else:atomic_json(summ,summary)
    say('I2_SAWGAD_TR40_12TUNE_FINAL_3x4_TRAINING_COMPLETE=PASS')
    say('I2_HELDOUT_TEST_EVALUATIONS=0')
    say('I2_NEXT_GATE=INDEPENDENT_PRETEST_AUDIT_AND_APPROVAL')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    require(sum((a.selftest,a.preflight,a.run))==1,'choose one mode')
    if a.selftest:selftest()
    if a.preflight:preflight()
    if a.run:run_all()
if __name__=='__main__':main()
