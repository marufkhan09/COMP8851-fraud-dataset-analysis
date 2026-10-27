#!/usr/bin/env python3
"""I1 T-Finance: independent Stage3/4 frozen integrity audit and backup.

Reads only validation labels and stored validation predictions; never test labels.
Never trains, scores, or writes to frozen experiment directories.
"""
from __future__ import annotations
import csv, datetime, hashlib, json, os, socket, sys, tarfile, tempfile
from pathlib import Path
import numpy as np
import torch
from dgl.data.utils import load_graphs
from sklearn.metrics import average_precision_score

ROOT=Path('/workspace/comp8851')
WORK=Path('/workspace')
STAGE3=ROOT/'results/eogfd/tfinance/stage3_tr40_tuning'
STAGE4=ROOT/'results/eogfd/tfinance/stage4_frozen_finals'
KIT3=WORK/'EOGFD_TFINANCE_STAGE3_I1_20261023'
KIT4=WORK/'EOGFD_TFINANCE_STAGE4_I1_20261023'
SRC=ROOT/'assets/tfinance/source/tfinance'
SPLIT=ROOT/'assets/tfinance/tfinance_seed2_nested_splits.npz'
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
SHA_SRC='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SHA_SPLIT='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
AUDIT_DIR=ROOT/'diagnostics/EOGFD_57212396_tfinance/pretest_audit'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def require(cond,why):
    if not cond:raise RuntimeError('STOP_'+str(why))

def verify_file(p,sha):
    require(p.is_file(),f'MISSING_{p}')
    require(digest(p)==sha,f'HASH_MISMATCH_{p}')

def create_new_json(path,dict_):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(dict_,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def check():
    require(socket.gethostname()=='60d39211b1fd','WRONG_I1_HOST')
    import subprocess
    require(subprocess.check_output(['git','-C',str(ROOT/'repos/EOGFD'),'rev-parse','HEAD'],text=True).strip()==PIN,'AUTHOR_COMMIT')
    require(not subprocess.check_output(['git','-C',str(ROOT/'repos/EOGFD'),'status','--porcelain'],text=True).strip(),'AUTHOR_DIRTY')
    # The I1 Stage3 launcher was patched during the *successful* import-path
    # recovery. Its exact original and repaired bytes are independently known.
    # Do NOT rewrite the old source kit or its signed manifest to conceal the
    # discrepancy. Verify the repaired launcher reproduces the original text
    # plus ONLY the two import preflight lines, and attests both hashes.
    import hashlib
    original_launcher_sha='3e37bf27966bc8c586cf98969eaf68750ef130c09c85d86ff273b52189117a21'
    repaired_launcher_sha='5882e569a43058fc47fe7edcb06b6b7dc84dcba904030908fd2de9dff5273c53'
    needle='source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"\n'
    import_fix=(
      'export PYTHONPATH="$ROOT/repos/EOGFD${PYTHONPATH:+:$PYTHONPATH}"\n'
      '"$PY" -c \'from models.eogfd import EOGFDHeteroModel; '
      'print("I1_EOGFD_AUTHOR_MODEL_IMPORT=PASS")\'\n'
    )
    launch=KIT3/'vast_i1_one_command.sh'
    repaired_bytes=launch.read_bytes()
    require(hashlib.sha256(repaired_bytes).hexdigest()==repaired_launcher_sha,'STAGE3_IMPORT_PATCH_HASH')
    repaired_text=repaired_bytes.decode('utf-8')
    require(repaired_text.count(import_fix)==1,'STAGE3_IMPORT_PATCH_COUNT')
    require(repaired_text.count(needle)==1,'STAGE3_IMPORT_SOURCE_COUNT')
    reconstructed=repaired_text.replace(needle+import_fix,needle)
    require(hashlib.sha256(reconstructed.encode('utf-8')).hexdigest()==original_launcher_sha,
       'STAGE3_IMPORT_PATCH_NOT_EXACT')
    for folder in (KIT3,KIT4):
        sums=(folder/'KIT_SHA256SUMS.txt').read_text().splitlines()
        for row in sums:
            sh,name=row.split(maxsplit=1)
            if folder==KIT3 and name.strip()=='vast_i1_one_command.sh':
                require(sh==original_launcher_sha,'STAGE3_ORIGINAL_MANIFEST_CHANGED')
                continue
            verify_file(folder/name.strip(),sh)
    print('I1_STAGE3_LAUNCHER_PATCH_FORENSIC_ATTESTATION=PASS '
          'original='+original_launcher_sha+' deployed='+repaired_launcher_sha,flush=True)
    verify_file(SRC,SHA_SRC);verify_file(SPLIT,SHA_SPLIT)
    with np.load(SPLIT,allow_pickle=False) as z:
        ids={k:np.asarray(z[k],dtype=np.int64).copy() for k in ('TR10','TR20','TR30','TR40','val','test')}
    require({k:len(v) for k,v in ids.items()}=={'TR10':3935,'TR20':7870,'TR30':11806,'TR40':15742,'val':7872,'test':15743},'SPLIT_COUNTS')
    require(not np.intersect1d(ids['TR40'],ids['val']).size,'TRAIN_VAL_OVERLAP')
    require(not np.intersect1d(ids['TR40'],ids['test']).size,'TRAIN_TEST_OVERLAP')
    require(not np.intersect1d(ids['val'],ids['test']).size,'VAL_TEST_OVERLAP')
    # Index ONLY validation labels; neither test labels nor test predictions are accessed.
    graphs,_=load_graphs(str(SRC));require(len(graphs)==1,'GRAPHS_COUNT')
    raw=graphs[0].ndata['label']
    val_ix=torch.from_numpy(ids['val'].copy())
    val_labels=(raw[val_ix].argmax(dim=1) if raw.shape==(39357,2) else raw[val_ix].view(-1)).long().numpy()
    require(len(val_labels)==7872 and int(val_labels.sum())==361,'VAL_LABELS')
    del graphs,raw
    frozen=json.loads((STAGE3/'EOGFD_TFinance_TR40_FrozenWinner.json').read_text())
    require(frozen['status']=='TR40_12_VALIDATION_ONLY_TRIALS_COMPLETE','STAGE3_STATUS')
    require(frozen['winner_trial']==8 and frozen['test_scored'] is False,'STAGE3_WINNER')
    require(frozen['source_sha256']==SHA_SRC and frozen['split_sha256']==SHA_SPLIT,'STAGE3_PROVENANCE')
    require(frozen['venus_methodological_approval_recorded'] is False,'SUPERVISOR_APPROVAL_FALSE_REQUIRED')
    winner_sha=digest(STAGE3/'EOGFD_TFinance_TR40_FrozenWinner.json')
    summary_path=STAGE4/'EOGFD_TFinance_Stage4_12FrozenFinalsSummary.json'
    summary=json.loads(summary_path.read_text())
    require(summary['status']=='EOGFD_TFINANCE_12_FROZEN_VALIDATION_ONLY_FINALS_COMPLETE','STAGE4_STATUS')
    require(summary['completed_finals']==12 and summary['test_scored'] is False and summary['test_evaluations']==0,'STAGE4_TEST_OR_COUNT')
    require(summary['frozen_winner_trial']==8 and summary['frozen_winner_sha256']==winner_sha,'WINNER_SHA_LOCK')
    cfg=frozen['winner_config']
    require(summary['config']==cfg and summary['source_sha256']==SHA_SRC and summary['split_sha256']==SHA_SPLIT,'STAGE4_CONFIG')
    require(summary['upstream_commit']==PIN and summary['venus_methodological_approval_recorded'] is False,'STAGE4_PROVENANCE')
    trials=[]
    for trial in range(1,13):
        root=STAGE3/f'trial_{trial:02d}'
        r=json.loads((root/'trial_result.json').read_text())
        require(r['trial']==trial and r['test_scored'] is False and r['test_labels_indexed'] is False,'STAGE3_TRIAL_FLAG')
        for name,key in [('best_validation.pt','checkpoint_sha256'),('best_validation_scores.npy','validation_scores_sha256'),('validation_history.json','history_sha256'),('epoch_times.csv','epoch_times_sha256')]:
            verify_file(root/name,r[key])
        require(not (root/'epoch_recovery.pt').exists(),'STAGE3_RECOVERY_EXISTS')
        trials.append(r)
    winner=max(trials,key=lambda r:(r['best_val_auprc'],-r['trial']))
    require(winner['trial']==8 and abs(winner['best_val_auprc']-0.883130)<1e-6,'STAGE3_OPTIMAL')
    require(frozen['winner_checkpoint_sha256']==winner['checkpoint_sha256'],'STAGE3_CHECKPOINT_LOCK')
    validated=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            name=f'{ratio}_seed{seed}'
            root=STAGE4/ratio/f'seed_{seed}'
            record=root/'frozen_training.json'
            r=json.loads(record.read_text())
            require(r['dataset']=='T-Finance' and r['ratio']==ratio and r['seed']==seed,'FROZEN_RUN_ID')
            require(r['config']==cfg and r['frozen_winner_sha256']==winner_sha,'FROZEN_CONFIG')
            require(r['source_sha256']==SHA_SRC and r['split_sha256']==SHA_SPLIT and r['upstream_commit']==PIN,'FROZEN_INPUT_SHA')
            require(r['test_scored'] is False and r['test_labels_indexed'] is False and r['test_evaluations']==0,'FROZEN_TEST_FLAG')
            require(r['venus_methodological_approval_recorded'] is False,'FROZEN_APPROVAL_FLAG')
            require(r['train_count']==len(ids[ratio]) and r['val_count']==7872,'FROZEN_TRAIN_COUNTS')
            require(1<=r['best_epoch']<=r['epochs_completed']<=100,'FROZEN_EPOCH')
            for name_f,key in [('best_validation.pt','checkpoint_sha256'),('best_validation_scores.npy','validation_scores_sha256'),('validation_history.json','history_sha256'),('epoch_times.csv','epoch_times_sha256')]:
                verify_file(root/name_f,r[key])
            require(not (root/'epoch_recovery.pt').exists(),'FROZEN_RECOVERY_EXISTS')
            require(digest(record)==summary['run_record_hashes'][name],'SUMMARY_RECORD_HASH')
            scores=np.load(root/'best_validation_scores.npy',allow_pickle=False)
            require(scores.shape==(7872,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(),'FROZEN_SCORES_VALID')
            ap=float(average_precision_score(val_labels,scores))
            require(abs(ap-r['best_val_auprc'])<1e-8,'FROZEN_VALIDATION_AP_REPLAY')
            history=json.loads((root/'validation_history.json').read_text())
            require(len(history)==r['epochs_completed'],'FROZEN_HISTORY_LEN')
            row=history[r['best_epoch']-1]
            require(row['epoch']==r['best_epoch'] and abs(row['val_auprc']-ap)<1e-8,'FROZEN_BEST_EPOCH')
            chk=torch.load(root/'best_validation.pt',map_location='cpu')
            require(chk['epoch']==r['best_epoch'] and chk['seed']==seed and chk['ratio']==ratio,'FROZEN_CHECKPOINT_ID')
            require(chk['config']==cfg and chk['frozen_winner_sha256']==winner_sha,'FROZEN_CHECKPOINT_LOCK')
            require(chk['test_scored'] is False and 'theta_last' not in chk['model_state'],'FROZEN_MODEL_STATE')
            require(chk['validation']==r['best_validation_metrics'],'FROZEN_VAL_METRICS_LOCK')
            # Independent validation replay of threshold uses exact same fixed grid/tie-breaks.
            ts=np.arange(1,100,dtype=np.float64)/100
            positives=val_labels.astype(bool)[:,None];pred=scores[:,None]>ts[None,:]
            tp=(pred&positives).sum(axis=0);tn=((~pred)&(~positives)).sum(axis=0)
            fp=(pred&(~positives)).sum(axis=0);fn=((~pred)&positives).sum(axis=0)
            fpos=2*tp/np.maximum(2*tp+fp+fn,1)
            fneg=2*tn/np.maximum(2*tn+fp+fn,1)
            fmean=(fpos+fneg)/2;recall=tp/np.maximum(tp+fn,1)
            ti=max(range(99),key=lambda j:(round(float(fmean[j]),12),round(float(recall[j]),12),-abs(float(ts[j])-.5),-float(ts[j])))
            require(abs(float(ts[ti])-r['frozen_validation_threshold'])<1e-12,'FROZEN_THRESHOLD_REPLAY')
            require(abs(float(fmean[ti])-r['best_validation_metrics']['val_macro_f1'])<1e-8,'FROZEN_F1_REPLAY')
            print('I1_TFINANCE_PRETEST_FROZEN_'+name+'=PASS val_AP='+f'{ap:.6f}'+' threshold='+f'{float(ts[ti]):.2f}',flush=True)
            validated.append(dict(ratio=ratio,seed=seed,validation_auprc=ap,threshold=float(ts[ti]),record_sha256=digest(record)))
    require(len(validated)==12,'FINAL_COUNT')
    out=dict(status='ALL_12_FROZEN_PRETEST_AUDITS_PASS',source_sha256=SHA_SRC,split_sha256=SHA_SPLIT,
         upstream_commit=PIN,mode='UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
         stage3_runtime_launcher_sha256='5882e569a43058fc47fe7edcb06b6b7dc84dcba904030908fd2de9dff5273c53',
         stage3_launch_import_patch_exactly_attested=True,
         frozen_winner_sha256=winner_sha,stage4_summary_sha256=digest(summary_path),
         training_seeds=list(SEEDS),ratios=list(RATIOS),completed_finals=12,
         test_labels_indexed=False,test_scored=False,test_evaluations=0,
         venus_methodological_approval_recorded=False,runs=validated)
    return out

def backup(audit):
    AUDIT_DIR.mkdir(parents=True,exist_ok=True)
    file=AUDIT_DIR/'EOGFD_TFinance_PreTest_Audit.json'
    if file.exists():
        require(json.loads(file.read_text())==audit,'EXISTING_AUDIT_CONFLICT')
    else:create_new_json(file,audit)
    # Private evidence backup: scripts, records, checkpoints, logs, diagnostics.
    # DOES NOT include raw dataset or split; Mac already retains canonical assets.
    sources=[STAGE3,STAGE4,AUDIT_DIR,
             WORK/'EOGFD_TFINANCE_PRETEST_AUDIT_I1_V2_20261023/i1_pretest_audit_backup_v2.py',
             WORK/'EOGFD_TFINANCE_PRETEST_AUDIT_I1_V2_20261023/vast_i1_audit_backup_v2.sh',
             KIT3/'stage3_tfinance.py',KIT3/'vast_i1_one_command.sh',
             KIT4/'stage4_tfinance.py',KIT4/'vast_i1_stage4.sh',
             ROOT/'diagnostics/EOGFD_57212396_tfinance/train_only_one_update.json']
    for path in sources:require(path.exists(),f'BACKUP_SOURCE_MISSING_{path}')
    for logname in ('EOGFD_57212396_TFinance_STAGE3_TR40.log','EOGFD_57212396_TFinance_STAGE4.log'):
        p=ROOT/'logs'/logname
        if p.exists():sources.append(p)
    items={}
    for root in sources:
        if root.is_dir():
            for p in root.rglob('*'):
                if p.is_file():
                    require(not p.is_symlink(),'BACKUP_SYMLINK')
                    items[p.relative_to(WORK).as_posix()]=p
        else:items[root.relative_to(WORK).as_posix()]=root
    require(len(items)>90,'TOO_FEW_BACKUP_ARTIFACTS')
    for name in items:
        require(not name.endswith('.writing'),'BACKUP_PARTIAL_FILE')
        require('test_scores' not in name and 'test_metrics' not in name,'BACKUP_PREMATURE_TEST_DATA')
    out=WORK/'EOGFD_TFINANCE_I1_PRETEST_BACKUP_20261023.tar.gz'
    checksum=Path(str(out)+'.sha256')
    require(not out.exists() and not checksum.exists(),'BACKUP_ALREADY_EXISTS_DO_NOT_OVERWRITE')
    manifest={name:{'sha256':digest(p),'bytes':p.stat().st_size} for name,p in sorted(items.items())}
    manifest_bytes=(json.dumps({'files':manifest,'audit_sha256':digest(file)},sort_keys=True,indent=2)+'\n').encode()
    tmp=Path(str(out)+'.writing')
    require(not tmp.exists(),'PARTIAL_BACKUP_PRESENT')
    with tarfile.open(tmp,'w:gz') as tar:
        import io
        info=tarfile.TarInfo('BACKUP_FILE_MANIFEST.json');info.size=len(manifest_bytes)
        tar.addfile(info,io.BytesIO(manifest_bytes))
        for name,p in sorted(items.items()):tar.add(p,arcname=name,recursive=False)
    with tarfile.open(tmp,'r:gz') as tar:
        members={m.name:m for m in tar.getmembers() if m.isfile()}
        require(set(members)==set(manifest)|{'BACKUP_FILE_MANIFEST.json'},'BACKUP_MEMBER_SET')
        for name,info in manifest.items():
            h=hashlib.sha256()
            f=tar.extractfile(members[name]);require(f is not None,'BACKUP_FILE_READ')
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            require(h.hexdigest()==info['sha256'],'BACKUP_IN_ARCHIVE_HASH')
    os.replace(tmp,out)
    with checksum.open('x') as f:f.write(digest(out)+'  '+out.name+'\n')
    print('I1_TFINANCE_PRETEST_BACKUP_VERIFIED=PASS archive='+str(out),flush=True)
    print('I1_TFINANCE_BACKUP_SHA256='+digest(out),flush=True)
    print('I1_TFINANCE_BACKUP_ITEMS='+str(len(items)),flush=True)
    print('I1_TFINANCE_BACKUP_INCLUDES_RAW_GRAPH=NO',flush=True)
    print('I1_TFINANCE_BACKUP_INCLUDES_FROZEN_SPLIT=NO',flush=True)
    return out

def main():
    if len(sys.argv)!=2 or sys.argv[1]!='--audit-and-backup':raise SystemExit('Usage: --audit-and-backup')
    audit=check()
    print('I1_TFINANCE_ALL_12_FROZEN_PRETEST_AUDIT=PASS TEST_SCORED=NO',flush=True)
    backup(audit)
    print('I1_TFINANCE_READY_FOR_SEPARATE_STAGE5_ONLY_AFTER_OFF_INSTANCE_BACKUP=YES',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print('I1_TFINANCE_PRETEST_AUDIT_STOP='+repr(e),flush=True)
        import traceback;traceback.print_exc();sys.exit(2)
