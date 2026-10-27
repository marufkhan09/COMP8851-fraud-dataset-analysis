#!/usr/bin/env python3
"""I2 EOGFD T-Social: audited frozen checkpoints; once-only test inference.

This is an adapted unified benchmark, NOT author-identical EOGFD. No test
labels for model/threshold selection. Per-run test claims are created before
accessing test labels or invoking the model. Never retry an incomplete claim.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, socket, statistics, sys, time, traceback, uuid, gc
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
        precision_score, recall_score, accuracy_score, balanced_accuracy_score,
        matthews_corrcoef, brier_score_loss, confusion_matrix)

ROOT=Path('/workspace/comp8851')
S4KIT=Path('/workspace/EOGFD_TSOCIAL_STAGE4_I2_20261023')
S4PATH=S4KIT/'stage4_tsocial.py'
S4RESULTS=ROOT/'results/eogfd/tsocial/stage4_frozen_finals_v3'
OUT=ROOT/'results/eogfd/tsocial/stage5_heldout_test_once_v1'
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
STAGE4_SHA='ee1fd00adb6fc0287e572d894481a9a06861a615f51cd7e56295bb2baf7fe914'
CLAIM_STATUS='EOGFD_TSOCIAL_HELDOUT_SINGLE_INFERENCE_CLAIM_CREATED'
COMPLETE_STATUS='EOGFD_TSOCIAL_HELDOUT_SINGLE_INFERENCE_COMPLETE'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(16*1024*1024),b''):h.update(x)
    return h.hexdigest()

def note(*a):print(*a,flush=True)

def jread(p):
    assert p.is_file(),('STOP_MISSING_FILE',str(p))
    return json.loads(p.read_text())

def jatomic(path,obj,only_new=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if only_new:
        with path.open('x') as f:
            json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        return
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def npatomic(path,arr):
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def load_s4():
    assert socket.gethostname()=='a92ee6fa0813','STOP_WRONG_INSTANCE'
    assert S4PATH.is_file() and sha(S4PATH)==STAGE4_SHA,'STOP_STAGE4_CODE_CHANGED'
    assert torch.__version__=='1.12.1+cu113','STOP_TORCH_VERSION_CHANGED'
    spec=importlib.util.spec_from_file_location('eogfd_stage4_frozen_verified',S4PATH)
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def frozen_audit():
    s4=load_s4()
    s3,ids,frozen,locksha=s4.audit_stage3()
    summary_path=S4RESULTS/'EOGFD_TSocial_Stage4_12FrozenFinalsSummary.json'
    summary=jread(summary_path)
    assert summary['status']==s4.STAGE4_STATUS
    assert summary['completed_finals']==12 and summary['frozen_winner_trial']==9
    assert summary['frozen_config']==s4.EXPECTED_CONFIG
    assert summary['frozen_winner_sha256']==locksha
    assert summary['source_sha256']==s3.SOURCE_SHA and summary['split_sha256']==s3.SPLIT_SHA
    assert summary['upstream_commit']==s3.PIN and summary['mode']==s3.MODE
    assert summary['test_scored'] is False and summary['test_labels_indexed'] is False
    assert summary['test_evaluations']==0
    records=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            d=S4RESULTS/ratio/f'seed_{seed}'
            path=d/'final_training.json'
            r=jread(path)
            assert r['ratio']==ratio and r['seed']==seed
            assert summary['run_record_hashes'][f'{ratio}_seed{seed}']==sha(path)
            assert r['frozen_winner_trial']==9 and r['config']==summary['frozen_config']
            assert r['frozen_winner_sha256']==locksha
            assert r['source_sha256']==s3.SOURCE_SHA and r['split_sha256']==s3.SPLIT_SHA
            assert r['upstream_commit']==s3.PIN and r['mode']==s3.MODE
            assert r['test_scored'] is False and r['test_labels_indexed'] is False and r['test_evaluations']==0
            assert r['validation_only_selection'] is True
            assert r['streamed_fitness_training_and_validation'] is True
            assert r['runtime_select_override'] is True
            assert r['best_val_auprc']>0.0 and r['best_val_auprc']<1.0
            v=r['best_validation_metrics']
            assert v['threshold']==r['frozen_validation_threshold'] and 0<v['threshold']<1
            assert abs(v['val_auprc']-r['best_val_auprc'])<1e-10
            assert not (d/'epoch_recovery.pt').exists(),('STOP_UNFINISHED_STAGE4_FINAL',ratio,seed)
            for fn,key in [('best_validation.pt','checkpoint_sha256'),
                           ('best_validation_scores.npy','validation_scores_sha256'),
                           ('validation_history.json','history_sha256'),
                           ('epoch_times.csv','epoch_times_sha256')]:
                assert sha(d/fn)==r[key],('STOP_STAGE4_INPUT_CHANGED',ratio,seed,fn)
            hist=jread(d/'validation_history.json')
            assert len(hist)==r['epochs_completed'] and len(hist)<=100
            assert abs(max(x['val_auprc'] for x in hist)-r['best_val_auprc'])<1e-10
            assert r['best_epoch'] in [x['epoch'] for x in hist]
            records.append(r)
    assert len(records)==12
    vals={f'{r["ratio"]}_seed{r["seed"]}':dict(
          checkpoint_sha256=r['checkpoint_sha256'],final_training_sha256=sha(S4RESULTS/r['ratio']/f'seed_{r["seed"]}'/'final_training.json'),
          threshold=r['frozen_validation_threshold'],validation_auprc=r['best_val_auprc']) for r in records}
    plan=dict(status='I2_EOGFD_TSOCIAL_STAGE5_PRETEST_INPUT_AUDIT_PASS',
      dataset='T-Social',model='EOGFDHeteroModel',instance='57231045',
      original_author_commit=s3.PIN,stage4_code_sha256=STAGE4_SHA,
      original_source_sha256=s3.SOURCE_SHA,frozen_split_sha256=s3.SPLIT_SHA,
      stage3_frozen_winner_sha256=locksha,stage4_summary_sha256=sha(summary_path),
      winner_trial=9,config=summary['frozen_config'],ratio_seed_runs=vals,
      fixed_eval_seed=s3.EVAL_SEED,method_mode=s3.MODE,
      author_identical_reproduction=False,supervisor_approval_recorded=False,
      test_evaluations_before_stage5=0,test_label_access_during_pretest_audit=False)
    note('I2_TSOCIAL_STAGE5_12_FROZEN_CHECKPOINTS_SHA_AUDIT=PASS')
    note('I2_TSOCIAL_STAGE5_FROZEN_THRESHOLDS=PASS '+str({k:v['threshold'] for k,v in vals.items()}))
    return s3,ids,records,plan

def fixed_test_metrics(y,s,thr):
    y=np.asarray(y,dtype=np.int8).reshape(-1)
    s=np.asarray(s,dtype=np.float64).reshape(-1)
    assert len(y)==len(s) and set(np.unique(y).tolist())=={0,1}
    assert np.isfinite(s).all() and np.min(s)>=0 and np.max(s)<=1
    assert 0<thr<1
    p=(s>thr).astype(np.int8)  # EXACT same strict > as Stage3 threshold selection
    tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    assert tn+fp+fn+tp==len(y)
    return dict(test_auprc=float(average_precision_score(y,s)),
        test_auroc=float(roc_auc_score(y,s)),
        test_macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),
        test_fraud_f1=float(f1_score(y,p,pos_label=1,zero_division=0)),
        test_fraud_precision=float(precision_score(y,p,pos_label=1,zero_division=0)),
        test_fraud_recall=float(recall_score(y,p,pos_label=1,zero_division=0)),
        test_accuracy=float(accuracy_score(y,p)),
        test_balanced_accuracy=float(balanced_accuracy_score(y,p)),
        test_mcc=float(matthews_corrcoef(y,p)),
        test_brier=float(brier_score_loss(y,s)),
        fixed_validation_threshold=float(thr),
        true_positives=tp,true_negatives=tn,false_positives=fp,false_negatives=fn,
        test_examples=len(y),test_positive=int(y.sum()),
        test_threshold_was_not_selected_on_test_labels=True)

def test_unit():
    from sklearn.metrics import f1_score
    y=np.array([0,0,0,1,1,1]);s=np.array([.05,.61,.20,.49,.92,.51])
    a=fixed_test_metrics(y,s,.50)
    assert (a['true_positives'],a['true_negatives'],a['false_positives'],a['false_negatives'])==(2,2,1,1)
    assert abs(a['test_macro_f1']-2/3)<1e-12
    assert a['fixed_validation_threshold']==.5
    note('I2_TSOCIAL_STAGE5_FIXED_THRESHOLD_UNIT_TEST=PASS')

def claim_new(dest,r,plan):
    claim=dest/'test_once_claim.json'
    assert not claim.exists(),'STOP_EXISTING_TEST_CLAIM; DO NOT RE-EVALUATE'
    assert not any(dest.glob('test_scores*')) and not (dest/'test_metrics.json').exists()
    data=dict(status=CLAIM_STATUS,dataset='T-Social',ratio=r['ratio'],seed=r['seed'],
       frozen_checkpoint_sha256=r['checkpoint_sha256'],
       frozen_training_record_sha256=plan['ratio_seed_runs'][f'{r["ratio"]}_seed{r["seed"]}']['final_training_sha256'],
       fixed_validation_threshold=r['frozen_validation_threshold'],
       stage4_summary_sha256=plan['stage4_summary_sha256'],
       stage5_test_inference_authorized_once=True,test_inference_attempted=True,
       heldout_test_labels_not_used_in_training_or_selection=True)
    jatomic(claim,data,only_new=True)
    return claim

def read_complete(dest,r,plan):
    record=jread(dest/'test_metrics.json')
    claim=jread(dest/'test_once_claim.json')
    assert claim['status']==CLAIM_STATUS and claim['test_inference_attempted'] is True
    assert record['status']==COMPLETE_STATUS
    assert record['ratio']==r['ratio'] and record['seed']==r['seed']
    assert record['test_inference_count']==1
    assert record['frozen_checkpoint_sha256']==r['checkpoint_sha256']
    assert record['frozen_validation_threshold']==r['frozen_validation_threshold']
    assert record['stage4_summary_sha256']==plan['stage4_summary_sha256']
    assert record['test_scores_sha256']==sha(dest/'test_scores.npy')
    assert record['test_once_claim_sha256']==sha(dest/'test_once_claim.json')
    assert record['test_labels_not_used_for_selection'] is True
    note(f'I2_TSOCIAL_STAGE5_{r["ratio"]}_SEED{r["seed"]}_PREVIOUS_TEST_REUSED_WITHOUT_INFERENCE=PASS')
    return record

def run_test(s3,ids,records,plan):
    OUT.mkdir(parents=True,exist_ok=True)
    check=OUT/'pretest_audit.json'
    assert check.is_file() and jread(check)==plan,'STOP_PRETEST_AUDIT_MISSING_OR_CHANGED'
    summary=OUT/'EOGFD_TSocial_Stage5_12TestOnceSummary.json'
    assert not summary.exists(),'STOP_STAGE5_FINAL_SUMMARY_EXISTS; DO NOT RE-EVALUATE'
    pending=[];completed=[]
    for r in records:
        dest=OUT/r['ratio']/f'seed_{r["seed"]}'
        if (dest/'test_metrics.json').exists():completed.append(read_complete(dest,r,plan))
        else:
            assert not (dest/'test_once_claim.json').exists(),('STOP_INCOMPLETE_TEST_ONCE_CLAIM_DO_NOT_REINFER',r['ratio'],r['seed'])
            assert not (dest/'test_scores.npy').exists(),('STOP_ORPHAN_TEST_SCORES',r['ratio'],r['seed'])
            pending.append((dest,r))
    if not pending:
        note('I2_TSOCIAL_STAGE5_NO_PENDING_TEST_INFERENCE=PASS')
    else:
        # Reserve FIRST inference before materializing any held-out labels.
        firstdest,firstrec=pending[0]
        firstdest.mkdir(parents=True,exist_ok=True)
        claim_new(firstdest,firstrec,plan)
        s3.install_scoped_optimizer()
        s3.assert_streamed_equivalence_on_small_graphs()
        s3.assert_streamed_forward_backward_equivalence()
        base=s3.dataset_gpu(ids)  # graph features only; train and validation labels only
        from dgl.data.utils import load_graphs
        graphs,_=load_graphs(str(s3.SRC))
        assert len(graphs)==1
        original=graphs[0]
        raw=original.ndata.pop('label')
        test_y=s3.decode_only(raw,ids['test']).cpu().numpy().astype(np.int8,copy=True)
        del raw, original, graphs
        assert len(test_y)==2312426
        test_ix=torch.from_numpy(np.asarray(ids['test'],dtype=np.int64).copy()).long().to('cuda:0')
        note('I2_TSOCIAL_STAGE5_TEST_LABELS_FIRST_ACCESSED_AFTER_WRITE_ONCE_CLAIM=PASS')
        for idx,(dest,r) in enumerate(pending):
            if idx:
                dest.mkdir(parents=True,exist_ok=True)
                claim_new(dest,r,plan)
            path=S4RESULTS/r['ratio']/f'seed_{r["seed"]}'/'best_validation.pt'
            assert sha(path)==r['checkpoint_sha256'],'STOP_CHECKPOINT_CHANGED_AFTER_CLAIM'
            ckpt=torch.load(path,map_location='cpu')
            assert ckpt['source_sha256']==s3.SOURCE_SHA and ckpt['split_sha256']==s3.SPLIT_SHA
            assert ckpt['frozen_winner_sha256']==plan['stage3_frozen_winner_sha256']
            assert ckpt['ratio']==r['ratio'] and ckpt['seed']==r['seed']
            assert ckpt['config']==plan['config'] and ckpt['epoch']==r['best_epoch']
            assert ckpt['test_scored'] is False and ckpt['test_labels_indexed'] is False
            s3.set_seeds(r['seed'])
            model=s3.build_model(plan['config'])
            s3.restore_state(model,ckpt['model_state'])
            model.eval()
            assert model.evolutionary_step.__func__ is s3.streamed_train_eval_evolutionary_step
            torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter()
            with torch.random.fork_rng(devices=[0],enabled=True):
                torch.manual_seed(s3.EVAL_SEED);torch.cuda.manual_seed_all(s3.EVAL_SEED)
                with torch.no_grad():
                    logits,_=model(base['graph'],base['features'])
                    scores=torch.softmax(logits[test_ix],dim=1)[:,1].detach().cpu().numpy().astype(np.float64)
                    del logits
            torch.cuda.synchronize()
            inf_s=time.perf_counter()-t
            assert scores.shape==(len(test_y),) and np.isfinite(scores).all()
            assert (scores>=0).all() and (scores<=1).all()
            peak=float(torch.cuda.max_memory_allocated()/2**30)
            assert peak<43,('STOP_UNSAFE_TEST_GPU_MEMORY',peak)
            npatomic(dest/'test_scores.npy',scores)
            score_sha=sha(dest/'test_scores.npy')
            # Test labels first interact with model outputs HERE, once and only
            # with the fixed validation-selected threshold and frozen weights.
            metrics=fixed_test_metrics(test_y,scores,r['frozen_validation_threshold'])
            record=dict(status=COMPLETE_STATUS,dataset='T-Social',model='EOGFDHeteroModel',
                ratio=r['ratio'],seed=r['seed'],frozen_winner_trial=9,
                method_mode=s3.MODE,author_identical_reproduction=False,
                original_author_commit=s3.PIN,source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,
                stage4_summary_sha256=plan['stage4_summary_sha256'],
                frozen_checkpoint_sha256=r['checkpoint_sha256'],
                frozen_validation_auprc=r['best_val_auprc'],frozen_validation_threshold=r['frozen_validation_threshold'],
                fixed_eval_seed=s3.EVAL_SEED,training_label_only_final_stage=True,
                test_labels_not_used_for_selection=True,test_inference_count=1,
                test_scores_sha256=score_sha,test_once_claim_sha256=sha(dest/'test_once_claim.json'),
                test_inference_seconds=inf_s,peak_cuda_allocated_gib=peak,**metrics)
            jatomic(dest/'test_metrics.json',record,only_new=True)
            completed.append(record)
            note(f'I2_TSOCIAL_STAGE5_{r["ratio"]}_SEED{r["seed"]}_TEST_ONCE=PASS AUPRC={record["test_auprc"]:.6f} AUROC={record["test_auroc"]:.6f} macro_F1={record["test_macro_f1"]:.6f} peak_gib={peak:.3f}')
            del model,ckpt,scores,metrics
            gc.collect();torch.cuda.empty_cache()
    assert len(completed)==12
    completed=sorted(completed,key=lambda r:(RATIOS.index(r['ratio']),SEEDS.index(r['seed'])))
    aggregates=[]
    for ratio in RATIOS:
        group=[r for r in completed if r['ratio']==ratio]
        assert len(group)==3
        ag={'ratio':ratio,'seeds':list(SEEDS)}
        for metric in ('test_auprc','test_auroc','test_macro_f1','test_fraud_f1',
                       'test_fraud_precision','test_fraud_recall','test_accuracy',
                       'test_balanced_accuracy','test_mcc','test_brier'):
            values=[r[metric] for r in group]
            ag[metric+'_mean']=statistics.mean(values)
            ag[metric+'_sample_sd']=statistics.stdev(values)
        aggregates.append(ag)
    final=dict(status='EOGFD_TSOCIAL_STAGE5_12_HELDOUT_TEST_ONCE_COMPLETE',
         dataset='T-Social',model='EOGFDHeteroModel',method_mode=plan['method_mode'],
         author_identical_reproduction=False,supervisor_approval_recorded=False,
         original_author_commit=plan['original_author_commit'],
         source_sha256=plan['original_source_sha256'],split_sha256=plan['frozen_split_sha256'],
         stage4_summary_sha256=plan['stage4_summary_sha256'],
         pretest_audit_sha256=sha(check),final_runs=12,heldout_test_inferences=12,
         single_inference_per_frozen_checkpoint=True,all_thresholds_validation_selected=True,
         records_sha256={f'{r["ratio"]}_seed{r["seed"]}':sha(OUT/r['ratio']/f'seed_{r["seed"]}'/'test_metrics.json') for r in completed},
         heldout_aggregates=aggregates)
    jatomic(summary,final,only_new=True)
    note('I2_TSOCIAL_STAGE5_12_TEST_ONCE=PASS TEST_EVALUATIONS=12 MODEL_SELECTION_FROM_TEST=NO')
    for ag in aggregates:
        note(f'I2_TSOCIAL_STAGE5_SUMMARY_{ag["ratio"]} AUPRC={ag["test_auprc_mean"]:.6f} SD={ag["test_auprc_sample_sd"]:.6f} AUROC={ag["test_auroc_mean"]:.6f} macro_F1={ag["test_macro_f1_mean"]:.6f}')

def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--selftest',action='store_true')
    g.add_argument('--audit-only',action='store_true')
    g.add_argument('--all',action='store_true')
    args=p.parse_args()
    if args.selftest:return test_unit()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','STOP_EXPECTED_GPU_0'
    torch.set_num_threads(min(8,os.cpu_count() or 2))
    s3,ids,records,plan=frozen_audit()
    check=OUT/'pretest_audit.json'
    if args.audit_only:
        assert not (OUT/'EOGFD_TSocial_Stage5_12TestOnceSummary.json').exists()
        if check.exists():assert jread(check)==plan,'STOP_PRETEST_RECORD_CHANGED'
        else:jatomic(check,plan,only_new=True)
        note('I2_TSOCIAL_STAGE5_PRETEST_AUDIT=PASS STAGE4_FINALS=12 TEST_LABELS_INDEXED=NO TEST_SCORED=NO')
        return
    run_test(s3,ids,records,plan)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        note('I2_TSOCIAL_STAGE5_STOP='+repr(e))
        traceback.print_exc()
        sys.exit(2)
