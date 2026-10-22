#!/usr/bin/env python3
"""EOGFD x FDCompCN/Elliptic: audit all frozen states then once-only test.

UNIFIED_CONDITIONAL_ADAPTED, Venus approval NOT recorded. Exact pinned author
model; fixed split ID; unchanged validation threshold; no training/tuning.
Write durable, exclusive claims BEFORE test forward; never forward again if
claim exists, including interruptions. Offline recovery from saved scores only.
"""
from __future__ import annotations
import argparse, csv, datetime, gc, hashlib, importlib.util, json, math, os, socket, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import dgl
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix)

BASE=Path('/workspace/comp8851')
STAGE3=Path('/workspace/EOGFD_NEWI_STAGE3')
STAGE4=Path('/workspace/EOGFD_NEWI_STAGE4')
REPO=BASE/'repos/EOGFD'
HOST='60d39211b1fd'
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
DATASETS=('FDCompCN','Elliptic')
EVAL_SEED=100921
AUDIT=BASE/'diagnostics/eogfd_newi_57212396'
OUTROOT=BASE/'results/eogfd'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json_new_or_same(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text())==data, ('EXISTING_JSON_CONFLICT',str(path))
        return
    with path.open('x') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def write_json_atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('x') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def save_npy_once(path,values):
    assert not path.exists(), ('TEST_SCORES_ALREADY_EXIST',str(path))
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('xb') as f:
        np.save(f,values,allow_pickle=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def get_stages():
    assert socket.gethostname()==HOST, ('WRONG_INSTANCE_HOST',socket.gethostname())
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    st3hash=json.loads((STAGE3/'KIT_SHA256_MANIFEST.json').read_text())
    st4hash=json.loads((STAGE4/'KIT_SHA256_MANIFEST.json').read_text())
    for folder,manifest in ((STAGE3,st3hash),(STAGE4,st4hash)):
        for file,expected in manifest.items():
            assert sha(folder/file)==expected, ('STAGE_SCRIPT_CHANGED',str(folder/file))
    spec=importlib.util.spec_from_file_location('eogfd_stage4_pinned_reader',STAGE4/'stage4_runner.py')
    st4=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st4)
    st4.load_stage3_verified()
    assert st4.PIN==PIN and st4.s3.EVAL_SEED==EVAL_SEED
    return st4


def offline_val_metrics(y, scores):
    from sklearn.metrics import average_precision_score,roc_auc_score
    s=np.asarray(scores,dtype=np.float64).reshape(-1)
    y=np.asarray(y,dtype=np.int64).reshape(-1)
    assert len(s)==len(y) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
    assert set(np.unique(y).tolist())=={0,1}
    thrs=np.arange(1,100,dtype=np.float64)/100
    pred=s[:,None]>thrs[None,:]
    positives=y.astype(bool)[:,None]
    tp=(pred&positives).sum(axis=0)
    tn=((~pred)&(~positives)).sum(axis=0)
    fp=(pred&(~positives)).sum(axis=0)
    fn=((~pred)&positives).sum(axis=0)
    fpos=2*tp/np.maximum(2*tp+fp+fn,1)
    fneg=2*tn/np.maximum(2*tn+fp+fn,1)
    macro=(fpos+fneg)/2
    recall=tp/np.maximum(tp+fn,1)
    i=max(range(99),key=lambda j:(round(float(macro[j]),12),round(float(recall[j]),12),-abs(float(thrs[j])-.5),-float(thrs[j])))
    return {'ap':float(average_precision_score(y,s)), 'auc':float(roc_auc_score(y,s)),
            'macro_f1':float(macro[i]),'threshold':float(thrs[i]),'recall':float(recall[i])}


def outfolder(dataset):
    return OUTROOT/dataset.lower()/'stage5_heldout_test_once'


def audit_dataset(dataset):
    st4=get_stages()
    mod=st4.s3.import_adapter()
    ids,pilot=st4.s3.validate_stage2(dataset,mod)
    winner,winner_sha=st4.verify_winner(dataset,mod,ids)
    b=st4.s3.fd_data(mod,ids) if dataset=='FDCompCN' else st4.s3.ell_data(mod,ids)
    val_y=b['val_y']
    summroot=OUTROOT/dataset.lower()/'stage4_frozen_finals'
    summaryfile=summroot/f'EOGFD_{dataset}_Stage4_12FrozenFinalsSummary.json'
    summary=json.loads(summaryfile.read_text())
    assert summary['status']=='EOGFD_12_CONDITIONAL_FROZEN_TRAININGS_COMPLETE'
    assert summary['completed_finals']==12 and summary['dataset']==dataset
    assert summary['ratios']==list(RATIOS) and summary['seeds']==list(SEEDS)
    assert summary['frozen_config']==winner['winner_config']
    assert summary['frozen_winner_sha256']==winner_sha
    assert summary['source_sha256']==winner['source_sha256']
    assert summary['split_sha256']==winner['split_sha256']
    assert summary['upstream_commit']==PIN
    assert summary['venus_approval_recorded'] is False and summary['test_scored'] is False
    assert summary['test_evaluations']==0 and summary['mode']=='UNIFIED_CONDITIONAL_ADAPTED'
    manifest=json.loads((summroot/'stage4_frozen_training_manifest.json').read_text())
    assert manifest['winner_sha256']==winner_sha
    assert manifest['frozen_config']==winner['winner_config']
    assert manifest['selection_metric']=='validation_AUPRC'
    assert manifest['threshold_metric']=='validation_macroF1'
    assert manifest['test_scored'] is False and manifest['venus_approval_recorded'] is False
    assert manifest['upstream_commit']==PIN
    if dataset=='Elliptic':
        assert manifest['elliptic_chronological_cutoffs']=={'TR40':20,'TR30':12,'TR20':7,'TR10':3}
    records=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            rd=summroot/ratio/f'seed_{seed}'
            rp=rd/'frozen_training.json'
            cp=rd/'best_validation.pt'
            vp=rd/'best_validation_scores.npy'
            hp=rd/'validation_history.json'
            ep=rd/'epoch_times.csv'
            assert all(p.is_file() for p in (rp,cp,vp,hp,ep)),('MISSING_FROZEN_ARTIFACT',dataset,ratio,seed)
            assert not (rd/'recovery_progress.pt').exists(),('INCOMPLETE_FROZEN_RUN',ratio,seed)
            r=json.loads(rp.read_text())
            assert r['dataset']==dataset and r['ratio']==ratio and r['seed']==seed
            assert r['mode']=='UNIFIED_CONDITIONAL_ADAPTED' and r['venus_approval_recorded'] is False
            assert r['author_identical_reproduction'] is False
            assert r['model']=='EOGFDHeteroModel' and r['upstream_commit']==PIN
            assert r['split_seed']==2 and r['train_count']==len(ids[ratio])
            assert r['validation_count']==len(ids['val'])
            assert r['config']==winner['winner_config'] and r['frozen_winner_sha256']==winner_sha
            assert r['source_sha256']==winner['source_sha256'] and r['split_sha256']==winner['split_sha256']
            assert r['test_labels_accessed'] is False and r['test_scored'] is False
            assert r['heldout_test_evaluations']==0 and r['test_pending'] is True
            assert r['validation_evolution_rng_seed']==EVAL_SEED
            assert 1<=r['best_epoch']<=r['epochs_completed']<=100
            assert r['max_epochs']==100 and r['patience']==20
            for p,key in ((cp,'checkpoint_sha256'),(vp,'validation_scores_sha256'),(hp,'history_sha256'),(ep,'epoch_times_sha256')):
                assert sha(p)==r[key], ('FROZEN_FILE_SHA_MISMATCH',dataset,ratio,seed,str(p))
            scores=np.asarray(np.load(vp,allow_pickle=False),dtype=np.float64)
            v=offline_val_metrics(val_y,scores)
            assert abs(v['ap']-r['best_val_auprc'])<1e-8, ('VAL_AP_REPLAY_MISMATCH',ratio,seed)
            assert abs(v['auc']-r['best_val_auroc'])<1e-8
            assert abs(v['macro_f1']-r['best_val_macro_f1'])<1e-8
            assert abs(v['threshold']-r['frozen_validation_threshold'])<1e-12
            assert abs(v['recall']-r['fraud_recall_validation'])<1e-8
            checkpoint=torch.load(cp,map_location='cpu')
            assert checkpoint['dataset']==dataset and checkpoint['ratio']==ratio and checkpoint['seed']==seed
            assert checkpoint['epoch']==r['best_epoch']
            assert checkpoint['config']==winner['winner_config']
            assert checkpoint['frozen_winner_sha256']==winner_sha
            assert checkpoint['source_sha256']==winner['source_sha256']
            assert checkpoint['split_sha256']==winner['split_sha256']
            assert checkpoint['upstream_commit']==PIN
            assert checkpoint['test_scored'] is False and checkpoint['theta_last_buffer_excluded'] is True
            assert 'theta_last' not in checkpoint['model_state']
            assert 'input_encoder.0.weight' in checkpoint['model_state']
            assert tuple(checkpoint['model_state']['input_encoder.0.weight'].shape)==(64,mod.SPECS[dataset]['d'])
            expected_metrics={
                'val_auprc':r['best_val_auprc'],
                'val_auroc':r['best_val_auroc'],
                'val_macro_f1':r['best_val_macro_f1'],
                'val_fraud_recall':r['fraud_recall_validation'],
                'threshold':r['frozen_validation_threshold']
            }
            assert set(checkpoint['validation'])==set(expected_metrics)
            for metric_key, value in expected_metrics.items():
                assert abs(float(checkpoint['validation'][metric_key])-float(value))<1e-8, ('FROZEN_CHECKPOINT_VAL_CONFLICT',ratio,seed,metric_key)
            if dataset=='Elliptic':
                assert r['graph']['cutoff']==st4.CUTS[ratio]
            else:
                assert r['graph']['cutoff'] is None and r['graph']['train_edges']==30752
            records.append({'ratio':ratio,'seed':seed,'checkpoint_sha256':r['checkpoint_sha256'],
               'record_sha256':sha(rp),'validation_scores_sha256':r['validation_scores_sha256'],
               'validation_AUPRC':r['best_val_auprc'], 'threshold':r['frozen_validation_threshold']})
            print(f'EOGFD_{dataset}_{ratio}_seed{seed}_FROZEN_PRETEST_AUDIT=PASS',flush=True)
            del checkpoint,scores
    assert len(records)==12
    for r in records:
        rd=outfolder(dataset)/r['ratio']/f'seed_{r["seed"]}'
        claim=rd/'test_once_claim.json'
        score=rd/'test_scores.npy'
        result=rd/'test_metrics.json'
        if score.exists():
            assert claim.is_file(),('UNCLAIMED_TEST_SCORE',str(score))
        if result.exists():
            assert score.exists() and claim.is_file(),('RESULT_WITHOUT_CLAIM_AND_SCORES',str(result))
        if claim.exists():
            c=json.loads(claim.read_text())
            assert c['checkpoint_sha256']==r['checkpoint_sha256']
            assert c['ratio']==r['ratio'] and c['seed']==r['seed']
            assert c['dataset']==dataset
            assert c['claimed_attempt_count']==1
            if not score.exists():
                raise RuntimeError(f'STOP_UNRECOVERABLE_TEST_CLAIM_WITHOUT_SCORES: {dataset}/{r["ratio"]}/seed{r["seed"]}. No repeated inference.')
    audit={'status':'ALL12_FROZEN_PRETEST_AUDITS_PASS', 'dataset':dataset,
           'model':'EOGFDHeteroModel','mode':'UNIFIED_CONDITIONAL_ADAPTED',
           'venus_approval_recorded':False,'source_sha256':winner['source_sha256'],
           'split_sha256':winner['split_sha256'],'upstream_commit':PIN,
           'stage4_summary_sha256':sha(summaryfile),'frozen_winner_sha256':winner_sha,
           'evaluation_rng_seed':EVAL_SEED,'test_labels_opened_during_audit':False,
           'ratios':list(RATIOS),'seeds':list(SEEDS),'runs':records}
    audit_path=outfolder(dataset)/f'EOGFD_{dataset}_Stage5_All12_PretestAudit.json'
    write_json_new_or_same(audit_path,audit)
    print(f'EOGFD_{dataset}_STAGE5_ALL12_FROZEN_PRETEST_AUDIT=PASS TEST_LABELS_NOT_OPENED=YES',flush=True)
    del b
    gc.collect();torch.cuda.empty_cache()
    return audit


def metric(y, s, threshold):
    assert s.ndim==1 and len(s)==len(y) and np.isfinite(s).all()
    assert np.all((s>=0)&(s<=1))
    assert 0.0<threshold<1.0 and set(np.unique(y).tolist())=={0,1}
    pred=np.asarray(s>threshold,dtype=np.int64)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    recall=tp/max(tp+fn,1); specificity=tn/max(tn+fp,1)
    return {'auprc':float(average_precision_score(y,s)),
            'auroc':float(roc_auc_score(y,s)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'fraud_f1':float(f1_score(y,pred,average='binary',pos_label=1,zero_division=0)),
            'fraud_precision':float(precision_score(y,pred,pos_label=1,zero_division=0)),
            'fraud_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),
            'g_mean':float(math.sqrt(recall*specificity)),
            'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'mcc':float(matthews_corrcoef(y,pred)),
            'brier_score':float(brier_score_loss(y,s)),
            'tp':tp,'tn':tn,'fp':fp,'fn':fn}


def test_view(dataset,ids,st4):
    from dgl.data.utils import load_graphs
    if dataset=='FDCompCN':
        source,_=load_graphs(str(BASE/'assets/fdcompcn/source/comp.dgl'))
        raw=source[0]
        assert raw.num_nodes()==5317 and raw.num_edges()==30752
        y=raw.ndata['label'][torch.from_numpy(ids['test'])].long().view(-1).numpy().astype(np.int64,copy=True)
        mod=st4.s3.import_adapter()
        g,x,_,_,metadata=mod.fdcompcn_graph(ids)
        assert len(g.canonical_etypes)==4 and g.num_edges()==30752
        assert not list(g.ndata.keys()) and not list(g.edata.keys())
        test_ix=np.asarray(ids['test'],dtype=np.int64).copy()
        assert len(y)==2127 and len(np.unique(test_ix))==2127
        return g,x,test_ix,y,{'graph_semantics':'Original FDCompCN 4 directed relations; transductive unlabeled full static graph',
                             'visible_nodes':5317,'visible_edges':30752,'temporal_cutoff':None,
                             'test_nodes':2127,'test_graph_labels_stripped':True}
    folder=BASE/'assets/elliptic/source'
    types={0:np.int64,1:np.int16};types.update({i:np.float32 for i in range(2,167)})
    frame=pd.read_csv(folder/'elliptic_txs_features.csv',header=None,dtype=types)
    assert frame.shape==(203769,167)
    txids=frame.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
    times=frame.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
    feats=frame.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
    del frame
    assert len(np.unique(txids))==203769 and times.min()==1 and times.max()==49
    assert np.all((times[ids['test']]>=32)&(times[ids['test']]<=49))
    ed=pd.read_csv(folder/'elliptic_txs_edgelist.csv',dtype=np.int64)
    assert len(ed)==234355 and tuple(ed.columns)==('txId1','txId2')
    idx=pd.Index(txids)
    src=idx.get_indexer(ed.txId1).astype(np.int64)
    dst=idx.get_indexer(ed.txId2).astype(np.int64)
    del ed
    assert np.all(src>=0) and np.all(dst>=0) and np.array_equal(times[src],times[dst])
    assert np.all(times[src]<=49) and np.all(times[dst]<=49)
    # No synthetic, reverse, cross-period, or extra edges; original graph only.
    g=dgl.heterograph({('tx','original_transaction_edge','tx'):(torch.from_numpy(src.copy()),torch.from_numpy(dst.copy()))},num_nodes_dict={'tx':203769})
    assert g.num_nodes()==203769 and g.num_edges()==234355 and len(g.canonical_etypes)==1
    # One full-period test graph; all original edges are *within* one timestep,
    # so no future-timestep edges/messages can reach an earlier test timestep.
    allowed=set(txids[ids['test']].tolist()); selected={}
    with (folder/'elliptic_txs_classes.csv').open(newline='') as f:
        rows=csv.DictReader(f)
        assert set(rows.fieldnames)=={'txId','class'}
        for row in rows:
            tx=int(row['txId'])
            if tx in allowed:
                assert tx not in selected and row['class'] in ('1','2')
                selected[tx]=int(row['class']=='1')
    assert len(selected)==18949
    y=np.asarray([selected[int(tx)] for tx in txids[ids['test']]],dtype=np.int64)
    return (g,torch.from_numpy(feats.copy()),np.asarray(ids['test'],dtype=np.int64).copy(),y,
           {'graph_semantics':'Single original directed within-timestep transaction relation; all timesteps <=49; no cross-time edges',
            'visible_nodes':203769,'visible_edges':234355,'temporal_cutoff':49,
            'test_nodes':18949,'test_graph_labels_stripped':True,
            'test_snapshot_includes_other_unlabeled_nodes_in_test_period':True,
            'supervisor_eligibility_unapproved':True})


def enforce_pretest_audits():
    for ds in DATASETS:
        p=outfolder(ds)/f'EOGFD_{ds}_Stage5_All12_PretestAudit.json'
        assert p.is_file(),('BOTH_DATASET_AUDITS_REQUIRED_BEFORE_ANY_TEST_LABEL_ACCESS',ds)
        a=json.loads(p.read_text())
        assert a['status']=='ALL12_FROZEN_PRETEST_AUDITS_PASS'
        assert a['test_labels_opened_during_audit'] is False
        assert len(a['runs'])==12
    print('EOGFD_BOTH_DATASETS_ALL24_PRETEST_AUDITS_EXIST=PASS',flush=True)


def score_dataset(dataset):
    assert os.environ.get('EOGFD_CONDITIONAL_TEST_ONCE')=='1', 'Explicit conditional one-time test consent not present'
    audit=audit_dataset(dataset)
    enforce_pretest_audits()
    st4=get_stages()
    mod=st4.s3.import_adapter()
    ids,_=st4.s3.validate_stage2(dataset,mod)
    winner,winner_sha=st4.verify_winner(dataset,mod,ids)
    assert winner_sha==audit['frozen_winner_sha256']
    # Everything above is train/validation-only. Test source labels open NOW.
    g,x,testix,y,graphmeta=test_view(dataset,ids,st4)
    assert len(y)==len(testix)==mod.SPECS[dataset]['train']['test']
    assert 0<int(y.sum())<len(y)
    assert len(np.unique(testix))==len(testix)
    print(f'EOGFD_{dataset}_STAGE5_HELDOUT_LABELS_OPENED_AFTER_24_FROZEN_AUDITS=PASS test_count={len(y)} test_positives={int(y.sum())}',flush=True)
    out=outfolder(dataset)
    source=winner['source_sha256'];split=winner['split_sha256']
    config=winner['winner_config']
    g=g.to('cuda:0');x=x.to('cuda:0')
    testtensor=torch.as_tensor(testix,dtype=torch.long,device='cuda:0')
    rows=[]
    for run in audit['runs']:
        ratio=run['ratio'];seed=run['seed']
        cp=OUTROOT/dataset.lower()/'stage4_frozen_finals'/ratio/f'seed_{seed}'/'best_validation.pt'
        dest=out/ratio/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        claimfile=dest/'test_once_claim.json';scorefile=dest/'test_scores.npy';resultfile=dest/'test_metrics.json'
        assert sha(cp)==run['checkpoint_sha256'],('CHECKPOINT_CHANGED_AFTER_AUDIT',ratio,seed)
        threshold=float(run['threshold'])
        if claimfile.exists():
            claim=json.loads(claimfile.read_text())
            assert claim['checkpoint_sha256']==run['checkpoint_sha256']
            assert claim['claimed_attempt_count']==1
            assert scorefile.is_file(),('STOP_NO_REINFERENCE_CLAIMED_WITHOUT_SCORES',ratio,seed)
            scores=np.asarray(np.load(scorefile,allow_pickle=False),dtype=np.float64)
            saved_sha=sha(scorefile)
            if claim['test_scores_sha256'] is None:
                assert claim['status']=='TEST_INFERENCE_CLAIMED_ONCE',('BAD_TEST_CLAIM_STATE',ratio,seed)
                claim['status']='TEST_SCORES_SAVED_ONCE'
                claim['test_scores_sha256']=saved_sha
                temp=claimfile.with_name(claimfile.name+'.writing')
                if temp.exists():
                    pending=json.loads(temp.read_text())
                    assert pending['test_scores_sha256']==saved_sha and pending['status']=='TEST_SCORES_SAVED_ONCE'
                    os.replace(temp,claimfile)
                else:
                    write_json_atomic(claimfile,claim)
            assert saved_sha==claim['test_scores_sha256'],('SAVED_TEST_SCORE_CHANGED',ratio,seed)
            print(f'EOGFD_{dataset}_{ratio}_seed{seed}_STAGE5_REUSE_SAVED_SCORES_ONLY=PASS NO_MODEL_FORWARD',flush=True)
        else:
            assert not scorefile.exists() and not resultfile.exists(),('UNCLAIMED_TEST_ARTIFACT',ratio,seed)
            model=st4.s3.make_model(config,x.shape[1])
            checkpoint=torch.load(cp,map_location='cpu')
            assert checkpoint['dataset']==dataset and checkpoint['ratio']==ratio and checkpoint['seed']==seed
            assert checkpoint['frozen_winner_sha256']==winner_sha
            st4.s3.load_model_state(model,checkpoint['model_state'])
            model.eval()
            del checkpoint
            claim={'status':'TEST_INFERENCE_CLAIMED_ONCE','model':'EOGFDHeteroModel','dataset':dataset,
                   'ratio':ratio,'seed':seed,'claimed_attempt_count':1,'checkpoint_sha256':run['checkpoint_sha256'],
                   'frozen_winner_sha256':winner_sha,'source_sha256':source,'split_sha256':split,
                   'fixed_threshold':threshold,'evaluation_rng_seed':EVAL_SEED,
                   'classification':'UNIFIED_CONDITIONAL_ADAPTED_NOT_VENUS_APPROVED',
                   'claim_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'test_scores_sha256':None}
            write_json_new_or_same(claimfile,claim)  # EXCLUSIVE, durable BEFORE held-out inference.
            print(f'EOGFD_{dataset}_{ratio}_seed{seed}_EXCLUSIVE_TEST_ONCE_CLAIM=PASS',flush=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();tic=time.perf_counter()
            with torch.random.fork_rng(devices=[0],enabled=True):
                torch.manual_seed(EVAL_SEED);torch.cuda.manual_seed_all(EVAL_SEED)
                with torch.no_grad():
                    logits,_=model(g,x)
                    scores=torch.softmax(logits[testtensor],dim=1)[:,1].detach().cpu().numpy().astype(np.float64)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-tic
            assert len(scores)==len(y) and np.isfinite(scores).all()
            save_npy_once(scorefile,scores)
            claim['status']='TEST_SCORES_SAVED_ONCE'
            claim['test_scores_sha256']=sha(scorefile)
            claim['inference_seconds']=float(elapsed)
            claim['peak_gpu_allocated_gib']=float(torch.cuda.max_memory_allocated()/2**30)
            # Atomic update after durable score save; interrupted update can be reconstructed from scores.
            write_json_atomic(claimfile,claim)
            del model,logits
            gc.collect();torch.cuda.empty_cache()
        metrics=metric(y,scores,threshold)
        record={'status':'FROZEN_HELDOUT_TEST_SCORED_ONCE','dataset':dataset,'model':'EOGFDHeteroModel',
                'ratio':ratio,'seed':seed,'model_config':config,'source_sha256':source,'split_sha256':split,
                'frozen_winner_sha256':winner_sha,'checkpoint_sha256':run['checkpoint_sha256'],
                'validation_AUPRC':run['validation_AUPRC'],'frozen_validation_threshold':threshold,
                'test_scores_sha256':sha(scorefile),'test_count':len(y),'test_positives':int(y.sum()),
                'test_inference_claims':1,'test_scored_once':True,'test_threshold_retuned':False,
                'training_during_test':False,'test_graph':graphmeta,
                'mode':'UNIFIED_CONDITIONAL_ADAPTED','venus_approval_recorded':False,
                'author_identical_reproduction':False,'upstream_commit':PIN,**metrics}
        write_json_new_or_same(resultfile,record)
        rows.append(record)
        print(f'EOGFD_{dataset}_{ratio}_seed{seed}_ONE_TIME_HELDOUT_TEST=PASS AUPRC={metrics["auprc"]:.6f} AUROC={metrics["auroc"]:.6f} macroF1={metrics["macro_f1"]:.6f}',flush=True)
        del scores
    assert len(rows)==12
    metrics_names=('auprc','auroc','macro_f1','fraud_f1','g_mean','fraud_precision','fraud_recall','accuracy','balanced_accuracy','mcc','brier_score')
    stats=[]
    for ratio in RATIOS:
        subset=[r for r in rows if r['ratio']==ratio]
        assert len(subset)==3 and [r['seed'] for r in subset]==list(SEEDS)
        vals={'ratio':ratio,'seeds':list(SEEDS)}
        for key in metrics_names:
            arr=np.asarray([r[key] for r in subset],dtype=np.float64)
            vals[key+'_mean']=float(np.mean(arr))
            vals[key+'_sample_sd']=float(np.std(arr,ddof=1))
        stats.append(vals)
        print(f'EOGFD_{dataset}_{ratio}_HELDOUT_TEST_AUPRC_MEAN={vals["auprc_mean"]:.6f} SAMPLE_SD={vals["auprc_sample_sd"]:.6f}',flush=True)
    summary={'status':'ALL12_CONDITIONAL_ONE_TIME_HELDOUT_TESTS_COMPLETE','dataset':dataset,
             'model':'EOGFDHeteroModel','mode':'UNIFIED_CONDITIONAL_ADAPTED',
             'venus_approval_recorded':False,'author_identical_reproduction':False,
             'source_sha256':source,'split_sha256':split,'upstream_commit':PIN,
             'frozen_winner_sha256':winner_sha,'frozen_config':config,
             'test_count':len(y),'test_positives':int(y.sum()),'test_scoring_per_final_checkpoint':1,
             'stage3_tuning_trials':12,'stage4_frozen_final_trainings':12,
             'ratios':list(RATIOS),'seeds':list(SEEDS),'metrics':stats,
             'elliptic_temporal_test_semantics_conditional_pending_venus_review':dataset=='Elliptic'}
    write_json_new_or_same(out/f'EOGFD_{dataset}_Stage5_AllRatios_Summary.json',summary)
    print(f'EOGFD_{dataset}_STAGE5_12_ONE_TIME_HELDOUT_TESTS_COMPLETE=PASS TEST_SCORING_ONCE_PER_CHECKPOINT=YES VENUS_APPROVAL_NOT_RECORDED',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=DATASETS)
    parser.add_argument('--phase',required=True,choices=('audit','score'))
    args=parser.parse_args()
    if args.phase=='audit':audit_dataset(args.dataset)
    else:score_dataset(args.dataset)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('EOGFD_STAGE5_FAIL_CLOSED_ERROR='+repr(exc),flush=True)
        traceback.print_exc()
        sys.exit(2)
