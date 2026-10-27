#!/usr/bin/env python3
"""EOGFD fixed-winner, 4 ratios x 3 seeds, CONDITIONAL Stage 4 training only.

Intentionally no test label extraction/evaluation, no hyperparameter retuning,
no Stage 5 scoring. Model author source is pinned and untouched. Results are
conditional, NOT Venus-approved; author-mode reproduction NOT claimed.
"""
from __future__ import annotations
import argparse,csv,datetime,gc,hashlib,importlib.util,json,os,random,socket,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import pandas as pd

BASE=Path('/workspace/comp8851')
STAGE3=Path('/workspace/EOGFD_NEWI_STAGE3')
REPO=BASE/'repos/EOGFD'
TUNE_ROOT=BASE/'results/eogfd'
FINAL_ROOT=BASE/'results/eogfd'
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
HOST='60d39211b1fd'
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
CUTS={'TR40':20,'TR30':12,'TR20':7,'TR10':3}
EXPECTED={'FDCompCN':{'winner_trial':1,'winner_val_ap':0.215919},
          'Elliptic':{'winner_trial':10,'winner_val_ap':0.708897}}
EPOCH_LIMIT=100
PATIENCE=20
s3=None


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_name(path.name+'.tmp')
    with part.open('w') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(part,path)


def atomic_pt(path,obj):
    part=path.with_name(path.name+'.tmp')
    torch.save(obj,part);os.replace(part,path)


def atomic_npy(path,arr):
    part=path.with_name(path.name+'.tmp')
    with part.open('wb') as f:
        np.save(f,arr,allow_pickle=False)
        f.flush();os.fsync(f.fileno())
    os.replace(part,path)


def model_state(model):
    return {k:v for k,v in model.state_dict().items() if k!='theta_last'}


def restore_model(model,state):
    x=model.load_state_dict(state,strict=False)
    assert x.missing_keys==['theta_last'] and not x.unexpected_keys,x


def load_stage3_verified():
    global s3
    assert socket.gethostname()==HOST,('WRONG_HOST',socket.gethostname())
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    manifest=json.loads((STAGE3/'KIT_SHA256_MANIFEST.json').read_text())
    for name,expected_hash in manifest.items():
        assert sha(STAGE3/name)==expected_hash,('STAGE3_SCRIPT_CHANGED',name)
    path=STAGE3/'stage3_runner.py'
    spec=importlib.util.spec_from_file_location('eogfd_57212396_stage3_verified',path)
    s3=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s3)
    assert s3.PIN==PIN and s3.GRID and len(s3.GRID)==12
    assert s3.EVAL_SEED==100921 and s3.MAX_EPOCHS==100 and s3.PATIENCE==20
    assert tuple(s3.GRID[0].keys())==('lr','population_size','mutation_rate','hidden_dim','order','beta_p','weight_decay','lamb')
    print('EOGFD_STAGE4_PINNED_MODEL_STAGE3_CODE_GPU0=PASS',flush=True)


def verify_winner(dataset,mod,splits):
    d=TUNE_ROOT/dataset.lower()/'stage3_tr40_tuning'
    wp=d/f'EOGFD_{dataset}_NewI_TR40_FrozenWinner.json'
    assert wp.is_file(),('MISSING_FROZEN_WINNER',dataset)
    w=json.loads(wp.read_text())
    assert w['dataset']==dataset and w['status']=='TR40_12_VALIDATION_ONLY_TRIALS_COMPLETED_CONDITIONAL'
    assert w['trial_count']==12 and w['test_scored'] is False and w['test_labels_accessed'] is False
    assert w['final_training_started'] is False and w['venus_approval_recorded'] is False
    assert w['mode']=='UNIFIED_CONDITIONAL_ADAPTED' and w['upstream_commit']==PIN
    assert w['split_sha256']==mod.SPECS[dataset]['split_sha']
    assert w['ratios']==list(RATIOS) and w['final_seeds']==list(SEEDS)
    assert len(splits['TR40'])==mod.SPECS[dataset]['train']['TR40']
    records=[]
    for trial,cfg in enumerate(s3.GRID,1):
        p=d/f'trial_{trial:02d}'
        rec=json.loads((p/'trial_result.json').read_text())
        assert rec['trial']==trial and rec['config']==cfg
        assert rec['dataset']==dataset and rec['ratio']=='TR40' and rec['seed']==2
        assert rec['selection_metric']=='validation_AUPRC'
        assert rec['source_sha256']==w['source_sha256'] and rec['split_sha256']==w['split_sha256']
        assert rec['upstream_commit']==PIN and rec['test_scored'] is False and rec['test_labels_accessed'] is False
        assert sha(p/'best_validation.pt')==rec['checkpoint_sha256'],('STAGE3_CHECKPOINT_CHANGED',trial)
        assert sha(p/'best_validation_scores.npy')==rec['validation_scores_sha256'],('STAGE3_SCORES_CHANGED',trial)
        assert (p/'epoch_times.csv').is_file() and (p/'validation_history.json').is_file()
        records.append(rec)
    winner=max(records,key=lambda r:(r['best_val_auprc'],-r['trial']))
    assert w['winner_trial']==winner['trial']==EXPECTED[dataset]['winner_trial']
    assert w['winner_config']==winner['config']
    assert abs(w['winner_val_auprc']-winner['best_val_auprc'])<1e-12
    assert abs(w['winner_val_auprc']-EXPECTED[dataset]['winner_val_ap'])<0.0000006
    assert w['winner_checkpoint_sha256']==winner['checkpoint_sha256']
    assert w['winner_config'] in s3.GRID
    assert all(w['winner_config'][k]==s3.GRID[0][k] for k in ('hidden_dim','order','beta_p','lamb','weight_decay'))
    # Publish a hash-stable freeze link rather than reselecting hyperparameters during finals.
    print(f'EOGFD_{dataset}_STAGE4_ALL12_TUNING_TRIALS_AND_WINNER_SHA=PASS winner={winner["trial"]} val_AUPRC={winner["best_val_auprc"]:.6f}',flush=True)
    return w,sha(wp)


def fdcompcn_prepare(mod,ids):
    b=s3.fd_data(mod,ids)
    original,_=dgl.data.utils.load_graphs(str(BASE/'assets/fdcompcn/source/comp.dgl'))
    raw=original[0].ndata['label'].long().flatten()
    train_y={r:raw[torch.from_numpy(ids[r])].clone().to('cuda:0') for r in RATIOS}
    del raw,original
    assert b['provenance']['train_edges']==30752
    assert len(b['train_g'].canonical_etypes)==4
    for ratio in RATIOS:
        assert train_y[ratio].numel()==len(ids[ratio])
        assert 0<int(train_y[ratio].sum())<len(ids[ratio])
    assert len(b['val_y'])==1064
    return b,train_y,None


def elliptic_prepare(mod,ids):
    # Stage 3 loader verifies TR40 prefix, and builds frozen validation <=t31.
    b=s3.ell_data(mod,ids)
    source=BASE/'assets/elliptic/source'
    dtypes={0:np.int64,1:np.int16}
    dtypes.update({i:np.float32 for i in range(2,167)})
    frame=pd.read_csv(source/'elliptic_txs_features.csv',header=None,dtype=dtypes)
    assert frame.shape==(203769,167)
    txids=frame.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
    times=frame.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
    feats=frame.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
    del frame
    assert len(np.unique(txids))==203769 and np.isfinite(feats).all()
    edges=pd.read_csv(source/'elliptic_txs_edgelist.csv',dtype=np.int64)
    assert len(edges)==234355 and tuple(edges.columns)==('txId1','txId2')
    idx=pd.Index(txids)
    edge_src=idx.get_indexer(edges.txId1).astype(np.int64)
    edge_dst=idx.get_indexer(edges.txId2).astype(np.int64)
    del edges
    assert np.all(edge_src>=0) and np.all(edge_dst>=0)
    assert np.array_equal(times[edge_src],times[edge_dst])
    assert np.all(times[ids['test']]>=32)
    allowed=set(int(v) for v in txids[ids['TR40']])
    labels={}
    with (source/'elliptic_txs_classes.csv').open(newline='') as f:
        rows=csv.DictReader(f)
        assert set(rows.fieldnames)=={'txId','class'}
        for row in rows:
            tx=int(row['txId'])
            if tx in allowed:
                assert tx not in labels and row['class'] in ('1','2')
                labels[tx]=int(row['class']=='1')
    assert len(labels)==18889
    train_y={r:torch.as_tensor([labels[int(t)] for t in txids[ids[r]]],dtype=torch.long,device='cuda:0') for r in RATIOS}
    for ratio in RATIOS:
        assert 0<int(train_y[ratio].sum())<len(ids[ratio])
        assert np.all(times[ids[ratio]]<=CUTS[ratio])
    del labels
    cache={'txids':txids,'times':times,'features':feats,'edge_src':edge_src,'edge_dst':edge_dst}
    assert b['provenance']['train_edges']==106807 and b['provenance']['val_edges']==145833
    assert len(b['val_y'])==8726
    print('EOGFD_Elliptic_STAGE4_TEST_PERIOD_EXCLUDED_FROM_ALL_TRAINING_PREFIXES=PASS',flush=True)
    return b,train_y,cache


def graph_for_ratio(dataset,ratio,b,ids,cache):
    if dataset=='FDCompCN':
        assert b['train_g'].num_nodes()==5317 and len(b['train_g'].canonical_etypes)==4
        return b['train_g'],b['train_x'],torch.as_tensor(ids[ratio],device='cuda:0',dtype=torch.long),{'train_nodes':5317,'train_edges':30752,'cutoff':None}
    if ratio=='TR40':
        return b['train_g'],b['train_x'],b['train_ix'],{'train_nodes':90762,'train_edges':106807,'cutoff':20}
    cutoff=CUTS[ratio]
    times=cache['times']
    visible=np.flatnonzero(times<=cutoff)
    assert visible.size and len(visible)<90762
    mapped=np.full(len(times),-1,dtype=np.int64)
    mapped[visible]=np.arange(len(visible))
    # Use ONLY original same-period directed transaction edges.
    sel=times[cache['edge_src']]<=cutoff
    src=mapped[cache['edge_src'][sel]]
    dst=mapped[cache['edge_dst'][sel]]
    assert src.size>0 and np.all(src>=0) and np.all(dst>=0)
    assert np.all(mapped[ids[ratio]]>=0)
    assert np.all(mapped[ids['val']]==-1) and np.all(mapped[ids['test']]==-1)
    g=dgl.heterograph({('tx','original_transaction_edge','tx'):(torch.from_numpy(src.copy()),torch.from_numpy(dst.copy()))},num_nodes_dict={'tx':len(visible)})
    x=torch.from_numpy(cache['features'][visible].copy())
    assert g.num_edges()==len(src) and len(g.canonical_etypes)==1
    assert g.canonical_etypes==b['val_g'].canonical_etypes
    print(f'EOGFD_Elliptic_{ratio}_TRAIN_PREFIX=PASS timestep_le_{cutoff} nodes={g.num_nodes()} edges={g.num_edges()} test_features_in_train_graph=NO',flush=True)
    return g.to('cuda:0'),x.to('cuda:0'),torch.from_numpy(mapped[ids[ratio]].copy()).long().to('cuda:0'),{'train_nodes':int(len(visible)),'train_edges':int(len(src)),'cutoff':cutoff}


def save_epochs(path,history):
    part=path.with_name(path.name+'.tmp')
    with part.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['epoch','training_seconds','validation_seconds','cumulative_training_seconds'])
        cumulative=0.
        for e in history:
            cumulative+=e['training_seconds']
            writer.writerow([e['epoch'],f'{e["training_seconds"]:.8f}',f'{e["validation_seconds"]:.8f}',f'{cumulative:.8f}'])
        f.flush();os.fsync(f.fileno())
    os.replace(part,path)


def fit_one(dataset,ratio,seed,b,ids,train_y,cache,config,winner_sha,winner):
    dest=FINAL_ROOT/dataset.lower()/'stage4_frozen_finals'/ratio/f'seed_{seed}'
    dest.mkdir(parents=True,exist_ok=True)
    record_path=dest/'frozen_training.json'
    best=dest/'best_validation.pt'
    scores_path=dest/'best_validation_scores.npy'
    recovery=dest/'recovery_progress.pt'
    history_path=dest/'validation_history.json'
    epoch_path=dest/'epoch_times.csv'
    original_sha=winner['source_sha256']
    split_sha=winner['split_sha256']
    if record_path.exists():
        row=json.loads(record_path.read_text())
        assert row['ratio']==ratio and row['seed']==seed and row['config']==config
        assert row['frozen_winner_sha256']==winner_sha
        assert row['source_sha256']==original_sha and row['split_sha256']==split_sha
        assert row['test_scored'] is False and row['test_labels_accessed'] is False
        assert row['mode']=='UNIFIED_CONDITIONAL_ADAPTED' and row['venus_approval_recorded'] is False
        assert sha(best)==row['checkpoint_sha256'] and sha(scores_path)==row['validation_scores_sha256']
        assert sha(history_path)==row['history_sha256'] and sha(epoch_path)==row['epoch_times_sha256']
        assert not recovery.exists(), 'STOP: stale recovery with completed final result'
        print(f'EOGFD_{dataset}_{ratio}_seed{seed}_STAGE4_FINAL_REUSED=PASS',flush=True)
        return row
    if best.exists() and not recovery.exists():
        raise AssertionError(('STOP: orphaned best checkpoint without recovery record',str(best)))
    if scores_path.exists() or history_path.exists() or epoch_path.exists():
        raise AssertionError(('STOP: orphaned final output without completion record',str(dest)))
    g,x,ix,graph_meta=graph_for_ratio(dataset,ratio,b,ids,cache)
    y=train_y[ratio]
    assert y.numel()==ix.numel()==len(ids[ratio]) and 0<int(y.sum())<len(y)
    w=torch.tensor([1.,float((y==0).sum().item())/float((y==1).sum().item())],device='cuda:0')
    s3.seed_all(seed)
    model=s3.make_model(config,x.shape[1]);opt=torch.optim.Adam(model.parameters(),lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=config['weight_decay'])
    best_ap=-1.;best_epoch=0;best_val=None;stale=0;hist=[];start=1
    if recovery.exists():
        checkpoint=torch.load(recovery,map_location='cpu')
        assert checkpoint['ratio']==ratio and checkpoint['seed']==seed
        assert checkpoint['config']==config and checkpoint['winner_sha256']==winner_sha
        assert checkpoint['source_sha256']==original_sha and checkpoint['split_sha256']==split_sha
        s3.load_model_state(model,checkpoint['model_state'])
        opt.load_state_dict(checkpoint['optimizer_state'])
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'],device='cuda:0')
        hist=checkpoint['history'];best_ap=checkpoint['best_ap'];best_epoch=checkpoint['best_epoch'];best_val=checkpoint['best_val'];stale=checkpoint['stale']
        start=checkpoint['epoch']+1
        assert best.exists() and sha(best)==checkpoint['best_checkpoint_sha256'], 'STOP: inconsistent recovery/best checkpoint'
        assert len(hist)==start-1
        print(f'EOGFD_{dataset}_{ratio}_seed{seed}_STAGE4_RESUME_EPOCH={start}',flush=True)
    torch.cuda.reset_peak_memory_stats();tic=time.perf_counter()
    bb={'val_g':b['val_g'],'val_x':b['val_x'],'val_ix':b['val_ix'],'val_y':b['val_y']}
    for epoch in range(start,EPOCH_LIMIT+1):
        model.train();opt.zero_grad(set_to_none=True)
        begin=time.perf_counter()
        logits,_=model(g,x)
        loss=F.cross_entropy(logits[ix],y,weight=w)
        assert torch.isfinite(loss).item(),('NONFINITE_LOSS',dataset,ratio,seed,epoch)
        loss.backward();opt.step();torch.cuda.synchronize()
        train_seconds=time.perf_counter()-begin
        begin=time.perf_counter()
        val,score=s3.validate_only(model,bb)
        torch.cuda.synchronize()
        val_seconds=time.perf_counter()-begin
        better=val['val_auprc']>best_ap+1e-12
        if better:
            best_ap=val['val_auprc'];best_epoch=epoch;best_val=val;stale=0
            atomic_pt(best,{'model_state':model_state(model),'epoch':epoch,'ratio':ratio,'seed':seed,
                'dataset':dataset,'config':config,'validation':val,'frozen_winner_sha256':winner_sha,
                'source_sha256':original_sha,'split_sha256':split_sha,'upstream_commit':PIN,
                'mode':'UNIFIED_CONDITIONAL_ADAPTED','test_scored':False,'theta_last_buffer_excluded':True})
        else:stale+=1
        hist.append({'epoch':epoch,'train_loss':float(loss.detach().cpu()),'training_seconds':train_seconds,'validation_seconds':val_seconds,
                     **val,'improved':bool(better),'best_ap_so_far':best_ap,'stale':stale})
        atomic_pt(recovery,{'dataset':dataset,'ratio':ratio,'seed':seed,'epoch':epoch,'config':config,
            'model_state':model_state(model),'optimizer_state':opt.state_dict(), 'history':hist,
            'best_ap':best_ap,'best_epoch':best_epoch,'best_val':best_val,'stale':stale,
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device='cuda:0'),
            'winner_sha256':winner_sha,'source_sha256':original_sha,'split_sha256':split_sha,
            'best_checkpoint_sha256':sha(best)})
        if epoch<=2 or epoch%10==0 or stale>=PATIENCE:
            print(f'EOGFD_{dataset}_{ratio}_seed{seed} epoch={epoch} loss={hist[-1]["train_loss"]:.5f} val_AUPRC={val["val_auprc"]:.6f} best={best_ap:.6f} stale={stale}',flush=True)
        del logits,loss,score
        if stale>=PATIENCE:break
    assert best.is_file() and best_val is not None and best_epoch>0
    # Replay best validation with same fixed RNG stream; never inspect test.
    checkpoint=torch.load(best,map_location='cpu')
    assert checkpoint['epoch']==best_epoch and checkpoint['ratio']==ratio and checkpoint['seed']==seed
    assert checkpoint['frozen_winner_sha256']==winner_sha
    s3.load_model_state(model,checkpoint['model_state'])
    verify,scores=s3.validate_only(model,bb)
    assert abs(verify['val_auprc']-best_ap)<1e-8,('VALIDATION_CHECKPOINT_REPLAY_MISMATCH',dataset,ratio,seed)
    assert abs(verify['threshold']-best_val['threshold'])<1e-12
    assert len(scores)==len(ids['val'])
    atomic_npy(scores_path,scores)
    atomic_json(history_path,hist)
    save_epochs(epoch_path,hist)
    rec={'model':'EOGFDHeteroModel','dataset':dataset,'ratio':ratio,'seed':seed,
        'split_seed':2,'train_count':len(ids[ratio]),'train_positives':int(y.sum().item()),
        'validation_count':len(ids['val']),'config':config,'frozen_winner_sha256':winner_sha,
        'mode':'UNIFIED_CONDITIONAL_ADAPTED','author_identical_reproduction':False,'venus_approval_recorded':False,
        'best_epoch':best_epoch,'epochs_completed':hist[-1]['epoch'],'best_val_auprc':best_ap,
        'best_val_auroc':verify['val_auroc'],'best_val_macro_f1':verify['val_macro_f1'],
        'frozen_validation_threshold':verify['threshold'],'fraud_recall_validation':verify['val_fraud_recall'],
        'checkpoint_sha256':sha(best),'validation_scores_sha256':sha(scores_path),'history_sha256':sha(history_path),
        'epoch_times_sha256':sha(epoch_path),'source_sha256':original_sha,'split_sha256':split_sha,
        'upstream_commit':PIN,'train_time_seconds':sum(e['training_seconds'] for e in hist),
        'validation_time_seconds':sum(e['validation_seconds'] for e in hist),
        'total_stage4_run_wall_seconds':time.perf_counter()-tic,'peak_gpu_allocated_gib':float(torch.cuda.max_memory_allocated()/2**30),
        'graph':graph_meta,'max_epochs':EPOCH_LIMIT,'patience':PATIENCE,
        'validation_evolution_rng_seed':s3.EVAL_SEED,'test_labels_accessed':False,'test_scored':False,
        'heldout_test_evaluations':0,'test_pending':True}
    atomic_json(record_path,rec)
    recovery.unlink()
    print(f'EOGFD_{dataset}_{ratio}_seed{seed}_FROZEN_TRAINING=PASS val_AUPRC={best_ap:.6f} threshold={verify["threshold"]:.2f} test_scored=NO',flush=True)
    del model,opt,checkpoint,g,x,ix,y,w
    gc.collect();torch.cuda.empty_cache()
    return rec


def main(dataset):
    load_stage3_verified()
    mod=s3.import_adapter()
    ids,pilot=s3.validate_stage2(dataset,mod)
    winner,winner_sha=verify_winner(dataset,mod,ids)
    config=winner['winner_config']
    root=FINAL_ROOT/dataset.lower()/'stage4_frozen_finals'
    root.mkdir(parents=True,exist_ok=True)
    attempt_id=(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+
                '_EOGFD_'+dataset+'_UNIFIED_CONDITIONAL_STAGE4_'+str(time.time_ns()))
    attempts=root/'attempts'
    attempts.mkdir(exist_ok=True)
    attempt=attempts/(attempt_id+'.json')
    with attempt.open('x') as f:
        json.dump({'attempt_id':attempt_id,'dataset':dataset,'status':'STARTED',
                   'venus_approval_recorded':False,'test_scored':False,'mode':'UNIFIED_CONDITIONAL_ADAPTED'},f,indent=2)
    try:
        source=winner['source_sha256'];split=winner['split_sha256']
        manifest={'dataset':dataset,'model':'EOGFDHeteroModel','mode':'UNIFIED_CONDITIONAL_ADAPTED',
             'classification':'CONDITIONAL_ADAPTED_UNIFIED_TRAINING_NOT_AUTHOR_REPRODUCTION',
             'venus_approval_recorded':False,'winner_trial':winner['winner_trial'],
             'winner_sha256':winner_sha,'frozen_config':config,'upstream_commit':PIN,
             'source_sha256':source,'split_sha256':split,'split_seed':2,
             'ratios':list(RATIOS),'training_seeds':list(SEEDS),'max_epochs':EPOCH_LIMIT,'patience':PATIENCE,
             'selection_metric':'validation_AUPRC','threshold_metric':'validation_macroF1',
             'original_stage3_grid_trial_count':12,'test_scored':False,'heldout_test_access':False,
             'elliptic_chronological_cutoffs':CUTS if dataset=='Elliptic' else None,
             'note':'Exploratory/conditional dataset adaptation; Venus methodological signoff NOT recorded.'}
        mf=root/'stage4_frozen_training_manifest.json'
        if mf.exists():assert json.loads(mf.read_text())==manifest,'STAGE4_MANIFEST_CONFLICT'
        else:atomic_json(mf,manifest)
        if dataset=='FDCompCN':b,labels,cache=fdcompcn_prepare(mod,ids)
        else:b,labels,cache=elliptic_prepare(mod,ids)
        print(f'EOGFD_{dataset}_STAGE4_12_FROZEN_TRAININGS_START=PASS VENUS_SIGNOFF_NOT_RECORDED TEST_SCORED=NO',flush=True)
        all_rows=[]
        for ratio in RATIOS:
            for seed in SEEDS:
                all_rows.append(fit_one(dataset,ratio,seed,b,ids,labels,cache,config,winner_sha,winner))
        assert len(all_rows)==12 and len({(r['ratio'],r['seed']) for r in all_rows})==12
        assert all(row['test_scored'] is False and row['test_labels_accessed'] is False for row in all_rows)
        stats=[]
        for ratio in RATIOS:
            arr=np.asarray([row['best_val_auprc'] for row in all_rows if row['ratio']==ratio],dtype=np.float64)
            assert len(arr)==3
            stats.append({'ratio':ratio,'validation_AUPRC_mean':float(arr.mean()),'validation_AUPRC_sample_sd':float(arr.std(ddof=1))})
        summary={'status':'EOGFD_12_CONDITIONAL_FROZEN_TRAININGS_COMPLETE','model':'EOGFDHeteroModel',
                 'dataset':dataset,'mode':'UNIFIED_CONDITIONAL_ADAPTED','venus_approval_recorded':False,
                 'completed_finals':12,'ratios':list(RATIOS),'seeds':list(SEEDS),'frozen_config':config,
                 'frozen_winner_sha256':winner_sha,'source_sha256':source,'split_sha256':split,
                 'upstream_commit':PIN,'validation_stats':stats,'test_scored':False,'test_evaluations':0}
        final=root/f'EOGFD_{dataset}_Stage4_12FrozenFinalsSummary.json'
        if final.exists():assert json.loads(final.read_text())==summary,'FINAL_SUMMARY_CONFLICT'
        else:atomic_json(final,summary)
        atomic_json(attempt,{'attempt_id':attempt_id,'dataset':dataset,'status':'COMPLETED',
                             'completed_finals':12,'venus_approval_recorded':False,'test_scored':False,
                             'mode':'UNIFIED_CONDITIONAL_ADAPTED'})
        for item in stats:
            print(f'EOGFD_{dataset}_{item["ratio"]}_STAGE4_VAL_AUPRC_MEAN={item["validation_AUPRC_mean"]:.6f} SAMPLE_SD={item["validation_AUPRC_sample_sd"]:.6f}',flush=True)
        print(f'EOGFD_{dataset}_STAGE4_12_FROZEN_TRAININGS_COMPLETE=PASS TEST_SCORED=NO VENUS_APPROVAL_NOT_RECORDED',flush=True)
    except BaseException as e:
        atomic_json(attempt,{'attempt_id':attempt_id,'dataset':dataset,'status':'FAILED_OR_INTERRUPTED',
                             'error':repr(e),'venus_approval_recorded':False,'test_scored':False,
                             'mode':'UNIFIED_CONDITIONAL_ADAPTED'})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['FDCompCN','Elliptic'])
    args=parser.parse_args()
    try:main(args.dataset)
    except BaseException as exc:
        print(f'EOGFD_{args.dataset}_STAGE4_STOP_ERROR={exc!r}',flush=True)
        traceback.print_exc();sys.exit(2)
