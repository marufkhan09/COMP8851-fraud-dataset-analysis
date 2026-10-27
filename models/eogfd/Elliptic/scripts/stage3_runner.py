#!/usr/bin/env python3
"""EOGFD original heterograph implementation, CONDITIONAL unified-mode 12-trial TR40 tuning.

No test labels, test predictions, evaluation of test nodes or test-driven choices.
FDCompCN uses original 4 directed relation types in static transductive graph.
Elliptic uses original directed single relation with training prefix <=t20,
validation-only snapshot <=t31 (test period begins t32).
Venus approval not claimed. No author-mode reproduction claimed.
"""
from __future__ import annotations
import argparse,csv,gc,hashlib,importlib.util,json,os,random,socket,subprocess,sys,time,traceback,itertools,datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

BASE=Path('/workspace/comp8851')
KIT=Path('/workspace/EOGFD_NEWI_KIT')
REPO=BASE/'repos/EOGFD'
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
PILOT_FIX_SHA='81112b076aa3936039422a297fae948f09753d1ed36138b940fdb244aba92160'
PILOT_RAW_SHA='70879ceaac03871676e408dbc2902581c26fba31bcf9d6991977702871ff3706'
EVAL_SEED=100921
PATIENCE=20
MAX_EPOCHS=100
GRID=[dict(lr=lr,population_size=pop,mutation_rate=mr,hidden_dim=64,order=3,beta_p=1,
           weight_decay=0.0,lamb=0.0)
      for lr,pop,mr in itertools.product((.001,.003,.01),(10,20),(.05,.10))]
assert len(GRID)==12
ROOT=BASE/'results/eogfd'
AUDIT=BASE/'diagnostics/eogfd_newi_57212396'
ATTEMPT_RECORD=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for p in iter(lambda:f.read(8*1024*1024),b''):h.update(p)
    return h.hexdigest()


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_name(path.name+'.tmp')
    with p.open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(p,path)


def atomic_pt(path,obj):
    p=path.with_name(path.name+'.tmp')
    torch.save(obj,p);os.replace(p,path)


def atomic_npy(path,arr):
    p=path.with_name(path.name+'.tmp')
    with p.open('wb') as f:
        np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(p,path)


def seed_all(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)


def model_state(model):
    # Original theta_last has data-dependent size and must not be strict-loaded.
    return {k:v for k,v in model.state_dict().items() if k!='theta_last'}


def load_model_state(model,state):
    status=model.load_state_dict(state,strict=False)
    assert status.missing_keys==['theta_last'] and not status.unexpected_keys,status


def import_adapter():
    original=KIT/'vast_preflight_and_realgraph_pilots.py'
    correct=KIT/'vast_preflight_and_realgraph_pilots_elliptic_corrected.py'
    assert sha(original)==PILOT_RAW_SHA,'Original Stage2 adapter changed'
    assert sha(correct)==PILOT_FIX_SHA,'Corrected Stage2 adapter changed'
    spec=importlib.util.spec_from_file_location('eogfd_stage2_verified',correct)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert mod.PIN==PIN
    return mod


def validate_stage2(dataset,mod):
    assert socket.gethostname()=='60d39211b1fd','STOP: Not original Vast instance 57212396'
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip(),'Pinned author source dirty'
    s=mod.check_inputs(dataset)  # all canonical source/split SHA checks and nested membership
    pilot_file=AUDIT/(dataset+'_Stage1_Stage2_ReadOnlyThenOneStep.json')
    p=json.loads(pilot_file.read_text())
    assert p['status']==dataset+'_REAL_GRAPH_PILOT_PASS'
    assert p['test_scored'] is False and p['tuning_started'] is False
    assert p['upstream_commit']==PIN and p['split_sha256']==mod.SPECS[dataset]['split_sha']
    assert p['train_count']==len(s['TR40'])
    print(f'EOGFD_{dataset}_PINNED_SOURCE_SPLIT_AND_REAL_GRAPH_PILOT=PASS TEST_SCORED=NO',flush=True)
    return s,p


def fd_data(mod,s):
    from dgl.data.utils import load_graphs
    g,x,tr,yt,meta=mod.fdcompcn_graph(s)
    source,_=load_graphs(str(BASE/'assets/fdcompcn/source/comp.dgl'))
    labels=source[0].ndata['label'].long().flatten()
    val_labels=labels[torch.from_numpy(s['val'])].numpy().astype(np.int64,copy=True)
    del labels,source
    assert np.unique(val_labels).tolist()==[0,1]
    assert len(tr)==2126 and len(s['val'])==1064
    assert len(g.canonical_etypes)==4 and meta['edge_count']==30752
    ix=torch.as_tensor(tr.copy(),dtype=torch.long,device='cuda:0')
    v_ix=torch.as_tensor(s['val'].copy(),dtype=torch.long,device='cuda:0')
    device='cuda:0'
    return dict(dataset='FDCompCN',train_g=g.to(device),val_g=g.to(device),
                train_x=x.to(device),val_x=x.to(device),train_ix=ix,train_y=yt.long().to(device),
                val_ix=v_ix,val_y=val_labels,
                provenance={'mode':'UNIFIED_CONDITIONAL_ADAPTED','graph_semantics':'FDCompCN original 4 directed relation types; full static unlabeled transductive topology',
                 'test_labels_in_graph':False,'train_edges':30752,'val_edges':30752,'train_nodes':5317,'val_nodes':5317,
                 'adapter_eligibility':'INPUT_ADAPTER_PENDING_VENUS_REVIEW','author_identical':False})


def elliptic_validation_snapshot(s,train_meta):
    folder=BASE/'assets/elliptic/source'
    dt={0:np.int64,1:np.int16};dt.update({i:np.float32 for i in range(2,167)})
    dataframe=pd.read_csv(folder/'elliptic_txs_features.csv',header=None,dtype=dt)
    assert dataframe.shape==(203769,167)
    txids=dataframe.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
    times=dataframe.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
    feats=dataframe.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
    del dataframe
    assert len(np.unique(txids))==203769
    assert np.isfinite(feats).all()
    assert np.all(times[s['val']]>=21) and np.all(times[s['val']]<=31)
    assert np.all(times[s['test']]>=32) and np.all(times[s['test']]<=49)
    visible=np.flatnonzero(times<=31)
    mapping=np.full(203769,-1,dtype=np.int64)
    mapping[visible]=np.arange(len(visible),dtype=np.int64)
    assert len(visible)==126103 and np.all(mapping[s['val']]>=0) and np.all(mapping[s['test']]==-1)
    assert np.all(mapping[s['TR40']]>=0)
    ed=pd.read_csv(folder/'elliptic_txs_edgelist.csv',dtype=np.int64)
    assert tuple(ed.columns)==('txId1','txId2') and len(ed)==234355
    indexed=pd.Index(txids)
    src=indexed.get_indexer(ed['txId1']).astype(np.int64)
    dst=indexed.get_indexer(ed['txId2']).astype(np.int64)
    del ed
    assert np.all((src>=0)&(dst>=0)) and np.array_equal(times[src],times[dst])
    kept=times[src]<=31
    uu=mapping[src[kept]];vv=mapping[dst[kept]]
    assert np.all(uu>=0)&np.all(vv>=0)
    assert len(uu)==145833,('Unexpected original Elliptic t<=31 edge count',len(uu))
    graph=dgl.heterograph({('tx','original_transaction_edge','tx'):(torch.from_numpy(uu.copy()),torch.from_numpy(vv.copy()))},num_nodes_dict={'tx':len(visible)})
    assert graph.num_edges()==145833
    allowed=set(int(x) for x in txids[s['val']])
    selected={}
    with (folder/'elliptic_txs_classes.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        assert set(reader.fieldnames)=={'txId','class'}
        for row in reader:
            tx=int(row['txId'])
            if tx in allowed:
                assert tx not in selected and row['class'] in ('1','2')
                selected[tx]=int(row['class']=='1')
    assert len(selected)==8726
    yval=np.asarray([selected[int(tx)] for tx in txids[s['val']]],dtype=np.int64)
    assert int(yval.sum())==1289
    assert train_meta['future_nodes_visible_during_training'] is False
    assert train_meta['future_edges_visible_during_training'] is False
    print('EOGFD_Elliptic_CAUSAL_VALIDATION_SNAPSHOT=PASS t<=31 nodes=126103 edges=145833 test_nodes_excluded=YES',flush=True)
    return graph,torch.from_numpy(feats[visible].copy()),mapping[s['val']].copy(),yval


def ell_data(mod,s):
    train_g,train_x,train_ix,train_y,train_meta=mod.elliptic_graph(s)
    val_g,val_x,v_ids,yval=elliptic_validation_snapshot(s,train_meta)
    assert train_g.num_nodes()==90762 and train_g.num_edges()==106807
    assert train_g.canonical_etypes==val_g.canonical_etypes
    return dict(dataset='Elliptic',train_g=train_g.to('cuda:0'),val_g=val_g.to('cuda:0'),
                train_x=train_x.to('cuda:0'),val_x=val_x.to('cuda:0'),
                train_ix=torch.as_tensor(train_ix.copy(),dtype=torch.long,device='cuda:0'),
                train_y=train_y.long().to('cuda:0'),
                val_ix=torch.as_tensor(v_ids,dtype=torch.long,device='cuda:0'),val_y=yval,
                provenance={'mode':'UNIFIED_CONDITIONAL_ADAPTED',
                'graph_semantics':'Single original directed transaction-edge relation; train prefix <=20, validation only snapshot <=31; held-out test starts at 32',
                'test_labels_in_graph':False,'train_edges':106807,'val_edges':145833,'train_nodes':90762,'val_nodes':126103,
                'adapter_eligibility':'CONDITIONAL_PENDING_VENUS_REVIEW', 'author_identical':False})


def data_for(dataset,mod,s):
    out=fd_data(mod,s) if dataset=='FDCompCN' else ell_data(mod,s)
    assert out['train_x'].shape[0]==out['train_g'].num_nodes()
    assert out['val_x'].shape[0]==out['val_g'].num_nodes()
    assert len(out['train_ix'])==len(s['TR40'])
    assert len(out['val_ix'])==len(s['val'])
    assert len(out['val_y'])==len(s['val'])
    assert out['train_y'].numel()==len(s['TR40'])
    return out


def make_model(cfg,feat_dim):
    from models.eogfd import EOGFDHeteroModel
    return EOGFDHeteroModel(in_feats=feat_dim,hid_feats=cfg['hidden_dim'],num_classes=2,
        d=cfg['order'],beta_p=cfg['beta_p'],population_size=cfg['population_size'],mutation_rate=cfg['mutation_rate']).to('cuda:0')


def validate_only(model,b):
    model.eval()
    # Same stochastic-evolution RNG stream every validation eval; fork avoids affecting train RNG.
    with torch.random.fork_rng(devices=[0],enabled=True):
        torch.manual_seed(EVAL_SEED);torch.cuda.manual_seed_all(EVAL_SEED)
        with torch.no_grad():
            logits,_=model(b['val_g'],b['val_x'])
            scores=torch.softmax(logits[b['val_ix']],dim=1)[:,1].detach().cpu().numpy().astype(np.float64)
    labels=b['val_y']
    assert len(scores)==len(labels) and np.isfinite(scores).all()
    thresholds=np.arange(1,100,dtype=np.float64)/100
    preds=scores[:,None]>thresholds[None,:]
    yy=labels.astype(bool)[:,None]
    tp=(preds&yy).sum(axis=0);tn=((~preds)&(~yy)).sum(axis=0)
    fp=(preds&(~yy)).sum(axis=0);fn=((~preds)&yy).sum(axis=0)
    fpos=2*tp/np.maximum(2*tp+fp+fn,1)
    fneg=2*tn/np.maximum(2*tn+fp+fn,1)
    mac=(fpos+fneg)/2
    recall=tp/np.maximum(tp+fn,1)
    best=max(range(99),key=lambda i:(round(float(mac[i]),12),round(float(recall[i]),12),-abs(float(thresholds[i])-.5),-float(thresholds[i])))
    metrics={'val_auprc':float(average_precision_score(labels,scores)),'val_auroc':float(roc_auc_score(labels,scores)),
             'val_macro_f1':float(mac[best]),'val_fraud_recall':float(recall[best]),'threshold':float(thresholds[best])}
    return metrics,scores


def build_weights(b):
    ys=b['train_y'];pos=int(ys.sum().item());neg=len(ys)-pos
    assert pos>0 and neg>0
    return torch.tensor([1.,neg/pos],device='cuda:0'),{'positives':pos,'negatives':neg,'class_weights':[1.,neg/pos]}


def cost_gate(dataset,K,b,weights,out,source_sha,split_sha):
    p=out/'population20_cost_gate.json'
    if p.exists():
        d=json.loads(p.read_text());assert d['status']=='PASS' and d['dataset']==dataset and d['source_sha256']==source_sha and d['split_sha256']==split_sha
        print(f'EOGFD_{dataset}_STAGE3_POP20_GATE_REUSED=PASS',flush=True);return
    seed_all(2);cfg=GRID[2];assert cfg['population_size']==20
    model=make_model(cfg,b['train_x'].shape[1]);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    torch.cuda.reset_peak_memory_stats();tic=time.perf_counter()
    model.train();opt.zero_grad(set_to_none=True)
    logits=model(b['train_g'],b['train_x'])[0]
    loss=F.cross_entropy(logits[b['train_ix']],b['train_y'],weight=weights)
    assert torch.isfinite(loss).item()
    loss.backward();opt.step();torch.cuda.synchronize()
    seconds=time.perf_counter()-tic;peak=torch.cuda.max_memory_allocated()/2**30
    del model,opt,loss,logits;gc.collect();torch.cuda.empty_cache()
    print(f'EOGFD_{dataset}_STAGE3_POP20_GATE seconds={seconds:.3f} gpu_peak_gib={peak:.3f}',flush=True)
    assert peak<12 and seconds<25,('STOP population20 runtime/memory budget exceeded',peak,seconds)
    atomic_json(p,dict(dataset=dataset,status='PASS',source_sha256=source_sha,split_sha256=split_sha,
            pop_size=20,seconds=seconds,gpu_peak_gib=peak,test_scored=False,train_labels_only=True))


def run_trial(dataset,index,cfg,b,weights,out,source_sha,split_sha):
    dirname=out/('trial_%02d'%index);dirname.mkdir(parents=True,exist_ok=True)
    final=dirname/'trial_result.json';best=dirname/'best_validation.pt'
    progress=dirname/'training_resume.pt';history_path=dirname/'validation_history.json'
    scores_path=dirname/'best_validation_scores.npy'
    if final.exists():
        record=json.loads(final.read_text())
        assert record['config']==cfg and record['source_sha256']==source_sha and record['split_sha256']==split_sha
        assert record['test_labels_accessed'] is False and record['test_scored'] is False
        assert best.is_file() and sha(best)==record['checkpoint_sha256']
        assert scores_path.is_file() and sha(scores_path)==record['validation_scores_sha256']
        print(f'EOGFD_{dataset}_TR40_TRIAL_{index:02d}_REUSED=PASS val_AUPRC={record["best_val_auprc"]:.6f}',flush=True)
        return record
    seed_all(2)
    model=make_model(cfg,b['train_x'].shape[1]);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],
                                    betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
    history=[];best_metric=-1.;best_epoch=0;best_details=None;stale=0;start_epoch=1
    if progress.exists():
        rec=torch.load(progress,map_location='cpu')
        assert rec['config']==cfg and rec['source_sha256']==source_sha and rec['split_sha256']==split_sha
        load_model_state(model,rec['model_state']);opt.load_state_dict(rec['optimizer_state'])
        torch.set_rng_state(rec['cpu_rng']);torch.cuda.set_rng_state(rec['cuda_rng'],device='cuda:0')
        history=rec['history'];best_metric=rec['best_metric'];best_epoch=rec['best_epoch'];best_details=rec['best_details'];stale=rec['stale'];start_epoch=rec['epoch']+1
        assert best.exists() or best_epoch==0
        print(f'EOGFD_{dataset}_TR40_TRIAL_{index:02d}_RESUME_FROM_EPOCH={start_epoch}',flush=True)
    else:
        assert not best.exists() and not scores_path.exists(),'Checkpoint exists without corresponding recovery file: STOP'
    torch.cuda.reset_peak_memory_stats();tic=time.perf_counter()
    if start_epoch>MAX_EPOCHS or stale>=PATIENCE:
        assert len(history)>0 and best_details is not None
    else:
        for epoch in range(start_epoch,MAX_EPOCHS+1):
            model.train();opt.zero_grad(set_to_none=True)
            t0=time.perf_counter()
            logits=model(b['train_g'],b['train_x'])[0]
            loss=F.cross_entropy(logits[b['train_ix']],b['train_y'],weight=weights)
            assert torch.isfinite(loss).item(),('NONFINITE TRAIN LOSS',index,epoch)
            loss.backward();opt.step();torch.cuda.synchronize()
            train_seconds=time.perf_counter()-t0
            tv=time.perf_counter()
            values,scores=validate_only(model,b)
            torch.cuda.synchronize()
            val_seconds=time.perf_counter()-tv
            improved=values['val_auprc']>best_metric+1e-12
            if improved:
                best_metric=values['val_auprc'];best_epoch=epoch;best_details=values;stale=0
                atomic_pt(best,dict(model_state=model_state(model),epoch=epoch,config=cfg,validation=values,
                    upstream_commit=PIN,source_sha256=source_sha,split_sha256=split_sha,dataset=dataset,
                    mode='UNIFIED_CONDITIONAL_ADAPTED',test_scored=False,theta_last_buffer_excluded=True))
                atomic_npy(scores_path,scores)
            else:stale+=1
            history.append(dict(epoch=epoch,train_loss=float(loss.detach().cpu()),train_seconds=train_seconds,
                validation_seconds=val_seconds,**values,improved=bool(improved),best_so_far=best_metric,stale=stale))
            atomic_pt(progress,dict(epoch=epoch,config=cfg,model_state=model_state(model),optimizer_state=opt.state_dict(),
                history=history,best_metric=best_metric,best_epoch=best_epoch,best_details=best_details,stale=stale,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device='cuda:0'),
                source_sha256=source_sha,split_sha256=split_sha))
            if epoch<=2 or epoch%10==0 or stale>=PATIENCE:
                print(f'EOGFD_{dataset}_TR40_TRIAL_{index:02d} epoch={epoch} loss={history[-1]["train_loss"]:.5f} val_AUPRC={values["val_auprc"]:.6f} best={best_metric:.6f} stale={stale}',flush=True)
            del logits,loss,scores
            if stale>=PATIENCE:break
    assert best.is_file() and scores_path.is_file() and best_details is not None
    atomic_json(history_path,history)
    # Full per-epoch timing CSV required by COMP8851 master execution plan v3.
    import csv as csvmod
    with (dirname/'epoch_times.csv.tmp').open('w',newline='') as f:
        w=csvmod.writer(f);w.writerow(['epoch','training_seconds','validation_seconds','cumulative_training_seconds'])
        acc=0.
        for item in history:
            acc+=item['train_seconds']
            w.writerow([item['epoch'],format(item['train_seconds'],'.6f'),format(item['validation_seconds'],'.6f'),format(acc,'.6f')])
    os.replace(dirname/'epoch_times.csv.tmp',dirname/'epoch_times.csv')
    rec=dict(dataset=dataset,model='EOGFDHeteroModel',trial=index,ratio='TR40',seed=2,config=cfg,
        mode='UNIFIED_CONDITIONAL_ADAPTED',venus_approval_recorded=False,
        epochs_completed=history[-1]['epoch'],best_epoch=best_epoch,best_val_auprc=best_metric,best_validation_metrics=best_details,
        checkpoint_sha256=sha(best),validation_scores_sha256=sha(scores_path),source_sha256=source_sha,
        split_sha256=split_sha,upstream_commit=PIN,selection_metric='validation_AUPRC',
        threshold_metric='validation_macroF1_recall_then_nearest_0.5',fixed_validation_rng_seed=EVAL_SEED,
        train_wall_seconds=sum(h['train_seconds'] for h in history),
        validation_wall_seconds=sum(h['validation_seconds'] for h in history),
        trial_wall_seconds=time.perf_counter()-tic,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        test_labels_accessed=False,test_scored=False)
    atomic_json(final,rec)
    if progress.exists():progress.unlink()
    print(f'EOGFD_{dataset}_TR40_TRIAL_{index:02d}_VALIDATION_ONLY=PASS AUPRC={best_metric:.6f} best_epoch={best_epoch} threshold={best_details["threshold"]:.2f}',flush=True)
    del model,opt;gc.collect();torch.cuda.empty_cache()
    return rec


def main(dataset):
    global ATTEMPT_RECORD
    assert len(GRID)==12 and len({json.dumps(c,sort_keys=True) for c in GRID})==12
    mod=import_adapter();s,pilot=validate_stage2(dataset,mod)
    out=ROOT/dataset.lower()/'stage3_tr40_tuning'
    out.mkdir(parents=True,exist_ok=True)
    attempt_id=(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+
        '_EOGFD_'+dataset+'_unified_conditional_TR40_splitS2_trainS2_'+str(time.time_ns()))
    ATTEMPT_RECORD=out/'attempts'/(attempt_id+'.json')
    ATTEMPT_RECORD.parent.mkdir(exist_ok=True,parents=True)
    with ATTEMPT_RECORD.open('x') as f:
        json.dump({'attempt_id':attempt_id,'dataset':dataset,'status':'STARTED',
          'mode':'UNIFIED_CONDITIONAL_ADAPTED','venus_approval_recorded':False,
          'test_scored':False},f,sort_keys=True,indent=2);f.write('\n')
    print(f'EOGFD_{dataset}_STAGE3_UNIQUE_ATTEMPT={attempt_id}',flush=True)
    expected_shas=mod.SPECS[dataset]
    source_sha=expected_shas['source'][0][1] if len(expected_shas['source'])==1 else hashlib.sha256(''.join(v for _,v in sorted(expected_shas['source'])).encode()).hexdigest()
    split_sha=expected_shas['split_sha']
    info=out/'stage3_run_manifest.json'
    manifest={'instance_id':'57212396','machine_hostname':'60d39211b1fd','dataset':dataset,
        'mode':'UNIFIED_CONDITIONAL_ADAPTED','author_identical_reproduction':False,'venus_approval_recorded':False,
        'source_sha256':source_sha,'individual_source_sha256':dict(expected_shas['source']),
        'split_sha256':split_sha,'upstream_commit':PIN,'split_seed':2,'tuning_seed':2,'grid':GRID,
        'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'checkpoint_metric':'validation_AUPRC',
        'threshold_selection':'validation_macroF1_max; tie fraud recall then nearest 0.5',
        'eval_rng_seed':EVAL_SEED,'stochastic_evolution_rng_fixed_on_validation':True,
        'heldout_test_access_during_stage3':False,'frozen_ratios':['TR40','TR30','TR20','TR10'],
        'final_seeds':[2,42,72],'model':'EOGFDHeteroModel',
        'note':'This is adapted, conditional unified-mode validation tuning only, not author-mode and not evidence of supervisor permission.'}
    if info.exists():assert json.loads(info.read_text())==manifest,'Manifest conflict: STOP'
    else:atomic_json(info,manifest)
    b=data_for(dataset,mod,s)
    weights,counts=build_weights(b)
    print(f'EOGFD_{dataset}_STAGE3_GRAPH_READY train_nodes={b["provenance"]["train_nodes"]} val_nodes={b["provenance"]["val_nodes"]} relations={len(b["train_g"].canonical_etypes)} weights={counts["class_weights"]}',flush=True)
    print(f'EOGFD_{dataset}_STAGE3_CONDITIONAL_NOT_YET_VENUS_APPROVED=YES',flush=True)
    gate_sha=split_sha
    cost_gate(dataset,None,b,weights,out,source_sha,gate_sha)
    records=[]
    for i,cfg in enumerate(GRID,1):
        records.append(run_trial(dataset,i,cfg,b,weights,out,source_sha,split_sha))
    selected=max(records,key=lambda r:(r['best_val_auprc'],-r['trial']))
    winner={'dataset':dataset,'status':'TR40_12_VALIDATION_ONLY_TRIALS_COMPLETED_CONDITIONAL',
            'winner_trial':selected['trial'],'winner_config':selected['config'],
            'winner_val_auprc':selected['best_val_auprc'],'winner_checkpoint_sha256':selected['checkpoint_sha256'],
            'source_sha256':source_sha,'split_sha256':split_sha,'upstream_commit':PIN,
            'mode':'UNIFIED_CONDITIONAL_ADAPTED','venus_approval_recorded':False,
            'trial_count':len(records),'test_scored':False,'test_labels_accessed':False,
            'final_training_started':False,'ratios':['TR40','TR30','TR20','TR10'],'final_seeds':[2,42,72],
            'adapter_provenance':b['provenance']}
    wp=out/f'EOGFD_{dataset}_NewI_TR40_FrozenWinner.json'
    if wp.exists():assert json.loads(wp.read_text())==winner,'Existing winner conflict'
    else:atomic_json(wp,winner)
    summary=out/f'EOGFD_{dataset}_NewI_TR40_TuningSummary.json'
    if summary.exists():
        assert json.loads(summary.read_text())['winner']==winner
    else:
        atomic_json(summary,dict(winner=winner,trial_metrics=[{'trial':r['trial'],'val_auprc':r['best_val_auprc'],
         'best_epoch':r['best_epoch'],'config':r['config']} for r in records]))
    print(f'EOGFD_{dataset}_STAGE3_12_VALIDATION_ONLY_TUNING_COMPLETE=PASS TEST_SCORED=NO',flush=True)
    print(f'EOGFD_{dataset}_STAGE3_WINNER=TRIAL_{selected["trial"]:02d} VAL_AUPRC={selected["best_val_auprc"]:.6f} CONFIG={json.dumps(selected["config"],sort_keys=True)}',flush=True)
    print(f'EOGFD_{dataset}_VENUS_METHODOLOGY_SIGNOFF=NOT_RECORDED MODE=UNIFIED_CONDITIONAL',flush=True)
    atomic_json(ATTEMPT_RECORD,dict(attempt_id=attempt_id,dataset=dataset,status='COMPLETED',
       mode='UNIFIED_CONDITIONAL_ADAPTED',venus_approval_recorded=False,
       trials_completed=12,winner_trial=selected['trial'],test_scored=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['FDCompCN','Elliptic'],required=True)
    args=parser.parse_args()
    try:main(args.dataset)
    except BaseException as e:
        if ATTEMPT_RECORD is not None and ATTEMPT_RECORD.exists():
            previous=json.loads(ATTEMPT_RECORD.read_text())
            if previous.get('status')=='STARTED':
                previous.update(status='FAILED_OR_INTERRUPTED',error=repr(e))
                atomic_json(ATTEMPT_RECORD,previous)
        print(f'EOGFD_{args.dataset}_STAGE3_STOP_ERROR={e!r}',flush=True)
        traceback.print_exc()
        sys.exit(2)
