#!/usr/bin/env python3
import os, sys, gc, csv, json, time, math, random, hashlib, tarfile, subprocess, contextlib, io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import torch
import torch.nn.functional as F
import dgl

WORK = Path('/workspace/bwgnn_vast')
AUDIT_ROOT = Path('/workspace/dataset_audit')
SPLIT_PATH = WORK / 'splits/elliptic_chronological_nested_splits.npz'

BASE = WORK / 'results/bwgnn/elliptic'
TUNING = BASE / 'tuning'
FINAL = BASE / 'final'
RUN_DIR = FINAL / 'runs'
TIME_DIR = FINAL / 'epoch_times'
CKPT_DIR = WORK / 'checkpoints/elliptic'
EVIDENCE = WORK / 'evidence/elliptic'
ARCHIVES = WORK / 'archives'
for d in [TUNING, RUN_DIR, TIME_DIR, CKPT_DIR, EVIDENCE, ARCHIVES, SPLIT_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORK/'adapters'))
from BWGNN_gpu import BWGNN

EXPECTED_COMMIT = 'de0631f039bbd19c1890b483cc01f1007f596af7'
RAW_HASHES = {
    'elliptic_txs_features.csv':'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
    'elliptic_txs_classes.csv':'93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
    'elliptic_txs_edgelist.csv':'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e',
}
EXPECTED_SPLIT_SHA = '1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc'

TRIALS = [
    {'hidden':64,  'order':2, 'lr':1e-2, 'weight_decay':0.0},
    {'hidden':64,  'order':2, 'lr':5e-3, 'weight_decay':0.0},
    {'hidden':64,  'order':1, 'lr':1e-2, 'weight_decay':0.0},
    {'hidden':64,  'order':3, 'lr':1e-2, 'weight_decay':0.0},
    {'hidden':32,  'order':2, 'lr':1e-2, 'weight_decay':0.0},
    {'hidden':32,  'order':3, 'lr':5e-3, 'weight_decay':0.0},
    {'hidden':128, 'order':2, 'lr':5e-3, 'weight_decay':0.0},
    {'hidden':128, 'order':3, 'lr':1e-3, 'weight_decay':0.0},
    {'hidden':64,  'order':2, 'lr':5e-3, 'weight_decay':1e-5},
    {'hidden':64,  'order':2, 'lr':5e-3, 'weight_decay':1e-4},
    {'hidden':64,  'order':3, 'lr':1e-3, 'weight_decay':1e-4},
    {'hidden':32,  'order':2, 'lr':1e-3, 'weight_decay':1e-3},
]
MAX_EPOCHS, PATIENCE = 100, 20
SEEDS = [2,42,72]
RATIOS = ['TR40','TR30','TR20','TR10']
WARMUPS, REPEATS = 3, 10
DEVICE = torch.device('cuda:0')

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def choose_threshold(y,p):
    c=[]
    for t in np.arange(0.01,1.00,0.01):
        pred=(p>=t).astype(np.int64)
        c.append((
            f1_score(y,pred,average='macro',zero_division=0),
            recall_score(y,pred,pos_label=1,zero_division=0),
            -abs(float(t)-0.5), float(t)
        ))
    return max(c)[3]

def metrics(y,p,t):
    pred=(p>=t).astype(np.int64)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else 0.0
    spec=tn/(tn+fp) if tn+fp else 0.0
    return {
        'auprc':float(average_precision_score(y,p)),
        'auroc':float(roc_auc_score(y,p)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'fraud_precision':float(precision_score(y,pred,pos_label=1,zero_division=0)),
        'fraud_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),
        'fraud_f1':float(f1_score(y,pred,pos_label=1,zero_division=0)),
        'gmean':float(math.sqrt(sens*spec)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)
    }

print('===== BWGNN VAST — ELLIPTIC 166-FEATURE FINAL =====', flush=True)

# ------------------------------------------------------------------
# 1. Source + hardware gate
# ------------------------------------------------------------------
commit=subprocess.check_output(
    ['git','-C',str(WORK/'repo/Rethinking-Anomaly-Detection'),'rev-parse','HEAD'],
    text=True
).strip()
assert commit == EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'A6000' in torch.cuda.get_device_name(0).upper()

# Resolve three canonical CSVs by exact SHA256.
raw={}
all_csv=list(AUDIT_ROOT.rglob('*.csv'))
for expected_name, expected_sha in RAW_HASHES.items():
    match=None
    # prefer matching basename
    for p in all_csv:
        if p.name==expected_name and sha256_file(p)==expected_sha:
            match=p; break
    # content-hash fallback under Elliptic-ish paths
    if match is None:
        for p in all_csv:
            if 'elliptic' not in str(p).lower():
                continue
            try:
                if sha256_file(p)==expected_sha:
                    match=p; break
            except Exception:
                pass
    if match is None:
        raise FileNotFoundError(f'Canonical {expected_name} not found by SHA256 {expected_sha}')
    raw[expected_name]=match
    print(expected_name, match, expected_sha, flush=True)

FEATURES=raw['elliptic_txs_features.csv']
CLASSES=raw['elliptic_txs_classes.csv']
EDGES=raw['elliptic_txs_edgelist.csv']

# ------------------------------------------------------------------
# 2. Load canonical 166-feature view
# ------------------------------------------------------------------
load_t0=time.perf_counter()
feat_df=pd.read_csv(FEATURES,header=None)
class_df=pd.read_csv(CLASSES)
edge_df=pd.read_csv(EDGES)
assert feat_df.shape==(203769,167), feat_df.shape
assert len(edge_df)==234355, len(edge_df)

tx_ids=feat_df.iloc[:,0].astype(np.int64).to_numpy()
timesteps=feat_df.iloc[:,1].astype(np.int64).to_numpy()

# FINAL v4.4 convention: drop transaction ID ONLY.
# time_step remains among model features and also drives chronology.
x_np=feat_df.iloc[:,1:].to_numpy(dtype=np.float32)
assert x_np.shape==(203769,166), x_np.shape
assert np.array_equal(x_np[:,0].astype(np.int64),timesteps)
print('MODEL_FEATURES=166',flush=True)
print('time_step retained as feature: YES',flush=True)

id_to_idx={int(tx):i for i,tx in enumerate(tx_ids)}
labels_np=np.full(len(tx_ids),-1,dtype=np.int64)
class_id_col,class_col=class_df.columns[:2]
for tx,raw_label in zip(class_df[class_id_col],class_df[class_col]):
    idx=id_to_idx[int(tx)]
    v=str(raw_label).strip()
    if v=='1': labels_np[idx]=1
    elif v=='2': labels_np[idx]=0
assert int((labels_np>=0).sum())==46564
assert int((labels_np==1).sum())==4545
assert int((labels_np==0).sum())==42019
assert int((labels_np<0).sum())==157205
print('Known/Fraud/Normal/Unknown:',46564,4545,42019,157205,flush=True)

# ------------------------------------------------------------------
# 3. Chronological frozen split
# ------------------------------------------------------------------
known=labels_np>=0
def known_between(a,b):
    return np.where(known & (timesteps>=a) & (timesteps<=b))[0].astype(np.int64)

splits={
    'TR10':known_between(1,3),
    'TR20':known_between(1,7),
    'TR30':known_between(1,12),
    'TR40':known_between(1,20),
    'val':known_between(21,31),
    'test':known_between(32,49),
}
expected_sizes={'TR10':4543,'TR20':9553,'TR30':13670,'TR40':18889,'val':8726,'test':18949}
for k,n in expected_sizes.items():
    assert len(splits[k])==n,(k,len(splits[k]),n)
assert set(splits['TR10']) <= set(splits['TR20']) <= set(splits['TR30']) <= set(splits['TR40'])
assert not(set(splits['TR40'])&set(splits['val']))
assert not(set(splits['TR40'])&set(splits['test']))
assert not(set(splits['val'])&set(splits['test']))
assert timesteps[splits['TR40']].max()<=20
assert timesteps[splits['val']].min()>=21 and timesteps[splits['val']].max()<=31
assert timesteps[splits['test']].min()>=32

# Prefer an existing exact registered artifact if present; otherwise recreate
# with the original Step 32B metadata and demand the exact file SHA.
registered=None
for p in AUDIT_ROOT.rglob('elliptic_chronological_nested_splits.npz'):
    try:
        if sha256_file(p)==EXPECTED_SPLIT_SHA:
            registered=p; break
    except Exception:
        pass

if registered is not None:
    SPLIT_PATH.write_bytes(registered.read_bytes())
    print('Frozen split source: existing registered artifact', registered, flush=True)
else:
    np.savez_compressed(
        SPLIT_PATH,
        **splits,
        split_seed=np.array([2],dtype=np.int64),
        tr10_max_timestep=np.array([3]),
        tr20_max_timestep=np.array([7]),
        tr30_max_timestep=np.array([12]),
        tr40_max_timestep=np.array([20]),
        val_start_timestep=np.array([21]),
        val_end_timestep=np.array([31]),
        test_start_timestep=np.array([32]),
    )
    print('Frozen split source: deterministically recreated from canonical chronology',flush=True)

split_sha=sha256_file(SPLIT_PATH)
print('Elliptic split SHA256 :',split_sha,flush=True)
print('Registered split SHA256:',EXPECTED_SPLIT_SHA,flush=True)
assert split_sha==EXPECTED_SPLIT_SHA, (
    'Elliptic frozen split hash mismatch. STOP before smoke/tuning/final.',
    split_sha,EXPECTED_SPLIT_SHA
)
print('ELLIPTIC_SPLIT_GATE=PASS',flush=True)

# ------------------------------------------------------------------
# 4. Directed graph + self loops; no reverse edges
# ------------------------------------------------------------------
src_tx=edge_df.iloc[:,0].astype(np.int64).to_numpy()
dst_tx=edge_df.iloc[:,1].astype(np.int64).to_numpy()
src_idx=np.fromiter((id_to_idx[int(x)] for x in src_tx),dtype=np.int64,count=len(src_tx))
dst_idx=np.fromiter((id_to_idx[int(x)] for x in dst_tx),dtype=np.int64,count=len(dst_tx))
graph=dgl.graph((src_idx,dst_idx),num_nodes=203769)
assert graph.num_edges()==234355
graph=dgl.add_self_loop(graph)
assert graph.num_edges()==438124
features=torch.from_numpy(x_np)
labels=torch.from_numpy(labels_np)
graph=graph.to(DEVICE); features=features.to(DEVICE); labels=labels.to(DEVICE)
val_ids=torch.tensor(splits['val'],dtype=torch.long,device=DEVICE)
test_ids=torch.tensor(splits['test'],dtype=torch.long,device=DEVICE)
torch.cuda.synchronize()
load_seconds=time.perf_counter()-load_t0
print('Directed edges:',234355,'| with self-loops:',438124,flush=True)
print('Reverse edges added: NO',flush=True)
print(f'Load/preprocess: {load_seconds:.3f}s',flush=True)

# ------------------------------------------------------------------
# 5. Mandatory 2-epoch smoke on TR40, test untouched
# ------------------------------------------------------------------
print('\n===== 2-EPOCH COMPATIBILITY SMOKE — 166 FEATURES (TEST NOT ACCESSED) =====',flush=True)
seed_all(2)
train_ids=torch.tensor(splits['TR40'],dtype=torch.long,device=DEVICE)
cfg=TRIALS[0]
with contextlib.redirect_stdout(io.StringIO()):
    model=BWGNN(in_feats=166,h_feats=cfg['hidden'],num_classes=2,graph=graph,d=cfg['order']).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],betas=(0.9,0.999),eps=1e-8)
ytr=labels[train_ids]; n0=int((ytr==0).sum()); n1=int((ytr==1).sum())
cw=torch.tensor([1.0,n0/n1],device=DEVICE)
torch.cuda.reset_peak_memory_stats()
smoke=[]
for ep in [1,2]:
    model.train(); torch.cuda.synchronize(); t0=time.perf_counter()
    logits=model(features)
    loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=cw)
    opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
    tr=time.perf_counter()-t0
    model.eval(); torch.cuda.synchronize(); v0=time.perf_counter()
    with torch.no_grad():
        vp=torch.softmax(model(features),1)[val_ids,1]
    torch.cuda.synchronize(); vs=time.perf_counter()-v0
    ap=float(average_precision_score(labels[val_ids].cpu().numpy(),vp.cpu().numpy()))
    smoke.append({'epoch':ep,'loss':float(loss.item()),'train_seconds':tr,'validation_seconds':vs,'val_auprc':ap})
    print(f'smoke epoch={ep} loss={loss.item():.6f} val_AUPRC={ap:.6f} train={tr:.4f}s val={vs:.4f}s',flush=True)
(EVIDENCE/'smoke_166.json').write_text(json.dumps({
    'status':'PASS','model_features':166,'test_accessed':False,
    'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2,
    'epochs':smoke
},indent=2))
del model,opt,logits,vp
gc.collect(); torch.cuda.empty_cache()
print('SMOKE_GATE=PASS',flush=True)

# ------------------------------------------------------------------
# 6. REQUIRED RETUNE: old 165-feature winner is pilot-only.
#    12 fixed TR40 trials, train seed 2, validation AUPRC only.
# ------------------------------------------------------------------
print('\n===== ELLIPTIC 166-FEATURE TR40 RETUNING — 12 FIXED TRIALS =====',flush=True)
tuning_rows=[]
train_ids=torch.tensor(splits['TR40'],dtype=torch.long,device=DEVICE)
for trial_no,cfg in enumerate(TRIALS,1):
    trial_file=TUNING/f'trial_{trial_no:02d}.json'
    if trial_file.exists():
        row=json.loads(trial_file.read_text())
        tuning_rows.append(row)
        print(f"SKIP tuning trial {trial_no:02d} complete | best_val_AUPRC={row['best_val_auprc']:.6f}",flush=True)
        continue

    seed_all(2); gc.collect(); torch.cuda.empty_cache()
    with contextlib.redirect_stdout(io.StringIO()):
        model=BWGNN(in_feats=166,h_feats=cfg['hidden'],num_classes=2,graph=graph,d=cfg['order']).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],betas=(0.9,0.999),eps=1e-8)
    ytr=labels[train_ids]; n0=int((ytr==0).sum()); n1=int((ytr==1).sum())
    cw=torch.tensor([1.0,n0/n1],device=DEVICE)

    best=-1.0; best_epoch=-1; stale=0
    start=time.perf_counter()
    for ep in range(1,MAX_EPOCHS+1):
        model.train()
        logits=model(features)
        loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=cw)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite tuning loss trial {trial_no} epoch {ep}')
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vp=torch.softmax(model(features),1)[val_ids,1]
        ap=float(average_precision_score(labels[val_ids].cpu().numpy(),vp.cpu().numpy()))
        if ap>best:
            best=ap; best_epoch=ep; stale=0
        else:
            stale+=1
        if ep==1 or ep%10==0 or stale>=PATIENCE:
            print(f'trial={trial_no:02d}/12 epoch={ep:03d} val_AP={ap:.6f} best={best:.6f}@{best_epoch} stale={stale}',flush=True)
        if stale>=PATIENCE:
            break
    row={
        'trial':trial_no,**cfg,'model_features':166,'train_seed':2,
        'split':'TR40','best_val_auprc':float(best),'best_epoch':int(best_epoch),
        'completed_epochs':int(ep),'wall_seconds':float(time.perf_counter()-start),
        'test_accessed':False
    }
    trial_file.write_text(json.dumps(row,indent=2))
    tuning_rows.append(row)
    del model,opt,logits,vp
    gc.collect(); torch.cuda.empty_cache()

winner=max(tuning_rows,key=lambda r:(r['best_val_auprc'],-r['trial']))
frozen={
    'dataset':'Elliptic','model':'BWGNN','model_features':166,
    'reason_for_retune':'Master Plan v4.4 changed final canonical view from earlier 165-feature pilot to locked 166-feature view.',
    'selection_metric':'validation AUPRC','test_accessed':False,
    'winner':winner,'trials_completed':len(tuning_rows)
}
(TUNING/'elliptic166_tuning_trials.json').write_text(json.dumps(tuning_rows,indent=2))
with (TUNING/'elliptic166_tuning_trials.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(tuning_rows[0].keys())); w.writeheader(); w.writerows(tuning_rows)
(TUNING/'elliptic166_frozen_config.json').write_text(json.dumps(frozen,indent=2))
print('TUNING_GATE=PASS',flush=True)
print('WINNER:',winner,flush=True)

HIDDEN=int(winner['hidden']); ORDER=int(winner['order'])
LR=float(winner['lr']); WEIGHT_DECAY=float(winner['weight_decay'])

# ------------------------------------------------------------------
# 7. Final 12 controlled runs with frozen 166-feature winner
# ------------------------------------------------------------------
print('\n===== ELLIPTIC 166-FEATURE FINAL GRID =====',flush=True)
for ratio in RATIOS:
    train_ids=torch.tensor(splits[ratio],dtype=torch.long,device=DEVICE)
    for seed in SEEDS:
        key=f'{ratio}_seed{seed}'
        result_file=RUN_DIR/f'{key}.json'
        epoch_file=TIME_DIR/f'{key}_epoch_times.csv'
        ckpt=CKPT_DIR/f'{key}.pt'

        if result_file.exists():
            print(f'SKIP complete {key}',flush=True)
            continue
        epoch_file.unlink(missing_ok=True); ckpt.unlink(missing_ok=True)

        print(f'\nSTART {key}',flush=True)
        seed_all(seed); gc.collect(); torch.cuda.empty_cache()
        with contextlib.redirect_stdout(io.StringIO()):
            model=BWGNN(in_feats=166,h_feats=HIDDEN,num_classes=2,graph=graph,d=ORDER).to(DEVICE)
        params=sum(p.numel() for p in model.parameters())
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,betas=(0.9,0.999),eps=1e-8)
        ytr=labels[train_ids]; n0=int((ytr==0).sum()); n1=int((ytr==1).sum())
        cw=torch.tensor([1.0,n0/n1],device=DEVICE)

        best=-1.0; best_epoch=-1; stale=0
        train_times=[]; val_times=[]
        torch.cuda.reset_peak_memory_stats()
        with epoch_file.open('w',newline='') as ef:
            ew=csv.DictWriter(ef,fieldnames=[
                'epoch','loss','train_seconds','validation_seconds',
                'cumulative_train_seconds','val_auprc','best_val_auprc','best_epoch'
            ])
            ew.writeheader(); cumulative=0.0
            for ep in range(1,MAX_EPOCHS+1):
                model.train(); torch.cuda.synchronize(); t0=time.perf_counter()
                logits=model(features)
                loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=cw)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss {key} epoch {ep}')
                opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
                tr=time.perf_counter()-t0; train_times.append(tr); cumulative+=tr

                model.eval(); torch.cuda.synchronize(); v0=time.perf_counter()
                with torch.no_grad():
                    val_logits=model(features)
                    vp=torch.softmax(val_logits,1)[val_ids,1]
                torch.cuda.synchronize(); vs=time.perf_counter()-v0; val_times.append(vs)
                ap=float(average_precision_score(labels[val_ids].cpu().numpy(),vp.cpu().numpy()))
                if ap>best:
                    best=ap; best_epoch=ep; stale=0
                    torch.save(model.state_dict(),ckpt)
                else:
                    stale+=1
                ew.writerow({
                    'epoch':ep,'loss':float(loss.item()),'train_seconds':tr,
                    'validation_seconds':vs,'cumulative_train_seconds':cumulative,
                    'val_auprc':ap,'best_val_auprc':best,'best_epoch':best_epoch
                }); ef.flush()
                if ep==1 or ep%10==0 or stale>=PATIENCE:
                    print(f'{key} epoch={ep:03d} loss={loss.item():.5f} val_AP={ap:.6f} best={best:.6f}@{best_epoch} train={tr:.4f}s val={vs:.4f}s stale={stale}',flush=True)
                if stale>=PATIENCE: break

        train_times=np.asarray(train_times,float); val_times=np.asarray(val_times,float)
        model.load_state_dict(torch.load(ckpt,map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            for _ in range(WARMUPS):
                _=model(features); torch.cuda.synchronize()
            latency=[]
            for _ in range(REPEATS):
                torch.cuda.synchronize(); q=time.perf_counter()
                _=model(features); torch.cuda.synchronize()
                latency.append((time.perf_counter()-q)*1000.0)
            prob=torch.softmax(model(features),1)[:,1]

        vy=labels[val_ids].cpu().numpy(); vp=prob[val_ids].cpu().numpy()
        threshold=choose_threshold(vy,vp)
        ty=labels[test_ids].cpu().numpy(); tp=prob[test_ids].cpu().numpy()
        m=metrics(ty,tp,threshold)

        result={
            'model':'BWGNN','dataset':'Elliptic','model_features':166,
            'chronology_preserved':True,'reverse_edges_added':False,
            'unknown_nodes_retained':True,'unknown_labels_supervised':False,
            'compatibility':'PASS_WITH_SHARED_ADAPTER',
            'hardware_profile_id':'vast-a6000-instance-51035671',
            'repository_commit':commit,'raw_sha256':RAW_HASHES,
            'split_sha256':split_sha,'split':ratio,'split_seed':2,'train_seed':seed,
            'hidden':HIDDEN,'order':ORDER,'learning_rate':LR,'weight_decay':WEIGHT_DECAY,
            'completed_epochs':int(len(train_times)),'best_epoch':int(best_epoch),
            'best_val_auprc':float(best),'parameter_count':int(params),
            'load_preprocess_seconds':float(load_seconds),
            'total_train_seconds':float(train_times.sum()),
            'mean_epoch_train_seconds':float(train_times.mean()),
            'median_epoch_train_seconds':float(np.median(train_times)),
            'std_epoch_train_seconds':float(train_times.std(ddof=1) if len(train_times)>1 else 0),
            'total_validation_seconds':float(val_times.sum()),
            'mean_validation_seconds':float(val_times.mean()),
            'inference_latency_mean_ms':float(np.mean(latency)),
            'inference_latency_std_ms':float(np.std(latency,ddof=1)),
            'peak_gpu_memory_mb':float(torch.cuda.max_memory_allocated()/1024**2),
            'validation_selected_threshold':float(threshold),
            'test_evaluated_once_after_freeze':True,**m
        }
        result_file.write_text(json.dumps(result,indent=2))
        ckpt.unlink(missing_ok=True)
        print(f"DONE {key} | test_AUPRC={m['auprc']:.6f} AUROC={m['auroc']:.6f} MacroF1={m['macro_f1']:.6f} | {result['mean_epoch_train_seconds']:.4f}s/epoch",flush=True)
        del model,opt,prob
        gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 8. Aggregate + evidence + archive
# ------------------------------------------------------------------
results=[]
for ratio in RATIOS:
    for seed in SEEDS:
        p=RUN_DIR/f'{ratio}_seed{seed}.json'
        assert p.exists(),f'Missing {p}'
        results.append(json.loads(p.read_text()))
(FINAL/'all_runs.json').write_text(json.dumps(results,indent=2))
with (FINAL/'all_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)

summary={
    'model':'BWGNN','dataset':'Elliptic','status':'COMPLETE',
    'model_features':166,'tuning_trials':12,'winner':winner,
    'completed_runs':len(results),'ratios':{}
}
metric_keys=[
    'auprc','auroc','macro_f1','fraud_precision','fraud_recall','fraud_f1','gmean',
    'mean_epoch_train_seconds','median_epoch_train_seconds','total_train_seconds',
    'inference_latency_mean_ms','peak_gpu_memory_mb'
]
for ratio in RATIOS:
    rows=[r for r in results if r['split']==ratio]
    summary['ratios'][ratio]={}
    for k in metric_keys:
        x=np.asarray([r[k] for r in rows],float)
        summary['ratios'][ratio][k]={'mean':float(x.mean()),'sd':float(x.std(ddof=1))}
(FINAL/'summary.json').write_text(json.dumps(summary,indent=2))

manifest={
    'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'model':'BWGNN','dataset':'Elliptic','benchmark_mode':'unified controlled',
    'compatibility':'INPUT ADAPTER ELIGIBLE','repository_commit':commit,
    'dataset_hashes':RAW_HASHES,'split_sha256':split_sha,
    'model_features':166,'time_step_used_as_model_feature':True,
    'time_step_used_for_split':True,'chronology_preserved':True,
    'canonical_nodes':203769,'canonical_directed_edges':234355,
    'edges_after_self_loops':438124,'known_labelled_nodes':46564,
    'unknown_nodes':157205,'fraud_nodes':4545,'normal_nodes':42019,
    'unknown_nodes_retained':True,'unknown_labels_used_for_supervision':False,
    'reverse_edges_added':False,'architecture_changed':False,
    'split_seed':2,'training_seeds':SEEDS,'training_ratios':RATIOS,
    'tuning_trials':12,'frozen_configuration':winner,
    'final_runs':12,'training_timing_excludes_validation':True,
    'validation_threshold_only':True,'test_used_for_tuning':False,
    'test_used_for_checkpoint_selection':False,'test_used_for_threshold_selection':False,
    'results_summary':summary
}
(EVIDENCE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))

archive=ARCHIVES/'bwgnn_elliptic166_vast_a6000_final.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p,arc in [
        (TUNING,'results/elliptic/tuning'),
        (FINAL,'results/elliptic/final'),
        (EVIDENCE,'evidence/elliptic'),
        (SPLIT_PATH,'splits/elliptic_chronological_nested_splits.npz'),
        (WORK/'adapters/BWGNN_gpu.py','adapters/BWGNN_gpu.py'),
        (Path(__file__),'runner/06_bwgnn_elliptic166_final.py')
    ]:
        t.add(p,arcname=str(arc))
digest=sha256_file(archive)
(archive.with_suffix(archive.suffix+'.sha256')).write_text(f'{digest}  {archive.name}\n')

print('\n===== ELLIPTIC 166-FEATURE FINAL COMPLETE =====',flush=True)
print(json.dumps(summary,indent=2),flush=True)
print('ARCHIVE:',archive,flush=True)
print('SHA256:',digest,flush=True)
