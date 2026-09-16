#!/usr/bin/env python3
import os, sys, gc, csv, json, time, math, random, hashlib, tarfile, subprocess, contextlib, io
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import torch
import torch.nn.functional as F
import dgl
from dgl.data.utils import load_graphs

WORK = Path('/workspace/bwgnn_vast')
AUDIT_ROOT = Path('/workspace/dataset_audit')
SPLIT_PATH = WORK / 'splits/fdcompcn_seed2_nested_splits.npz'
ROOT = WORK / 'results/bwgnn/fdcompcn/final'
RUN_DIR, TIME_DIR = ROOT/'runs', ROOT/'epoch_times'
CKPT_DIR = WORK / 'checkpoints/fdcompcn'
EVIDENCE = WORK / 'evidence/fdcompcn'
ARCHIVES = WORK / 'archives'
for d in [RUN_DIR, TIME_DIR, CKPT_DIR, EVIDENCE, ARCHIVES, SPLIT_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORK/'adapters'))
from BWGNN_gpu import BWGNN

EXPECTED_COMMIT = 'de0631f039bbd19c1890b483cc01f1007f596af7'
EXPECTED_DATA_SHA = 'e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa'
EXPECTED_SPLIT_SHA = '5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168'

HIDDEN, ORDER, LR, WEIGHT_DECAY = 64, 1, 0.01, 0.0
MAX_EPOCHS, PATIENCE = 100, 20
SEEDS, RATIOS = [2,42,72], ['TR40','TR30','TR20','TR10']
WARMUPS, REPEATS = 3, 10
DEVICE = torch.device('cuda:0')

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def choose_threshold(y,p):
    choices=[]
    for t in np.arange(0.01,1.00,0.01):
        pred=(p>=t).astype(np.int64)
        choices.append((
            f1_score(y,pred,average='macro',zero_division=0),
            recall_score(y,pred,pos_label=1,zero_division=0),
            -abs(float(t)-0.5),
            float(t)
        ))
    return max(choices)[3]

def calc_metrics(y,p,t):
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

print('===== BWGNN VAST — FDCOMPCN =====', flush=True)

# ------------------------------------------------------------------
# 1. Frozen source/hardware gate
# ------------------------------------------------------------------
commit=subprocess.check_output(
    ['git','-C',str(WORK/'repo/Rethinking-Anomaly-Detection'),'rev-parse','HEAD'],
    text=True
).strip()
assert commit==EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'A6000' in torch.cuda.get_device_name(0).upper()

# Resolve comp.dgl by content hash, not assumed folder.
candidates=sorted(AUDIT_ROOT.rglob('comp.dgl'))
DATA_PATH=None
seen=[]
for p in candidates:
    h=sha256_file(p)
    seen.append((str(p),h,p.stat().st_size))
    if h==EXPECTED_DATA_SHA:
        DATA_PATH=p
        break

# Fallback: if renamed, scan only reasonably sized files under fdcompcn folders.
if DATA_PATH is None:
    for p in sorted(AUDIT_ROOT.rglob('*')):
        if not p.is_file() or 'fdcompcn' not in str(p).lower():
            continue
        try:
            h=sha256_file(p)
        except Exception:
            continue
        seen.append((str(p),h,p.stat().st_size))
        if h==EXPECTED_DATA_SHA:
            DATA_PATH=p
            break

if DATA_PATH is None:
    raise FileNotFoundError(
        'Canonical FDCompCN comp.dgl was not found by SHA256. '
        f'Expected {EXPECTED_DATA_SHA}. Candidates={seen[:30]}'
    )

data_sha=sha256_file(DATA_PATH)
print('Data:',DATA_PATH,flush=True)
print('Dataset SHA256:',data_sha,flush=True)
print('GPU:',torch.cuda.get_device_name(0),flush=True)
print('Frozen config:',{'hidden':HIDDEN,'order':ORDER,'lr':LR,'weight_decay':WEIGHT_DECAY},flush=True)

# ------------------------------------------------------------------
# 2. Load canonical relation-preserving asset
# ------------------------------------------------------------------
load_t0=time.perf_counter()
graphs,_=load_graphs(str(DATA_PATH))
assert len(graphs)>=1
source_g=graphs[0]
assert source_g.num_nodes()==5317, source_g.num_nodes()
assert source_g.num_edges()==30752, source_g.num_edges()

# Labels/features are preserved through the approved deterministic adapter.
raw_labels=source_g.ndata['label']
if isinstance(raw_labels, dict):
    assert len(raw_labels)==1
    raw_labels=next(iter(raw_labels.values()))
labels_cpu=raw_labels.argmax(1).long() if raw_labels.ndim==2 else raw_labels.reshape(-1).long()

normal=int((labels_cpu==0).sum().item())
fraud=int((labels_cpu==1).sum().item())
assert (normal,fraud)==(4758,559),(normal,fraud)

# ------------------------------------------------------------------
# 3. Recreate frozen COMP8851 nested split and VERIFY exact hash
# ------------------------------------------------------------------
labels_np=labels_cpu.cpu().numpy().astype(int)
all_ids=np.arange(len(labels_np),dtype=np.int64)

remaining,test_ids=train_test_split(
    all_ids,test_size=0.40,stratify=labels_np,
    random_state=2,shuffle=True
)
tr40,val_ids=train_test_split(
    remaining,test_size=(1/3),stratify=labels_np[remaining],
    random_state=2,shuffle=True
)
tr30,_=train_test_split(
    tr40,train_size=0.75,stratify=labels_np[tr40],
    random_state=2,shuffle=True
)
tr20,_=train_test_split(
    tr30,train_size=(2/3),stratify=labels_np[tr30],
    random_state=2,shuffle=True
)
tr10,_=train_test_split(
    tr20,train_size=0.50,stratify=labels_np[tr20],
    random_state=2,shuffle=True
)
splits={
    'TR40':np.sort(tr40),
    'TR30':np.sort(tr30),
    'TR20':np.sort(tr20),
    'TR10':np.sort(tr10),
    'val':np.sort(val_ids),
    'test':np.sort(test_ids),
}
expected_sizes={'TR40':2126,'TR30':1594,'TR20':1062,'TR10':531,'val':1064,'test':2127}
for k,n in expected_sizes.items():
    assert len(splits[k])==n,(k,len(splits[k]),n)
assert set(splits['TR10']) <= set(splits['TR20']) <= set(splits['TR30']) <= set(splits['TR40'])
assert not (set(splits['TR40']) & set(splits['val']))
assert not (set(splits['TR40']) & set(splits['test']))
assert not (set(splits['val']) & set(splits['test']))

# Match the original registered artifact byte-for-byte.
np.savez_compressed(SPLIT_PATH, **splits, seed=np.array([2]))
split_sha=sha256_file(SPLIT_PATH)
print('Recreated split SHA256 :',split_sha,flush=True)
print('Registered split SHA256:',EXPECTED_SPLIT_SHA,flush=True)
assert split_sha==EXPECTED_SPLIT_SHA, (
    'FDCompCN frozen split hash mismatch. Training STOPPED before smoke/final runs.',
    split_sha, EXPECTED_SPLIT_SHA
)
print('FDCOMPCN_SPLIT_GATE=PASS',flush=True)

# ------------------------------------------------------------------
# 4. Approved A1 deterministic representation adapter for BWGNN
#    Preserve nodes/features/labels/edges; convert heterograph -> homogeneous
#    and add one self-loop per node, exactly as prior validated BWGNN path.
# ------------------------------------------------------------------
graph=dgl.to_homogeneous(source_g, ndata=['feature','label'])
assert graph.num_nodes()==5317
assert graph.num_edges()==30752
graph=dgl.add_self_loop(graph)
assert graph.num_edges()==36069

features=graph.ndata['feature'].float()
raw_h_labels=graph.ndata['label']
labels=raw_h_labels.argmax(1).long() if raw_h_labels.ndim==2 else raw_h_labels.reshape(-1).long()
assert tuple(features.shape)==(5317,57),tuple(features.shape)
assert int((labels==0).sum())==4758 and int((labels==1).sum())==559

graph=graph.to(DEVICE)
features=features.to(DEVICE)
labels=labels.to(DEVICE)
val_ids_t=torch.tensor(splits['val'],dtype=torch.long,device=DEVICE)
test_ids_t=torch.tensor(splits['test'],dtype=torch.long,device=DEVICE)
torch.cuda.synchronize()
load_seconds=time.perf_counter()-load_t0

print('Source graph:',5317,'nodes |',30752,'stored relation entries',flush=True)
print('BWGNN adapted graph:',graph.num_nodes(),'nodes |',graph.num_edges(),'edges incl. self-loops',flush=True)
print('Normal/Fraud:',normal,fraud,flush=True)
print('Features:',features.shape[1],flush=True)
print(f'Load/preprocess: {load_seconds:.3f}s',flush=True)

manifest={
    'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'model':'BWGNN','dataset':'FDCompCN',
    'compatibility':'PASS_WITH_SHARED_ADAPTER',
    'adapter_class':'A1 deterministic representation',
    'adapter':'DGL heterogeneous-to-homogeneous + one self-loop per node',
    'architecture_changed':False,
    'nodes_preserved':True,'features_preserved':True,'labels_preserved':True,
    'original_edges_preserved':True,
    'repository_commit':commit,
    'dataset_path':str(DATA_PATH),'dataset_sha256':data_sha,
    'canonical_nodes':5317,'canonical_stored_edges':30752,
    'adapted_edges_with_self_loops':36069,
    'features':57,'normal':4758,'fraud':559,
    'split_seed':2,'split_sha256':split_sha,
    'registered_split_sha256':EXPECTED_SPLIT_SHA,
    'split_sizes':expected_sizes,
    'gpu':torch.cuda.get_device_name(0),
    'torch':torch.__version__,'dgl':dgl.__version__,
    'frozen_config':{'hidden':HIDDEN,'order':ORDER,'lr':LR,'weight_decay':WEIGHT_DECAY},
    'train_seeds':SEEDS,'ratios':RATIOS,
    'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
    'checkpoint_metric':'validation AUPRC',
    'threshold_rule':'validation Macro-F1; ties fraud recall then closest to 0.5',
    'test_isolation':True,'load_preprocess_seconds':load_seconds,
}
(EVIDENCE/'input_manifest.json').write_text(json.dumps(manifest,indent=2))

# ------------------------------------------------------------------
# 5. Two-epoch compatibility smoke; test untouched
# ------------------------------------------------------------------
print('\n===== 2-EPOCH COMPATIBILITY SMOKE (TEST NOT ACCESSED) =====',flush=True)
seed_all(2)
train_ids=torch.tensor(splits['TR40'],dtype=torch.long,device=DEVICE)
with contextlib.redirect_stdout(io.StringIO()):
    model=BWGNN(57,HIDDEN,2,graph,d=ORDER).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,betas=(0.9,0.999),eps=1e-8)
ytr=labels[train_ids]
n0=int((ytr==0).sum()); n1=int((ytr==1).sum())
weight=torch.tensor([1.0,n0/n1],device=DEVICE)
torch.cuda.reset_peak_memory_stats()
smoke=[]
for epoch in [1,2]:
    model.train()
    torch.cuda.synchronize(); t0=time.perf_counter()
    logits=model(features)
    loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=weight)
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); tr=time.perf_counter()-t0

    model.eval()
    torch.cuda.synchronize(); v0=time.perf_counter()
    with torch.no_grad():
        vp=torch.softmax(model(features),1)[val_ids_t,1]
    torch.cuda.synchronize(); vs=time.perf_counter()-v0
    ap=float(average_precision_score(labels[val_ids_t].cpu().numpy(),vp.cpu().numpy()))
    smoke.append({'epoch':epoch,'loss':float(loss.item()),'train_seconds':tr,'validation_seconds':vs,'val_auprc':ap})
    print(f'smoke epoch={epoch} loss={loss.item():.6f} val_AUPRC={ap:.6f} train={tr:.4f}s val={vs:.4f}s',flush=True)

(EVIDENCE/'smoke.json').write_text(json.dumps({
    'status':'PASS','test_accessed':False,
    'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2,
    'epochs':smoke
},indent=2))
del model,opt,logits,vp
gc.collect(); torch.cuda.empty_cache()
print('SMOKE_GATE=PASS',flush=True)

# ------------------------------------------------------------------
# 6. Final 12-run controlled grid; resume-safe
# ------------------------------------------------------------------
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
        epoch_file.unlink(missing_ok=True)
        ckpt.unlink(missing_ok=True)

        print(f'\nSTART {key}',flush=True)
        seed_all(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            model=BWGNN(57,HIDDEN,2,graph,d=ORDER).to(DEVICE)
        params=sum(p.numel() for p in model.parameters())
        opt=torch.optim.Adam(
            model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,
            betas=(0.9,0.999),eps=1e-8
        )
        ytr=labels[train_ids]
        n0=int((ytr==0).sum()); n1=int((ytr==1).sum())
        weight=torch.tensor([1.0,n0/n1],device=DEVICE)

        best_ap=-1.0; best_epoch=-1; stale=0
        train_times=[]; val_times=[]
        torch.cuda.reset_peak_memory_stats()

        with epoch_file.open('w',newline='') as ef:
            ew=csv.DictWriter(ef,fieldnames=[
                'epoch','loss','train_seconds','validation_seconds',
                'cumulative_train_seconds','val_auprc',
                'best_val_auprc','best_epoch'
            ])
            ew.writeheader()
            cumulative=0.0
            for epoch in range(1,MAX_EPOCHS+1):
                model.train()
                torch.cuda.synchronize(); t0=time.perf_counter()
                logits=model(features)
                loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss {key} epoch {epoch}')
                opt.zero_grad(); loss.backward(); opt.step()
                torch.cuda.synchronize(); tr=time.perf_counter()-t0
                train_times.append(tr); cumulative+=tr

                model.eval()
                torch.cuda.synchronize(); v0=time.perf_counter()
                with torch.no_grad():
                    val_logits=model(features)
                    vp=torch.softmax(val_logits,1)[val_ids_t,1]
                torch.cuda.synchronize(); vs=time.perf_counter()-v0
                val_times.append(vs)
                vy=labels[val_ids_t].cpu().numpy()
                vp_np=vp.cpu().numpy()
                val_ap=float(average_precision_score(vy,vp_np))

                if val_ap>best_ap:
                    best_ap=val_ap; best_epoch=epoch; stale=0
                    torch.save(model.state_dict(),ckpt)
                else:
                    stale+=1

                ew.writerow({
                    'epoch':epoch,'loss':float(loss.item()),
                    'train_seconds':tr,'validation_seconds':vs,
                    'cumulative_train_seconds':cumulative,
                    'val_auprc':val_ap,'best_val_auprc':best_ap,
                    'best_epoch':best_epoch
                })
                ef.flush()

                if epoch==1 or epoch%10==0 or stale>=PATIENCE:
                    print(
                        f'{key} epoch={epoch:03d} loss={loss.item():.5f} '
                        f'val_AP={val_ap:.6f} best={best_ap:.6f}@{best_epoch} '
                        f'train={tr:.4f}s val={vs:.4f}s stale={stale}',
                        flush=True
                    )
                if stale>=PATIENCE:
                    break

        train_times=np.asarray(train_times,dtype=float)
        val_times=np.asarray(val_times,dtype=float)
        model.load_state_dict(torch.load(ckpt,map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            for _ in range(WARMUPS):
                _=model(features); torch.cuda.synchronize()
            latency=[]
            for _ in range(REPEATS):
                torch.cuda.synchronize(); q=time.perf_counter()
                _=model(features)
                torch.cuda.synchronize()
                latency.append((time.perf_counter()-q)*1000.0)
            prob=torch.softmax(model(features),1)[:,1]

        vy=labels[val_ids_t].cpu().numpy()
        vp=prob[val_ids_t].cpu().numpy()
        threshold=choose_threshold(vy,vp)

        ty=labels[test_ids_t].cpu().numpy()
        tp=prob[test_ids_t].cpu().numpy()
        metrics=calc_metrics(ty,tp,threshold)

        result={
            'model':'BWGNN','dataset':'FDCompCN',
            'compatibility':'PASS_WITH_SHARED_ADAPTER',
            'adapter_class':'A1 deterministic representation',
            'hardware_profile_id':'vast-a6000-instance-51035671',
            'repository_commit':commit,
            'dataset_sha256':data_sha,'split_sha256':split_sha,
            'split':ratio,'split_seed':2,'train_seed':seed,
            'hidden':HIDDEN,'order':ORDER,
            'learning_rate':LR,'weight_decay':WEIGHT_DECAY,
            'completed_epochs':int(len(train_times)),
            'best_epoch':int(best_epoch),'best_val_auprc':float(best_ap),
            'parameter_count':int(params),
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
            'test_evaluated_once_after_freeze':True,
            **metrics
        }
        result_file.write_text(json.dumps(result,indent=2))
        ckpt.unlink(missing_ok=True)
        print(
            f"DONE {key} | test_AUPRC={metrics['auprc']:.6f} "
            f"AUROC={metrics['auroc']:.6f} MacroF1={metrics['macro_f1']:.6f} "
            f"| {result['mean_epoch_train_seconds']:.4f}s/epoch",
            flush=True
        )
        del model,opt,prob
        gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 7. Aggregate and archive
# ------------------------------------------------------------------
results=[]
for ratio in RATIOS:
    for seed in SEEDS:
        p=RUN_DIR/f'{ratio}_seed{seed}.json'
        assert p.exists(),f'Missing {p}'
        results.append(json.loads(p.read_text()))

(ROOT/'all_runs.json').write_text(json.dumps(results,indent=2))
fields=list(results[0].keys())
with (ROOT/'all_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader(); w.writerows(results)

summary={
    'model':'BWGNN','dataset':'FDCompCN','status':'COMPLETE',
    'compatibility':'PASS_WITH_SHARED_ADAPTER',
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
        x=np.asarray([r[k] for r in rows],dtype=float)
        summary['ratios'][ratio][k]={'mean':float(x.mean()),'sd':float(x.std(ddof=1))}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))

archive=ARCHIVES/'bwgnn_fdcompcn_vast_a6000_final.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p,arc in [
        (ROOT,'results/fdcompcn'),
        (EVIDENCE,'evidence/fdcompcn'),
        (SPLIT_PATH,'splits/fdcompcn_seed2_nested_splits.npz'),
        (WORK/'adapters/BWGNN_gpu.py','adapters/BWGNN_gpu.py'),
        (Path(__file__),'runner/05_bwgnn_fdcompcn_final.py')
    ]:
        t.add(p,arcname=str(arc))

digest=sha256_file(archive)
(archive.with_suffix(archive.suffix+'.sha256')).write_text(f'{digest}  {archive.name}\n')

print('\n===== FDCOMPCN FINAL COMPLETE =====',flush=True)
print(json.dumps(summary,indent=2),flush=True)
print('ARCHIVE:',archive,flush=True)
print('SHA256:',digest,flush=True)
