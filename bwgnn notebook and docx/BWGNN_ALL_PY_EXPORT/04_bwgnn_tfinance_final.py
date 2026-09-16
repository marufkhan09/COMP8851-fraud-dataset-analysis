#!/usr/bin/env python3
import os, sys, gc, csv, json, time, math, random, hashlib, tarfile, subprocess, contextlib, io
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn.functional as F
import dgl
from dgl.data.utils import load_graphs

WORK = Path('/workspace/bwgnn_vast')
RAW_ROOT = Path('/workspace/dataset_audit/raw/tfinance')
SPLIT_PATH = WORK / 'splits/tfinance_seed2_nested_splits.npz'
ROOT = WORK / 'results/bwgnn/tfinance/final'
RUN_DIR, TIME_DIR = ROOT/'runs', ROOT/'epoch_times'
CKPT_DIR = WORK / 'checkpoints/tfinance'
EVIDENCE = WORK / 'evidence/tfinance'
ARCHIVES = WORK / 'archives'
for d in [RUN_DIR, TIME_DIR, CKPT_DIR, EVIDENCE, ARCHIVES, SPLIT_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORK/'adapters'))
from BWGNN_gpu import BWGNN

EXPECTED_COMMIT = 'de0631f039bbd19c1890b483cc01f1007f596af7'
EXPECTED_DATA_SHA = 'b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
EXPECTED_REGISTERED_SPLIT_SHA = 'de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'

HIDDEN, ORDER, LR, WEIGHT_DECAY = 64, 3, 0.005, 0.0
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

def choose_threshold(y,p):
    c=[]
    for t in np.arange(0.01,1.00,0.01):
        pred=(p>=t).astype(np.int64)
        c.append((
            f1_score(y,pred,average='macro',zero_division=0),
            recall_score(y,pred,pos_label=1,zero_division=0),
            -abs(float(t)-0.5),
            float(t)
        ))
    return max(c)[3]

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

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

print('===== BWGNN VAST — T-FINANCE =====', flush=True)

# ------------------------------------------------------------------
# 1. Frozen source / hardware gate
# ------------------------------------------------------------------
commit=subprocess.check_output(
    ['git','-C',str(WORK/'repo/Rethinking-Anomaly-Detection'),'rev-parse','HEAD'],
    text=True
).strip()
assert commit==EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'A6000' in torch.cuda.get_device_name(0).upper()

# Resolve canonical T-Finance BY HASH, not folder/file name.
candidates=[p for p in RAW_ROOT.rglob('*') if p.is_file()]
DATA_PATH=None
candidate_records=[]
for p in sorted(candidates):
    try:
        h=sha256_file(p)
    except Exception:
        continue
    candidate_records.append((str(p),h,p.stat().st_size))
    if h==EXPECTED_DATA_SHA:
        DATA_PATH=p
        break
if DATA_PATH is None:
    raise FileNotFoundError(
        'Canonical T-Finance file not found by SHA256 under '
        f'{RAW_ROOT}. Expected {EXPECTED_DATA_SHA}. Candidates={candidate_records[:20]}'
    )
data_sha=sha256_file(DATA_PATH)
print('Data:',DATA_PATH,flush=True)
print('Dataset SHA256:',data_sha,flush=True)
print('GPU:',torch.cuda.get_device_name(0),flush=True)
print('Frozen config:',{'hidden':HIDDEN,'order':ORDER,'lr':LR,'weight_decay':WEIGHT_DECAY},flush=True)

# ------------------------------------------------------------------
# 2. Load canonical graph and correct labels
# ------------------------------------------------------------------
load_t0=time.perf_counter()
graphs,_=load_graphs(str(DATA_PATH))
assert len(graphs)>=1
g=graphs[0]
features=g.ndata['feature'].float()
raw_labels=g.ndata['label']
labels = raw_labels.argmax(1).long() if raw_labels.ndim==2 else raw_labels.long().squeeze(-1)

assert g.num_nodes()==39357, g.num_nodes()
assert g.num_edges()==42445086, g.num_edges()
assert tuple(features.shape)==(39357,10), tuple(features.shape)
normal=int((labels==0).sum().item())
fraud=int((labels==1).sum().item())
assert (normal,fraud)==(37553,1804), (normal,fraud)
print('Nodes/edges/features:',g.num_nodes(),g.num_edges(),tuple(features.shape),flush=True)
print('Normal/Fraud:',normal,fraud,flush=True)

# ------------------------------------------------------------------
# 3. REQUIRED v4.4 T-Finance split REVALIDATION
#    Recreate with exact previously used generator and assert the
#    byte-level split SHA matches the registered split artifact.
# ------------------------------------------------------------------
labels_np=labels.cpu().numpy().astype(int)
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

expected_sizes={'TR40':15742,'TR30':11806,'TR20':7870,'TR10':3935,'val':7872,'test':15743}
for k,n in expected_sizes.items():
    assert len(splits[k])==n,(k,len(splits[k]),n)
assert set(splits['TR10']) <= set(splits['TR20']) <= set(splits['TR30']) <= set(splits['TR40'])
assert not (set(splits['TR40']) & set(splits['val']))
assert not (set(splits['TR40']) & set(splits['test']))
assert not (set(splits['val']) & set(splits['test']))

np.savez_compressed(
    SPLIT_PATH,
    **splits,
    seed=np.array([2]),
    source_nodes=np.array([39357]),
    source_normal=np.array([37553]),
    source_fraud=np.array([1804]),
)
split_sha=sha256_file(SPLIT_PATH)
print('Revalidated split SHA256:',split_sha,flush=True)
print('Registered split SHA256 :',EXPECTED_REGISTERED_SPLIT_SHA,flush=True)
assert split_sha==EXPECTED_REGISTERED_SPLIT_SHA, (
    'T-Finance split revalidation failed. Training STOPPED before smoke/final runs.',
    split_sha, EXPECTED_REGISTERED_SPLIT_SHA
)
print('TFINANCE_SPLIT_REVALIDATION=PASS',flush=True)
print('Reason frozen tuning remains reusable: canonical source, corrected 1,804 labels, exact split generator and registered split bytes all match.',flush=True)

# ------------------------------------------------------------------
# 4. GPU graph
# ------------------------------------------------------------------
g=g.to(DEVICE)
features=features.to(DEVICE)
labels=labels.to(DEVICE)
val_ids_t=torch.tensor(splits['val'],dtype=torch.long,device=DEVICE)
test_ids_t=torch.tensor(splits['test'],dtype=torch.long,device=DEVICE)
torch.cuda.synchronize()
load_seconds=time.perf_counter()-load_t0

manifest={
    'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'model':'BWGNN','dataset':'T-Finance','compatibility':'PASS_NATIVE',
    'repository_commit':commit,
    'dataset_path':str(DATA_PATH),'dataset_sha256':data_sha,
    'nodes':39357,'stored_edges':42445086,'features':10,
    'normal':37553,'fraud':1804,
    'split_seed':2,'split_sha256':split_sha,
    'registered_split_sha256':EXPECTED_REGISTERED_SPLIT_SHA,
    'split_revalidation_status':'PASS',
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
print(f'Load/preprocess: {load_seconds:.3f}s',flush=True)

# ------------------------------------------------------------------
# 5. Two-epoch compatibility smoke; test untouched.
# ------------------------------------------------------------------
print('\n===== 2-EPOCH COMPATIBILITY SMOKE (TEST NOT ACCESSED) =====',flush=True)
seed_all(2)
train_ids=torch.tensor(splits['TR40'],dtype=torch.long,device=DEVICE)
with contextlib.redirect_stdout(io.StringIO()):
    model=BWGNN(10,HIDDEN,2,g,d=ORDER).to(DEVICE)
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
        # partial timing from an interrupted run must not contaminate a fresh run
        epoch_file.unlink(missing_ok=True)
        ckpt.unlink(missing_ok=True)

        print(f'\nSTART {key}',flush=True)
        seed_all(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            model=BWGNN(10,HIDDEN,2,g,d=ORDER).to(DEVICE)
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
                torch.cuda.synchronize()
                vy=labels[val_ids_t].cpu().numpy()
                vp_np=vp.cpu().numpy()
                val_ap=float(average_precision_score(vy,vp_np))
                vs=time.perf_counter()-v0
                val_times.append(vs)

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
            'model':'BWGNN','dataset':'T-Finance',
            'hardware_profile_id':'vast-a6000-instance-51035671',
            'repository_commit':commit,
            'dataset_sha256':data_sha,
            'split_sha256':split_sha,
            'split_revalidation_status':'PASS',
            'split':ratio,'split_seed':2,'train_seed':seed,
            'hidden':HIDDEN,'order':ORDER,
            'learning_rate':LR,'weight_decay':WEIGHT_DECAY,
            'completed_epochs':int(len(train_times)),
            'best_epoch':int(best_epoch),
            'best_val_auprc':float(best_ap),
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

summary={'model':'BWGNN','dataset':'T-Finance','status':'COMPLETE','completed_runs':len(results),'ratios':{}}
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

archive=ARCHIVES/'bwgnn_tfinance_vast_a6000_final.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p,arc in [
        (ROOT,'results/tfinance'),
        (EVIDENCE,'evidence/tfinance'),
        (SPLIT_PATH,'splits/tfinance_seed2_nested_splits.npz'),
        (WORK/'adapters/BWGNN_gpu.py','adapters/BWGNN_gpu.py'),
        (Path(__file__),'runner/04_bwgnn_tfinance_final.py')
    ]:
        t.add(p,arcname=str(arc))

digest=sha256_file(archive)
(archive.with_suffix(archive.suffix+'.sha256')).write_text(f'{digest}  {archive.name}\n')
print('\n===== T-FINANCE FINAL COMPLETE =====',flush=True)
print(json.dumps(summary,indent=2),flush=True)
print('ARCHIVE:',archive,flush=True)
print('SHA256:',digest,flush=True)
