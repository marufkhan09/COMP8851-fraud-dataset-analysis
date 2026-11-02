#!/usr/bin/env python3
import os, sys, gc, csv, json, time, math, random, hashlib, tarfile, subprocess, contextlib, io
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn.functional as F
import dgl

WORK = Path('/workspace/bwgnn_vast')
RAW_ROOT = Path('/workspace/dataset_audit/raw/amazon')
SPLIT_PATH = WORK / 'splits/amazon_seed2_nested_splits.npz'
ROOT = WORK / 'results/bwgnn/amazon/final'
RUN_DIR, TIME_DIR = ROOT/'runs', ROOT/'epoch_times'
CKPT_DIR = WORK / 'checkpoints/amazon'
EVIDENCE = WORK / 'evidence/amazon'
ARCHIVES = WORK / 'archives'
for d in [RUN_DIR, TIME_DIR, CKPT_DIR, EVIDENCE, ARCHIVES, SPLIT_PATH.parent]: d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORK/'adapters'))
from BWGNN_gpu import BWGNN

EXPECTED_COMMIT = 'de0631f039bbd19c1890b483cc01f1007f596af7'
EXPECTED_DATA_SHA = '4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63'
EXPECTED_SPLIT_ARRAY_SHA = {
 'TR40':'70a01b19199879ca3e0f101d881f56649ab31b1ba89e7f50dd242b2c3119f158',
 'TR30':'82878022897719840efc5f03d3ffcb8e66c9d9c3dc2c3297ef5021a1cb3939d6',
 'TR20':'5d0f06fd65ae95bdf8ba4d88e4ce7ed88eb5538b88d220192dc1c417579e01d0',
 'TR10':'00217d9320f4c4951db6de1f96e1dbaba30acf00ed8e07ea681dabb5a102d813',
 'val':'cbda0925852b9467cce189d21d67f0f9b6a7a77629c56d791ff97329d5d28bb7',
 'test':'e51d3764bd71a07b02c27bd4da334b131c9f95a9882bb97dab64c8ae5e7ad0d3',
}
HIDDEN, ORDER, LR, WEIGHT_DECAY = 32, 2, 0.01, 0.0
MAX_EPOCHS, PATIENCE = 100, 20
SEEDS, RATIOS = [2,42,72], ['TR40','TR30','TR20','TR10']
WARMUPS, REPEATS = 3, 10
DEVICE = torch.device('cuda:0')


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def sha_array(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def choose_threshold(y,p):
    candidates=[]
    for t in np.arange(0.01,1.00,0.01):
        pred=(p>=t).astype(np.int64)
        candidates.append((f1_score(y,pred,average='macro',zero_division=0), recall_score(y,pred,pos_label=1,zero_division=0), -abs(float(t)-0.5), float(t)))
    return max(candidates)[3]

def calc_metrics(y,p,t):
    pred=(p>=t).astype(np.int64)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else 0.0; spec=tn/(tn+fp) if tn+fp else 0.0
    return {
      'auprc':float(average_precision_score(y,p)), 'auroc':float(roc_auc_score(y,p)),
      'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
      'fraud_precision':float(precision_score(y,pred,pos_label=1,zero_division=0)),
      'fraud_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),
      'fraud_f1':float(f1_score(y,pred,pos_label=1,zero_division=0)),
      'gmean':float(math.sqrt(sens*spec)), 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)
    }

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

# Source identity.
commit=subprocess.check_output(['git','-C',str(WORK/'repo/Rethinking-Anomaly-Detection'),'rev-parse','HEAD'],text=True).strip()
assert commit==EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)

# Canonical Amazon bytes.
candidates=list(RAW_ROOT.rglob('Amazon.mat'))
assert len(candidates)==1, f'Expected exactly one Amazon.mat under {RAW_ROOT}, found {candidates}'
DATA_PATH=candidates[0]
data_sha=sha256_file(DATA_PATH)
assert data_sha==EXPECTED_DATA_SHA, f'Amazon SHA mismatch: {data_sha}'

print('===== BWGNN VAST — AMAZON =====', flush=True)
print('Data:',DATA_PATH,flush=True); print('SHA256:',data_sha,flush=True)
print('GPU:',torch.cuda.get_device_name(0),flush=True)
print('Frozen config:',{'hidden':HIDDEN,'order':ORDER,'lr':LR,'weight_decay':WEIGHT_DECAY},flush=True)

load_t0=time.perf_counter()
m=loadmat(DATA_PATH)
features_np=m['features']
if sp.issparse(features_np): features_np=features_np.toarray()
features_np=np.asarray(features_np,dtype=np.float32)
labels_np=np.asarray(m['label']).reshape(-1).astype(np.int64)
assert features_np.shape==(11944,25)
assert labels_np.shape==(11944,)
eligible_ids=np.arange(3305,11944,dtype=np.int64)
eligible_labels=labels_np[eligible_ids]
assert len(eligible_ids)==8639
assert (int((eligible_labels==0).sum()),int((eligible_labels==1).sum()))==(7818,821)

# Recreate exact corrected Amazon supervised splits (unlabelled IDs 0-3304 excluded).
remaining,test=train_test_split(eligible_ids,test_size=0.40,stratify=labels_np[eligible_ids],random_state=2,shuffle=True)
tr40,val=train_test_split(remaining,test_size=(1/3),stratify=labels_np[remaining],random_state=2,shuffle=True)
tr30,_=train_test_split(tr40,train_size=0.75,stratify=labels_np[tr40],random_state=2,shuffle=True)
tr20,_=train_test_split(tr30,train_size=(2/3),stratify=labels_np[tr30],random_state=2,shuffle=True)
tr10,_=train_test_split(tr20,train_size=0.50,stratify=labels_np[tr20],random_state=2,shuffle=True)
splits={'TR40':np.sort(tr40),'TR30':np.sort(tr30),'TR20':np.sort(tr20),'TR10':np.sort(tr10),'val':np.sort(val),'test':np.sort(test)}
for k,v in splits.items():
    assert int(v.min())>=3305, f'Unlabelled Amazon node entered {k}'
    actual=sha_array(v); assert actual==EXPECTED_SPLIT_ARRAY_SHA[k], f'{k} split-content mismatch: {actual}'
np.savez_compressed(SPLIT_PATH,**splits,seed=np.array([2]),source_nodes=np.array([11944]),excluded_first_id=np.array([0]),excluded_last_id=np.array([3304]),eligible_nodes=np.array([8639]))

# Author-compatible homogeneous execution: preserve relation entries, then add one self-loop/node.
relations={}; edge_total=0
for rel,key in [('upu','net_upu'),('usu','net_usu'),('uvu','net_uvu')]:
    a=sp.coo_matrix(m[key]); src=torch.from_numpy(a.row.astype(np.int64)); dst=torch.from_numpy(a.col.astype(np.int64))
    edge_total += len(src); relations[('user',rel,'user')]=(src,dst)
assert edge_total==9557648
hg=dgl.heterograph(relations,num_nodes_dict={'user':11944})
hg.nodes['user'].data['feature']=torch.from_numpy(features_np)
hg.nodes['user'].data['label']=torch.from_numpy(labels_np)
g=dgl.to_homogeneous(hg,ndata=['feature','label']); g=dgl.add_self_loop(g)
assert g.num_nodes()==11944 and g.num_edges()==9569592
features=g.ndata['feature'].float(); labels=g.ndata['label'].long()
g=g.to(DEVICE); features=features.to(DEVICE); labels=labels.to(DEVICE)
val_ids=torch.tensor(splits['val'],dtype=torch.long,device=DEVICE); test_ids=torch.tensor(splits['test'],dtype=torch.long,device=DEVICE)
torch.cuda.synchronize(); load_seconds=time.perf_counter()-load_t0

manifest={
 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'model':'BWGNN','dataset':'Amazon',
 'repository_commit':commit,'dataset_path':str(DATA_PATH),'dataset_sha256':data_sha,
 'split_seed':2,'split_array_sha256':EXPECTED_SPLIT_ARRAY_SHA,'nodes_total':11944,'eligible_supervised_nodes':8639,
 'excluded_unlabelled_nodes':3305,'relation_edges':9557648,'homogeneous_edges_with_self_loops':9569592,
 'features':25,'eligible_normal':7818,'eligible_fraud':821,
 'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'dgl':dgl.__version__,
 'frozen_config':{'hidden':HIDDEN,'order':ORDER,'lr':LR,'weight_decay':WEIGHT_DECAY},
 'train_seeds':SEEDS,'ratios':RATIOS,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
 'checkpoint_metric':'validation AUPRC','threshold_rule':'validation Macro-F1; ties fraud recall then closest to 0.5',
 'test_isolation':True,'load_preprocess_seconds':load_seconds
}
(EVIDENCE/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'Load/preprocess: {load_seconds:.3f}s | corrected split identity PASS',flush=True)

del m,hg,relations,features_np,labels_np,eligible_labels; gc.collect()

# Mandatory 2-epoch smoke; test untouched.
print('\n===== 2-EPOCH COMPATIBILITY SMOKE (TEST NOT ACCESSED) =====',flush=True)
seed_all(2); train_ids=torch.tensor(splits['TR40'],dtype=torch.long,device=DEVICE)
with contextlib.redirect_stdout(io.StringIO()): model=BWGNN(25,HIDDEN,2,g,d=ORDER).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
ytr=labels[train_ids]; n0=int((ytr==0).sum()); n1=int((ytr==1).sum()); weight=torch.tensor([1.0,n0/n1],device=DEVICE)
torch.cuda.reset_peak_memory_stats(); smoke=[]
for epoch in [1,2]:
    model.train(); torch.cuda.synchronize(); t0=time.perf_counter(); logits=model(features); loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=weight); opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize(); tr=time.perf_counter()-t0
    model.eval(); torch.cuda.synchronize(); v0=time.perf_counter()
    with torch.no_grad(): vp=torch.softmax(model(features),1)[val_ids,1]
    torch.cuda.synchronize(); vs=time.perf_counter()-v0; ap=float(average_precision_score(labels[val_ids].cpu().numpy(),vp.cpu().numpy()))
    smoke.append({'epoch':epoch,'loss':float(loss.item()),'train_seconds':tr,'validation_seconds':vs,'val_auprc':ap})
    print(f'smoke epoch={epoch} loss={loss.item():.6f} val_AUPRC={ap:.6f} train={tr:.4f}s val={vs:.4f}s',flush=True)
(EVIDENCE/'smoke.json').write_text(json.dumps({'status':'PASS','test_accessed':False,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2,'epochs':smoke},indent=2))
del model,opt,logits,vp; gc.collect(); torch.cuda.empty_cache(); print('SMOKE_GATE=PASS',flush=True)

# Final 12-run controlled grid; resume-safe.
for ratio in RATIOS:
  train_ids=torch.tensor(splits[ratio],dtype=torch.long,device=DEVICE)
  for seed in SEEDS:
    key=f'{ratio}_seed{seed}'; result_file=RUN_DIR/f'{key}.json'; epoch_file=TIME_DIR/f'{key}_epoch_times.csv'; ckpt=CKPT_DIR/f'{key}.pt'
    if result_file.exists(): print(f'SKIP complete {key}',flush=True); continue
    if epoch_file.exists(): epoch_file.unlink()
    if ckpt.exists(): ckpt.unlink()
    seed_all(seed); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(io.StringIO()): model=BWGNN(25,HIDDEN,2,g,d=ORDER).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,betas=(0.9,0.999),eps=1e-8)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    ytr=labels[train_ids]; n0=int((ytr==0).sum()); n1=int((ytr==1).sum()); weight=torch.tensor([1.0,n0/n1],dtype=torch.float32,device=DEVICE)
    train_times=[]; val_times=[]; best_ap=-1.0; best_epoch=-1; stale=0; cumulative=0.0
    print(f'\nSTART {key}',flush=True)
    with epoch_file.open('w',newline='') as f:
      w=csv.writer(f); w.writerow(['epoch','loss','train_seconds','validation_seconds','cumulative_train_seconds','val_auprc','best_val_auprc','best_epoch'])
      for epoch in range(1,MAX_EPOCHS+1):
        model.train(); torch.cuda.synchronize(); t0=time.perf_counter(); logits=model(features); loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=weight); opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize(); tr=time.perf_counter()-t0; train_times.append(tr); cumulative+=tr
        model.eval(); torch.cuda.synchronize(); v0=time.perf_counter()
        with torch.no_grad(): vp=torch.softmax(model(features),1)[val_ids,1]
        torch.cuda.synchronize(); vs=time.perf_counter()-v0; val_times.append(vs); vy=labels[val_ids].cpu().numpy(); vp_np=vp.cpu().numpy(); val_ap=float(average_precision_score(vy,vp_np))
        if val_ap>best_ap: best_ap=val_ap; best_epoch=epoch; stale=0; torch.save(model.state_dict(),ckpt)
        else: stale+=1
        w.writerow([epoch,float(loss.item()),tr,vs,cumulative,val_ap,best_ap,best_epoch]); f.flush()
        if epoch==1 or epoch%10==0 or stale>=PATIENCE: print(f'{key} epoch={epoch:03d} val_AP={val_ap:.6f} best={best_ap:.6f}@{best_epoch} train={tr:.4f}s stale={stale}',flush=True)
        if stale>=PATIENCE: break
    train_times=np.asarray(train_times); val_times=np.asarray(val_times)
    model.load_state_dict(torch.load(ckpt,map_location=DEVICE)); model.eval()
    with torch.no_grad():
      for _ in range(WARMUPS): _=model(features); torch.cuda.synchronize()
      latency=[]
      for _ in range(REPEATS): torch.cuda.synchronize(); q=time.perf_counter(); _=model(features); torch.cuda.synchronize(); latency.append((time.perf_counter()-q)*1000)
      prob=torch.softmax(model(features),1)[:,1]
    vy=labels[val_ids].cpu().numpy(); vp=prob[val_ids].cpu().numpy(); threshold=choose_threshold(vy,vp)
    ty=labels[test_ids].cpu().numpy(); tp=prob[test_ids].cpu().numpy(); metrics=calc_metrics(ty,tp,threshold)
    result={
      'model':'BWGNN','dataset':'Amazon','hardware_profile_id':'vast-a6000-instance-51035671','repository_commit':commit,
      'dataset_sha256':data_sha,'split':ratio,'split_seed':2,'train_seed':seed,'hidden':HIDDEN,'order':ORDER,'learning_rate':LR,'weight_decay':WEIGHT_DECAY,
      'completed_epochs':int(len(train_times)),'best_epoch':int(best_epoch),'best_val_auprc':float(best_ap),'parameter_count':int(params),
      'load_preprocess_seconds':float(load_seconds),'total_train_seconds':float(train_times.sum()),'mean_epoch_train_seconds':float(train_times.mean()),
      'median_epoch_train_seconds':float(np.median(train_times)),'std_epoch_train_seconds':float(train_times.std(ddof=1) if len(train_times)>1 else 0),
      'total_validation_seconds':float(val_times.sum()),'mean_validation_seconds':float(val_times.mean()),
      'inference_latency_mean_ms':float(np.mean(latency)),'inference_latency_std_ms':float(np.std(latency,ddof=1)),'peak_gpu_memory_mb':float(torch.cuda.max_memory_allocated()/1024**2),
      'validation_selected_threshold':float(threshold),'test_evaluated_once_after_freeze':True,**metrics
    }
    result_file.write_text(json.dumps(result,indent=2)); ckpt.unlink(missing_ok=True)
    print(f"DONE {key} | test_AUPRC={metrics['auprc']:.6f} AUROC={metrics['auroc']:.6f} MacroF1={metrics['macro_f1']:.6f} | {result['mean_epoch_train_seconds']:.4f}s/epoch",flush=True)
    del model,opt,prob; gc.collect(); torch.cuda.empty_cache()

results=[]
for ratio in RATIOS:
  for seed in SEEDS:
    p=RUN_DIR/f'{ratio}_seed{seed}.json'; assert p.exists(),f'Missing {p}'; results.append(json.loads(p.read_text()))
(ROOT/'all_runs.json').write_text(json.dumps(results,indent=2))
fields=[k for k in results[0].keys()]
with (ROOT/'all_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
summary={'model':'BWGNN','dataset':'Amazon','status':'COMPLETE','completed_runs':len(results),'ratios':{}}
metric_keys=['auprc','auroc','macro_f1','fraud_precision','fraud_recall','fraud_f1','gmean','mean_epoch_train_seconds','median_epoch_train_seconds','total_train_seconds','inference_latency_mean_ms','peak_gpu_memory_mb']
for ratio in RATIOS:
    rows=[r for r in results if r['split']==ratio]; summary['ratios'][ratio]={}
    for k in metric_keys:
        x=np.asarray([r[k] for r in rows],dtype=float); summary['ratios'][ratio][k]={'mean':float(x.mean()),'sd':float(x.std(ddof=1))}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))

archive=ARCHIVES/'bwgnn_amazon_vast_a6000_final.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p,arc in [(ROOT,'results/amazon'),(EVIDENCE,'evidence/amazon'),(SPLIT_PATH,'splits/amazon_seed2_nested_splits.npz'),(WORK/'adapters/BWGNN_gpu.py','adapters/BWGNN_gpu.py'),(Path(__file__),'runner/03_bwgnn_amazon_final.py')]:
        t.add(p,arcname=str(arc))
digest=sha256_file(archive); (archive.with_suffix(archive.suffix+'.sha256')).write_text(f'{digest}  {archive.name}\n')
print('\n===== AMAZON FINAL COMPLETE =====',flush=True)
print(json.dumps(summary,indent=2),flush=True)
print('ARCHIVE:',archive,flush=True); print('SHA256:',digest,flush=True)
