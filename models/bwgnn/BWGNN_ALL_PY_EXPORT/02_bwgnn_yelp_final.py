#!/usr/bin/env python3
import os, sys, gc, csv, json, time, math, random, hashlib, tarfile, platform, subprocess, contextlib, io
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
RAW_ROOT = Path('/workspace/dataset_audit/raw/yelpchi')
SPLIT_PATH = WORK / 'splits/yelp_seed2_nested_splits.npz'
ROOT = WORK / 'results/bwgnn/yelpchi/final'
RUN_DIR, TIME_DIR = ROOT/'runs', ROOT/'epoch_times'
CKPT_DIR = WORK / 'checkpoints/yelpchi'
EVIDENCE = WORK / 'evidence/yelpchi'
ARCHIVES = WORK / 'archives'
for d in [RUN_DIR, TIME_DIR, CKPT_DIR, EVIDENCE, ARCHIVES, SPLIT_PATH.parent]: d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORK/'adapters'))
from BWGNN_gpu import BWGNN

EXPECTED_COMMIT = 'de0631f039bbd19c1890b483cc01f1007f596af7'
EXPECTED_DATA_SHA = 'fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42'
EXPECTED_SPLIT_ARRAY_SHA = {
 'TR40':'c3230ae44436166a5019804bcb7df06754619b3b93d299ac637088a21fd47591',
 'TR30':'cc7e06783946c8556c0b4b9e067fa585cf75454fa30f1ea47860a68b04f4ce0d',
 'TR20':'93c56ec731b010dbe21095bc377c964db6672b14b29f00a58c84127ccb8c9fb2',
 'TR10':'fcce9fccdf120a6399157ad01c615b542ab67b5f51bf09efb7225df30c89da2b',
 'val':'f776169b20dcda5dcd68098074412226bd2534f1e090f2057684c547f3902ade',
 'test':'828144cf88b8faf8d6145ab06b4e6fe65bcb5ca18699d447b95de3ad2f1e9036',
}
HIDDEN, ORDER, LR, WEIGHT_DECAY = 64, 3, 0.005, 0.0
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

# Exact source identity.
commit=subprocess.check_output(['git','-C',str(WORK/'repo/Rethinking-Anomaly-Detection'),'rev-parse','HEAD'],text=True).strip()
assert commit==EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)

# Find and verify the already-audited canonical YelpChi bytes.
candidates=list(RAW_ROOT.rglob('YelpChi.mat'))
assert len(candidates)==1, f'Expected exactly one YelpChi.mat under {RAW_ROOT}, found {candidates}'
DATA_PATH=candidates[0]
data_sha=sha256_file(DATA_PATH)
assert data_sha==EXPECTED_DATA_SHA, f'YelpChi SHA mismatch: {data_sha}'

print('===== BWGNN VAST — YELPCHI =====', flush=True)
print('Data:',DATA_PATH,flush=True); print('SHA256:',data_sha,flush=True)
print('GPU:',torch.cuda.get_device_name(0),flush=True)
print('Frozen config:',{'hidden':HIDDEN,'order':ORDER,'lr':LR,'weight_decay':WEIGHT_DECAY},flush=True)

load_t0=time.perf_counter()
m=loadmat(DATA_PATH)
features_np=m['features']
if sp.issparse(features_np): features_np=features_np.toarray()
features_np=np.asarray(features_np,dtype=np.float32)
labels_np=np.asarray(m['label']).reshape(-1).astype(np.int64)
assert features_np.shape==(45954,32)
assert (int((labels_np==0).sum()),int((labels_np==1).sum()))==(39277,6677)

# Recreate the exact frozen nested split content; validate against archived array hashes.
ids=np.arange(len(labels_np),dtype=np.int64)
remaining,test=train_test_split(ids,test_size=0.40,stratify=labels_np,random_state=2,shuffle=True)
tr40,val=train_test_split(remaining,test_size=(1/3),stratify=labels_np[remaining],random_state=2,shuffle=True)
tr30,_=train_test_split(tr40,train_size=0.75,stratify=labels_np[tr40],random_state=2,shuffle=True)
tr20,_=train_test_split(tr30,train_size=(2/3),stratify=labels_np[tr30],random_state=2,shuffle=True)
tr10,_=train_test_split(tr20,train_size=0.50,stratify=labels_np[tr20],random_state=2,shuffle=True)
splits={'TR40':np.sort(tr40),'TR30':np.sort(tr30),'TR20':np.sort(tr20),'TR10':np.sort(tr10),'val':np.sort(val),'test':np.sort(test)}
for k,v in splits.items():
    actual=sha_array(v); assert actual==EXPECTED_SPLIT_ARRAY_SHA[k], f'{k} split-content mismatch: {actual}'
np.savez_compressed(SPLIT_PATH,**splits,seed=np.array([2]),source_nodes=np.array([45954]))

relations={}; edge_total=0
for rel,key in [('rsr','net_rsr'),('rtr','net_rtr'),('rur','net_rur')]:
    a=sp.coo_matrix(m[key]); src=torch.from_numpy(a.row.astype(np.int64)); dst=torch.from_numpy(a.col.astype(np.int64))
    edge_total += len(src); relations[('review',rel,'review')]=(src,dst)
assert edge_total==8051348
hg=dgl.heterograph(relations,num_nodes_dict={'review':45954})
hg.nodes['review'].data['feature']=torch.from_numpy(features_np)
hg.nodes['review'].data['label']=torch.from_numpy(labels_np)
g=dgl.to_homogeneous(hg,ndata=['feature','label']); g=dgl.add_self_loop(g)
assert g.num_nodes()==45954 and g.num_edges()==8097302
features=g.ndata['feature'].float(); labels=g.ndata['label'].long()
g=g.to(DEVICE); features=features.to(DEVICE); labels=labels.to(DEVICE)
val_ids=torch.tensor(splits['val'],dtype=torch.long,device=DEVICE); test_ids=torch.tensor(splits['test'],dtype=torch.long,device=DEVICE)
torch.cuda.synchronize(); load_seconds=time.perf_counter()-load_t0

manifest={
 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'model':'BWGNN','dataset':'YelpChi',
 'repository_commit':commit,'dataset_path':str(DATA_PATH),'dataset_sha256':data_sha,
 'split_seed':2,'split_array_sha256':EXPECTED_SPLIT_ARRAY_SHA,'nodes':45954,'relation_edges':8051348,
 'homogeneous_edges_with_self_loops':8097302,'features':32,'normal':39277,'fraud':6677,
 'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'dgl':dgl.__version__,
 'frozen_config':{'hidden':HIDDEN,'order':ORDER,'lr':LR,'weight_decay':WEIGHT_DECAY},
 'train_seeds':SEEDS,'ratios':RATIOS,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
 'checkpoint_metric':'validation AUPRC','threshold_rule':'validation Macro-F1; ties fraud recall then closest to 0.5',
 'test_isolation':True,'load_preprocess_seconds':load_seconds
}
(EVIDENCE/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'Load/preprocess: {load_seconds:.3f}s | split identity PASS',flush=True)

del m,hg,relations,features_np,labels_np; gc.collect()

# Mandatory two-epoch compatibility smoke; no test access.
print('\n===== 2-EPOCH COMPATIBILITY SMOKE (TEST NOT ACCESSED) =====',flush=True)
seed_all(2); train_ids=torch.tensor(splits['TR40'],dtype=torch.long,device=DEVICE)
with contextlib.redirect_stdout(io.StringIO()): model=BWGNN(32,HIDDEN,2,g,d=ORDER).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
ytr=labels[train_ids]; n0=int((ytr==0).sum()); n1=int((ytr==1).sum()); weight=torch.tensor([1.0,n0/n1],device=DEVICE)
torch.cuda.reset_peak_memory_stats()
smoke=[]
for epoch in [1,2]:
    model.train(); torch.cuda.synchronize(); t0=time.perf_counter(); logits=model(features); loss=F.cross_entropy(logits[train_ids],labels[train_ids],weight=weight); opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize(); tr=time.perf_counter()-t0
    model.eval(); torch.cuda.synchronize(); v0=time.perf_counter()
    with torch.no_grad(): vp=torch.softmax(model(features),1)[val_ids,1]
    torch.cuda.synchronize(); vs=time.perf_counter()-v0; ap=float(average_precision_score(labels[val_ids].cpu().numpy(),vp.cpu().numpy()))
    smoke.append({'epoch':epoch,'loss':float(loss.item()),'train_seconds':tr,'validation_seconds':vs,'val_auprc':ap})
    print(f'smoke epoch={epoch} loss={loss.item():.6f} val_AUPRC={ap:.6f} train={tr:.4f}s val={vs:.4f}s',flush=True)
smoke_record={'status':'PASS','test_accessed':False,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2,'epochs':smoke}
(EVIDENCE/'smoke.json').write_text(json.dumps(smoke_record,indent=2)); del model,opt,logits,vp; gc.collect(); torch.cuda.empty_cache()
print('SMOKE_GATE=PASS',flush=True)

# Final controlled 12-run grid. Resume-safe at completed run JSON boundary.
for ratio in RATIOS:
  train_ids=torch.tensor(splits[ratio],dtype=torch.long,device=DEVICE)
  for seed in SEEDS:
    key=f'{ratio}_seed{seed}'; result_file=RUN_DIR/f'{key}.json'; epoch_file=TIME_DIR/f'{key}_epoch_times.csv'; ckpt=CKPT_DIR/f'{key}.pt'
    if result_file.exists(): print(f'SKIP complete {key}',flush=True); continue
    if epoch_file.exists(): epoch_file.unlink()
    if ckpt.exists(): ckpt.unlink()
    seed_all(seed); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(io.StringIO()): model=BWGNN(32,HIDDEN,2,g,d=ORDER).to(DEVICE)
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
      'model':'BWGNN','dataset':'YelpChi','hardware_profile_id':'vast-a6000-instance-51035671','repository_commit':commit,
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

# Aggregate + mean/SD summary.
results=[]
for ratio in RATIOS:
  for seed in SEEDS:
    p=RUN_DIR/f'{ratio}_seed{seed}.json'; assert p.exists(),f'Missing {p}'; results.append(json.loads(p.read_text()))
(ROOT/'all_runs.json').write_text(json.dumps(results,indent=2))
fields=[k for k in results[0].keys()]
with (ROOT/'all_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
summary={'model':'BWGNN','dataset':'YelpChi','status':'COMPLETE','completed_runs':len(results),'ratios':{}}
metric_keys=['auprc','auroc','macro_f1','fraud_precision','fraud_recall','fraud_f1','gmean','mean_epoch_train_seconds','median_epoch_train_seconds','total_train_seconds','inference_latency_mean_ms','peak_gpu_memory_mb']
for ratio in RATIOS:
    rows=[r for r in results if r['split']==ratio]; summary['ratios'][ratio]={}
    for k in metric_keys:
        x=np.asarray([r[k] for r in rows],dtype=float); summary['ratios'][ratio][k]={'mean':float(x.mean()),'sd':float(x.std(ddof=1))}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))

# Lightweight evidence archive and hash.
archive=ARCHIVES/'bwgnn_yelpchi_vast_a6000_final.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p,arc in [(ROOT,'results/yelpchi'),(EVIDENCE,'evidence/yelpchi'),(SPLIT_PATH,'splits/yelp_seed2_nested_splits.npz'),(WORK/'adapters/BWGNN_gpu.py','adapters/BWGNN_gpu.py'),(Path(__file__),'runner/02_bwgnn_yelp_final.py')]:
        t.add(p,arcname=str(arc))
digest=sha256_file(archive); (archive.with_suffix(archive.suffix+'.sha256')).write_text(f'{digest}  {archive.name}\n')
print('\n===== YELPCHI FINAL COMPLETE =====',flush=True)
print(json.dumps(summary,indent=2),flush=True)
print('ARCHIVE:',archive,flush=True); print('SHA256:',digest,flush=True)
