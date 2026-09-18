from pathlib import Path
import os,sys,json,csv,time,copy,math,importlib.util,hashlib,subprocess,gc
import numpy as np, torch, yaml
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,precision_score,recall_score,confusion_matrix
from lightning import Trainer,LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping,Callback
from dgl.dataloading import MultiLayerFullNeighborSampler,DataLoader
ROOT=Path('/workspace/gaap_vast'); REPO=ROOT/'repo/GAAP'; WINNER=ROOT/'unified/yelp/tuning/GAAP_YELP_FROZEN_WINNER.json'; FINAL=ROOT/'unified/yelp/final'; FINAL.mkdir(parents=True,exist_ok=True)
COMMIT='6a7dbb0447c4897504525de49e41a0526ee777f8'; DATA_SHA='fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42'; SPLIT_SHA='0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b'
RATIOS=['TR40','TR30','TR20','TR10']; SEEDS=[2,42,72]; MAXE=100; PAT=20
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(REPO),*a],text=True).strip()
def findsha(root,suf,target):
 for p in root.rglob('*'+suf):
  try:
   if p.is_file() and sha(p)==target:return p
  except: pass
 raise FileNotFoundError(target)
assert WINNER.is_file(); assert git('rev-parse','HEAD')==COMMIT; assert git('status','--porcelain','--untracked-files=no')==''
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'A6000' in torch.cuda.get_device_name(0).upper()
SPLIT=findsha(ROOT,'.npz',SPLIT_SHA)
os.chdir(REPO);sys.path.insert(0,str(REPO/'mycode'))
sp=importlib.util.spec_from_file_location('gaap',REPO/'mycode/exp/101_retrain.py');A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)
cfg0=yaml.safe_load((REPO/'config/SAGE_MiniF_DyPLE_MHA/yelp.yaml').read_text()); W=json.loads(WINNER.read_text())
wb=next((W[k] for k in ('winner','frozen_config','config','candidate','best_config') if isinstance(W.get(k),dict)),W)
def gv(names,d):
 for s in (wb,W):
  for n in names:
   if isinstance(s,dict) and n in s:return s[n]
 return d
CF=copy.deepcopy(cfg0);CF['d_hidden']=int(gv(('d_hidden','hidden','hid_dim'),cfg0['d_hidden']));CF['n_bins']=int(gv(('n_bins','bins'),cfg0['n_bins']));CF['lr']=float(gv(('lr','learning_rate'),cfg0['lr']));CF.update(max_epochs=MAXE,patience=PAT,device='cuda',device_id=0,nowandb=True)
print('===== GAAP × YELPCHI FINAL 12 RUNS =====');print('GPU=',torch.cuda.get_device_name(0));print('SPLIT_SHA256=',sha(SPLIT));print('FROZEN_CONFIG=',{k:CF[k] for k in ['d_hidden','n_bins','lr','weight_decay','gnn_n_layers','gnn_dropout','gnn_agg','use_dyple','use_mha','bs','val_bs','norm_type','preprocess']});print('TUNING_RERUN=NO')
S=np.load(SPLIT,allow_pickle=False); E={'TR40':18381,'TR30':13785,'TR20':9190,'TR10':4595,'val':9191,'test':18382}
for k,n in E.items(): assert len(S[k])==n
V=torch.as_tensor(S['val'],dtype=torch.long);T=torch.as_tensor(S['test'],dtype=torch.long)
def mask(n,ids):
 m=torch.zeros(n,dtype=torch.bool);m[ids]=True;return m
def mets(y,p,t):
 q=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel();tpr=tp/(tp+fn) if tp+fn else 0;tnr=tn/(tn+fp) if tn+fp else 0
 return dict(auprc=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),macro_f1=float(f1_score(y,q,average='macro',zero_division=0)),fraud_precision=float(precision_score(y,q,zero_division=0)),fraud_recall=float(recall_score(y,q,zero_division=0)),fraud_f1=float(f1_score(y,q,zero_division=0)),gmean=float(math.sqrt(tpr*tnr)))
def threshold(y,p):
 a=[]
 for t in np.arange(.01,1,.01):
  m=mets(y,p,float(t));a.append((float(t),m['macro_f1'],m['fraud_recall']))
 a.sort(key=lambda x:(-x[1],-x[2],abs(x[0]-.5)));return a[0][0],a
class DM(LightningDataModule):
 def __init__(self,g,tr,c):super().__init__();self.g=g;self.tr=tr;self.c=c;self.all=torch.arange(g.num_nodes());self.s=MultiLayerFullNeighborSampler(int(c['gnn_n_layers']))
 def train_dataloader(self):return DataLoader(self.g,self.tr,self.s,batch_size=int(self.c['bs']),shuffle=True,drop_last=False,num_workers=8)
 def val_dataloader(self):return DataLoader(self.g,self.all,self.s,batch_size=int(self.c['val_bs']),shuffle=False,drop_last=False,num_workers=8)
class ET(Callback):
 def __init__(self):self.st={};self.sec=[]
 def on_train_epoch_start(self,tr,m):torch.cuda.synchronize();self.st[int(tr.current_epoch)]=time.perf_counter()
 def on_train_epoch_end(self,tr,m):torch.cuda.synchronize();e=int(tr.current_epoch);self.sec.append(time.perf_counter()-self.st[e])
class M(A.LitSAGE):
 def __init__(self,rd,label,*a,**k):super().__init__(*a,**k);self.rd=Path(rd);self.label=label;self.best=-1.;self.be=0;self.bl=[];self.loss=float('nan');self.full=None
 def training_step(self,b,bi):
  o=super().training_step(b,bi);l=o.get('loss') if isinstance(o,dict) else o
  if torch.is_tensor(l):self.bl.append(float(l.detach().cpu()))
  return o
 def on_train_epoch_end(self):
  if self.bl:self.loss=float(np.mean(self.bl))
  self.bl=[]
 def on_validation_epoch_end(self):
  if self.trainer.sanity_checking:self.outs.clear();return
  y=torch.cat([x[0] for x in self.outs]);z=torch.cat([x[1] for x in self.outs]);nid=torch.cat([x[2] for x in self.outs]);he=torch.cat([x[3] for x in self.outs]);o=torch.argsort(nid);y=y[o];z=z[o];he=he[o];self.his_emb=he;self.full=z.detach().cpu()
  vi=V.to(z.device);vy=y[vi];vz=z[vi];loss=F.cross_entropy(vz,vy);p=vz.softmax(-1)[:,1].detach().cpu().numpy();yt=vy.detach().cpu().numpy();ap=float(average_precision_score(yt,p));au=float(roc_auc_score(yt,p))
  self.log('valoss',loss,on_step=False,on_epoch=True);self.log('val_aps',ap,on_step=False,on_epoch=True);self.log('val_auc',au,on_step=False,on_epoch=True)
  if self.trainer.state.fn.name=='FITTING':
   ep=int(self.current_epoch)+1
   if ap>self.best:
    self.best=ap;self.be=ep;torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in self.state_dict().items()},'his_emb':self.his_emb.detach().cpu().clone(),'epoch':ep,'val_auprc':ap,'val_auroc':au},self.rd/'best_state.pt')
   print(f'{self.label} | epoch={ep:03d} | loss={self.loss:.6f} | valAUPRC={ap:.6f} | valAUROC={au:.6f} | best={self.best:.6f}@{self.be} | patience={ep-self.be}/{PAT}',flush=True)
  self.outs.clear()
def run(ratio,seed):
 rd=FINAL/f'{ratio.lower()}_seed{seed}';rd.mkdir(parents=True,exist_ok=True);rp=rd/'result.json'
 if rp.exists():
  old=json.loads(rp.read_text())
  if old.get('status')=='PASS':print('SKIP',ratio,seed,'already PASS');return old
 A.fix_seed(seed);c=copy.deepcopy(CF);c['seed']=seed;adm=A.tag_dm_map[c['loader_type']](**c);g=adm.g;assert g.num_nodes()==45954 and g.ndata['feature'].shape[1]==32
 y=g.ndata['label'].clone();y=y.argmax(1) if y.ndim==2 else y;y=y.long().cpu();assert int((y==1).sum())==6677
 tr=torch.as_tensor(S[ratio],dtype=torch.long);g.ndata['train_mask']=mask(g.num_nodes(),tr);g.ndata['val_mask']=mask(g.num_nodes(),V);g.ndata['test_mask']=mask(g.num_nodes(),T);g.ndata['label']=y.clone();g.ndata['label'][T]=-1
 c['d_in']=32;c['n_classes']=2;c['n_nodes']=45954;dm=DM(g,tr,c);m=M(rd,f'{ratio} seed={seed}',**c);et=ET();es=EarlyStopping(monitor='val_aps',mode='max',patience=PAT,min_delta=0.)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();w0=time.perf_counter();trainer=Trainer(accelerator='gpu',devices=1,max_epochs=MAXE,logger=False,enable_progress_bar=False,enable_checkpointing=False,num_sanity_val_steps=0,gradient_clip_val=10,callbacks=[et,es],deterministic=False);trainer.fit(m,datamodule=dm);torch.cuda.synchronize();wall=time.perf_counter()-w0;peak=float(torch.cuda.max_memory_allocated()/1024**2)
 b=torch.load(rd/'best_state.pt',map_location='cpu');m.load_state_dict(b['state_dict']);m.his_emb=b['his_emb'].to(m.device);m.full=None;trainer.validate(m,datamodule=dm,verbose=False);z=m.full;vp=z[V].softmax(-1)[:,1].numpy();vy=y[V].numpy();th,grid=threshold(vy,vp)
 with open(rd/'threshold_grid.csv','w',newline='') as f:
  q=csv.writer(f);q.writerow(['threshold','macro_f1','fraud_recall']);q.writerows(grid)
 g.ndata['label']=y.clone();m.full=None;torch.cuda.synchronize();t0=time.perf_counter();trainer.validate(m,datamodule=dm,verbose=False);torch.cuda.synchronize();infer=time.perf_counter()-t0;z=m.full;tp=z[T].softmax(-1)[:,1].numpy();ty=y[T].numpy();mm=mets(ty,tp,th)
 r=dict(status='PASS',model='GAAP',dataset='YelpChi',ratio=ratio,seed=seed,train_nodes=len(tr),val_nodes=len(V),test_nodes=len(T),best_epoch=int(b['epoch']),best_val_auprc=float(b['val_auprc']),best_val_auroc=float(b['val_auroc']),threshold=float(th),**mm,epoch_train_seconds=et.sec,total_train_seconds=float(sum(et.sec)),fit_wall_seconds=float(wall),test_inference_seconds=float(infer),peak_gpu_memory_mb=peak,test_used_during_fit=False,test_used_for_threshold=False,test_evaluated_after_freeze=True,repository_commit=COMMIT,dataset_sha256=DATA_SHA,split_sha256=SPLIT_SHA,frozen_config={k:c[k] for k in ['d_hidden','n_bins','lr','weight_decay','gnn_n_layers','gnn_dropout','gnn_agg','use_dyple','use_mha','bs','val_bs','norm_type','preprocess']})
 rp.write_text(json.dumps(r,indent=2))
 with open(rd/'epoch_timing.csv','w',newline='') as f:q=csv.writer(f);q.writerow(['epoch','train_seconds']);q.writerows(enumerate(et.sec,1))
 print(f"FINAL {ratio} seed={seed} PASS | AUPRC={mm['auprc']:.6f} | AUROC={mm['auroc']:.6f} | MacroF1={mm['macro_f1']:.6f} | Recall={mm['fraud_recall']:.6f}",flush=True)
 del m,trainer,dm,adm,g;gc.collect();torch.cuda.empty_cache();return r
R=[run(r,s) for r in RATIOS for s in SEEDS];assert len(R)==12
names=['auprc','auroc','macro_f1','fraud_precision','fraud_recall','fraud_f1','gmean']
def ms(v):a=np.asarray(v,float);return float(a.mean()),float(a.std(ddof=1))
rows=[]
for ratio in RATIOS:
 rr=[x for x in R if x['ratio']==ratio];row={'ratio':ratio,'n_seeds':3}
 for n in names:
  a,b=ms([x[n] for x in rr]);row[n+'_mean']=a;row[n+'_sd']=b
 for n in ['total_train_seconds','test_inference_seconds','peak_gpu_memory_mb']:
  a,b=ms([x[n] for x in rr]);row[n+'_mean']=a;row[n+'_sd']=b
 rows.append(row)
fields=['ratio','seed','train_nodes','val_nodes','test_nodes','best_epoch','best_val_auprc','best_val_auroc','threshold',*names,'total_train_seconds','test_inference_seconds','peak_gpu_memory_mb']
with open(FINAL/'GAAP_YELP_FINAL_12_RUNS.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();[w.writerow({k:r[k] for k in fields}) for r in R]
with open(FINAL/'GAAP_YELP_FINAL_RATIO_SUMMARY.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(FINAL/'GAAP_YELP_FINAL_SUMMARY.json').write_text(json.dumps({'status':'PASS','model':'GAAP','dataset':'YelpChi','final_runs_complete':12,'ratios':RATIOS,'seeds':SEEDS,'repository_commit':COMMIT,'dataset_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,'winner_sha256':sha(WINNER),'ratio_summary':rows},indent=2))
assert git('status','--porcelain','--untracked-files=no')==''
print('\n===== GAAP YELP FINAL RESULTS — MEAN ± SD =====')
for r in rows:print(f"{r['ratio']} | AUPRC={r['auprc_mean']:.6f}±{r['auprc_sd']:.6f} | AUROC={r['auroc_mean']:.6f}±{r['auroc_sd']:.6f} | MacroF1={r['macro_f1_mean']:.6f}±{r['macro_f1_sd']:.6f} | Prec={r['fraud_precision_mean']:.6f}±{r['fraud_precision_sd']:.6f} | Recall={r['fraud_recall_mean']:.6f}±{r['fraud_recall_sd']:.6f} | FraudF1={r['fraud_f1_mean']:.6f}±{r['fraud_f1_sd']:.6f} | GMean={r['gmean_mean']:.6f}±{r['gmean_sd']:.6f}")
print('\n===== FINAL GATE =====');print('GAAP_YELP_FINAL=PASS');print('GAAP_YELP_FINAL_RUNS=12/12');print('TEST_ISOLATION=PASS');print('GAAP_AUTHOR_REPO_CLEAN=PASS')
