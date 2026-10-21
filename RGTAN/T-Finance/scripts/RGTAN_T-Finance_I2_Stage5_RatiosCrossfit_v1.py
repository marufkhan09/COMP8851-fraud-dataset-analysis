#!/usr/bin/env python3
"""RGTAN x T-Finance: build TR30/TR20/TR10 OOF risk from original canonical graph.

Keeps complete 42,445,086-edge source for risk computation and graph sampling.
Uses ONE GPU exact-2-hop reachability computation for all missing ratios. No
validation/test labels are decoded, retained, or used in risk features.
Existing completed, hash-validated ratio assets are always reused.

Provisional method: downstream RGTAN model samples [10,10] fanouts and requires
supervisor approval before this candidate is reported as an accepted benchmark.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, socket, subprocess, sys, time
from pathlib import Path
import dgl
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path('/workspace/comp8851')
SOURCE=ROOT/'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
PREVIOUS=ROOT/'pipeline/RGTAN_T-Finance_TR40_Candidate_Pilot_Crossfit_v1.py'
EXISTING=ROOT/'assets/rgtan/tfinance/RGTAN_T-Finance_TR40_ExactRiskCrossfit_v1.npz'
ASSETS=ROOT/'assets/rgtan/tfinance'
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
DATA_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
TR40_SHA='798daba5c4a131159208d242b71e03ef3d1c79aa44f5cfd599c7fd97b98dab81'
N=39357; EDGES=42445086
COUNTS={'TR40':15742,'TR30':11806,'TR20':7870,'TR10':3935,'val':7872}
POSITIVES={'TR40':721,'TR30':541,'TR20':361,'TR10':181}
RATIOS=('TR30','TR20','TR10')
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def atomic_json(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=Path(str(p)+'.partial')
 with tmp.open('x') as f:
  json.dump(data,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)


def preflight():
 assert socket.gethostname()=='f24d1bb2672d','STOP: Wrong instance, only I2'
 assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
 assert torch.cuda.is_available() and 'A6000' in torch.cuda.get_device_name(0)
 assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==PIN
 assert sha(SOURCE)==DATA_SHA and sha(SPLIT)==SPLIT_SHA
 assert sha(EXISTING)==TR40_SHA,'STOP: Stage3 canonical TR40 crossfit has changed'
 with np.load(SPLIT,allow_pickle=False) as z:
  assert int(z['seed'].item())==2 and int(z['source_fraud'].item())==1804
  parts={k:np.asarray(z[k],dtype=np.int64).copy() for k in (*COUNTS,'test')}
 for k,v in COUNTS.items():assert len(parts[k])==v and len(np.unique(parts[k]))==v
 for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):assert np.isin(parts[a],parts[b]).all()
 for a,b in (('TR40','val'),('TR40','test'),('val','test')):assert not np.intersect1d(parts[a],parts[b]).size
 print('RGTAN_T-Finance_I2_ALL_RATIOS_CANONICAL_SPLIT=PASS',flush=True)
 return parts


def generate():
 parts=preflight()
 missing=[];ready=[]
 for ratio in RATIOS:
  out=ASSETS/f'RGTAN_T-Finance_{ratio}_ExactRiskCrossfit_v1.npz'
  manifest=out.with_suffix('.json')
  if out.exists() or manifest.exists():
   assert out.is_file() and manifest.is_file(),'STOP: Partial asset pair, investigate '+ratio
   m=json.loads(manifest.read_text())
   assert m['status']=='RGTAN_TFINANCE_'+ratio+'_EXACT_RISK_CROSSFIT_PASS'
   assert m['artifact_sha256']==sha(out) and m['source_sha256']==DATA_SHA and m['split_sha256']==SPLIT_SHA
   assert m['train_count']==COUNTS[ratio] and m['training_positive_count']==POSITIVES[ratio]
   assert m['validation_labels_used'] is False and m['test_labels_used'] is False
   with np.load(out,allow_pickle=False) as z:
    assert np.array_equal(z['train_ids'],parts[ratio]) and np.array_equal(z['val_ids'],parts['val'])
   print(f'RGTAN_T-Finance_{ratio}_CROSSFIT_VERIFIED_REUSE=PASS',flush=True)
   ready.append(ratio)
  else:missing.append(ratio)
 if not missing:
  print('RGTAN_T-Finance_I2_TR30_TR20_TR10_CROSSFITS_READY=PASS (all reused)',flush=True)
  return
 # Import pinned Stage3 math implementation without re-running Stage3.
 assert PREVIOUS.is_file()
 spec=importlib.util.spec_from_file_location('RGTAN_TFinance_Stage3',PREVIOUS)
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 assert old.N==N and old.EDGES==EDGES and tuple(old.NAMES)==NAMES
 graphs,_=dgl.load_graphs(str(SOURCE))
 assert len(graphs)==1
 graph=graphs[0];assert graph.is_homogeneous and graph.num_nodes()==N and graph.num_edges()==EDGES
 feat=graph.ndata['feature'].numpy().astype(np.float32,copy=True)
 assert feat.shape==(N,10) and np.isfinite(feat).all()
 onehot=graph.ndata['label'];assert tuple(onehot.shape)==(N,2)
 # Only read training labels. No heldout truth enters these assets.
 trainlabels={}
 for ratio in missing:
  tr=parts[ratio]
  y=onehot[torch.from_numpy(tr.copy())].argmax(dim=1).numpy().astype(np.int64,copy=True)
  assert int(y.sum())==POSITIVES[ratio]
  trainlabels[ratio]=y
 for k in list(graph.ndata):del graph.ndata[k]
 for k in list(graph.edata):del graph.edata[k]
 assert not graph.ndata and not graph.edata
 row_values=[];folds_by_ratio={};contexts_by_ratio={}
 for ratio in missing:
  tr=parts[ratio];y=trainlabels[ratio]
  folds=np.full(N,-1,dtype=np.int8)
  skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
  for k,(_,held) in enumerate(skf.split(np.zeros(len(tr)),y)):
   folds[tr[held]]=k
  assert set(folds[tr])==set(range(5))
  ctx=np.full((6,N),2,dtype=np.int8)
  view=np.zeros((N,6),dtype=np.float32)
  for k in range(5):
   used=folds[tr]!=k;ids=tr[used]
   ctx[k,ids]=y[used];view[ids,k]=y[used]
   heldout=tr[~used]
   assert np.all(ctx[k,heldout]==2) and np.all(view[heldout,k]==0)
  ctx[5,tr]=y;view[tr,5]=y
  assert np.all(ctx[:,parts['val']]==2) and np.all(ctx[:,parts['test']]==2)
  assert np.all(view[parts['val']]==0) and np.all(view[parts['test']]==0)
  row_values.append(view);folds_by_ratio[ratio]=folds;contexts_by_ratio[ratio]=ctx
  print(f'RGTAN_T-Finance_{ratio}_TRAIN_ONLY_5_FOLDS=PASS',flush=True)
 # Original full directed source; remove duplicates/self only in the risk view.
 u,v=graph.edges(order='eid')
 adj=sp.coo_matrix((np.ones(len(u),dtype=np.int8),(v.numpy(),u.numpy())),shape=(N,N)).tocsr()
 del u,v,graph,graphs,onehot
 adj.sum_duplicates();adj.data.fill(1);adj.setdiag(0);adj.eliminate_zeros();adj.sort_indices()
 assert 21000000<adj.nnz<=EDGES
 print('RGTAN_T-Finance_I2_ORIGINAL_GRAPH_ONE_PASS_TWOHOP_START columns='+str(6*len(missing)),flush=True)
 t0=time.perf_counter()
 pos=np.concatenate(row_values,axis=1).astype(np.float32,copy=False)
 degree,one_risk,first_degree,second_degree,first_risk,second_risk=old.exact_risk(adj,pos)
 del pos,adj
 assert one_risk.shape==first_risk.shape==second_risk.shape==(N,6*len(missing))
 ASSETS.mkdir(parents=True,exist_ok=True)
 for j,ratio in enumerate(missing):
  sl=slice(6*j,6*(j+1));tr=parts[ratio];y=trainlabels[ratio]
  raw_views=[]
  for k in range(6):
   col=6*j+k
   raw_views.append(np.column_stack((degree,one_risk[:,col],first_degree,second_degree,
     first_risk[:,col],second_risk[:,col])).astype(np.float32))
  folds=folds_by_ratio[ratio];oof=np.empty((len(tr),6),dtype=np.float32)
  for k in range(5):
   mask=folds[tr]==k;oof[mask]=raw_views[k][tr[mask]]
  scaler=StandardScaler().fit(oof)
  views=[scaler.transform(z).astype(np.float32) for z in raw_views]
  assert all(np.isfinite(z).all() for z in views)
  payload={'train_ids':tr,'train_labels':y,'val_ids':parts['val'],
   'fold_of_node':folds,'features':feat,'risk_feature_names':np.asarray(NAMES),
   'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_}
  for k in range(6):payload[f'risk_view_{k}']=views[k];payload[f'label_context_{k}']=contexts_by_ratio[ratio][k]
  out=ASSETS/f'RGTAN_T-Finance_{ratio}_ExactRiskCrossfit_v1.npz'
  manifest=out.with_suffix('.json')
  tmp=Path(str(out)+'.partial');assert not tmp.exists()
  with tmp.open('xb') as f:
   np.savez_compressed(f,**payload);f.flush();os.fsync(f.fileno())
  os.replace(tmp,out)
  meta={'status':'RGTAN_TFINANCE_'+ratio+'_EXACT_RISK_CROSSFIT_PASS','model':'RGTAN','dataset':'T-Finance',
   'ratio':ratio,'train_count':COUNTS[ratio],'training_positive_count':POSITIVES[ratio],
   'validation_count':COUNTS['val'],'original_graph_edges':EDGES,
   'risk_features':'original_directed_edges_binary_exact_2hop_OOF_5fold',
   'OOF_scaler_fit_train_only':True,'test_labels_used':False,'validation_labels_used':False,
   'full_neighbor_sampling_replaced_for_downstream_model':True,'fanouts':[10,10],
   'benchmark_supervisor_approval_required':True,'author_commit':PIN,
   'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,
   'artifact_sha256':sha(out),'script_sha256':sha(__file__),'combined_gpu_twohop_seconds':round(time.perf_counter()-t0,3)}
  assert not manifest.exists();atomic_json(manifest,meta)
  print(f'RGTAN_T-Finance_{ratio}_CROSSFIT_BUILT=PASS SHA256={meta["artifact_sha256"]}',flush=True)
 print('RGTAN_T-Finance_I2_TR30_TR20_TR10_CROSSFITS_READY=PASS',flush=True)

if __name__=='__main__':generate()
