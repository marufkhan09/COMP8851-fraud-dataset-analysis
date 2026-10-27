#!/usr/bin/env python3
"""Original EOGFD, train-only diagnostic pilots on FDCompCN and Elliptic.
Never tunes a parameter, reads test labels, or scores test examples.
"""
from __future__ import annotations
import os, sys, json, csv, hashlib, time, socket, subprocess, traceback
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/comp8851')
AUDIT=ROOT/'diagnostics/eogfd_newi_57212396'
REPO=ROOT/'repos/EOGFD'
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
SPLIT_SEED=2
SPECS={
 'FDCompCN':{
  'source':[('assets/fdcompcn/source/comp.dgl','e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa')],
  'split':'fdcompcn_seed2_nested_splits.npz','split_path':'/workspace/uploads/ass2_split_candidates/FDCompCN/fdcompcn_seed2_nested_splits.npz',
  'split_sha':'5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168',
  'n':5317,'d':57,'train':{'TR10':531,'TR20':1062,'TR30':1594,'TR40':2126,'val':1064,'test':2127},'pilot_d':3},
 'Elliptic':{
  'source':[('assets/elliptic/source/elliptic_txs_features.csv','fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0'),('assets/elliptic/source/elliptic_txs_classes.csv','93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493'),('assets/elliptic/source/elliptic_txs_edgelist.csv','a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e')],
  'split':'elliptic_seed2_nested_splits.npz','split_path':'/workspace/uploads/ass2_split_candidates/Elliptic/elliptic_seed2_nested_splits.npz',
  'split_sha':'1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc',
  'n':203769,'d':165,'train':{'TR10':4543,'TR20':9553,'TR30':13670,'TR40':18889,'val':8726,'test':18949},'pilot_d':3}
}

# no source labels enter graph.ndata, no test labels indexed

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def say(*a):print(*a,flush=True)

def check_inputs(dataset):
 c=SPECS[dataset]
 assert (REPO/'.git').is_dir(),'STOP missing pinned EOGFD repo'
 commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 assert commit==PIN,('UPSTREAM_COMMIT_CHANGED',commit)
 status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
 assert not status,('UPSTREAM_SOURCE_DIRTY',status)
 for path,expected in c['source']:
  p=ROOT/path
  assert p.is_file(),('MISSING_CANONICAL_SOURCE',str(p))
  assert sha(p)==expected,('SOURCE_SHA_MISMATCH',str(p))
 p=Path(c['split_path']);assert p.is_file(),('MISSING_FROZEN_SPLIT',str(p))
 assert sha(p)==c['split_sha'],'FROZEN_SPLIT_SHA_MISMATCH'
 with np.load(p,allow_pickle=False) as z:
  sk='split_seed' if dataset=='Elliptic' else 'seed'
  assert int(z[sk].item())==2,'WRONG_SPLIT_SEED'
  s={k:np.asarray(z[k],dtype=np.int64).copy() for k in c['train']}
 n=c['n']
 for key,actual in s.items():
  assert actual.shape==(c['train'][key],),(dataset,key,'shape')
  assert len(np.unique(actual))==len(actual),(dataset,key,'duplicates')
  assert len(actual)>0 and actual.min()>=0 and actual.max()<n,(dataset,key,'out of range')
 for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
  assert np.isin(s[a],s[b]).all(),(dataset,a,b,'NOT_NESTED')
 for a,b in (('TR40','val'),('TR40','test'),('val','test')):
  assert not np.intersect1d(s[a],s[b]).size,(dataset,a,b,'OVERLAP')
 say(dataset+'_PINNED_AUTHOR_DATA_SPLIT_AND_NESTING=PASS')
 return s

def fdcompcn_graph(s):
 import torch, dgl
 from dgl.data.utils import load_graphs
 f=ROOT/'assets/fdcompcn/source/comp.dgl'
 g,_=load_graphs(str(f));assert len(g)==1
 source=g[0]
 assert len(source.ntypes)==1,('Unexpected node types',source.ntypes)
 node_type=source.ntypes[0]
 assert all(e[0]==node_type and e[2]==node_type for e in source.canonical_etypes),('FDCompCN cross-type edges',source.canonical_etypes)
 assert source.num_nodes()==5317 and source.num_edges()==30752
 assert 'feature' in source.ndata and 'label' in source.ndata
 xx=source.ndata['feature'].float()
 assert xx.shape==(5317,57) and bool(xx.isfinite().all())
 # Only training + validation labels. Never index s['test'] from source label tensor.
 y=source.ndata['label'].long().view(-1)
 labels=torch.cat([y[torch.from_numpy(s['TR40'])],y[torch.from_numpy(s['val'])]])
 assert bool(((labels==0)|(labels==1)).all()) and 0<int(labels.sum())<len(labels)
 reldata={e:source.edges(etype=e,order='eid') for e in source.canonical_etypes}
 assert len(reldata)>=2,('FDCompCN relations unexpectedly collapsed',list(reldata))
 # Strip every original node/edge field from model graph; preserve directed edges and native etype names.
 safe=dgl.heterograph(reldata,num_nodes_dict={node_type:5317})
 assert not list(safe.ndata.keys()) and not list(safe.edata.keys())
 assert safe.num_edges()==30752
 say('FDCompCN_RELATIONS_PRESERVED',[(e,int(safe.num_edges(e))) for e in safe.canonical_etypes])
 ids=s['TR40'].copy(); yt=y[torch.from_numpy(ids)].clone(); assert yt.numel()==2126
 return safe,xx.clone(),ids,yt, {'canonical_etypes':[list(e) for e in safe.canonical_etypes], 'edge_count':int(safe.num_edges()),'node_count':int(safe.num_nodes()),'features':57,'static_transductive_unlabeled_test_node_features_present':True,'test_labels_in_model_graph':False,'model_graph_labels_stripped':True}

def elliptic_graph(s):
 import torch, dgl, pandas as pd
 folder=ROOT/'assets/elliptic/source'
 dt={0:np.int64,1:np.int16};dt.update({i:np.float32 for i in range(2,167)})
 f=pd.read_csv(folder/'elliptic_txs_features.csv',header=None,dtype=dt)
 assert f.shape==(203769,167)
 txids=f.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
 times=f.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
 xx=f.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
 del f
 assert np.unique(txids).size==203769 and int(times.min())==1 and int(times.max())==49
 assert bool(np.isfinite(xx).all())
 cutoffs={'TR10':3,'TR20':7,'TR30':12,'TR40':20}
 for k,v in cutoffs.items():assert bool((times[s[k]]<=v).all()),(k,'future train ids')
 assert bool(((times[s['val']]>=21)&(times[s['val']]<=31)).all())
 assert bool((times[s['test']]>=32).all())
 # Parse labels of frozen TRAIN (TR40) and VAL only; every other label value is ignored.
 allowed_ids=set(txids[np.r_[s['TR40'],s['val']]].tolist()); actual={}
 with (folder/'elliptic_txs_classes.csv').open(newline='') as file:
  reader=csv.DictReader(file);assert set(reader.fieldnames)=={'txId','class'}
  for row in reader:
   id=int(row['txId'])
   if id in allowed_ids:
    assert id not in actual
    assert row['class'] in ('1','2')
    actual[id]=int(row['class']=='1')
 assert len(actual)==18889+8726
 yy=torch.tensor([actual[int(id)] for id in txids[s['TR40']]],dtype=torch.long)
 assert int(yy.sum())==1771
 vpos=sum(actual[int(id)] for id in txids[s['val']]);assert vpos==1289
 del actual, allowed_ids
 ed=pd.read_csv(folder/'elliptic_txs_edgelist.csv',dtype=np.int64)
 assert tuple(ed.columns)==('txId1','txId2') and len(ed)==234355
 order=pd.Index(txids)
 src=order.get_indexer(ed['txId1']).astype(np.int64)
 dst=order.get_indexer(ed['txId2']).astype(np.int64)
 del ed
 assert bool((src>=0).all() and (dst>=0).all())
 assert bool((times[src]==times[dst]).all()),'unexpected cross-time original edge'
 visible=np.flatnonzero(times<=20)
 mapping=np.full(203769,-1,dtype=np.int64);mapping[visible]=np.arange(len(visible))
 kept=(times[src]<=20)&(times[dst]<=20)
 uu=mapping[src[kept]];vv=mapping[dst[kept]]
 assert uu.min()>=0 and vv.min()>=0
 assert len(visible)==90762 and len(uu)==197569,(len(visible),len(uu))
 # Single truthful original directed relation; no pseudo-label or synthetic relation added.
 safe=dgl.heterograph({('tx','original_transaction_edge','tx'):(torch.from_numpy(uu.copy()),torch.from_numpy(vv.copy()))},num_nodes_dict={'tx':len(visible)})
 assert len(safe.canonical_etypes)==1 and safe.num_edges()==197569
 train_local=mapping[s['TR40']]
 assert bool((train_local>=0).all())
 assert bool((mapping[s['val']]==-1).all()) and bool((mapping[s['test']]==-1).all())
 # Existing original features are used unchanged, without future-based feature fitting.
 x=torch.from_numpy(xx[visible].copy())
 assert bool(x.isfinite().all())
 say('ELLIPTIC_CAUSAL_TRAIN_PREFIX=PASS times_le_20 nodes=',len(visible),'edges=',safe.num_edges())
 return safe,x,train_local,yy, {'canonical_etypes':[list(e) for e in safe.canonical_etypes],'edge_count':int(safe.num_edges()),'node_count':int(safe.num_nodes()),'features':165,'train_cutoff':20,'validation_cutoff':31,'test_start':32,'future_nodes_visible_during_training':False,'future_edges_visible_during_training':False,'test_labels_accessed':False,'single_original_relation_conditional_methodology_requires_venus_confirmation':True}

def pilot(dataset, graph, x, ids, yt, metadata):
 import torch, dgl
 sys.path.insert(0,str(REPO))
 from models.eogfd import EOGFDHeteroModel
 assert torch.cuda.device_count()==1,('STOP: expected one visible GPU',torch.cuda.device_count())
 seed=2;torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
 device=torch.device('cuda:0')
 gpu_g=graph.to(device);gpu_x=x.to(device)
 model=EOGFDHeteroModel(in_feats=x.size(1),hid_feats=64,num_classes=2,d=3,beta_p=1,population_size=3,mutation_rate=.1).to(device)
 optim=torch.optim.Adam(model.parameters(),lr=.003)
 pos=torch.tensor(ids,dtype=torch.long,device=device);train_y=yt.to(device)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 start=time.perf_counter();model.train();optim.zero_grad()
 logits,_=model(gpu_g,gpu_x)
 assert tuple(logits.shape)==(graph.num_nodes(),2)
 loss=torch.nn.functional.cross_entropy(logits[pos],train_y)
 assert torch.isfinite(loss).item()
 loss.backward();optim.step();torch.cuda.synchronize()
 elapsed=time.perf_counter()-start
 say(dataset+'_REAL_FULL_GRAPH_TRAIN_LABELS_ONLY_PILOT=PASS seconds=%.3f peak_gpu_gib=%.3f loss=%.5f'%(elapsed,torch.cuda.max_memory_allocated()/2**30,float(loss)))
 return {'loss':float(loss.item()),'seconds':round(elapsed,3),'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/2**30,4),'training_labels_only':True,'test_scored':False,'tuning_started':False,'model':'EOGFDHeteroModel','pilot_population_size':3,'pilot_degree':3,'pilot_hidden_dim':64,'graph':metadata}

def main():
 import torch, dgl, scipy
 AUDIT.mkdir(parents=True,exist_ok=True)
 assert torch.cuda.is_available(),'STOP: CUDA unavailable'
 assert torch.__version__=='1.12.1+cu113',torch.__version__
 assert dgl.__version__=='0.8.1',dgl.__version__
 say('EOGFD_NEWI_GPU_ENV=PASS hostname='+socket.gethostname()+' gpu='+torch.cuda.get_device_name(0))
 for dataset in ('FDCompCN','Elliptic'):
  output=AUDIT/(dataset+'_Stage1_Stage2_ReadOnlyThenOneStep.json')
  assert not output.exists(),('STOP existing successful pilot; do not repeat',str(output))
  say('===',dataset,'CANONICAL_PREFLIGHT_START ===')
  ids=check_inputs(dataset)
  g,x,train_nodes,y,meta=fdcompcn_graph(ids) if dataset=='FDCompCN' else elliptic_graph(ids)
  assert len(train_nodes)==SPECS[dataset]['train']['TR40']
  say(dataset+'_CANONICAL_GRAPH_ADAPTER=PASS TEST_SCORED=NO')
  result=pilot(dataset,g,x,train_nodes,y,meta)
  result.update({'dataset':dataset,'instance_id_user_identified':'57212396','hostname':socket.gethostname(),'upstream_commit':PIN,'source_sha256':dict(SPECS[dataset]['source']),'split_sha256':SPECS[dataset]['split_sha'],'train_count':len(train_nodes),'validation_count':len(ids['val']),'test_count':len(ids['test']),'tuning_started':False,'heldout_test_evaluated':False,'status':dataset+'_REAL_GRAPH_PILOT_PASS'})
  with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
  say(dataset+'_STAGE1_STAGE2=PASS AUDIT='+str(output))
 say('EOGFD_NEWI_FDCOMPCN_ELLIPTIC_BOTH_PILOTS=PASS TEST_SCORED=NO TUNING_STARTED=NO')
if __name__=='__main__':
 try:main()
 except Exception as e:
  say('EOGFD_NEWI_STAGE12_STOP_ERROR='+repr(e));traceback.print_exc();sys.exit(2)
