#!/usr/bin/env python3
"""Read-only identity/graph-capacity preflight. Does not tune or score held-out test."""
import hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import dgl
ROOT=Path('/workspace/comp8851')
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SEEDS=(2,42,72)
RATIOS=('TR10','TR20','TR30','TR40')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_file(path, expected):
    assert path.is_file(), 'STOP missing: '+str(path)
    actual=sha(path)
    assert actual==expected,(str(path),actual,expected)
    print('SHA256_PASS',path.name,flush=True)

def check_env():
    assert torch.__version__=='1.12.1+cu113',torch.__version__
    assert dgl.__version__=='0.8.1',dgl.__version__
    assert torch.cuda.is_available(),'No available GPU'
    head=subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()
    assert head==PIN,head
    print('RGTAN_AUTHOR_ENV_GPU_PIN=PASS',flush=True)

def split_load(path, counts, n):
    with np.load(path,allow_pickle=False) as z:
        print('FROZEN_SPLIT_KEYS',list(z.files),flush=True)
        parts={k:np.asarray(z[k],dtype=np.int64).copy() for k in counts}
    for k,ids in parts.items():
        assert len(ids)==counts[k],(k,len(ids),counts[k])
        assert ids.ndim==1 and np.unique(ids).size==ids.size,(k,'duplicate')
        assert ids.min()>=0 and ids.max()<n,(k,'out of bounds')
    for a,b in zip(RATIOS,RATIOS[1:]):
        assert np.isin(parts[a],parts[b]).all(),(a,b,'not nested')
    for k in ('val','test'):
        assert np.intersect1d(parts['TR40'],parts[k]).size==0,(k,'overlaps train')
    assert np.intersect1d(parts['val'],parts['test']).size==0
    print('FROZEN_SPLIT_NESTING_AND_ISOLATION=PASS',flush=True)
    return parts

def save_report(dataset,report):
    out=ROOT/'diagnostics/rgtan'/dataset/'preflight_v1.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    report['dataset']=dataset
    report['tuning_started']=False
    report['test_scored']=False
    report['author_commit']=PIN
    if out.exists():
        old=json.loads(out.read_text())
        # Preflight runs are repeatable; log a new numbered copy, never overwrite.
        out=out.with_name('preflight_v1_repeat_'+str(int(time.time()))+'.json')
    with open(out,'x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print('REPORT',out,flush=True)

from dgl.data.utils import load_graphs
SOURCE=ROOT/'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
SRC_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
N=39357
COUNTS={'TR10':3935,'TR20':7870,'TR30':11806,'TR40':15742,'val':7872,'test':15743}

def main():
    assert os.uname().nodename=='f24d1bb2672d','STOP: wrong I2 machine'
    check_env()
    verify_file(SOURCE,SRC_SHA)
    verify_file(SPLIT,SPLIT_SHA)
    s=split_load(SPLIT,COUNTS,N)
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(z['seed'])==2 and int(z['source_nodes'])==N
        assert (int(z['source_normal']),int(z['source_fraud']))==(37553,1804)
    print('TFINANCE_LOADING_CANONICAL_GRAPH',flush=True)
    gs,_=load_graphs(str(SOURCE))
    assert len(gs)==1
    g=gs[0]
    assert g.is_homogeneous and g.num_nodes()==N and g.num_edges()==42445086
    assert {'feature','label'}.issubset(set(g.ndata))
    features=g.ndata['feature']
    raw=g.ndata['label']
    assert tuple(features.shape)==(N,10) and bool(torch.isfinite(features).all())
    assert tuple(raw.shape)==(N,2) and bool(((raw==0)|(raw==1)).all())
    assert bool((raw.sum(1)==1).all())
    y=raw.argmax(1).numpy().astype(np.int8)
    assert y[27028]==1
    positives={k:int(y[v].sum()) for k,v in s.items() if k!='test'}
    assert positives=={'TR10':181,'TR20':361,'TR30':541,'TR40':721,'val':361},positives
    # The source embeds labels but these are never forwarded into the RGTAN model.
    # Immediately drop all graph-side metadata. Test labels are not used for scoring.
    for k in list(g.ndata):del g.ndata[k]
    for k in list(g.edata):del g.edata[k]
    assert not list(g.ndata) and not list(g.edata)
    print('TFINANCE_TRAIN_VALIDATION_LABEL_AND_SPLIT_AUDIT=PASS; TEST_LABELS_NOT_USED_FOR_MODEL',flush=True)
    print('TFINANCE_FULL_GRAPH_LOADED_NODES',g.num_nodes(),'EDGES',g.num_edges(),flush=True)
    # Quantify graph fanout with no 2-hop sampling: full-neighbour 2-hop on this graph
    # could allocate approximately the entire 42M-edge graph for even a small batch.
    deg=g.in_degrees().numpy()
    assert len(deg)==N
    vals=np.percentile(deg,[50,90,95,99]).tolist()
    print('TFINANCE_IN_DEGREE_STATS mean=',round(float(deg.mean()),2),
      'p50_p90_p95_p99=',[int(round(x)) for x in vals],
      'max=',int(deg.max()),flush=True)
    # Bounded one-hop CPU sampling only: does not forward or train the model.
    from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler
    seed=int(s['TR40'][0])
    tic=time.perf_counter()
    loader=NodeDataLoader(g,torch.tensor([seed]),MultiLayerFullNeighborSampler(1),
       batch_size=1,shuffle=False,drop_last=False,num_workers=0)
    inputs,outputs,blocks=next(iter(loader))
    assert len(blocks)==1 and len(outputs)==1 and int(outputs[0])==seed
    assert int(blocks[0].num_edges())==int(deg[seed])
    print('TFINANCE_ONE_HOP_ONLY_SAMPLE=PASS edges=',int(blocks[0].num_edges()),
      'seconds=',round(time.perf_counter()-tic,3),flush=True)
    print('TFINANCE_FULL_TWO_HOP_SAMPLE=NOT_ATTEMPTED (memory safeguard)',flush=True)
    print('TRAINING_STARTED=NO TEST_SCORING=NO',flush=True)
    save_report('tfinance',{'model':'RGTAN','ratio':'TR40','nodes':N,'features':10,
      'original_directed_edges':int(g.num_edges()),'train_count':len(s['TR40']),
      'val_count':len(s['val']),'test_count':len(s['test']),
      'train_positive':positives['TR40'],'validation_positive':positives['val'],
      'source_sha256':SRC_SHA,'split_sha256':SPLIT_SHA,'canonical_fraud_label_count_provenance':1804,
      'graph_label_metadata_dropped':True,'in_degree_mean':float(deg.mean()),
      'in_degree_percentiles_50_90_95_99':vals,'max_in_degree':int(deg.max()),
      'one_hop_seed':seed,'one_hop_edge_count':int(blocks[0].num_edges()),
      'full_two_hop_sampler_executed':False,'full_model_initialized':False,
      'real_train_crossfit_built':False,'test_scoring_performed':False})
    print('I2_TFINANCE_RGTAN_PREFLIGHT=PASS',flush=True)
if __name__=='__main__':main()
