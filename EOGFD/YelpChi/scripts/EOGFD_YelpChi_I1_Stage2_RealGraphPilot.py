#!/usr/bin/env python3
"""One (only one) real full-graph EOGFD forward/backprop cost pilot.
Requires pinned unmodified author implementation. Never indexes held-out test labels,
never reads an intermediate tuning result, no checkpoint writes, no test inference.
"""
import sys, os, time, socket, json, hashlib, subprocess, resource, traceback
from pathlib import Path

CFG=json.loads(Path(__file__).with_name('EOGFD_YelpChi_I1_Stage2_Config.json').read_text())
PFX='EOGFD_'+CFG['dataset']+'_'+CFG['id']
LOG=Path('/workspace/comp8851/logs/eogfd'); LOG.mkdir(parents=True,exist_ok=True)
OUTPUT=LOG/('EOGFD_'+CFG['dataset']+'_'+CFG['id']+'_Stage2_FullGraphPilot.json')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''): h.update(x)
    return h.hexdigest()

def main():
    import numpy as np
    import torch, scipy.io, scipy.sparse as sp, dgl
    assert socket.gethostname()==CFG['host'],('WRONG INSTANCE',socket.gethostname())
    assert torch.cuda.is_available(),'CUDA unavailable. No CPU full graph pilot.'
    assert not OUTPUT.exists(),('PILOT ALREADY COMPLETED, DO NOT RERUN',str(OUTPUT))
    src=Path('/workspace/comp8851/assets/eogfd')/CFG['dataset']/CFG['src_name']
    spl=Path('/workspace/uploads/ass2_split_candidates')/CFG['dataset']/CFG['split_name']
    for k,p in [('source',src),('split',spl)]:
        assert p.is_file(),('MISSING',str(p))
        s=sha(p)
        assert s==CFG[k+'_sha'],('WRONG HASH',k,s)
        print(PFX+'_'+k.upper()+'_SHA256=PASS',flush=True)
    repo=Path('/workspace/comp8851/repos/EOGFD')
    assert repo.is_dir(),('RUN STAGE1 PINNED SOURCE SMOKE FIRST',str(repo))
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert commit==CFG['author_commit'],('WRONG UPSTREAM COMMIT',commit)
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(), 'UPSTREAM TREE HAS LOCAL MODIFICATIONS'
    sys.path.insert(0,str(repo))
    from models.eogfd import EOGFDHeteroModel
    assert not hasattr(__import__('models.eogfd',fromlist=['EOGFDHeteroModel']),'EOGFDHetero'), 'Upstream source changed: re-audit class aliases'
    with np.load(spl,allow_pickle=False) as z:
        ratios={r:z[r].copy() for r in ('TR40','TR30','TR20','TR10')}
        val_idx=z['val'].copy()
        test_idx=z['test'].copy()  # Membership audit ONLY. Do not use for labels/model evaluation.
        assert int(z['seed'].reshape(-1)[0])==2
        assert int(z['source_nodes'].reshape(-1)[0])==CFG['nodes']
    for r,n in zip(('TR40','TR30','TR20','TR10'),CFG['train_lengths']):
        assert len(ratios[r])==n,(r,len(ratios[r]),n)
        assert ratios[r].dtype.kind in 'iu'
        assert np.unique(ratios[r]).size==n
        assert min(ratios[r])>=0 and max(ratios[r])<CFG['nodes']
    assert len(val_idx)==CFG['val_n'] and len(test_idx)==CFG['test_n']
    assert np.unique(val_idx).size==len(val_idx) and np.unique(test_idx).size==len(test_idx)
    assert not np.intersect1d(val_idx,test_idx).size
    for r in ratios:
        assert not np.intersect1d(ratios[r],test_idx).size
        assert not np.intersect1d(ratios[r],val_idx).size
    for a,b in [('TR10','TR20'),('TR20','TR30'),('TR30','TR40')]:
        assert np.isin(ratios[a],ratios[b]).all(),(a,b,'NOT NESTED')
    print(PFX+'_FROZEN_SPLIT_MEMBERSHIP_NESTING_AND_TEST_ISOLATION=PASS',flush=True)
    start=time.perf_counter()
    mat=scipy.io.loadmat(src)
    feature=mat['features']
    if sp.issparse(feature): feature=feature.toarray()
    feature=np.asarray(feature,dtype=np.float32)
    assert feature.shape==(CFG['nodes'],CFG['input_dim']), feature.shape
    assert np.isfinite(feature).all(),'NONFINITE FEATURE'
    # Original .mat loader necessarily materializes label vector, but we index ONLY TR40
    # training labels for loss; never index test or use test for any normalization.
    all_labels=np.asarray(mat['label']).reshape(-1)
    assert len(all_labels)==CFG['nodes']
    train_idx=np.asarray(ratios['TR40'],dtype=np.int64)
    y_train=np.asarray(all_labels[train_idx],dtype=np.int64)
    assert set(np.unique(y_train)).issubset({0,1}), np.unique(y_train).tolist()
    assert np.unique(y_train).size==2
    # Remove labels from graph/source container and do not construct test labels.
    del all_labels
    mat.pop('label',None)
    edge_data={}
    counts={}
    for rel in CFG['relations']:
        mx=mat[rel]
        assert sp.issparse(mx) and mx.shape==(CFG['nodes'],CFG['nodes']), (rel,getattr(mx,'shape',None))
        coo=mx.tocoo(copy=False)
        row=np.asarray(coo.row,dtype=np.int64)
        col=np.asarray(coo.col,dtype=np.int64)
        counts[rel]=int(len(row))
        edge_data[('user',rel,'user')]=(torch.from_numpy(row),torch.from_numpy(col))
    del mat
    print(PFX+'_REAL_GRAPH_RELATION_EDGES='+json.dumps(counts,sort_keys=True),flush=True)
    print(PFX+'_STAGE2_BUILDING_FULL_HETEROGRAPH=START',flush=True)
    graph=dgl.heterograph(edge_data,num_nodes_dict={'user':CFG['nodes']})
    del edge_data
    assert graph.num_nodes()==CFG['nodes']
    device=torch.device('cuda:0')
    graph=graph.to(device)
    x=torch.from_numpy(feature).to(device)
    train_ix=torch.as_tensor(train_idx,dtype=torch.long,device=device)
    y=torch.as_tensor(y_train,dtype=torch.long,device=device)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    torch.cuda.reset_peak_memory_stats(device)
    model=EOGFDHeteroModel(CFG['input_dim'],64,2,d=CFG['order'],beta_p=1,population_size=3,mutation_rate=0.1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    weight=torch.tensor([1.,float((y==0).sum())/max(1,float((y==1).sum()))],device=device,dtype=torch.float32)
    t=time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    print(PFX+'_STAGE2_ONE_FULL_GRAPH_FORWARD_BACKWARD=START population_size=3 training_labels_only=YES',flush=True)
    logits,theta=model(graph,x)
    assert tuple(logits.shape)==(CFG['nodes'],2)
    # TR40 *train-only* loss: no val/test labels or test logits indexed here.
    loss=torch.nn.functional.cross_entropy(logits[train_ix],y,weight=weight)
    assert bool(torch.isfinite(loss).item()),'NONFINITE TRAIN LOSS'
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-t
    peak=torch.cuda.max_memory_allocated(device)/(1024**3)
    reserved=torch.cuda.max_memory_reserved(device)/(1024**3)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
    evidence={
        'model':'EOGFD','dataset':CFG['dataset'],'instance_id':CFG['id'],
        'upstream_commit':CFG['author_commit'],'source_sha256':CFG['source_sha'],
        'split_sha256':CFG['split_sha'],'graph_relation_edge_counts':counts,
        'nodes':CFG['nodes'],'input_features':CFG['input_dim'],
        'training_ratio':'TR40','train_n':int(len(train_idx)),'train_positives':int(y_train.sum()),
        'pilot_model':'EOGFDHeteroModel','population_size':3,'mutation_rate':0.1,
        'order':CFG['order'],'beta_p':1,'hidden_dim':64,'iterations':1,
        'epochs':1,'pilot_only_not_hparam_tuning':True,
        'full_graph_forward_backward_s':round(elapsed,5),
        'end_to_end_s':round(time.perf_counter()-start,5),
        'gpu_peak_allocated_gib':round(peak,5),
        'gpu_peak_reserved_gib':round(reserved,5),
        'cpu_peak_rss_gib':round(rss,5),
        'train_only_loss':float(loss.detach().cpu()),
        'test_labels_indexed':False,'test_scores_computed':False,
        'validation_used_for_selection':False,'training_checkpoint_saved':False,
        'note':'Pilot population 3 rather than planned full 12-trial population grid, for cost measurement only.'}
    import tempfile, os
    with tempfile.NamedTemporaryFile('w',dir=LOG,prefix='._eogfd_stage2_',delete=False) as out:
        json.dump(evidence,out,indent=2,sort_keys=True);out.write('\n');tmp=Path(out.name)
    os.replace(tmp,OUTPUT)
    print(PFX+'_STAGE2_FULL_REAL_HETEROGRAPH_FORWARD_BACKWARD=PASS',flush=True)
    print(PFX+'_STAGE2_ONE_STEP_SECONDS='+str(round(elapsed,3)),flush=True)
    print(PFX+'_STAGE2_PEAK_GPU_GIB='+str(round(peak,3)),flush=True)
    print(PFX+'_STAGE2_FULL_GRAPH_PILOT=PASS TEST_SCORED=NO TUNING_STARTED=NO',flush=True)
    print(PFX+'_STAGE2_RECORD='+str(OUTPUT),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(PFX+'_STAGE2_STOP_ERROR='+repr(e),flush=True)
        traceback.print_exc()
        sys.exit(2)
