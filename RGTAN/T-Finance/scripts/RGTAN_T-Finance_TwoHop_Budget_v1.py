#!/usr/bin/env python3
"""T-Finance full-two-hop expected block-size audit, NO unbounded sampler.

Uses original graph topology and frozen TR40 node IDs, never test labels or
training; predicts 2-layer full-neighbor block edge counts from exact in-neighbor
sets, for representative training nodes, without materializing two-hop blocks.
"""
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import dgl

ROOT=Path('/workspace/comp8851')
DATA=ROOT/'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SRC_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
OUT=ROOT/'diagnostics/rgtan/tfinance/RGTAN_T-Finance_TwoHop_ExpansionBudget_v1.json'


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def bound():
    # Cgroup v2 memory limit applies to rented container, unlike host free -h.
    p=Path('/sys/fs/cgroup/memory.max')
    if p.is_file():
        x=p.read_text().strip()
        return 'unlimited' if x=='max' else int(x)
    p=Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    if p.is_file():return int(p.read_text().strip())
    return 'not reported'


def main():
    assert socket.gethostname()=='f24d1bb2672d','Wrong instance: I2 only'
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
    assert torch.cuda.is_available()
    assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==PIN
    assert digest(DATA)==SRC_SHA and digest(SPLIT)==SPLIT_SHA
    report=ROOT/'diagnostics/rgtan/tfinance/preflight_v1.json'
    assert report.is_file() and json.loads(report.read_text()).get('full_two_hop_sampler_executed') is False
    with np.load(SPLIT,allow_pickle=False) as z:
        tr=np.asarray(z['TR40'],dtype=np.int64).copy()
        assert tr.size==15742
    start=time.perf_counter()
    graphs,_=dgl.load_graphs(str(DATA))
    assert len(graphs)==1
    g=graphs[0]
    assert g.num_nodes()==39357 and g.num_edges()==42445086
    # Protect against source graph label exposure beyond graph-loading stage.
    for k in list(g.ndata):del g.ndata[k]
    for k in list(g.edata):del g.edata[k]
    deg=g.in_degrees().numpy().astype(np.int64)
    assert deg.size==39357
    train_deg=deg[tr]
    ordered=np.argsort(train_deg,kind='stable')
    # Pick representative seeds from low through high quantiles.
    quantiles=(0.0,0.1,0.25,0.5,0.75,0.9,0.99,1.0)
    examples=[]
    proc=psutil.Process(os.getpid())
    for q in quantiles:
        rank=min(len(ordered)-1,int(q*(len(ordered)-1)))
        seed=int(tr[ordered[rank]])
        inbound=g.in_edges(seed,form='uv')[0].numpy()
        nbrs=np.unique(inbound)
        # DGL's two-layer FULL-neighbor blocks need all inbound edges of the
        # unique layer-1 source nodes. Exactly equals this edge count.
        second_block_edges=int(deg[nbrs].sum(dtype=np.int64))
        entry={
          'degree_quantile':q,'training_node':seed,
          'layer2_seed_in_degree_edges':int(deg[seed]),
          'unique_layer1_source_nodes':int(len(nbrs)),
          'layer1_predecessor_edges':second_block_edges,
          'total_edges_in_two_layer_blocks':int(deg[seed])+second_block_edges,
          'fraction_of_original_graph_edges':round(second_block_edges/int(g.num_edges()),6),
          'process_peak_current_rss_gib':round(proc.memory_info().rss/1024**3,3)
        }
        assert second_block_edges <= int(g.num_edges())
        examples.append(entry)
        print('TFINANCE_DEGREE_QUANTILE_AND_TWOHOP_EXPANSION',json.dumps(entry,sort_keys=True),flush=True)
    meta={
     'status':'RGTAN_TFINANCE_TWOHOP_EXPANSION_BUDGET_PASS',
     'description':'Analytical exact block edge counts, no two-hop block allocated',
     'train_seed_ratio':'TR40',
     'graph_nodes':int(g.num_nodes()),'graph_directed_edges':int(g.num_edges()),
     'container_memory_cgroup_bytes':bound(),
     'gpu_name':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
     'read_only_graph_source_sha256':SRC_SHA,'split_sha256':SPLIT_SHA,
     'author_commit':PIN,'seeds':examples,
     'two_hop_sampler_executed':False,'model_forward_executed':False,
     'train_or_tune_started':False,'test_labels_accessed':False,'test_scored':False,
     'elapsed_seconds':round(time.perf_counter()-start,2),'script_sha256':digest(__file__)
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():
        old=json.loads(OUT.read_text())
        assert old.get('status')=='RGTAN_TFINANCE_TWOHOP_EXPANSION_BUDGET_PASS', 'Existing diagnostic status mismatch'
        assert old.get('script_sha256')==digest(__file__), 'Existing diagnostic source changed'
        assert old.get('read_only_graph_source_sha256')==SRC_SHA, 'Existing diagnostic graph changed'
        assert old.get('split_sha256')==SPLIT_SHA, 'Existing diagnostic split changed'
        assert old.get('two_hop_sampler_executed') is False and old.get('test_scored') is False
        print('RGTAN_T-Finance_EXISTING_TWOHOP_BUDGET_REUSED=PASS',flush=True)
        return
    with OUT.open('x') as f:
        json.dump(meta,f,indent=2,sort_keys=True);f.write('\n')
    print('RGTAN_TFINANCE_TWOHOP_EXPANSION_BUDGET=PASS',flush=True)
    print('REPORT',OUT,flush=True)
    print('TUNING_STARTED=NO TEST_SCORED=NO',flush=True)

if __name__=='__main__':main()
