"""Read-only YelpChi graph probe. Sample-only and optional bounded one-step model GPU test.
No checkpoint, model selection, trained weights, or test score is produced.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import dgl
from dgl.dataloading import NodeDataLoader,MultiLayerFullNeighborSampler

from rgtan_yelpchi_core_v1 import (N,NF,EXPECTED_EDGES,ROOT,SRC,
                                    assert_environment,verify_input_files,load_splits,
                                    get_graph_and_features)

names=['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['sample','model-one-batch'])
    args=p.parse_args()
    assert_environment();verify_input_files()
    parts=load_splits()
    t0=time.perf_counter()
    adj,features=get_graph_and_features()
    g=dgl.from_scipy(adj)
    assert g.num_nodes()==N and g.num_edges()==EXPECTED_EDGES and len(g.ndata)==0
    print('CANONICAL_GRAPH_AND_SPLITS=PASS edges='+str(g.num_edges()),flush=True)
    print('BUILD_GRAPH_SECONDS=%.2f'%(time.perf_counter()-t0),flush=True)
    torch.set_num_threads(8)
    ids=parts['TR40']
    records=[]
    largest=None
    for bsz in (1,4,32):
        tic=time.perf_counter()
        loader=NodeDataLoader(g,torch.from_numpy(ids[:bsz].copy()),
               MultiLayerFullNeighborSampler(2),batch_size=bsz,shuffle=False,
               drop_last=False,num_workers=0)
        inputs,outputs,blocks=next(iter(loader))
        sec=time.perf_counter()-tic
        assert len(blocks)==2 and len(outputs)==bsz
        assert torch.equal(blocks[0].srcdata[dgl.NID],inputs)
        assert torch.equal(blocks[-1].dstdata[dgl.NID],outputs)
        edges=[int(block.num_edges()) for block in blocks]
        rec={'batch':bsz,'input_nodes':int(len(inputs)),
             'block_edges':edges,'total_block_edges':sum(edges),
             'sampling_seconds':round(sec,3)}
        records.append(rec)
        print('FULLNEIGHBOR_SAMPLE '+json.dumps(rec,sort_keys=True),flush=True)
        if bsz==32:largest=(inputs,outputs,blocks)
    print('GRAPH_SAMPLE_ALL_THREE_BATCHES=PASS',flush=True)
    print('EXPECTED_TR40_TRAIN_BATCHES_PER_EPOCH='+str((len(ids)+31)//32),flush=True)
    print('EXPECTED_VALIDATION_BATCHES_PER_EPOCH='+str((len(parts['val'])+31)//32),flush=True)
    if args.mode=='sample':
        print('MODEL_TRAINING_PERFORMED=NO TEST_SCORING_PERFORMED=NO',flush=True)
        print('I3_RGTAN_YELPCHI_GRAPH_PROBE=PASS',flush=True)
        return
    from methods.rgtan.rgtan_model import RGTAN
    random.seed(2);np.random.seed(2);torch.manual_seed(2);torch.cuda.manual_seed_all(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.reset_peak_memory_stats()
    x_ids,out_ids,blocks=largest
    zero=np.zeros((N,6),dtype=np.float32)
    model=RGTAN(in_feats=NF,hidden_dim=32,n_layers=2,n_classes=2,
        heads=[4,4],activation=torch.nn.PReLU(),drop=[.2,.1],
        device='cuda:0',gated=True,ref_df=pd.DataFrame(features),
        cat_features={},neigh_features={n:torch.from_numpy(zero[:,j]).to('cuda:0') for j,n in enumerate(names)},
        nei_att_head=4).to('cuda:0')
    model.train()
    # Dummy *unlabeled* label context and ZERO risks for an architecture-only
    # GPU memory test; these are NOT final training/risk features.
    input_ids=x_ids.numpy()
    tensor=torch.from_numpy(features[input_ids]).to('cuda:0')
    ctx=torch.full((len(x_ids),),2,dtype=torch.long,device='cuda:0')
    risk={name:torch.zeros((len(x_ids),),device='cuda:0') for name in names}
    gpu_blocks=[b.to('cuda:0') for b in blocks]
    optim=torch.optim.Adam(model.parameters(),lr=.001*(32/1024)**.5,weight_decay=.0001)
    tic=time.perf_counter()
    logits=model(gpu_blocks,tensor,ctx,{},risk)
    targets=torch.zeros(len(out_ids),dtype=torch.long,device='cuda:0')
    loss=torch.nn.functional.cross_entropy(logits,targets)
    loss.backward();optim.step();torch.cuda.synchronize()
    print('ARCHITECTURE_ONLY_STEP_SECONDS=%.3f'%(time.perf_counter()-tic),flush=True)
    print('ARCHITECTURE_ONLY_GPU_PEAK_GIB=%.3f'%(torch.cuda.max_memory_allocated()/1024**3),flush=True)
    print('DUMMY_DATA_NOT_A_TRAINING_TRIAL=YES; TEST_SCORING_PERFORMED=NO',flush=True)
    print('I3_RGTAN_YELPCHI_MODEL_ONE_BATCH=PASS',flush=True)

if __name__=='__main__':
    main()
