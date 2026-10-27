#!/usr/bin/env python3
"""RGTAN x T-Finance, frozen TR40 and validation-only candidate 12-trial tune.

Original canonical graph and author model, train-only 5-fold OOF label/risk
contexts. Deviation: capped 10+10 neighbor sampling rather than full neighbors.
No test label extraction, no test scoring, no evaluation against test nodes.
The candidate configuration is NOT a validated author-identical benchmark;
benchmark treatment requires supervisor approval of sampling method change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from sklearn.metrics import average_precision_score

ROOT = Path('/workspace/comp8851')
DATA = ROOT/'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
ARTIFACT = ROOT/'assets/rgtan/tfinance/RGTAN_T-Finance_TR40_ExactRiskCrossfit_v1.npz'
ARTIFACT_META = ARTIFACT.with_suffix('.json')
PILOT_META = ROOT/'diagnostics/rgtan/tfinance/RGTAN_T-Finance_TR40_Fanout10x10_Pilot_v1.json'
OUT = ROOT/'results/rgtan/tfinance/RGTAN_T-Finance_TR40_Tune12_Fanout10x10_Candidate_v1'
AUTHOR_REPO = ROOT/'repos/antifraud'
AUTHOR_COMMIT = '65e583f03ac5c5983e49cc35179fe7a65908b9c1'
DATA_SHA = 'b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA = 'de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
RISK_SHA = '798daba5c4a131159208d242b71e03ef3d1c79aa44f5cfd599c7fd97b98dab81'
N, FEATURES, EDGE_COUNT = 39357, 10, 42445086
NAMES = ('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')
FANOUTS = (10,10)
BATCH = 1024
SEED = 2
MAX_EPOCHS = 100
PATIENCE = 20
DEVICE = 'cuda:0'
GRID = [dict(hidden_dim=h, drop=[d,d/2], base_lr=lr)
        for h in (32,64) for d in (.2,.3) for lr in (.001,.003,.005)]
assert len(GRID) == 12


def checksum(p):
    h=hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.partial')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def save_checkpoint(path,obj):
    tmp=Path(str(path)+'.partial')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def validate_all():
    assert socket.gethostname()=='f24d1bb2672d','STOP: Wrong machine (I2 only)'
    assert torch.__version__=='1.12.1+cu113',torch.__version__
    assert dgl.__version__=='0.8.1',dgl.__version__
    assert torch.cuda.is_available() and 'A6000' in torch.cuda.get_device_name(0)
    actual=subprocess.check_output(['git','-C',str(AUTHOR_REPO),'rev-parse','HEAD'],text=True).strip()
    assert actual==AUTHOR_COMMIT,'STOP: Author source commit mismatch'
    for path,expected in ((DATA,DATA_SHA),(SPLIT,SPLIT_SHA),(ARTIFACT,RISK_SHA)):
        assert path.is_file() and checksum(path)==expected, f'STOP: SHA-256 mismatch for {path}'
    meta=json.loads(ARTIFACT_META.read_text())
    assert meta['status']=='RGTAN_TFINANCE_TR40_EXACT_RISK_CROSSFIT_PASS'
    assert meta['artifact_sha256']==RISK_SHA
    assert meta['source_sha256']==DATA_SHA and meta['split_sha256']==SPLIT_SHA
    assert meta['test_labels_used'] is False and meta['validation_labels_used'] is False
    assert meta['benchmark_supervisor_approval_required'] is True
    pilot=json.loads(PILOT_META.read_text())
    assert pilot['status']=='RGTAN_TFINANCE_CAPPED_FANOUT_PILOT_PASS'
    assert pilot['fanouts']==list(FANOUTS) and pilot['optimizer_steps']==0
    assert pilot['test_scored'] is False
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(z['seed'].item())==2
        assert int(z['source_fraud'].item())==1804
        assert int(z['source_normal'].item())==37553
        parts={k:np.asarray(z[k],dtype=np.int64).copy() for k in ('TR40','TR30','TR20','TR10','val')}
    assert len(parts['TR40'])==15742 and len(parts['val'])==7872
    for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
        assert np.isin(parts[a],parts[b]).all()
    assert not np.intersect1d(parts['TR40'],parts['val']).size
    with np.load(ARTIFACT,allow_pickle=False) as z:
        fields={k:z[k] for k in z.files}
    assert tuple(map(str,fields['risk_feature_names']))==NAMES
    assert np.array_equal(fields['train_ids'],parts['TR40'])
    assert np.array_equal(fields['val_ids'],parts['val'])
    assert fields['features'].shape==(N,FEATURES)
    assert len(fields['train_labels'])==15742 and int(fields['train_labels'].sum())==721
    assert fields['fold_of_node'].shape==(N,)
    assert set(fields['fold_of_node'][parts['TR40']])==set(range(5))
    val=parts['val'];tr=parts['TR40']
    for k in range(6):
        assert fields[f'risk_view_{k}'].shape==(N,6)
        assert fields[f'label_context_{k}'].shape==(N,)
        assert np.isfinite(fields[f'risk_view_{k}']).all()
        assert np.all(fields[f'label_context_{k}'][val]==2)
    assert np.all(fields['label_context_5'][tr]==fields['train_labels'])
    for k in range(5):
        held=tr[fields['fold_of_node'][tr]==k]
        assert len(held)>0
        assert np.all(fields[f'label_context_{k}'][held]==2)
    print('RGTAN_T-Finance_TR40_PINNED_DATA_SPLIT_RISK_AND_OOF=PASS',flush=True)
    return parts,fields


def load_source_graph_and_validation_only(parts, fields):
    graphs,_=dgl.load_graphs(str(DATA))
    assert len(graphs)==1
    g=graphs[0]
    assert g.is_homogeneous and g.num_nodes()==N and g.num_edges()==EDGE_COUNT
    assert 'label' in g.ndata and 'feature' in g.ndata
    # Only training labels and validation labels are selected. No test indices or
    # test truth are loaded or retained in the model, feature contexts or outputs.
    onehot=g.ndata['label']
    assert tuple(onehot.shape)==(N,2)
    assert bool((((onehot==0)|(onehot==1)).all()) and (onehot.sum(1)==1).all())
    # Decode ONLY selected training and validation rows, never the test partition.
    tr_y=onehot[torch.from_numpy(parts['TR40'].copy())].argmax(1).numpy().astype(np.int64,copy=True)
    val_y=onehot[torch.from_numpy(parts['val'].copy())].argmax(1).numpy().astype(np.int64,copy=True)
    assert np.array_equal(tr_y,fields['train_labels'])
    assert int(val_y.sum())==361
    del onehot
    # Explicitly remove all labels and original feature tensors from DGL graph;
    # the model receives features and masked context from pinned crossfit file.
    for key in list(g.ndata): del g.ndata[key]
    for key in list(g.edata): del g.edata[key]
    assert not g.ndata and not g.edata
    assert fields['features'].dtype in (np.float32,np.float64)
    assert np.isfinite(fields['features']).all()
    print('RGTAN_T-Finance_ORIGINAL_GRAPH_42M_EDGES_VALIDATION_ONLY=PASS',flush=True)
    return g,tr_y,val_y


class Engine:
    def __init__(self,g,fields,train_labels,val_labels,config,seed=SEED):
        from methods.rgtan.rgtan_model import RGTAN
        self.g=g;self.fields=fields;self.y=train_labels;self.vy=val_labels
        self.cfg=config;self.seed=seed
        self.tr=fields['train_ids'];self.val=fields['val_ids']
        self.folds=fields['fold_of_node'][self.tr]
        self.targets=[];self.seed_nodes=[]
        for k in range(5):
            ids=self.tr[self.folds==k]
            self.seed_nodes.append(torch.from_numpy(ids.copy()))
            self.targets.append(dict(zip(map(int,ids),map(int,self.y[self.folds==k]))))
        self.val_positions=dict(zip(map(int,self.val),range(len(self.val))))
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.set_num_threads(6)
        dgl.seed(seed)
        initial={name:torch.zeros(N,device=DEVICE) for name in NAMES}
        self.model=RGTAN(in_feats=FEATURES,hidden_dim=config['hidden_dim'],n_layers=2,n_classes=2,
                         heads=[4,4],activation=torch.nn.PReLU(),drop=config['drop'],
                         device=DEVICE,gated=True,ref_df=pd.DataFrame(fields['features']),
                         cat_features={},neigh_features=initial,nei_att_head=2).to(DEVICE)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=config['base_lr']*(BATCH/1024)**.5,weight_decay=0.0001)
        self.sam=MultiLayerNeighborSampler(list(FANOUTS))
        self.train_loaders=[NodeDataLoader(self.g,ids,self.sam,batch_size=BATCH,shuffle=True,
                                          drop_last=False,num_workers=0) for ids in self.seed_nodes]
        self.val_loader=NodeDataLoader(self.g,torch.from_numpy(self.val.copy()),self.sam,
                                       batch_size=BATCH,shuffle=False,drop_last=False,num_workers=0)
        self.steps=0

    def forward(self,batch,stage,fold=5):
        in_nodes,out_nodes,blocks=batch
        ins=in_nodes.numpy();outs=out_nodes.numpy()
        if stage=='train':
            x=self.fields['features'][ins]
            risk=self.fields[f'risk_view_{fold}'][ins]
            context=self.fields[f'label_context_{fold}'][ins]
            assert np.all(context[np.isin(ins,outs)]==2),'STOP: Training target leaked label context'
        else:
            x=self.fields['features'][ins]
            risk=self.fields['risk_view_5'][ins]
            context=self.fields['label_context_5'][ins]
            assert np.all(context[np.isin(ins,outs)]==2),'STOP: Validation target leaked label context'
        xx=torch.from_numpy(x.astype(np.float32,copy=False).copy()).to(DEVICE)
        cc=torch.from_numpy(context.astype(np.int64,copy=False).copy()).to(DEVICE)
        rd={name:torch.from_numpy(risk[:,j].copy()).to(DEVICE) for j,name in enumerate(NAMES)}
        model_blocks=[b.to(DEVICE) for b in blocks]
        logits=self.model(model_blocks,xx,cc,{},rd)
        assert logits.shape==(len(outs),2) and bool(torch.isfinite(logits).all())
        return logits,outs

    def one_epoch(self,epoch):
        self.model.train();loss_sum=0.;count=0
        order=np.random.RandomState(self.seed+epoch).permutation(5)
        for k in order:
            # Reproducible but epoch-specific training neighbour samples.
            dgl.seed(self.seed+epoch*17+int(k))
            for batch in self.train_loaders[k]:
                logits,outs=self.forward(batch,'train',k)
                target=torch.as_tensor([self.targets[k][int(i)] for i in outs],dtype=torch.long,device=DEVICE)
                loss=F.cross_entropy(logits,target)
                assert bool(torch.isfinite(loss))
                self.opt.zero_grad(set_to_none=True)
                loss.backward();self.opt.step()
                self.steps+=1;loss_sum+=float(loss.item())*len(outs);count+=len(outs)
        assert count==15742,(count,'TRAIN_COUNT')
        return loss_sum/count

    def validation(self):
        # Fixed validation fanout sampling across epochs/trials.
        # This prevents Monte Carlo neighbour changes from distorting selection.
        dgl.seed(90210)
        self.model.eval();scores=np.empty(len(self.val),dtype=np.float32)
        with torch.no_grad():
            for batch in self.val_loader:
                logits,ids=self.forward(batch,'val')
                score=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
                for i,node in enumerate(ids):scores[self.val_positions[int(node)]]=score[i]
        assert np.isfinite(scores).all()
        return float(average_precision_score(self.vy,scores)),scores


def run_trial(g,fields,tr_y,val_y,trial,cfg,mode):
    directory=OUT/f'trial_{trial:02d}'
    if mode=='tune':directory.mkdir(parents=True,exist_ok=True)
    else:directory=OUT/'pilot_only';directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'RGTAN_T-Finance_TR40_CandidateTrialResult.json'
    checkpoint=directory/'RGTAN_T-Finance_TR40_CandidateBestModel.pt'
    if mode=='tune' and manifest.is_file():
        data=json.loads(manifest.read_text())
        assert data['status']=='COMPLETE' and data['risk_sha256']==RISK_SHA
        assert data['config']==cfg and data['script_sha256']==checksum(__file__)
        assert data['test_scored'] is False and data['test_labels_used'] is False
        assert checkpoint.is_file()
        assert data['checkpoint_sha256']==checksum(checkpoint)
        print(f'RGTAN_T-Finance_TR40_CANDIDATE_TRIAL_{trial:02d}_REUSED=PASS val_AUPRC={data["best_val_auprc"]:.6f}',flush=True)
        return data
    eng=Engine(g,fields,tr_y,val_y,cfg)
    start=time.perf_counter();best=-1.;best_epoch=0;stale=0
    history=[]
    for epoch in range(1,(1 if mode=='pilot' else MAX_EPOCHS)+1):
        loss=eng.one_epoch(epoch)
        ap,scores=eng.validation()
        history.append({'epoch':epoch,'train_loss':loss,'val_auprc':ap})
        if ap>best+1e-9:
            best=ap;best_epoch=epoch;stale=0
            save_checkpoint(checkpoint,{'model':eng.model.state_dict(),'config':cfg,
              'seed':SEED,'epoch':epoch,'val_auprc':ap,'risk_sha256':RISK_SHA,
              'author_pin':AUTHOR_COMMIT,'dataset':'T-Finance','model_name':'RGTAN',
              'fanouts':list(FANOUTS),'benchmark_supervisor_approval_required':True,
              'test_scored':False})
            temp=directory/'RGTAN_T-Finance_TR40_CandidateBestValidationScores.npy.partial'
            with temp.open('wb') as f:np.save(f,scores);f.flush();os.fsync(f.fileno())
            os.replace(temp,directory/'RGTAN_T-Finance_TR40_CandidateBestValidationScores.npy')
        else:stale+=1
        if epoch<=2 or epoch%5==0 or stale>=PATIENCE:
            print(f'RGTAN_T-Finance_TR40_CANDIDATE trial={trial}/12 epoch={epoch} loss={loss:.5f} '
                  f'val_AUPRC={ap:.6f} best={best:.6f} no_improve={stale}',flush=True)
        if mode=='tune' and stale>=PATIENCE:break
    row={'status':'COMPLETE' if mode=='tune' else 'PILOT_ONLY',
      'model':'RGTAN','dataset':'T-Finance','ratio':'TR40','trial':trial,
      'candidate_method_only':True,'benchmark_supervisor_approval_required':True,
      'sampler':'MultiLayerNeighborSampler','fanouts':list(FANOUTS),
      'original_full_neighbor_sampling_replaced':True,
      'original_graph_edges':EDGE_COUNT,'risk_features':'exact_full_graph_training_only_5fold_crossfit',
      'config':cfg,'batch':BATCH,'seed':SEED,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
      'best_epoch':best_epoch,'epochs_completed':epoch,'best_val_auprc':best,
      'optimizer_steps':eng.steps,'wall_seconds':round(time.perf_counter()-start,3),
      'risk_sha256':RISK_SHA,'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,
      'script_sha256':checksum(__file__),
      'test_scored':False,'test_labels_used':False,
      'checkpoint_sha256':checksum(checkpoint)}
    save_json(directory/'RGTAN_T-Finance_TR40_CandidateHistory.json',history)
    if mode=='tune':save_json(manifest,row)
    else:save_json(directory/'RGTAN_T-Finance_TR40_CandidateOneEpochPilot.json',row)
    print(f'RGTAN_T-Finance_TR40_{mode.upper()}_TRIAL_{trial:02d}=PASS',flush=True)
    del eng;torch.cuda.empty_cache()
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('pilot','tune'),required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(AUTHOR_REPO))
    parts,fields=validate_all()
    g,tr_y,val_y=load_source_graph_and_validation_only(parts,fields)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='pilot':
        run_trial(g,fields,tr_y,val_y,1,GRID[0],'pilot')
        print('RGTAN_T-Finance_TR40_ONE_EPOCH_CANDIDATE_PILOT=PASS TEST_SCORED=NO',flush=True)
        return
    rows=[]
    for i,cfg in enumerate(GRID,1): rows.append(run_trial(g,fields,tr_y,val_y,i,cfg,'tune'))
    best=max(rows,key=lambda x:(x['best_val_auprc'],-x['trial']))
    save_json(OUT/'RGTAN_T-Finance_TR40_CandidateTune12Winner.json',{
        'status':'RGTAN_TFINANCE_TR40_CANDIDATE_12_VAL_ONLY_TUNE_COMPLETE',
        'model':'RGTAN','dataset':'T-Finance','ratio':'TR40','trial_count':12,
        'winner_trial':best['trial'],'winner_val_auprc':best['best_val_auprc'],
        'winner_epoch':best['best_epoch'],'winner_configuration':best['config'],
        'risk_sha256':RISK_SHA,'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,
        'original_full_neighbor_sampling_replaced':True,
        'sampler':'MultiLayerNeighborSampler','fanouts':list(FANOUTS),
        'benchmark_supervisor_approval_required':True,
        'test_scored':False,'test_labels_used':False,'author_commit':AUTHOR_COMMIT})
    print('RGTAN_T-Finance_I2_TR40_CANDIDATE_12_VALIDATION_TRIALS=PASS TEST_SCORED=NO',flush=True)

if __name__=='__main__': main()
