#!/usr/bin/env python3
"""Frozen-protocol RGTAN YelpChi TR40: 12 validation-only tuning trials.

Direct adaptation from I2 Amazon official-model runner. This runner never reads
or scores YelpChi test truth. Use `sample` first, `preflight`, then `pilot`
prior to any real tuning.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score

from methods.rgtan.rgtan_model import RGTAN
from rgtan_yelpchi_core_v1 import (
    ROOT,SRC,MAT,SPLIT,SOURCE_SHA,SPLIT_SHA,AUTHOR_COMMIT,
    N,NF,EXPECTED_EDGES,ASSET_DIR,OUT,EXPECTED_TRAIN,EXPECTED_POS,
    sha256,load_splits,get_graph_and_features,verify_input_files,assert_environment)

ASSET=ASSET_DIR/'yelpchi_tr40_crossfit.npz'
MANIFEST=ASSET_DIR/'yelpchi_tr40_crossfit_manifest.json'
DEVICE='cuda:0'
SOURCE_COMMIT=AUTHOR_COMMIT
SEED=2
BATCH=1024
OUT=ROOT / 'results/rgtan/yelpchi_tr40_tuning_batch1024_v1'
MAX_EPOCHS=100
PATIENCE=20
GRID=[dict(hidden_dim=h,drop=[d,d/2],base_lr=lr)
      for h in (32,64) for d in (.2,.3) for lr in (.001,.003,.005)]
assert len(GRID)==12

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
    os.replace(str(temp), str(path))


def save_torch(path, obj):
    temp = str(path) + '.partial'
    torch.save(obj, temp)
    os.replace(temp, str(path))


def check_inputs():
    verify_input_files()
    assert_environment()
    parts=load_splits()
    assert ASSET.is_file() and MANIFEST.is_file(), 'Build TR40 crossfit before tuning'
    meta=json.loads(MANIFEST.read_text())
    asset_hash=sha256(ASSET)
    assert meta['status']=='RGTAN_YELPCHI_CROSSFIT_PASS'
    assert meta['ratio']=='TR40' and meta['artifact_sha256']==asset_hash
    assert meta['source_sha256']==SOURCE_SHA and meta['split_sha256']==SPLIT_SHA
    assert meta['edges']==EXPECTED_EDGES and meta['nodes']==N
    assert meta['fold_labels_masked_from_their_own_risk_views'] is True
    assert meta['validation_test_labels_used_in_features'] is False
    assert meta['normalization_fit_on_oof_training_only'] is True
    with np.load(ASSET,allow_pickle=False) as z:
        features=z['features'].astype(np.float32)
        train=z['train_ids'].astype(np.int64)
        y_train=z['train_labels'].astype(np.int64)
        val=z['val_ids'].astype(np.int64)
        fold=z['fold_of_node'].astype(np.int64)
        views=[z[f'risk_view_{k}'].astype(np.float32) for k in range(6)]
        contexts=[z[f'label_context_{k}'].astype(np.int64) for k in range(6)]
        names=[str(x) for x in z['risk_feature_names']]
    assert np.array_equal(train,parts['TR40']) and np.array_equal(val,parts['val'])
    assert len(train)==18381 and len(val)==9191 and len(parts['test'])==18382
    assert features.shape==(N,NF) and np.isfinite(features).all()
    assert y_train.shape==train.shape and int(y_train.sum())==EXPECTED_POS['TR40']
    assert np.array_equal(np.unique(fold[train]),np.arange(5))
    assert np.all(fold[np.setdiff1d(np.arange(N),train)]==-1)
    assert names==['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']
    for k in range(6):
        assert views[k].shape==(N,6) and np.isfinite(views[k]).all()
        assert contexts[k].shape==(N,)
        assert np.all(contexts[k][parts['val']]==2) and np.all(contexts[k][parts['test']]==2)
        assert np.all(contexts[k][np.setdiff1d(np.arange(N),train)]==2)
        if k<5:
            heldout=train[fold[train]==k]
            assert np.all(contexts[k][heldout]==2)
            mask=fold[train]!=k
            assert np.array_equal(contexts[k][train[mask]],y_train[mask])
        else:
            assert np.array_equal(contexts[5][train],y_train)
    print('SOURCES_SPLITS_CROSSFIT_ISOLATION=PASS',flush=True)
    adj,f=get_graph_and_features()
    assert np.array_equal(f,features), 'Adapter original features differ from MAT'
    del f
    graph=dgl.from_scipy(adj)
    del adj
    assert graph.num_nodes()==N and graph.num_edges()==EXPECTED_EDGES
    assert len(graph.ndata)==0
    print('RGTAN_YELPCHI_GRAPH_TOPOLOGY=PASS',flush=True)
    raw_labels=np.asarray(loadmat(MAT,variable_names=['label'])['label']).reshape(-1)
    assert np.array_equal(raw_labels[train],y_train)
    val_y=np.array(raw_labels[val],dtype=np.int64,copy=True)
    del raw_labels
    assert set(val_y.tolist())=={0,1}
    return graph,features,train,y_train,val,val_y,fold,views,contexts,names

class Experiment:
    def __init__(self, data, config):
        (self.graph, self.features, self.train, self.y_train, self.val,
         self.val_y, self.fold, self.views, self.contexts, self.names) = data
        self.config = config
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(8)
        self.model = RGTAN(
            in_feats=NF,
            hidden_dim=config['hidden_dim'],
            n_layers=2,
            n_classes=2,
            heads=[4, 4],
            activation=torch.nn.PReLU(),
            drop=config['drop'],
            device=DEVICE,
            gated=True,
            ref_df=pd.DataFrame(self.features),
            cat_features={},
            neigh_features={
                n: torch.from_numpy(self.views[5][:, j]).to(DEVICE)
                for j, n in enumerate(self.names)
            },
            nei_att_head=4,
        ).to(DEVICE)
        self.learning_rate = config['base_lr'] * (BATCH / 1024.0) ** .5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.supervision = np.full(N, -1, dtype=np.int64)
        self.supervision[self.train] = self.y_train

    def loader(self, ids, shuffle):
        return NodeDataLoader(
            self.graph,
            torch.from_numpy(np.asarray(ids, dtype=np.int64)),
            MultiLayerFullNeighborSampler(2),
            batch_size=BATCH,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
        )

    def predict(self, batch, view):
        input_nodes, seeds, blocks = batch
        x_ids = input_nodes.numpy()
        target_ids = seeds.numpy()
        assert torch.equal(blocks[0].srcdata[dgl.NID], input_nodes)
        assert torch.equal(blocks[-1].dstdata[dgl.NID], seeds)
        assert np.all(self.contexts[view][target_ids] == 2), 'target label leakage'
        embedded = self.contexts[view][x_ids].copy()
        embedded[np.isin(x_ids, target_ids)] = 2
        node_features = torch.from_numpy(self.features[x_ids]).to(DEVICE)
        risk = torch.from_numpy(self.views[view][x_ids]).to(DEVICE)
        label_context = torch.from_numpy(embedded).long().to(DEVICE)
        risk_dict = {name: risk[:, j] for j, name in enumerate(self.names)}
        blocks_gpu = [b.to(DEVICE) for b in blocks]
        logits = self.model(blocks_gpu, node_features, label_context, {}, risk_dict)
        assert logits.shape == (len(target_ids), 2)
        assert torch.isfinite(logits).all(), 'non-finite logits'
        return target_ids, logits

    def epoch(self):
        losses = []
        seen = []
        for k in range(5):
            ids = self.train[self.fold[self.train] == k]
            self.model.train()
            for batch in self.loader(ids, shuffle=True):
                self.optimizer.zero_grad(set_to_none=True)
                seed_ids, logits = self.predict(batch, k)
                target = self.supervision[seed_ids]
                assert np.all((target == 0) | (target == 1))
                y = torch.from_numpy(target).long().to(DEVICE)
                loss = F.cross_entropy(logits, y)
                assert torch.isfinite(loss).item()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().item()))
                seen.extend(seed_ids.tolist())
        assert len(seen) == len(self.train)
        assert set(seen) == set(self.train.tolist())
        return float(np.mean(losses)), len(losses)

    def validate(self):
        self.model.eval()
        scores = np.full(N, np.nan, dtype=np.float32)
        seen = []
        with torch.no_grad():
            for batch in self.loader(self.val, shuffle=False):
                ids, logits = self.predict(batch, 5)
                scores[ids] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                seen.extend(ids.tolist())
        assert len(seen) == len(self.val) and len(set(seen)) == len(self.val)
        assert set(seen) == set(self.val.tolist())
        assert np.isfinite(scores[self.val]).all()
        ap = float(average_precision_score(self.val_y, scores[self.val]))
        auc = float(roc_auc_score(self.val_y, scores[self.val]))
        assert np.isfinite(ap) and np.isfinite(auc)
        return ap, auc


def run_trial(data, idx):
    config = GRID[idx - 1]
    directory = OUT / 'tuning' / f'trial_{idx:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / 'tuning.json'
    if final.exists():
        result = json.loads(final.read_text())
        assert result['status'] == 'TUNING_COMPLETE'
        assert result['trial'] == idx and result['config'] == config
        assert digest(directory / 'best_val_auprc.pt') == result['checkpoint_sha256']
        print(f'TUNING_ALREADY_COMPLETE={idx}/12 valAUPRC={result["best_val_auprc"]:.6f}', flush=True)
        return result
    exp = Experiment(data, config)
    latest = directory / 'latest.pt'
    best_ckpt = directory / 'best_val_auprc.pt'
    started_at = time.perf_counter()
    epoch_start, best_ap, best_auc, best_ep, wait = 1, -1.0, -1.0, 0, 0
    history = []
    if latest.exists():
        state = torch.load(latest, map_location=DEVICE)
        assert state['trial'] == idx and state['config'] == config
        assert state['adapter_sha256'] == sha256(ASSET)
        exp.model.load_state_dict(state['model'])
        exp.optimizer.load_state_dict(state['optimizer'])
        epoch_start = state['epoch'] + 1
        best_ap = float(state['best_val_auprc'])
        best_auc = float(state['best_val_auroc'])
        best_ep = state['best_epoch']
        wait = state['patience_wait']
        history = state['history']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device=DEVICE)
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        assert best_ckpt.exists() and digest(best_ckpt) == state['best_checkpoint_sha256']
        if wait >= PATIENCE:
            epoch_start = MAX_EPOCHS + 1
        print(f'RESUMING_TRIAL={idx:02d} epoch={epoch_start}', flush=True)
    else:
        assert not best_ckpt.exists(), 'best checkpoint exists without resumable state; stop'
        print(f'STARTING_TRIAL={idx:02d} config={json.dumps(config,sort_keys=True)}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epoch_start, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss, steps = exp.epoch()
        ap, auc = exp.validate()
        improved = ap > best_ap + 1e-12
        if improved:
            best_ap, best_auc, best_ep, wait = ap, auc, epoch, 0
            save_torch(best_ckpt, {
                'model_state_dict': {k: v.detach().cpu().clone() for k, v in exp.model.state_dict().items()},
                'config': config, 'trial': idx, 'epoch': epoch,
                'best_val_auprc': ap, 'best_val_auroc': auc,
                'adapter_sha256': sha256(ASSET),
            })
        else:
            wait += 1
        rec = dict(epoch=epoch, train_loss=train_loss, val_auprc=ap, val_auroc=auc,
                   optimizer_steps=steps, seconds=round(time.perf_counter() - t0, 3))
        history.append(rec)
        save_torch(latest, {
            'trial': idx, 'config': config, 'epoch': epoch,
            'model': exp.model.state_dict(), 'optimizer': exp.optimizer.state_dict(),
            'best_val_auprc': best_ap, 'best_val_auroc': best_auc, 'best_epoch': best_ep,
            'patience_wait': wait, 'history': history,
            'adapter_sha256': sha256(ASSET),
            'best_checkpoint_sha256': digest(best_ckpt),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(DEVICE),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()
        })
        print(f'TRIAL={idx:02d}/12 EPOCH={epoch:03d}/{MAX_EPOCHS} LOSS={train_loss:.5f} '
              f'valAUPRC={ap:.6f} best={best_ap:.6f}@{best_ep} '
              f'wait={wait}/{PATIENCE} seconds={rec["seconds"]:.1f}', flush=True)
        if wait >= PATIENCE:
            print(f'EARLY_STOP=YES trial={idx:02d} epoch={epoch}', flush=True)
            break
    assert best_ckpt.exists() and best_ep > 0
    result = dict(status='TUNING_COMPLETE', model='RGTAN', dataset='YelpChi', ratio='TR40',
                  split_seed=2, seed=SEED, trial=idx, config=config,
                  base_learning_rate=config['base_lr'], effective_lr=exp.learning_rate,
                  batch_size=BATCH, max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
                  completed_epochs=len(history), best_epoch=best_ep,
                  best_val_auprc=best_ap, best_val_auroc=best_auc,
                  peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                  wall_seconds=round(time.perf_counter() - started_at, 3),
                  adapter_sha256=sha256(ASSET), source_sha256=SOURCE_SHA,
                  split_sha256=SPLIT_SHA, author_commit=SOURCE_COMMIT,
                  checkpoint_sha256=digest(best_ckpt),
                  training_method='one model, five out-of-fold label/risk contexts',
                  test_scoring_performed=False, test_used_for_selection=False)
    write_json(directory / 'history.json', history)
    write_json(final, result)
    # keep best checkpoint and compact result; removes temporary resumed optimizer state
    latest.unlink(missing_ok=True)
    print(f'TUNING_FINISHED={idx}/12 valAUPRC={best_ap:.6f} epoch={best_ep}', flush=True)
    return result


def pilot_one_batch(data):
    """One real 32-target fold batch + one validation batch, no metrics/test.

    No durable checkpoints; this is an isolated capacity/time probe, not a
    valid full epoch or part of any tuning trial.
    """
    exp=Experiment(data,GRID[0])
    ids=exp.train[exp.fold[exp.train]==0]
    assert len(ids)>BATCH
    torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    batch=next(iter(exp.loader(ids[:BATCH],shuffle=False)))
    t_sample=time.perf_counter()
    exp.model.train()
    exp.optimizer.zero_grad(set_to_none=True)
    seeds,logits=exp.predict(batch,0)
    target=torch.from_numpy(exp.supervision[seeds]).long().to(DEVICE)
    loss=F.cross_entropy(logits,target)
    loss.backward();exp.optimizer.step();torch.cuda.synchronize()
    t_train=time.perf_counter()
    exp.model.eval()
    with torch.no_grad():
        batch_val=next(iter(exp.loader(exp.val[:BATCH],shuffle=False)))
        val_seeds,val_logits=exp.predict(batch_val,5)
        assert torch.isfinite(val_logits).all()
        torch.cuda.synchronize()
    t_val=time.perf_counter()
    assert len(seeds)==BATCH and len(val_seeds)==BATCH
    print('PILOT_SAMPLE_SECONDS=%.3f'%(t_sample-t0),flush=True)
    print('PILOT_FORWARD_BACKWARD_OPTIMIZER_SECONDS=%.3f'%(t_train-t_sample),flush=True)
    print('PILOT_SINGLE_VALIDATION_BATCH_SECONDS=%.3f'%(t_val-t_train),flush=True)
    print('PILOT_GPU_PEAK_ALLOCATED_GIB=%.3f'%(torch.cuda.max_memory_allocated()/1024**3),flush=True)
    print('PILOT_NOT_TUNING_TRIAL=YES TEST_SCORING_PERFORMED=NO',flush=True)
    print('I3_RGTAN_YELPCHI_REAL_FEATURE_PILOT=PASS',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['preflight','pilot','tune-one','tune-all','status'])
    args=ap.parse_args()
    if args.mode=='status':
        for t in range(1,13):
            f=OUT/'tuning'/f'trial_{t:02d}'/'tuning.json'
            if f.exists():
                report=json.loads(f.read_text())
                assert report['source_sha256']==SOURCE_SHA and report['split_sha256']==SPLIT_SHA
                print(f'{t:02d}/12 COMPLETE valAUPRC={report["best_val_auprc"]:.6f}')
            else: print(f'{t:02d}/12 PENDING')
        return
    data=check_inputs()
    if args.mode=='preflight':
        print('I3_RGTAN_YELPCHI_TUNER_PREFLIGHT=PASS TEST_SCORING_PERFORMED=NO',flush=True)
        return
    if args.mode=='pilot':
        pilot_one_batch(data)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='tune-one':
        pending=[i for i in range(1,13) if not (OUT/'tuning'/f'trial_{i:02d}'/'tuning.json').exists()]
        if pending:run_trial(data,pending[0])
        else:print('ALL_TUNING_ALREADY_COMPLETE',flush=True)
        print('TEST_SCORING_PERFORMED=NO',flush=True)
        return
    rows=[run_trial(data,i) for i in range(1,13)]
    assert len(rows)==12
    winner=max(rows,key=lambda d:(d['best_val_auprc'],-d['trial']))
    meta=json.loads(MANIFEST.read_text())
    record=dict(status='WINNER_FROZEN',ratio='TR40',split_seed=2,seed=2,
                winner_trial=winner['trial'],config=winner['config'],
                best_val_auprc=winner['best_val_auprc'],source_sha256=SOURCE_SHA,
                split_sha256=SPLIT_SHA,adapter_sha256=meta['artifact_sha256'],
                test_scoring_performed=False)
    winner_file=OUT/'winner.json'
    if winner_file.exists():
        assert json.loads(winner_file.read_text())==record, 'Frozen winner differs'
    else:write_json(winner_file,record)
    print('RGTAN_YELPCHI_TR40_TUNING_COMPLETE=12/12',flush=True)
    print('WINNER_FROZEN='+json.dumps(record,sort_keys=True),flush=True)
    print('TEST_SCORING_PERFORMED=NO',flush=True)

if __name__=='__main__':main()
