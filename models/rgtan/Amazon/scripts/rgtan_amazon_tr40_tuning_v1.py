#!/usr/bin/env python3
"""RGTAN Amazon TR40 tuning, using frozen COMP8851 crossfit feature views.

Source graph/topology and all label contexts are read-only. Never evaluates test labels.
Creates one model per trial and applies held-out-fold contexts to the same model.
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

ROOT = Path('/workspace/comp8851')
SRC = ROOT / 'repos/antifraud'
ASSET = ROOT / 'assets/rgtan/amazon/amazon_tr40_crossfit.npz'
MANIFEST = ROOT / 'assets/rgtan/amazon/amazon_tr40_crossfit_manifest.json'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/Amazon/amazon_seed2_nested_splits.npz')
MAT = Path('/root/.dgl/amazon_1e446145/Amazon.mat')
OUT = ROOT / 'results/rgtan/amazon_tr40_tuning_v1'
DEVICE = 'cuda:0'
SOURCE_COMMIT = '65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SOURCE_SHA = '4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63'
SPLIT_SHA = '0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd'
ADAPTER_SHA = 'fd3f871171c795999e6a1470857b270d5a605bb3f7cb3c3ea296cd5b3d1c6efe'
SEED = 2
BATCH = 32
MAX_EPOCHS = 100
PATIENCE = 20
GRID = [dict(hidden_dim=h, drop=[d, d / 2], base_lr=lr)
        for h in (32, 64) for d in (.2, .3) for lr in (.001, .003, .005)]
assert len(GRID) == 12


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
    import subprocess
    assert torch.cuda.is_available(), 'CUDA unavailable'
    assert torch.__version__ == '1.12.1+cu113', torch.__version__
    assert dgl.__version__ == '0.8.1', dgl.__version__
    assert subprocess.check_output(['git', '-C', str(SRC), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_COMMIT
    assert digest(MAT) == SOURCE_SHA, 'raw Amazon source mismatch'
    assert digest(SPLIT) == SPLIT_SHA, 'frozen Amazon split mismatch'
    assert digest(ASSET) == ADAPTER_SHA, 'crossfit adapter mismatch'
    meta = json.loads(MANIFEST.read_text())
    assert meta['status'] == 'RGTAN_CROSSFIT_ADAPTER_PASS'
    assert meta['artifact_sha256'] == ADAPTER_SHA
    assert meta['source_sha256'] == SOURCE_SHA
    assert meta['split_sha256'] == SPLIT_SHA
    with np.load(ASSET, allow_pickle=False) as z:
        features = z['features'].astype(np.float32)
        train = z['train_ids'].astype(np.int64)
        y_train = z['train_labels'].astype(np.int64)
        val = z['val_ids'].astype(np.int64)
        fold = z['fold_of_node'].astype(np.int64)
        views = [z[f'risk_view_{i}'].astype(np.float32) for i in range(6)]
        contexts = [z[f'label_context_{i}'].astype(np.int64) for i in range(6)]
        names = [str(x) for x in z['risk_feature_names']]
    with np.load(SPLIT, allow_pickle=False) as z:
        test = z['test'].astype(np.int64)
        assert int(z['seed'].item()) == 2
        assert np.array_equal(train, z['TR40']) and np.array_equal(val, z['val'])
    assert features.shape == (11944, 25) and train.shape == (3455,) and val.shape == (1728,)
    assert test.shape == (3456,) and y_train.shape == train.shape
    assert np.all(np.isin(y_train, [0, 1]))
    assert set(fold[train].tolist()) == {0, 1, 2, 3, 4}
    assert len(np.unique(train)) == len(train)
    assert not np.intersect1d(train, val).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(val, test).size
    assert names == ['degree', 'riskstat', '1hop_degree', '2hop_degree', '1hop_riskstat', '2hop_riskstat']
    for k in range(6):
        assert views[k].shape == (11944, 6) and np.isfinite(views[k]).all()
        assert contexts[k].shape == (11944,)
        assert np.all(contexts[k][test] == 2) and np.all(contexts[k][val] == 2)
        assert np.all(contexts[k][:3305] == 2)
    for k in range(5):
        assert np.all(contexts[k][train[fold[train] == k]] == 2)
    assert np.all(contexts[5][test] == 2) and np.all(contexts[5][val] == 2)
    print('SOURCES_SPLITS_CROSSFIT_ISOLATION=PASS', flush=True)
    raw = loadmat(MAT, variable_names=['homo'])
    adj = sp.csr_matrix(raw['homo'], dtype=np.float32)
    del raw
    adj = (adj + adj.T).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0
    adj = adj.tolil()
    adj.setdiag(1.0)
    adj = adj.tocsr()
    adj.eliminate_zeros()
    adj.sort_indices()
    assert adj.nnz == 8808728, adj.nnz
    graph = dgl.from_scipy(adj)
    assert graph.num_nodes() == 11944 and len(graph.ndata) == 0
    print('RGTAN_AUTHOR_GRAPH_TOPOLOGY=PASS', flush=True)
    # Read only fixed validation truth; no test node is scored or used for selection.
    raw_labels = np.asarray(loadmat(MAT, variable_names=['label'])['label']).reshape(-1)
    val_y = np.array(raw_labels[val], dtype=np.int64, copy=True)
    del raw_labels
    assert set(val_y.tolist()) == {0, 1}
    return graph, features, train, y_train, val, val_y, fold, views, contexts, names


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
            in_feats=25,
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
            nei_att_head=5,
        ).to(DEVICE)
        self.learning_rate = config['base_lr'] * (BATCH / 1024.0) ** .5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.supervision = np.full(11944, -1, dtype=np.int64)
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
        scores = np.full(11944, np.nan, dtype=np.float32)
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
        assert state['adapter_sha256'] == ADAPTER_SHA
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
                'adapter_sha256': ADAPTER_SHA,
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
            'adapter_sha256': ADAPTER_SHA,
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
    result = dict(status='TUNING_COMPLETE', model='RGTAN', dataset='Amazon', ratio='TR40',
                  split_seed=2, seed=SEED, trial=idx, config=config,
                  base_learning_rate=config['base_lr'], effective_lr=exp.learning_rate,
                  batch_size=BATCH, max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
                  completed_epochs=len(history), best_epoch=best_ep,
                  best_val_auprc=best_ap, best_val_auroc=best_auc,
                  peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                  wall_seconds=round(time.perf_counter() - started_at, 3),
                  adapter_sha256=ADAPTER_SHA, source_sha256=SOURCE_SHA,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['tune-one', 'tune-all', 'status'])
    args = ap.parse_args()
    if args.mode == 'status':
        for t in range(1, 13):
            f = OUT / 'tuning' / f'trial_{t:02d}' / 'tuning.json'
            if f.exists():
                j = json.loads(f.read_text())
                print(f'{t:02d}/12: COMPLETE valAUPRC={j["best_val_auprc"]:.6f}')
            else:
                print(f'{t:02d}/12: PENDING')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    data = check_inputs()
    if args.mode == 'tune-one':
        pending = [i for i in range(1, 13) if not (OUT/'tuning'/f'trial_{i:02d}'/'tuning.json').exists()]
        if pending:
            run_trial(data, pending[0])
        else:
            print('ALL_TUNING_ALREADY_COMPLETE', flush=True)
        print('TEST_EVALUATED=NO', flush=True)
        return
    rows = [run_trial(data, i) for i in range(1, 13)]
    assert len(rows) == 12
    winner = max(rows, key=lambda d: (d['best_val_auprc'], -d['trial']))
    record = dict(status='WINNER_FROZEN', ratio='TR40', split_seed=2, seed=2,
                  winner_trial=winner['trial'], config=winner['config'],
                  best_val_auprc=winner['best_val_auprc'],
                  source_sha256=SOURCE_SHA, split_sha256=SPLIT_SHA,
                  adapter_sha256=ADAPTER_SHA, test_scoring_performed=False)
    winner_file = OUT/'winner.json'
    if winner_file.exists():
        assert json.loads(winner_file.read_text()) == record, 'frozen winner differs'
    else:
        write_json(winner_file, record)
    print('RGTAN_AMAZON_TR40_TUNING_COMPLETE=12/12', flush=True)
    print('WINNER_FROZEN='+json.dumps(record, sort_keys=True), flush=True)
    print('TEST_EVALUATED=NO', flush=True)


if __name__ == '__main__':
    main()
