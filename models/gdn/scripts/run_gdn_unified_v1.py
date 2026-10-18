"""
GDN (Gao et al., WSDM 2023) under the COMP8851 unified benchmark protocol.

Usage:
    python run_gdn_unified.py --dataset yelp --ratio 40 --seed 2
    python run_gdn_unified.py --dataset yelp --ratio 40 --seed 2 --track smoke --max_epoch 2

What this script does NOT change: GDN's model code, losses, constraint mechanism,
optimiser groups or hyperparameters. It imports models/ from the author's repo
unmodified and replays the author's per-batch training step exactly.

What it does change (protocol harness only, recorded in run_config.yml):
  - canonical nested TR40/30/20/10 splits (team generate_splits, split seed 2),
    verified mask-for-mask against SplitGNN's stored split_indices.npz
  - validation every epoch, checkpoint by best validation AUPRC
  - max 100 epochs, early stopping patience 20
  - decision threshold chosen by best validation Macro-F1 (0.05..0.95 grid)
  - test set evaluated exactly once, all 7 metrics computed directly
  - GDN's prototype vectors are snapshotted before and restored after each
    evaluation pass, so per-epoch validation does not alter training dynamics

Results go to /workspace/gdn_vast/results/gdn/<dataset>/<track>/tr<ratio>/seed_<seed>/
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

GDN_ROOT = '/workspace/gdn'
VAST_ROOT = '/workspace/gdn_vast'
TEAM_ROOT = '/workspace/team'
MODEL_NAME = 'gdn'
AUTHOR_CONFIG = f'{GDN_ROOT}/config/gdn_yelpchi.yml'  # the only config the authors ship

sys.path.insert(0, GDN_ROOT)
sys.path.insert(0, f'{TEAM_ROOT}/models/splitgnn/splits')
from models.layers import InterAgg, IntraAgg  # noqa: E402  (author code, unmodified)
from models.model import GDNLayer  # noqa: E402
from utils.utils import normalize  # noqa: E402
from generate_splits import generate_splits, verify_splits  # noqa: E402  (team canonical splitter)

# Native three-relation datasets read straight from the author's preprocessing output.
NATIVE = {
    'yelp': ('YelpChi.mat', 'yelp', ['rur', 'rtr', 'rsr']),
    'amazon': ('Amazon.mat', 'amz', ['upu', 'usu', 'uvu']),
}

PROTOCOL_DEVIATIONS = [
    'splits: team canonical nested TR40/30/20/10 (seed 2) instead of author train_test_split 40/20/40',
    'validation every epoch instead of every 5 epochs',
    'checkpoint selected by validation AUPRC instead of validation AUROC',
    'max 100 epochs with early stopping patience 20 instead of fixed 101 epochs',
    'threshold chosen by best validation Macro-F1 (0.05..0.95) instead of fixed 0.2',
    'training node order: canonical train mask shuffled once with RandomState(2) (author: train_test_split order, random_state=2)',
    'prototype vectors snapshotted/restored around evaluation passes',
    'environment: python 3.10, torch 2.5.1+cu121 instead of README pins (torch 1.4, numpy 1.16)',
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_commit(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'UNKNOWN'


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux reports KB


# ---------------------------------------------------------------- data
def load_dataset(name):
    """Returns features (ndarray), labels (int64 ndarray), [adj1, adj2, adj3], source files, adapter id."""
    data_dir = f'{GDN_ROOT}/data/'
    if name in NATIVE:
        mat_file, prefix, rels = NATIVE[name]
        from scipy.io import loadmat
        d = loadmat(data_dir + mat_file)
        labels = d['label'].flatten().astype(np.int64)
        feat = d['features'].todense().A
        adj, sources = [], [data_dir + mat_file]
        for r in rels:
            p = f'{data_dir}{prefix}_{r}_adjlists.pickle'
            with open(p, 'rb') as f:
                adj.append(pickle.load(f))
            sources.append(p)
        adapter = 'none'
    else:
        # Adapter bundle: <name>_bundle.npz with features, labels, relation_files, adapter_version.
        # A pickle listed more than once is loaded once and the same object reused
        # (homogeneous graphs fill all three relation slots without tripling RAM).
        bundle = f'{data_dir}{name}_bundle.npz'
        b = np.load(bundle, allow_pickle=False)
        feat = b['features'].astype(np.float32)
        labels = b['labels'].astype(np.int64)
        rel_files = [str(x) for x in b['relation_files']]
        assert len(rel_files) == 3, 'GDN needs exactly three relation slots'
        cache = {}
        for rf in rel_files:
            if rf not in cache:
                with open(data_dir + rf, 'rb') as f:
                    cache[rf] = pickle.load(f)
        adj = [cache[rf] for rf in rel_files]
        sources = [bundle] + [data_dir + rf for rf in dict.fromkeys(rel_files)]
        adapter = str(b['adapter_version'])
    if name == 'amazon':
        feat = normalize(feat)  # author convention (model_handler.py)
    return np.asarray(feat, dtype=np.float32), labels, adj, sources, adapter


def canonical_splits(name, labels, log):
    """Team generate_splits over labelled nodes, exactly as SplitGNN's run_evidence.py does."""
    os.makedirs(f'{VAST_ROOT}/splits', exist_ok=True)
    path = f'{VAST_ROOT}/splits/{name}_seed2_nested_splits.npz'
    if not os.path.exists(path):
        labeled = np.where(labels != -1)[0]
        if len(labeled) < len(labels):
            log(f'NOTE: {len(labels) - len(labeled)} unlabelled (-1) nodes excluded from splits')
        sub = generate_splits(torch.from_numpy(labels[labeled]))
        masks = {}
        for k, v in sub.items():
            m = np.zeros(len(labels), dtype=bool)
            m[labeled[v.numpy()]] = True
            masks[k] = m
        verify_splits({k: torch.from_numpy(v) for k, v in masks.items()}, torch.from_numpy(labels))
        np.savez(path, **masks)
        log(f'Generated canonical splits: {path}')
    return np.load(path), path


# ---------------------------------------------------------------- model helpers
def build_model(feat, adj, train_pos, train_neg, cfg):
    """Same construction order as model_handler.ModelHandler.train (keeps init RNG identical)."""
    features = nn.Embedding(feat.shape[0], feat.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat), requires_grad=True)
    features.cuda()
    d, e = feat.shape[1], cfg['emb_size']
    intra1 = IntraAgg(features, d, e, train_pos, cuda=True)
    intra2 = IntraAgg(features, d, e, train_pos, cuda=True)
    intra3 = IntraAgg(features, d, e, train_pos, cuda=True)
    inter1 = InterAgg(features, d, e, train_pos, train_neg, adj, [intra1, intra2, intra3],
                      inter=cfg['multi_relation'], cuda=True)
    model = GDNLayer(2, inter1).cuda()

    group_1, group_2 = [], []
    for pname, param in model.named_parameters():
        (group_2 if pname == 'inter1.features.weight' else group_1).append(param)
    optimizer = torch.optim.Adam([
        dict(params=group_1, weight_decay=cfg['weight_decay'], lr=cfg['lr_1']),
        dict(params=group_2, weight_decay=cfg['weight_decay_2'], lr=cfg['lr_2']),
    ])
    return model, optimizer


def train_one_epoch(model, optimizer, idx_train, labels, feat_dim, cfg):
    """One epoch of the author's training step (model_handler.py lines 156-208), unchanged in effect."""
    bs = cfg['batch_size']
    num_batches = int(len(idx_train) / bs) + 1
    losses = []
    for batch in range(num_batches):
        i_start, i_end = batch * bs, min((batch + 1) * bs, len(idx_train))
        batch_nodes = idx_train[i_start:i_end]
        if not batch_nodes:  # author code would crash on an empty trailing batch
            continue
        batch_label = torch.as_tensor(labels[np.array(batch_nodes)], dtype=torch.long, device='cuda')
        optimizer.zero_grad()
        loss = model.loss(batch_nodes, batch_label)
        loss.backward(retain_graph=True)

        if cfg['add_constraint']:
            # fl step: top-k features by |dLoss/dX|
            grad = torch.abs(torch.autograd.grad(outputs=loss, inputs=model.inter1.features.weight)[0])
            grads_idx = grad.mean(dim=0).topk(k=cfg['topk']).indices
            top = set(grads_idx.tolist())
            non_grads_idx = torch.tensor([i for i in range(feat_dim) if i not in top], dtype=torch.long)

            loss_pos, loss_neg = model.inter1.fl_loss(grads_idx)
            loss_stable = cfg['Beta'] * torch.exp(loss_pos - loss_neg)
            loss_stable.backward(retain_graph=True)

            # fn step
            fn_pos, fn_neg = model.inter1.fn_loss(batch_nodes, non_grads_idx)
            loss_fn = cfg['Beta'] * torch.exp(fn_pos - fn_neg)
            loss_fn.backward()

        optimizer.step()
        lv = loss.item()
        if not math.isfinite(lv):
            raise FloatingPointError(f'non-finite loss at batch {batch}: {lv}')
        losses.append(lv)
    return float(np.mean(losses))


def predict(model, nodes, bs):
    """Fraud probabilities for `nodes`. Prototype state is restored afterwards."""
    inter = model.inter1
    saved = (inter.pos_vector, inter.neg_vector, getattr(inter, 'unique_nodes', None))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(nodes), bs):
            out.append(model.to_prob(nodes[i:i + bs], None)[:, 1].cpu().numpy())
    model.train()
    inter.pos_vector, inter.neg_vector = saved[0], saved[1]
    if saved[2] is not None:
        inter.unique_nodes = saved[2]
    return np.concatenate(out)


def full_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        'auprc': float(average_precision_score(y, p)),
        'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
        'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
        'precision_fraud': float(precision_score(y, pred, zero_division=0)),
        'recall_fraud': float(recall_score(y, pred, zero_division=0)),
        'f1_fraud': float(f1_score(y, pred, zero_division=0)),
        'g_mean': float(math.sqrt(tpr * tnr)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'threshold_used': float(round(thr, 2)),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='yelp, amazon, comp, tfinance, elliptic, tsocial')
    ap.add_argument('--ratio', required=True, type=int, choices=[40, 30, 20, 10])
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--track', default='unified', choices=['unified', 'smoke'])
    ap.add_argument('--gpu', default='a6000')
    ap.add_argument('--max_epoch', type=int, default=100)
    ap.add_argument('--early_stop', type=int, default=20)
    ap.add_argument('--topk', type=int, default=None, help='override author topk (only if feature dim <= 10)')
    ap.add_argument('--threads', type=int, default=4, help='torch CPU threads (lets seeds run in parallel)')
    ap.add_argument('--allow_split_mismatch', action='store_true')
    cli = ap.parse_args()

    torch.set_num_threads(cli.threads)
    with open(AUTHOR_CONFIG) as f:
        author_cfg = yaml.safe_load(f)
    cfg = {k: author_cfg[k] for k in ['emb_size', 'multi_relation', 'optimizer', 'lr_1', 'lr_2', 'weight_decay',
                                        'weight_decay_2', 'batch_size', 'test_batch_size', 'add_constraint',
                                        'Beta', 'topk']}
    if cli.topk is not None:
        cfg['topk'] = cli.topk

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_name = f'{MODEL_NAME}_{cli.dataset}_{cli.track}_tr{cli.ratio}_seed{cli.seed}_{cli.gpu}_{timestamp}'
    result_dir = f'{VAST_ROOT}/results/{MODEL_NAME}/{cli.dataset}/{cli.track}/tr{cli.ratio}/seed_{cli.seed}'
    os.makedirs(f'{result_dir}/checkpoints', exist_ok=True)
    for stale in ['summary.json', 'test_metrics.json']:  # a rerun must never leave an old "done" marker
        if os.path.exists(f'{result_dir}/{stale}'):
            os.remove(f'{result_dir}/{stale}')

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    def flush_log():
        with open(f'{result_dir}/terminal.log', 'w') as f:
            f.write('\n'.join(log_lines) + '\n')

    log(f'=== Run: {run_name} ===')
    log(f'Result dir: {result_dir}')

    # ---- seeds (author sets torch, cuda, numpy and python random) ----
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    # ---- data ----
    t_load = time.perf_counter()
    feat, labels, adj, sources, adapter = load_dataset(cli.dataset)
    dataset_load_seconds = time.perf_counter() - t_load
    n, feat_dim = feat.shape
    log(f'Loaded {cli.dataset}: nodes={n} features={feat_dim} fraud={int((labels == 1).sum())} '
        f'adapter={adapter} ({dataset_load_seconds:.1f}s)')
    if cfg['topk'] >= feat_dim:
        raise ValueError(f"topk={cfg['topk']} >= feature dim {feat_dim}: GDN's fn step would get an empty "
                         f"feature set. Pass --topk explicitly and record it as a deviation.")

    # ---- splits ----
    splits, split_path = canonical_splits(cli.dataset, labels, log)
    train_mask = splits[f'train_mask_tr{cli.ratio}']
    valid_mask, test_mask = splits['valid_mask'], splits['test_mask']
    assert not (train_mask & valid_mask).any(), 'train/valid overlap'
    assert not (train_mask & test_mask).any(), 'train/test overlap'
    assert not (valid_mask & test_mask).any(), 'valid/test overlap'
    log('Split overlap checks: PASSED')

    ref = f'{TEAM_ROOT}/results/unified/splitgnn_results/{cli.dataset}/unified/tr{cli.ratio}/seed_{cli.seed}/split_indices.npz'
    if os.path.exists(ref):
        r = np.load(ref)
        split_match = bool(np.array_equal(r['train_mask'], train_mask) and np.array_equal(r['valid_mask'], valid_mask)
                           and np.array_equal(r['test_mask'], test_mask))
        log(f'Split identical to SplitGNN reference: {split_match}')
        if not split_match and not cli.allow_split_mismatch:
            flush_log()
            raise SystemExit('Split differs from SplitGNN reference (node order mismatch?). Stopping.')
    else:
        split_match = None
        log(f'No SplitGNN reference split at {ref}')

    idx_train = np.where(train_mask)[0]
    np.random.RandomState(2).shuffle(idx_train)
    idx_train = [int(i) for i in idx_train]
    idx_valid = [int(i) for i in np.where(valid_mask)[0]]
    idx_test = [int(i) for i in np.where(test_mask)[0]]
    y_valid, y_test = labels[idx_valid], labels[idx_test]
    train_pos = [i for i in idx_train if labels[i] == 1]
    train_neg = [i for i in idx_train if labels[i] != 1]
    log(f'train={len(idx_train)} (fraud {len(train_pos)}) valid={len(idx_valid)} test={len(idx_test)}')

    # ---- model ----
    model, optimizer = build_model(feat, adj, train_pos, train_neg, cfg)

    # ---- evidence written before training ----
    effective = {'dataset': cli.dataset, 'ratio': cli.ratio, 'seed': cli.seed, 'track': cli.track,
                 'max_epoch': cli.max_epoch, 'early_stop': cli.early_stop, **cfg,
                 'selection_metric': 'val_auprc', 'threshold_rule': 'best val macro-F1 on 0.05..0.95 step 0.05',
                 'author_config': os.path.basename(AUTHOR_CONFIG),
                 'protocol_deviations_from_native': PROTOCOL_DEVIATIONS
                 + ([f"topk overridden to {cli.topk} (author 10)"] if cli.topk is not None else [])}
    config_sha256 = sha256_bytes(json.dumps(effective, sort_keys=True).encode())
    with open(f'{result_dir}/run_config.yml', 'w') as f:
        yaml.safe_dump(effective, f, sort_keys=False)

    split_sha256 = sha256_file(split_path)
    source_hashes = {os.path.basename(s): sha256_file(s) for s in sources}
    protocol_identity = {
        'protocol_version': 'vast-v4.4',
        'hardware_profile_id': 'PENDING_TEAM_MANIFEST',
        'container_digest': 'PENDING_TEAM_MANIFEST',
        'model_source_commit': git_commit(GDN_ROOT),
        'model_source_repo': 'https://github.com/blacksingular/wsdm_GDN',
        'dataset_manifest_version': 'PENDING_TEAM_MANIFEST',
        'source_sha256': source_hashes[os.path.basename(sources[0])],
        'dataset_view_id': f'{cli.dataset}_gdn_adjlists_v1',
        'split_registry_version': 'team_generate_splits_seed2',
        'split_sha256': split_sha256,
        'split_identical_to_splitgnn': split_match,
        'evaluator': 'run_gdn_unified.py',
        'evaluator_sha256': sha256_file(os.path.abspath(__file__)),
        'adapter_version': adapter,
        'config_sha256': config_sha256,
    }
    with open(f'{result_dir}/protocol_identity.json', 'w') as f:
        json.dump(protocol_identity, f, indent=2)
    with open(f'{result_dir}/dataset_manifest.json', 'w') as f:
        json.dump({'dataset': cli.dataset, 'files_sha256': source_hashes, 'n_nodes': int(n),
                   'n_features': int(feat_dim), 'n_fraud': int((labels == 1).sum()),
                   'n_unlabelled': int((labels == -1).sum())}, f, indent=2)
    np.savez(f'{result_dir}/split_indices.npz', train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    with open(f'{result_dir}/split_summary.json', 'w') as f:
        json.dump({'n_train': len(idx_train), 'n_valid': len(idx_valid), 'n_test': len(idx_test),
                   'train_fraud': len(train_pos), 'valid_fraud': int(y_valid.sum()), 'test_fraud': int(y_test.sum()),
                   'split_sha256': split_sha256, 'split_seed': 2}, f, indent=2)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    with open(f'{result_dir}/hardware.json', 'w') as f:
        json.dump({'gpu_query': gpu, 'cpu_count': os.cpu_count(), 'torch_threads': cli.threads,
                   'torch': torch.__version__, 'cuda': torch.version.cuda}, f, indent=2)
    with open(f'{result_dir}/environment.txt', 'w') as f:
        f.write(f'python {sys.version}\n')
        f.write(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)

    log(f"Model source commit: {protocol_identity['model_source_commit']}")
    log(f'Config SHA256: {config_sha256}')
    log(f'Split SHA256: {split_sha256}')
    flush_log()

    # ---- training ----
    epoch_rows, valid_rows = [], []
    best_auprc, best_epoch, stale_epochs = -1.0, -1, 0
    train_total = train_valid_total = 0.0
    ckpt = f'{result_dir}/checkpoints/best_model.pt'
    test_bs = cfg['test_batch_size']

    for epoch in range(cli.max_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = train_one_epoch(model, optimizer, idx_train, labels, feat_dim, cfg)
        torch.cuda.synchronize()
        train_s = time.perf_counter() - t0

        tv = time.perf_counter()
        p_valid = predict(model, idx_valid, test_bs)
        torch.cuda.synchronize()
        valid_s = time.perf_counter() - tv
        train_total += train_s
        train_valid_total += train_s + valid_s

        v_auprc = float(average_precision_score(y_valid, p_valid))
        v_auc = float(roc_auc_score(y_valid, p_valid))
        v_pred = (p_valid >= 0.5).astype(int)
        v_f1 = float(f1_score(y_valid, v_pred, average='macro', zero_division=0))
        v_rec = float(recall_score(y_valid, v_pred, zero_division=0))

        epoch_rows.append({'epoch': epoch, 'train_seconds': train_s, 'valid_seconds': valid_s,
                           'peak_memory_bytes': torch.cuda.max_memory_allocated(),
                           'peak_rss_bytes': peak_rss_bytes(), 'loss': loss})
        valid_rows.append({'epoch': epoch, 'val_auprc': v_auprc, 'val_auc': v_auc,
                           'val_f1_macro': v_f1, 'val_recall': v_rec})

        if v_auprc > best_auprc:
            best_auprc, best_epoch, stale_epochs = v_auprc, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale_epochs += 1
        log(f'epoch {epoch}: loss={loss:.4f} train={train_s:.1f}s valid={valid_s:.1f}s val_auprc={v_auprc:.4f} '
            f'val_auc={v_auc:.4f} val_f1_macro@0.5={v_f1:.4f} (best_epoch={best_epoch}, best_auprc={best_auprc:.4f})')
        if epoch % 5 == 0:
            flush_log()
        if stale_epochs >= cli.early_stop:
            log(f'Early stopping at epoch {epoch} (no improvement for {cli.early_stop} epochs)')
            break
    n_epochs = epoch + 1

    # ---- threshold on validation, then test exactly once ----
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    p_valid = predict(model, idx_valid, test_bs)
    best_thr, best_vf1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        f1v = f1_score(y_valid, (p_valid >= thr).astype(int), average='macro', zero_division=0)
        if f1v > best_vf1:
            best_vf1, best_thr = f1v, float(thr)
    val_auprc_reload = float(average_precision_score(y_valid, p_valid))
    log(f'Selected threshold (best val macro-F1): {best_thr:.2f} (val F1={best_vf1:.4f})')
    log(f'Reloaded checkpoint val AUPRC {val_auprc_reload:.6f} vs logged {best_auprc:.6f}')

    t_inf = time.perf_counter()
    p_test = predict(model, idx_test, test_bs)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t_inf

    test_metrics = full_metrics(y_test, p_test, best_thr)
    test_metrics['best_epoch'] = best_epoch
    log(f'TEST metrics: {json.dumps(test_metrics)}')
    if test_metrics['tp'] + test_metrics['fp'] == 0:
        log('WARNING: model predicted no fraud on the test set at the selected threshold')

    # ---- remaining artefacts ----
    with open(f'{result_dir}/epoch_times.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        w.writeheader()
        w.writerows(epoch_rows)
    with open(f'{result_dir}/validation_metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(valid_rows[0].keys()))
        w.writeheader()
        w.writerows(valid_rows)
    np.save(f'{result_dir}/valid_probs.npy', p_valid.astype(np.float32))
    np.save(f'{result_dir}/test_probs.npy', p_test.astype(np.float32))
    with open(f'{result_dir}/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    summary = {
        'run_name': run_name, 'model': MODEL_NAME, 'dataset': cli.dataset, 'track': cli.track,
        'ratio': cli.ratio, 'seed': cli.seed, 'n_completed_epochs': n_epochs, 'best_epoch': best_epoch,
        'best_val_auprc': best_auprc, 'reloaded_val_auprc': val_auprc_reload,
        'dataset_load_seconds': dataset_load_seconds, 'train_only_total_seconds': train_total,
        'train_plus_valid_total_seconds': train_valid_total, 'inference_seconds': inference_seconds,
        'peak_memory_bytes': max(r['peak_memory_bytes'] for r in epoch_rows),
        'peak_rss_bytes': peak_rss_bytes(), 'test_metrics': test_metrics,
    }
    assert len(epoch_rows) == n_epochs
    log(f'=== Run complete: {run_name} ===')
    flush_log()
    with open(f'{result_dir}/summary.json', 'w') as f:  # written last: marks the run as complete
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
