"""
AMNet (Chai et al., IJCAI 2022) under the COMP8851 unified benchmark protocol.

Usage:
    python run_amnet_unified.py --dataset yelp --ratio 40 --seed 2
    python run_amnet_unified.py --dataset comp --ratio 40 --seed 2 --track smoke --max_epoch 2
    python run_amnet_unified.py --dataset elliptic --ratio 40 --seed 2 --track preflight --max_epoch 2 --device cpu

What this script does NOT change: AMNet's model code (model.py, bernconv.py imported
unmodified from the author's repo), its loss (cross-entropy + beta * marginal constraint),
its optimiser groups, or its hyperparameters. The training step replays train.py's train().

What it does change (protocol harness only, recorded in run_config.yml):
  - canonical nested TR40/30/20/10 splits (team generate_splits, split seed 2), verified
    by file hash and mask-for-mask against SplitGNN's stored split_indices.npz
  - max 100 epochs, early stopping patience 20 (author: 2000 / 200)
  - checkpoint by best validation AUPRC (sklearn average_precision_score)
  - threshold chosen by best validation Macro-F1 (0.05..0.95 grid)
  - test set evaluated exactly once (author evaluates test at every validation improvement)
  - python, numpy and torch seeds set (author code sets none)

Results go to /workspace/amnet_vast/results/amnet/<dataset>/<track>/tr<ratio>/seed_<seed>/
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

AMNET_ROOT = '/workspace/amnet_vast/AMNet'
VAST_ROOT = '/workspace/amnet_vast'
TEAM_ROOT = '/workspace/team'
GDN_SPLITS = '/workspace/gdn_vast/splits'
MODEL_NAME = 'amnet'

sys.path.insert(0, f'{TEAM_ROOT}/models/splitgnn/splits')
sys.path.insert(0, AMNET_ROOT)
from model import AMNet  # noqa: E402  (author code, unmodified)
from config import dataset_config  # noqa: E402  (author tuned profiles)
from generate_splits import generate_splits, verify_splits  # noqa: E402  (team canonical splitter)

# Author-tuned profiles exist for yelp and elliptic only. Every other dataset uses the
# README's "Run on your own dataset" default profile, unchanged (no tuning).
README_CUSTOM_PROFILE = {'K': 2, 'M': 5, 'hidden_channels': 64, 'lr_f': 5e-2, 'lr': 5e-4,
                         'weight_decay': 1e-5, 'beta': 1.}
AUTHOR_PROFILE_FOR = {'yelp': 'yelp', 'elliptic': 'elliptic'}

# Nested split files, identical to SplitGNN's recorded split_sha256 (team registry).
EXPECTED_SPLIT_SHA256 = {
    'yelp': '448e01069d60e85ede228ac68b6d7208b0a153087127598c26b383b82606bb1f',
    'amazon': '194b49f7cac03b0081747e62cd5d422718e854d4350e200337c19b7663dca8dc',
    'comp': '193f52954bc953878d02f1334a94c4137fb8c2a7e3d37d0aaff51b9f11f1ae02',
    'tfinance': 'e7f78da52dc324c283f864744cd85105fe1c5d0feab065727930a50ba1c1971c',
    'tsocial': '197cd8a496ca269cfa2affeea08837bb97bc55181f878b36cc724036ba6d001b',
    'elliptic': '77008be57543be620a444bf233a229fe6f09d3d5b4fcf427a8481f7972e5ebf5',
}

PROTOCOL_DEVIATIONS = [
    'splits: team canonical nested TR40/30/20/10 (seed 2) instead of the author-shipped splits',
    'max 100 epochs with early stopping patience 20 instead of 2000 / 200',
    'checkpoint selected by validation AUPRC via sklearn average_precision_score '
    '(author: trapezoidal auc(recall, precision)); strict improvement required (author: >=)',
    'decision threshold chosen by best validation Macro-F1 (0.05..0.95); author reports AUC metrics only',
    'test set evaluated once after training (author evaluates test at every validation improvement, for logging)',
    'python/numpy/torch seeds set from --seed (author code sets no seeds)',
    'dataset view: single undirected edge set (AMNet is single-relation); multi-relation datasets use the '
    'stored homogeneous adjacency',
    'environment: torch 1.11.0+cu113, torch-geometric 2.0.4, torch-scatter 2.0.9, torch-sparse 0.6.13, '
    'nvidia-cusparse-cu11 for torch-sparse (author tested on CUDA 11.2)',
    'GPU scatter-add in message passing is non-deterministic, so reruns with the same seed can differ slightly',
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


def is_oom(e):
    return isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()


# ---------------------------------------------------------------- splits
def load_splits(name, labels, log):
    """Prefer the shared canonical file (hash-checked); otherwise generate exactly as the GDN runner does."""
    shared = f'{GDN_SPLITS}/{name}_seed2_nested_splits.npz'
    if os.path.exists(shared):
        sha = sha256_file(shared)
        if sha != EXPECTED_SPLIT_SHA256[name]:
            raise SystemExit(f'Canonical split file hash mismatch for {name}: {sha}. Stopping.')
        log(f'Splits: shared canonical file {shared} (SHA256 verified)')
        return dict(np.load(shared)), shared, sha, 'shared_canonical_file_sha_verified'

    own = f'{VAST_ROOT}/splits/{name}_seed2_nested_splits.npz'
    if not os.path.exists(own):
        os.makedirs(f'{VAST_ROOT}/splits', exist_ok=True)
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
        np.savez(own, **masks)
        log(f'Generated canonical splits: {own}')
    log(f'Splits: generated locally at {own} (verified only by the SplitGNN mask comparison)')
    return dict(np.load(own)), own, sha256_file(own), 'generated_locally'


# ---------------------------------------------------------------- helpers
def predict(net, x, edge_index):
    """Fraud probability for every node (full-graph forward, as AMNet's evaluating())."""
    net.eval()
    with torch.no_grad():
        p = F.softmax(net(x, edge_index), dim=1)[:, 1]
    net.train()
    return p.detach().cpu().numpy()


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
    ap.add_argument('--dataset', required=True, choices=list(EXPECTED_SPLIT_SHA256))
    ap.add_argument('--ratio', required=True, type=int, choices=[40, 30, 20, 10])
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--track', default='unified', choices=['unified', 'smoke', 'preflight'])
    ap.add_argument('--gpu', default='a6000')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--max_epoch', type=int, default=100)
    ap.add_argument('--early_stop', type=int, default=20)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--allow_unverified_split', action='store_true')
    cli = ap.parse_args()

    torch.set_num_threads(cli.threads)
    use_cuda = cli.device == 'cuda'
    if use_cuda and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available. Stopping.')
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    profile_source = AUTHOR_PROFILE_FOR.get(cli.dataset)
    if profile_source:
        prof = {k: v for k, v in dataset_config[profile_source].items() if k not in ('epochs', 'patience')}
        profile_label = f'author config.py[{profile_source!r}]'
    else:
        prof = dict(README_CUSTOM_PROFILE)
        profile_label = 'author README custom-dataset default profile'

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_name = f'{MODEL_NAME}_{cli.dataset}_{cli.track}_tr{cli.ratio}_seed{cli.seed}_{cli.gpu}_{timestamp}'
    result_dir = f'{VAST_ROOT}/results/{MODEL_NAME}/{cli.dataset}/{cli.track}/tr{cli.ratio}/seed_{cli.seed}'
    os.makedirs(f'{result_dir}/checkpoints', exist_ok=True)
    for stale in ['summary.json', 'test_metrics.json', 'failure.json']:  # a rerun never leaves an old marker
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

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(cli.seed)

    # ---- data ----
    t_load = time.perf_counter()
    view_path = f'{VAST_ROOT}/views/{cli.dataset}_view.npz'
    if not os.path.exists(view_path):
        raise SystemExit(f'Missing view {view_path}. Run build_amnet_views.py first.')
    v = np.load(view_path, allow_pickle=False)
    x_np, labels, ei_np = v['x'].astype(np.float32), v['y'].astype(np.int64), v['edge_index'].astype(np.int64)
    adapter = str(v['adapter_version'])
    view_stats = json.loads(str(v['stats_json']))
    n, feat_dim = x_np.shape
    log(f'Loaded {cli.dataset}: nodes={n} features={feat_dim} edges(both dirs)={ei_np.shape[1]} '
        f'fraud={int((labels == 1).sum())} unlabelled={int((labels == -1).sum())} adapter={adapter}')

    # ---- splits ----
    splits, split_path, split_sha256, split_source = load_splits(cli.dataset, labels, log)
    train_mask = splits[f'train_mask_tr{cli.ratio}'].astype(bool)
    valid_mask, test_mask = splits['valid_mask'].astype(bool), splits['test_mask'].astype(bool)
    assert len(train_mask) == len(valid_mask) == len(test_mask) == n, 'split length != number of nodes'
    assert not (train_mask & valid_mask).any(), 'train/valid overlap'
    assert not (train_mask & test_mask).any(), 'train/test overlap'
    assert not (valid_mask & test_mask).any(), 'valid/test overlap'
    assert (labels[train_mask | valid_mask | test_mask] != -1).all(), 'unlabelled node inside a split'
    log('Split overlap and label checks: PASSED')

    ref = f'{TEAM_ROOT}/results/unified/splitgnn_results/{cli.dataset}/unified/tr{cli.ratio}/seed_{cli.seed}/split_indices.npz'
    if os.path.exists(ref):
        r = np.load(ref)
        split_match = bool(np.array_equal(r['train_mask'], train_mask) and np.array_equal(r['valid_mask'], valid_mask)
                           and np.array_equal(r['test_mask'], test_mask))
        log(f'Split identical to SplitGNN reference: {split_match}')
        if not split_match:
            flush_log()
            raise SystemExit('Split differs from SplitGNN reference (node order mismatch?). Stopping.')
    else:
        split_match = None
        log(f'No SplitGNN reference split at {ref}')
        if split_source != 'shared_canonical_file_sha_verified' and not cli.allow_unverified_split:
            flush_log()
            raise SystemExit('Split is neither hash-verified nor checked against SplitGNN. Stopping.')

    idx_train, idx_valid, idx_test = (np.where(m)[0] for m in (train_mask, valid_mask, test_mask))
    y_valid, y_test = labels[idx_valid], labels[idx_test]
    log(f'train={len(idx_train)} (fraud {int(labels[idx_train].sum())}) valid={len(idx_valid)} '
        f'(fraud {int(y_valid.sum())}) test={len(idx_test)} (fraud {int(y_test.sum())})')

    x = torch.from_numpy(x_np).to(device)
    edge_index = torch.from_numpy(ei_np).to(device)
    y = torch.from_numpy(labels).to(device)
    train_t = torch.from_numpy(train_mask).to(device)
    anomaly, normal = (y == 1), (y == 0)
    label_masks = (train_t & anomaly, train_t & normal)
    dataset_load_seconds = time.perf_counter() - t_load

    # ---- model and optimiser, exactly as train.py ----
    net = AMNet(in_channels=feat_dim, hid_channels=prof['hidden_channels'], num_class=2,
                K=prof['M'], filter_num=prof['K']).to(device)
    optimizer = torch.optim.Adam([
        dict(params=net.filters.parameters(), lr=prof['lr_f']),
        dict(params=net.lin, lr=prof['lr'], weight_decay=prof['weight_decay']),
        dict(params=net.attn, lr=prof['lr'], weight_decay=prof['weight_decay'])])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1., 1.]).to(device))

    # ---- evidence written before training ----
    effective = {'dataset': cli.dataset, 'ratio': cli.ratio, 'seed': cli.seed, 'track': cli.track,
                 'device': cli.device, 'max_epoch': cli.max_epoch, 'early_stop': cli.early_stop,
                 'hyperparameter_profile': profile_label, **prof,
                 'selection_metric': 'val_auprc', 'threshold_rule': 'best val macro-F1 on 0.05..0.95 step 0.05',
                 'adapter_version': adapter, 'feature_transform': 'none',
                 'protocol_deviations_from_native': PROTOCOL_DEVIATIONS
                 + ([] if profile_source else ['no author-tuned profile for this dataset: README default profile used'])
                 + (['T-Social: 15% stratified node subsample (team canonical view), not the full graph']
                    if cli.dataset == 'tsocial' else [])
                 + (['Elliptic: canonical view (all 203,769 nodes, features = CSV columns 2:, unknown nodes kept '
                     'unlabelled), not the author-shipped labelled-only 93-feature subgraph']
                    if cli.dataset == 'elliptic' else [])}
    config_sha256 = sha256_bytes(json.dumps(effective, sort_keys=True).encode())
    with open(f'{result_dir}/run_config.yml', 'w') as f:
        yaml.safe_dump(effective, f, sort_keys=False)

    source_sha = view_stats['source_sha256']
    protocol_identity = {
        'protocol_version': 'vast-v4.4',
        'hardware_profile_id': 'PENDING_TEAM_MANIFEST',
        'container_digest': 'PENDING_TEAM_MANIFEST',
        'model_source_commit': git_commit(AMNET_ROOT),
        'model_source_repo': 'https://github.com/zjunet/AMNet',
        'dataset_manifest_version': 'PENDING_TEAM_MANIFEST',
        'source_sha256': source_sha,
        'dataset_view_id': f'{cli.dataset}_amnet_{adapter}',
        'dataset_view_sha256': sha256_file(view_path),
        'split_registry_version': 'team_generate_splits_seed2',
        'split_sha256': split_sha256,
        'split_source': split_source,
        'split_identical_to_splitgnn': split_match,
        'evaluator': 'run_amnet_unified.py',
        'evaluator_sha256': sha256_file(os.path.abspath(__file__)),
        'adapter_version': adapter,
        'config_sha256': config_sha256,
    }
    with open(f'{result_dir}/protocol_identity.json', 'w') as f:
        json.dump(protocol_identity, f, indent=2)
    with open(f'{result_dir}/dataset_manifest.json', 'w') as f:
        json.dump({'dataset': cli.dataset, 'files_sha256': source_sha, 'source_path': view_stats.get('source_path'),
                   'n_nodes': int(n), 'n_features': int(feat_dim), 'n_fraud': int((labels == 1).sum()),
                   'n_unlabelled': int((labels == -1).sum())}, f, indent=2)
    with open(f'{result_dir}/dataset_view_manifest.json', 'w') as f:
        json.dump({'view_file': view_path, 'view_sha256': protocol_identity['dataset_view_sha256'],
                   **view_stats}, f, indent=2)
    np.savez(f'{result_dir}/split_indices.npz', train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    with open(f'{result_dir}/split_summary.json', 'w') as f:
        json.dump({'n_train': int(len(idx_train)), 'n_valid': int(len(idx_valid)), 'n_test': int(len(idx_test)),
                   'train_fraud': int(labels[idx_train].sum()), 'valid_fraud': int(y_valid.sum()),
                   'test_fraud': int(y_test.sum()), 'split_sha256': split_sha256, 'split_seed': 2}, f, indent=2)
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    except OSError:
        gpu = 'nvidia-smi unavailable'
    with open(f'{result_dir}/hardware.json', 'w') as f:
        json.dump({'gpu_query': gpu, 'device_used': cli.device, 'cpu_count': os.cpu_count(),
                   'torch_threads': cli.threads, 'torch': torch.__version__, 'cuda': torch.version.cuda}, f, indent=2)
    with open(f'{result_dir}/environment.txt', 'w') as f:
        f.write(f'python {sys.version}\n')
        f.write(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)

    log(f"Model source commit: {protocol_identity['model_source_commit']}")
    log(f'Hyperparameters: {profile_label} {prof}')
    log(f'Config SHA256: {config_sha256}')
    log(f'Split SHA256: {split_sha256}')
    flush_log()

    # ---- training ----
    epoch_rows, valid_rows = [], []
    best_auprc, best_epoch, stale_epochs = -1.0, -1, 0
    train_total = train_valid_total = 0.0
    ckpt = f'{result_dir}/checkpoints/best_model.pt'
    epoch = -1
    try:
        for epoch in range(cli.max_epoch):
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
            sync()
            t0 = time.perf_counter()
            # train.py train(): one full-graph step
            net.train()
            optimizer.zero_grad()
            output, bias_loss = net(x, edge_index, label=label_masks)
            loss = criterion(output[train_t], y[train_t]) + bias_loss * prof['beta']
            loss.backward()
            optimizer.step()
            sync()
            train_s = time.perf_counter() - t0
            loss_v = float(loss.item())
            if not math.isfinite(loss_v):
                raise FloatingPointError(f'non-finite loss at epoch {epoch}: {loss_v}')

            tv = time.perf_counter()
            p_all = predict(net, x, edge_index)
            sync()
            valid_s = time.perf_counter() - tv
            train_total += train_s
            train_valid_total += train_s + valid_s

            p_valid = p_all[idx_valid]
            v_auprc = float(average_precision_score(y_valid, p_valid))
            v_auc = float(roc_auc_score(y_valid, p_valid))
            v_pred = (p_valid >= 0.5).astype(int)
            v_f1 = float(f1_score(y_valid, v_pred, average='macro', zero_division=0))
            v_rec = float(recall_score(y_valid, v_pred, zero_division=0))

            epoch_rows.append({'epoch': epoch, 'train_seconds': train_s, 'valid_seconds': valid_s,
                               'peak_memory_bytes': torch.cuda.max_memory_allocated() if use_cuda else 0,
                               'peak_rss_bytes': peak_rss_bytes(), 'loss': loss_v})
            valid_rows.append({'epoch': epoch, 'val_auprc': v_auprc, 'val_auc': v_auc,
                               'val_f1_macro': v_f1, 'val_recall': v_rec})

            if v_auprc > best_auprc:
                best_auprc, best_epoch, stale_epochs = v_auprc, epoch, 0
                torch.save(net.state_dict(), ckpt)
            else:
                stale_epochs += 1
            log(f'epoch {epoch}: loss={loss_v:.4f} train={train_s:.2f}s valid={valid_s:.2f}s '
                f'val_auprc={v_auprc:.4f} val_auc={v_auc:.4f} val_f1_macro@0.5={v_f1:.4f} '
                f'(best_epoch={best_epoch}, best_auprc={best_auprc:.4f})')
            if epoch % 5 == 0:
                flush_log()
            if stale_epochs >= cli.early_stop:
                log(f'Early stopping at epoch {epoch} (no improvement for {cli.early_stop} epochs)')
                break
    except Exception as e:  # record the failure as evidence, then exit non-zero
        status = 'FAILED_TECHNICAL_OOM' if is_oom(e) else 'FAILED_TECHNICAL'
        peak = torch.cuda.max_memory_allocated() if use_cuda else 0
        log(f'{status} at epoch {epoch}: {type(e).__name__}: {e}')
        log(f'peak GPU memory before failure: {peak / 2**30:.2f} GiB')
        with open(f'{result_dir}/failure.json', 'w') as f:
            json.dump({'status': status, 'epoch': epoch, 'error': f'{type(e).__name__}: {e}',
                       'peak_memory_bytes': peak, 'run_name': run_name}, f, indent=2)
        flush_log()
        sys.exit(3)
    n_epochs = epoch + 1

    # ---- threshold on validation, then test exactly once ----
    net.load_state_dict(torch.load(ckpt, map_location=device))
    sync()
    t_inf = time.perf_counter()
    p_all = predict(net, x, edge_index)
    sync()
    inference_seconds = time.perf_counter() - t_inf
    p_valid = p_all[idx_valid]
    best_thr, best_vf1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        f1v = f1_score(y_valid, (p_valid >= thr).astype(int), average='macro', zero_division=0)
        if f1v > best_vf1:
            best_vf1, best_thr = f1v, float(thr)
    val_auprc_reload = float(average_precision_score(y_valid, p_valid))
    log(f'Selected threshold (best val macro-F1): {best_thr:.2f} (val F1={best_vf1:.4f})')
    log(f'Reloaded checkpoint val AUPRC {val_auprc_reload:.6f} vs logged {best_auprc:.6f}')

    p_test = p_all[idx_test]  # the only time test entries are read
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
