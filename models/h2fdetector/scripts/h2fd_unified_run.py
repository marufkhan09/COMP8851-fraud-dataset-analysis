#!/usr/bin/env python
"""
H2-FDetector unified-protocol run: one dataset x training ratio x seed.

Protocol (identical to the group's SplitGNN unified runs, run_evidence.py):
  * max 100 epochs, early stopping patience 20 on validation AUPRC
  * checkpoint = best validation AUPRC (strict improvement)
  * decision threshold = best validation macro-F1 over np.arange(0.05, 0.96, 0.05)
  * no test data used for any selection
Model, loss, optimiser (Adam) and all author hyperparameters come unchanged from the
authors' config yaml; only seed / epochs / patience / device are overridden.
Training step and feature normalisation mirror the authors' train.py line for line.

Evidence bundle written to
  <results-root>/<dataset>/unified/tr<R>/seed_<S>/
with the same file names and keys as the SplitGNN bundle, plus
full_test_metrics.json (all 7 metrics) and predictions.npz (val/test probabilities).

Idempotent: a run whose summary.json exists is skipped unless --force is given.

Usage:
  python h2fd_unified_run.py --dataset yelp --ratio 40 --seed 2
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
import dgl
import scipy
import sklearn
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

PROTOCOL_VERSION = 'vast-v4.4'
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f'UNAVAILABLE ({e})'


def full_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    rec_pos = recall_score(labels, preds, pos_label=1, zero_division=0)
    rec_neg = recall_score(labels, preds, pos_label=0, zero_division=0)
    return {
        'auprc': float(average_precision_score(labels, probs)),
        'auc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        'f1_macro': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'fraud_precision': float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        'fraud_recall': float(rec_pos),
        'fraud_f1': float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        'g_mean': float(np.sqrt(max(rec_pos, 0) * max(rec_neg, 0))),
        'threshold_used': float(threshold),
        'n_pred_fraud': int(preds.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--ratio', type=int, required=True, choices=[40, 30, 20, 10])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--h2fd-root', default='/workspace/H2-FDetector')
    p.add_argument('--views-dir', default='/workspace/h2fd_vast/views')
    p.add_argument('--results-root', default='/workspace/h2fd_vast/results/h2fdetector_results')
    p.add_argument('--max-epoch', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--force', action='store_true')
    cli = p.parse_args()

    ds, r, s = cli.dataset, cli.ratio, cli.seed
    result_dir = f'{cli.results_root}/{ds}/unified/tr{r}/seed_{s}'
    if os.path.exists(f'{result_dir}/summary.json') and not cli.force:
        print(f'SKIP (already complete): {result_dir}')
        return 0
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)  # remove a partial / forced run so no stale rows remain
    os.makedirs(f'{result_dir}/checkpoints', exist_ok=True)

    log_f = open(f'{result_dir}/terminal.log', 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')

    sys.path.insert(0, f'{cli.h2fd_root}/src')
    from model import H2FDetector          # authors' model, unchanged
    from utils import setup_seed, normalize  # authors' helpers, unchanged

    # ---------------- config: author yaml + protocol overrides only
    cfg_path = f'{cli.h2fd_root}/config/{ds}.yaml'
    with open(cfg_path) as f:
        author_cfg = yaml.safe_load(f)
    eff = dict(author_cfg)
    eff.update({'seed': s, 'epoch': cli.max_epoch, 'early_stop': cli.patience, 'cuda': 0,
                'ratio': r, 'track': 'unified', 'selection_metric': 'val_auprc',
                'threshold_rule': 'best_val_macro_f1_grid_0.05_0.95'})
    with open(f'{result_dir}/run_config.yml', 'w') as f:
        yaml.safe_dump(eff, f, sort_keys=True)
    config_sha256 = hashlib.sha256(json.dumps(eff, sort_keys=True).encode()).hexdigest()

    device = torch.device('cuda:0')
    args = argparse.Namespace(**eff)
    args.device = device

    # ---------------- dataset view
    view_path = f'{cli.views_dir}/{ds}_tr{r}.dgl'
    split_src = f'{cli.views_dir}/{ds}_tr{r}_split.npz'
    manifest_path = f'{cli.views_dir}/{ds}_views_manifest.json'
    for pth in (view_path, split_src, manifest_path):
        if not os.path.exists(pth):
            log(f'ERROR: missing {pth} (run h2fd_build_views.py --dataset {ds})')
            return 1
    with open(manifest_path) as f:
        vmanifest = json.load(f)
    vinfo = vmanifest['views'][f'tr{r}']

    setup_seed(s)

    t_load = time.perf_counter()
    g = dgl.load_graphs(view_path)[0][0]
    features = g.ndata['feature'].numpy()          # same three lines as authors' train.py
    features = normalize(features)
    g.ndata['feature'] = torch.from_numpy(features).float()
    g = g.to(device)
    torch.cuda.synchronize()
    dataset_load_seconds = time.perf_counter() - t_load

    labels = g.ndata['label']
    train_mask = g.ndata['train_mask'].bool()
    valid_mask = g.ndata['valid_mask'].bool()
    test_mask = g.ndata['test_mask'].bool()
    valid_labels = labels[valid_mask].cpu().numpy()
    test_labels = labels[test_mask].cpu().numpy()

    # split evidence: copy the view's split file so split_sha256 is the same for every seed
    shutil.copyfile(split_src, f'{result_dir}/split_indices.npz')
    split_sha256 = sha256_file(f'{result_dir}/split_indices.npz')
    sz = np.load(f'{result_dir}/split_indices.npz')
    assert np.array_equal(sz['train_mask'].astype(bool), train_mask.cpu().numpy())
    assert np.array_equal(sz['valid_mask'].astype(bool), valid_mask.cpu().numpy())
    assert np.array_equal(sz['test_mask'].astype(bool), test_mask.cpu().numpy())
    n_train, n_valid, n_test = int(train_mask.sum()), int(valid_mask.sum()), int(test_mask.sum())
    with open(f'{result_dir}/split_summary.json', 'w') as f:
        json.dump({'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
                   'split_sha256': split_sha256, 'split_seed': 2,
                   'split_content_sha256': vinfo['split_content_sha256']}, f, indent=2)

    # ---------------- provenance
    gpu_name = torch.cuda.get_device_name(0)
    gpu_tag = gpu_name.lower().replace('nvidia', '').replace('geforce', '').replace('rtx', '').strip().replace(' ', '_')
    hw_query = run_cmd(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                        '--format=csv,noheader'])
    with open(f'{result_dir}/hardware.json', 'w') as f:
        json.dump({'gpu_query': hw_query, 'note': 'hardware_profile_id PENDING_TEAM_MANIFEST'}, f, indent=2)
    with open(f'{result_dir}/environment.txt', 'w') as f:
        f.write(f'Package versions used for this run (Vast.ai {gpu_name}):\n')
        f.write(f'numpy: {np.__version__}\ntorch: {torch.__version__}\ndgl: {dgl.__version__}\n')
        f.write(f'scipy: {scipy.__version__}\nscikit-learn: {sklearn.__version__}\n')
        f.write(f'python: {platform.python_version()}\n')
        f.write(f'CUDA available: {torch.cuda.is_available()}\n')
        f.write(f'CUDA (torch build): {torch.version.cuda}\n')
        f.write(f'Hardware: {gpu_name} (GPU 0)\n')

    model_commit = run_cmd(['git', '-C', cli.h2fd_root, 'rev-parse', 'HEAD'])
    local_changes = run_cmd(['git', '-C', cli.h2fd_root, 'status', '--porcelain', '--untracked-files=no'])
    protocol_identity = {
        'protocol_version': PROTOCOL_VERSION,
        'hardware_profile_id': 'PENDING_TEAM_MANIFEST',
        'container_digest': 'PENDING_TEAM_MANIFEST',
        'model_source_commit': model_commit,
        'model_source_local_changes': local_changes.splitlines(),
        'dataset_manifest_version': 'PENDING_TEAM_MANIFEST',
        'source_sha256': vmanifest['source_mat_sha256'],
        'dataset_view_id': f'{ds}_h2fd_native_canonical_tr{r}_v1',
        'view_sha256': vinfo['view_sha256'],
        'split_registry_version': vmanifest['split_registry_version'],
        'split_sha256': split_sha256,
        'evaluator_commit': 'PENDING_TEAM_MANIFEST',
        'adapter_version': vmanifest['builder_version'],
        'adapter_sha256': vmanifest['builder_sha256'],
        'config_sha256': config_sha256,
    }
    with open(f'{result_dir}/protocol_identity.json', 'w') as f:
        json.dump(protocol_identity, f, indent=2)
    with open(f'{result_dir}/dataset_manifest.json', 'w') as f:
        json.dump({'dataset': ds, 'dgl_path': view_path, 'dgl_sha256': vinfo['view_sha256'],
                   'native_dgl_path': vmanifest['native_dgl_path'],
                   'native_dgl_sha256': vmanifest['native_dgl_sha256'],
                   'source_mat_path': vmanifest['source_mat_path'],
                   'source_mat_sha256': vmanifest['source_mat_sha256'],
                   'n_nodes': vmanifest['n_nodes'], 'n_edges': vmanifest['n_edges'],
                   'edge_label_verification': vmanifest['edge_label_verification']}, f, indent=2)
    with open(f'{result_dir}/dataset_view_manifest.json', 'w') as f:
        json.dump({'view_id': protocol_identity['dataset_view_id'],
                   'view_sha256': protocol_identity['view_sha256']}, f, indent=2)

    run_name = (f'h2fdetector_{ds}_unified_tr{r}_seed{s}_{gpu_tag}_'
                f'{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}')
    log(f'Run: {run_name}')
    log(f'Model commit: {model_commit}  local changes: {local_changes.splitlines()}')
    log(f'Config SHA256: {config_sha256}')
    log(f'Split SHA256: {split_sha256}')
    log(f'train={n_train} valid={n_valid} test={n_test}  dataset_load={dataset_load_seconds:.2f}s')

    # ---------------- model (authors' construction)
    model = H2FDetector(args, g).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt_path = f'{result_dir}/checkpoints/best_model.pt'

    ep_f = open(f'{result_dir}/epoch_times.csv', 'w', newline='', buffering=1)
    ep_w = csv.writer(ep_f)
    ep_w.writerow(['epoch', 'train_seconds', 'valid_seconds', 'peak_memory_bytes', 'loss'])
    va_f = open(f'{result_dir}/validation_metrics.csv', 'w', newline='', buffering=1)
    va_w = csv.writer(va_f)
    va_w.writerow(['epoch', 'val_auprc', 'val_auc', 'val_f1_macro', 'val_recall'])

    best_auprc, best_epoch, bad, n_done = -1.0, -1, 0, 0
    train_total, valid_total, peak_overall = 0.0, 0.0, 0
    stop_reason = 'max_epoch'
    for e in range(cli.max_epoch):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        loss = model.loss(g)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            model.eval()
            logits = model(g)
            vprobs = torch.softmax(logits[valid_mask], dim=1)[:, 1].cpu().numpy()
        torch.cuda.synchronize()
        valid_s = time.perf_counter() - t1
        peak = int(torch.cuda.max_memory_allocated(device))

        v_auprc = float(average_precision_score(valid_labels, vprobs))
        v_auc = float(roc_auc_score(valid_labels, vprobs))
        vpred = (vprobs >= 0.5).astype(int)
        v_f1 = float(f1_score(valid_labels, vpred, average='macro', zero_division=0))
        v_rec = float(recall_score(valid_labels, vpred, zero_division=0))

        ep_w.writerow([e, train_s, valid_s, peak, float(loss.item())])
        va_w.writerow([e, v_auprc, v_auc, v_f1, v_rec])
        train_total += train_s
        valid_total += valid_s
        peak_overall = max(peak_overall, peak)
        n_done = e + 1

        if v_auprc > best_auprc:
            best_auprc, best_epoch, bad = v_auprc, e, 0
            torch.save(model.state_dict(), ckpt_path)
            flag = ' *'
        else:
            bad += 1
            flag = ''
        log(f'{e}: loss={loss.item():.4f} val_auprc={v_auprc:.4f} val_auc={v_auc:.4f} '
            f'val_f1@0.5={v_f1:.4f} val_recall@0.5={v_rec:.4f} '
            f'train={train_s:.2f}s valid={valid_s:.2f}s peak={peak / 2**30:.2f}GiB '
            f'best={best_auprc:.4f}@{best_epoch}{flag}')
        if bad >= cli.patience:
            stop_reason = f'early_stop_patience_{cli.patience}'
            log(f'Early stopping at epoch {e} (best epoch {best_epoch})')
            break
    ep_f.close()
    va_f.close()

    # ---------------- reload best checkpoint, select threshold on validation, evaluate test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    torch.cuda.synchronize()
    t_inf = time.perf_counter()
    with torch.no_grad():
        logits = model(g)
        probs_all = torch.softmax(logits, dim=1)[:, 1]
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t_inf
    vprobs = probs_all[valid_mask].cpu().numpy()
    tprobs = probs_all[test_mask].cpu().numpy()
    reload_val_auprc = float(average_precision_score(valid_labels, vprobs))

    best_thresh, best_f1 = 0.5, -1.0
    for th in THRESHOLD_GRID:
        f1 = f1_score(valid_labels, (vprobs >= th).astype(int), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(th)
    log(f'Selected threshold (best val macro-F1): {best_thresh:.2f} (val F1={best_f1:.4f})')
    log(f'Reload check: val AUPRC at best epoch {best_auprc:.6f}, after reload {reload_val_auprc:.6f}')

    full = full_metrics(test_labels, tprobs, best_thresh)
    test_metrics = {  # same keys as the SplitGNN bundle
        'auprc': full['auprc'], 'auc': full['auc'], 'f1_macro': full['f1_macro'],
        'recall': full['fraud_recall'], 'threshold_used': best_thresh, 'best_epoch': best_epoch,
    }
    with open(f'{result_dir}/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    with open(f'{result_dir}/full_test_metrics.json', 'w') as f:
        json.dump({**full, 'best_epoch': best_epoch, 'val_macro_f1_at_threshold': float(best_f1)}, f, indent=2)
    np.savez(f'{result_dir}/predictions.npz', val_probs=vprobs, val_labels=valid_labels,
             test_probs=tprobs, test_labels=test_labels)

    summary = {
        'run_name': run_name, 'model': 'h2fdetector', 'dataset': ds, 'track': 'unified',
        'ratio': r, 'seed': s,
        'n_completed_epochs': n_done, 'best_epoch': best_epoch,
        'best_val_auprc': best_auprc, 'reload_val_auprc': reload_val_auprc,
        'stop_reason': stop_reason,
        'dataset_load_seconds': dataset_load_seconds,
        'train_only_total_seconds': train_total,
        'train_plus_valid_total_seconds': train_total + valid_total,
        'inference_seconds': inference_seconds,
        'peak_memory_bytes': peak_overall,
        'gpu': gpu_name,
        'test_metrics': test_metrics,
    }
    with open(f'{result_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'TEST: ' + ' '.join(f'{k}={v:.4f}' for k, v in full.items() if isinstance(v, float)))
    log(f'Done: {result_dir}')
    log_f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
