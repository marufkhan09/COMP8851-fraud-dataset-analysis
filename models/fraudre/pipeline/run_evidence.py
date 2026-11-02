"""
FRAUDRE evidence pipeline: proper TR40/30/20/10 x seed 2/42/72 runs with
validation-based checkpoint/threshold selection and required artefacts.

Usage:
  python run_evidence.py --data amazon --ratio 40 --seed 2
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, precision_score,
                               average_precision_score, recall_score, f1_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utlis import load_data, normalize, sparse_to_adjlist
from model import MODEL
from layers import IntraAgg, InterAgg, MLP_
from generate_splits import load_or_generate_splits, verify_splits

VAST_ROOT = '/workspace/FRAUDRE'
SPLITS_DIR = os.path.join(VAST_ROOT, 'splits')
RESULTS_ROOT = os.path.join(VAST_ROOT, 'results')

BATCH_LR_DEFAULTS = {
    'yelp':     {'batch_size': 1024, 'lr': 0.001},
    'amazon':   {'batch_size': 256,  'lr': 0.1},
    'comp':     {'batch_size': 256,  'lr': 0.1},
    'tfinance': {'batch_size': 512,  'lr': 0.01},
    'elliptic': {'batch_size': 512,  'lr': 0.01},
}

MAX_EPOCHS = 100
PATIENCE = 20
EMBED_DIM = 64
LAMBDA_1 = 1e-4


def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=VAST_ROOT
        ).decode().strip()
    except Exception:
        return 'UNKNOWN'


def sha256_of_array(arr):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def evaluate(idx, y_true, gnn_model, cuda, eval_batch_size=256):
    """Batched evaluation -- the original FRAUDRE code (and my first draft)
    pushed the ENTIRE idx list through to_prob() in one unbatched forward
    pass, which is a real OOM risk on large validation/test sets (YelpChi's
    9190-node valid set every epoch; T-Finance/Elliptic would be worse).
    This chunks idx into eval_batch_size pieces and concatenates results,
    wrapped in torch.no_grad() since no gradient is needed here, which also
    cuts memory versus the original (ungraded) forward passes.

    Returns auc, precision, auprc(a_p), recall, f1, probs, extra_metrics.
    """
    idx = list(idx)
    y_true = np.asarray(y_true)
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(idx), eval_batch_size):
            batch_idx = idx[i:i + eval_batch_size]
            gnn_prob = gnn_model.to_prob(batch_idx, train_flag=False)
            batch_arr = gnn_prob.data.cpu().numpy()
            all_probs.append(batch_arr[:, 1])
            all_preds.append(batch_arr.argmax(axis=1))
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)

    auc = roc_auc_score(y_true, probs)
    precision = precision_score(y_true, preds, average='macro', zero_division=0)
    auprc = average_precision_score(y_true, probs)
    recall = recall_score(y_true, preds, average='macro', zero_division=0)
    f1 = f1_score(y_true, preds, average='macro', zero_division=0)

    # fraud-class-specific (positive class = 1) metrics, not macro-averaged
    fraud_precision = precision_score(y_true, preds, pos_label=1,
                                       average='binary', zero_division=0)
    fraud_recall = recall_score(y_true, preds, pos_label=1,
                                 average='binary', zero_division=0)
    fraud_f1 = f1_score(y_true, preds, pos_label=1,
                         average='binary', zero_division=0)

    # G-Mean = sqrt(sensitivity * specificity) = sqrt(recall_pos * recall_neg)
    recall_neg = recall_score(y_true, preds, pos_label=0,
                               average='binary', zero_division=0)
    g_mean = float(np.sqrt(fraud_recall * recall_neg))

    extra_metrics = {
        'fraud_precision': float(fraud_precision),
        'fraud_recall': float(fraud_recall),
        'fraud_f1': float(fraud_f1),
        'g_mean': g_mean,
    }
    return auc, precision, auprc, recall, f1, probs, extra_metrics


def select_threshold(y_true, probs):
    """Sweep thresholds, pick the one maximizing macro-F1 on the given (val) set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                         choices=['yelp', 'amazon', 'comp', 'tfinance', 'elliptic'])
    parser.add_argument('--ratio', type=int, required=True, choices=[40, 30, 20, 10])
    parser.add_argument('--seed', type=int, required=True, choices=[2, 42, 72])
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()

    run_id = f"fraudre_{args.data}_unified_tr{args.ratio}_seed{args.seed}_" \
             f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    result_dir = os.path.join(RESULTS_ROOT, args.data, 'unified',
                               f'tr{args.ratio}', f'seed_{args.seed}')
    ckpt_dir = os.path.join(result_dir, 'checkpoints')

    summary_path = os.path.join(result_dir, 'summary.json')
    if os.path.exists(summary_path):
        print(f"SKIP: {run_id} already has summary.json")
        return

    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"=== Run: {run_id} ===")
    print(f"Result dir: {result_dir}")

    # ---- seed everything for this run ----
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # ---- load data ----
    homo, relation1, relation2, relation3, feat_data, labels = load_data(args.data)
    labels = np.asarray(labels)

    timesteps = None
    if args.data == 'elliptic':
        timesteps = np.load(os.path.join(VAST_ROOT, 'data', 'elliptic_timesteps.npy'))

    # ---- canonical split (seed=2 fixed, regenerated/cached) ----
    split = load_or_generate_splits(args.data, labels, timesteps, SPLITS_DIR)
    verify_splits(split)

    pool = split['train_pool_order']
    n_tr = split['tr_sizes'][args.ratio]
    idx_train = pool[:n_tr].tolist()
    idx_valid = split['valid_idx'].tolist()
    idx_test = split['test_idx'].tolist()

    y_train = labels[np.array(idx_train)]
    y_valid = labels[np.array(idx_valid)]
    y_test = labels[np.array(idx_test)]

    print(f"train={len(idx_train)} valid={len(idx_valid)} test={len(idx_test)}")

    split_sha = sha256_of_array(np.concatenate([idx_train, idx_valid, idx_test]))

    # ---- prior from TRAIN split only ----
    num_1 = max(1, len(np.where(y_train == 1)[0]))
    num_2 = max(1, len(np.where(y_train == 0)[0]))
    p0 = num_1 / (num_1 + num_2)
    p1 = 1 - p0
    prior = np.array([p1, p0])
    prior = torch.from_numpy(prior + 1e-8)
    if cuda:
        prior = prior.cuda()

    # ---- model input ----
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    feat_data_norm = normalize(feat_data)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data_norm), requires_grad=False)
    if cuda:
        features.cuda()

    adj_lists = [relation1, relation2, relation3]

    hp = BATCH_LR_DEFAULTS[args.data]
    batch_size = hp['batch_size']
    lr = hp['lr']

    mlp = MLP_(features, feat_data.shape[1], EMBED_DIM, cuda=cuda)

    intra1 = [IntraAgg(cuda=cuda) for _ in range(3)]
    agg1 = InterAgg(lambda nodes: mlp(nodes), EMBED_DIM, adj_lists, intra1, cuda=cuda)

    intra2 = [IntraAgg(cuda=cuda) for _ in range(3)]
    agg2 = InterAgg(lambda nodes: agg1(nodes), EMBED_DIM * 2, adj_lists, intra2, cuda=cuda)

    gnn_model = MODEL(2, 2, EMBED_DIM, agg2, prior)
    if cuda:
        gnn_model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gnn_model.parameters()),
        lr=lr, weight_decay=LAMBDA_1)

    # ---- training loop with early stopping on val AUPRC ----
    best_val_auprc = -1.0
    best_epoch = -1
    non_improving = 0
    epoch_time_rows = []
    val_metric_rows = []
    best_ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')

    overall_time = 0.0
    for epoch in range(MAX_EPOCHS):
        idx_train_shuf = idx_train.copy()
        random.shuffle(idx_train_shuf)
        num_batches = int(len(idx_train_shuf) / batch_size) + 1

        epoch_time = 0.0
        last_loss = None
        for batch in range(num_batches):
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(idx_train_shuf))
            if i_start >= i_end:
                continue
            batch_nodes = idx_train_shuf[i_start:i_end]
            batch_label = labels[np.array(batch_nodes)]

            optimizer.zero_grad()
            t0 = time.time()
            if cuda:
                loss = gnn_model.loss(batch_nodes, torch.cuda.LongTensor(batch_label))
            else:
                loss = gnn_model.loss(batch_nodes, torch.LongTensor(batch_label))
            t1 = time.time()
            epoch_time += (t1 - t0)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        overall_time += epoch_time

        # ---- validation every epoch (needed for early stopping / checkpoint) ----
        val_auc, val_prec, val_auprc, val_recall, val_f1, val_probs, val_extra = evaluate(
            idx_valid, y_valid, gnn_model, cuda, eval_batch_size=batch_size)
        val_metric_rows.append({
            'epoch': epoch, 'loss': last_loss, 'epoch_time_s': epoch_time,
            'val_auc': val_auc, 'val_precision': val_prec, 'val_auprc': val_auprc,
            'val_recall': val_recall, 'val_f1_macro': val_f1,
            'val_fraud_precision': val_extra['fraud_precision'],
            'val_fraud_recall': val_extra['fraud_recall'],
            'val_fraud_f1': val_extra['fraud_f1'],
            'val_g_mean': val_extra['g_mean'],
        })
        epoch_time_rows.append({'epoch': epoch, 'time_s': epoch_time})

        print(f"epoch {epoch}: loss={last_loss:.4f} val_auprc={val_auprc:.4f} "
              f"val_auc={val_auc:.4f} val_f1_macro={val_f1:.4f} "
              f"(best_epoch={best_epoch}, best_auprc={best_val_auprc:.4f})")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            non_improving = 0
            torch.save(gnn_model.state_dict(), best_ckpt_path)
        else:
            non_improving += 1
            if non_improving >= PATIENCE:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # ---- reload best checkpoint ----
    gnn_model.load_state_dict(torch.load(best_ckpt_path))

    # ---- threshold selection on VALID set only ----
    _, _, _, _, _, val_probs_final, _ = evaluate(idx_valid, y_valid, gnn_model, cuda,
                                                   eval_batch_size=batch_size)
    best_threshold, val_f1_at_threshold = select_threshold(y_valid, val_probs_final)
    print(f"Selected threshold (best val macro-F1): {best_threshold:.2f} "
          f"(val F1={val_f1_at_threshold:.4f})")

    # ---- test set touched exactly once, here ----
    test_auc, _, test_auprc, _, _, test_probs, _ = evaluate(idx_test, y_test, gnn_model, cuda,
                                                               eval_batch_size=batch_size)
    test_preds = (test_probs >= best_threshold).astype(int)
    y_test = np.asarray(y_test)
    test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(y_test, test_preds, average='macro', zero_division=0)

    # fraud-class-specific metrics and G-Mean, computed at the CHOSEN
    # threshold (consistent with test_f1/test_recall above), not the
    # argmax-based predictions evaluate() uses internally for val logging
    test_fraud_precision = precision_score(y_test, test_preds, pos_label=1,
                                            average='binary', zero_division=0)
    test_fraud_recall = recall_score(y_test, test_preds, pos_label=1,
                                      average='binary', zero_division=0)
    test_fraud_f1 = f1_score(y_test, test_preds, pos_label=1,
                              average='binary', zero_division=0)
    test_recall_neg = recall_score(y_test, test_preds, pos_label=0,
                                    average='binary', zero_division=0)
    test_g_mean = float(np.sqrt(test_fraud_recall * test_recall_neg))

    test_metrics = {
        'auprc': float(test_auprc), 'auc': float(test_auc),
        'f1_macro': float(test_f1), 'recall_macro': float(test_recall),
        'fraud_precision': float(test_fraud_precision),
        'fraud_recall': float(test_fraud_recall),
        'fraud_f1': float(test_fraud_f1),
        'g_mean': test_g_mean,
        'threshold_used': float(best_threshold), 'best_epoch': int(best_epoch),
    }
    print("TEST metrics:", test_metrics)

    # ---- write required artefacts ----
    with open(os.path.join(result_dir, 'run_config.yml'), 'w') as f:
        f.write(f"data: {args.data}\nratio: {args.ratio}\nseed: {args.seed}\n"
                f"batch_size: {batch_size}\nlr: {lr}\nembed_dim: {EMBED_DIM}\n"
                f"lambda_1: {LAMBDA_1}\nmax_epochs: {MAX_EPOCHS}\npatience: {PATIENCE}\n")

    with open(os.path.join(result_dir, 'protocol_identity.json'), 'w') as f:
        json.dump({
            'model_source_commit': get_git_commit(),
            'split_sha256': split_sha,
            'run_id': run_id,
            'hardware_profile_id': 'PENDING_TEAM_MANIFEST',
            'container_digest': 'PENDING_TEAM_MANIFEST',
            'dataset_manifest_version': 'PENDING_TEAM_MANIFEST',
            'source_sha256': 'PENDING_TEAM_MANIFEST',
            'evaluator_commit': 'PENDING_TEAM_MANIFEST',
        }, f, indent=2)

    with open(os.path.join(result_dir, 'environment.txt'), 'w') as f:
        f.write(f"torch: {torch.__version__}\n"
                f"cuda_available: {cuda}\n"
                f"python: {sys.version}\n")

    with open(os.path.join(result_dir, 'split_summary.json'), 'w') as f:
        json.dump({
            'n_train': len(idx_train), 'n_valid': len(idx_valid), 'n_test': len(idx_test),
            'ratio': args.ratio, 'split_seed': 2, 'training_seed': args.seed,
        }, f, indent=2)

    np.savez(os.path.join(result_dir, 'split_indices.npz'),
             idx_train=idx_train, idx_valid=idx_valid, idx_test=idx_test)

    import csv
    with open(os.path.join(result_dir, 'epoch_times.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'time_s'])
        w.writeheader()
        w.writerows(epoch_time_rows)

    with open(os.path.join(result_dir, 'validation_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(val_metric_rows[0].keys()))
        w.writeheader()
        w.writerows(val_metric_rows)

    with open(os.path.join(result_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
        json.dump({
            'run_id': run_id, 'data': args.data, 'ratio': args.ratio, 'seed': args.seed,
            'best_epoch': int(best_epoch), 'epochs_run': len(epoch_time_rows),
            'overall_time_s': overall_time, 'test_metrics': test_metrics,
        }, f, indent=2)

    print(f"epoch_times.csv row count matches completed epochs: {len(epoch_time_rows)}")
    print(f"=== Run complete: {run_id} ===")
    print(f"Artefacts written to: {result_dir}")


if __name__ == '__main__':
    main()
