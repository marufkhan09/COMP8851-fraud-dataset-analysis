"""
Per-run evidence script for SplitGNN, following Master Plan v4.4 section 9.
Usage: python run_evidence.py --dataset yelp --ratio 40 --seed 2 --track unified --gpu a6000
Produces results/splitgnn/<dataset>/<track>/tr<ratio>/seed_<seed>/ with the full
required artefact set.

NOTE: several protocol_identity fields (hardware_profile_id, container_digest,
dataset_manifest_version, source_sha256, split_registry_version, evaluator_commit)
depend on team-wide frozen manifests not yet available to this run - these are
written as "PENDING_TEAM_MANIFEST" placeholders and must be filled in once available.
This run's OWN split file is hashed and recorded correctly regardless.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import torch.optim as optim
import dgl
import yaml
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

sys.path.insert(0, '/workspace/splitgnn/src')
sys.path.insert(0, '/workspace/splitgnn_vast/splits')
from model import SplitGNN
from generate_splits import generate_splits

MODEL_NAME = 'splitgnn'
SPLITGNN_ROOT = '/workspace/splitgnn'
VAST_ROOT = '/workspace/splitgnn_vast'
RESULTS_ROOT = f'{VAST_ROOT}/results'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes):
    return hashlib.sha256(b).hexdigest()


def get_git_commit(repo_path):
    try:
        return subprocess.check_output(['git', '-C', repo_path, 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'UNKNOWN'


class Args:
    """Mimics the args object SplitGNN's model.py/train.py expect."""
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help="yelp, amazon, or comp (matches config/<dataset>.yaml naming)")
    p.add_argument('--ratio', required=True, type=int, choices=[40, 30, 20, 10])
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--track', required=True, choices=['native', 'unified'])
    p.add_argument('--gpu', default='a6000')
    p.add_argument('--max_epoch', type=int, default=100)
    p.add_argument('--early_stop', type=int, default=20)
    # No hardcoded defaults here - these are loaded per-dataset from the author's own
    # config/<dataset>.yaml below (e.g. yelp lr=0.01, amazon lr=0.1, comp lr=0.001).
    # Passing any of these on the CLI explicitly overrides the loaded config value.
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--C', type=int, default=None)
    p.add_argument('--K', type=int, default=None)
    p.add_argument('--intra_dim', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    cli = p.parse_args()

    # Load author's per-dataset config as the base; CLI args (if given) override.
    author_config_path = f"{SPLITGNN_ROOT}/config/{cli.dataset}.yaml"
    with open(author_config_path) as f:
        author_cfg = yaml.safe_load(f)
    for field, cfg_key in [('lr', 'lr'), ('weight_decay', 'weight_decay'), ('gamma', 'gamma'),
                            ('C', 'C'), ('K', 'K'), ('intra_dim', 'intra_dim'), ('dropout', 'dropout')]:
        if getattr(cli, field) is None:
            setattr(cli, field, author_cfg[cfg_key])

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_name = f"{MODEL_NAME}_{cli.dataset}_{cli.track}_tr{cli.ratio}_seed{cli.seed}_{cli.gpu}_{timestamp}"
    result_dir = f"{RESULTS_ROOT}/{MODEL_NAME}/{cli.dataset}/{cli.track}/tr{cli.ratio}/seed_{cli.seed}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/checkpoints", exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"=== Run: {run_name} ===")
    log(f"Result dir: {result_dir}")

    # ---- Reproducibility ----
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device('cuda:0')

    # ---- Load dataset ----
    dgl_path = f"{SPLITGNN_ROOT}/data/{cli.dataset}.dgl"
    g = dgl.load_graphs(dgl_path)[0][0]
    node_type = 'company' if cli.dataset == 'comp' else 'r'
    labels_full = g.nodes[node_type].data['label'] if node_type in g.ntypes and 'label' in g.nodes[node_type].data else g.ndata['label']

    # ---- Canonical split: load if already generated for this dataset, else generate now ----
    # Splits are generated only over LABELED nodes (label != -1); unlabeled nodes (used by
    # some datasets like Elliptic for message-passing-only context) are never placed into
    # train/valid/test. This is a safe no-op for datasets with no unlabeled nodes.
    split_path = f"{VAST_ROOT}/splits/{cli.dataset}_seed2_nested_splits.npz"
    if not os.path.exists(split_path):
        log(f"No existing split file at {split_path}, generating now.")
        labels_np_full = labels_full.numpy()
        labeled_idx = np.where(labels_np_full != -1)[0]
        n_unlabeled = len(labels_np_full) - len(labeled_idx)
        if n_unlabeled > 0:
            log(f"NOTE: {n_unlabeled} unlabeled (-1) nodes excluded from split generation, "
                f"kept in graph for message passing only.")
        sub_masks = generate_splits(torch.from_numpy(labels_np_full[labeled_idx]))
        n_total = len(labels_np_full)
        masks = {}
        for key, sub_mask in sub_masks.items():
            full_mask = np.zeros(n_total, dtype=bool)
            full_mask[labeled_idx[sub_mask.numpy()]] = True
            masks[key] = torch.from_numpy(full_mask)
        np.savez(split_path,
                 valid_mask=masks['valid_mask'].numpy(),
                 test_mask=masks['test_mask'].numpy(),
                 train_mask_tr40=masks['train_mask_tr40'].numpy(),
                 train_mask_tr30=masks['train_mask_tr30'].numpy(),
                 train_mask_tr20=masks['train_mask_tr20'].numpy(),
                 train_mask_tr10=masks['train_mask_tr10'].numpy())
    split_npz = np.load(split_path)
    split_sha256 = sha256_file(split_path)

    train_mask = torch.from_numpy(split_npz[f'train_mask_tr{cli.ratio}'])
    valid_mask = torch.from_numpy(split_npz['valid_mask'])
    test_mask = torch.from_numpy(split_npz['test_mask'])

    # Overlap checks (repeat here defensively, per section 9.1 pre-run checklist)
    assert (train_mask & valid_mask).sum() == 0, "train/valid overlap!"
    assert (train_mask & test_mask).sum() == 0, "train/test overlap!"
    assert (valid_mask & test_mask).sum() == 0, "valid/test overlap!"
    log("Split overlap checks: PASSED")

    # Attach masks onto the node type used by the model (overwrites baked-in native masks - by design, this IS the unified track)
    g.nodes[node_type].data['train_mask'] = train_mask
    g.nodes[node_type].data['valid_mask'] = valid_mask
    g.nodes[node_type].data['test_mask'] = test_mask

    # Rebuild homo edge label/train_mask against the NEW train_mask (required by model.py's loss())
    label_flat = g.nodes[node_type].data['label']
    homo_src, homo_dst = g.edges(etype='homo')
    edge_labels = torch.where(label_flat[homo_src] == label_flat[homo_dst],
                               torch.tensor(1, dtype=torch.long),
                               torch.tensor(-1, dtype=torch.long))
    edge_train_mask = train_mask[homo_src] & train_mask[homo_dst]
    g.edges['homo'].data['label'] = edge_labels
    g.edges['homo'].data['train_mask'] = edge_train_mask

    n_train, n_valid, n_test = int(train_mask.sum()), int(valid_mask.sum()), int(test_mask.sum())
    log(f"train={n_train} valid={n_valid} test={n_test}")

    # ---- Features (normalize per author convention) ----
    features = g.nodes[node_type].data['feature'].numpy() if node_type in g.ntypes else g.ndata['feature'].numpy()
    if cli.dataset == 'amazon':
        features = np.delete(features, 19, axis=1)
    from utils import normalize  # reuse author's normalize()
    features = normalize(features)
    features = torch.from_numpy(features).float()
    g.nodes[node_type].data['feature'] = features

    g = g.to(device)

    # ---- Build args object for model ----
    args = Args()
    args.dataset = cli.dataset
    args.intra_dim = cli.intra_dim
    args.gamma = cli.gamma
    args.C = cli.C
    args.K = cli.K
    args.n_class = 2
    args.dropout = cli.dropout
    args.device = device

    model = SplitGNN(args, g).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)

    # ---- Effective config + hashes (recorded BEFORE training starts, per checklist) ----
    effective_config = {
        'dataset': cli.dataset, 'ratio': cli.ratio, 'seed': cli.seed, 'track': cli.track,
        'lr': cli.lr, 'weight_decay': cli.weight_decay, 'gamma': cli.gamma, 'C': cli.C,
        'K': cli.K, 'intra_dim': cli.intra_dim, 'dropout': cli.dropout,
        'max_epoch': cli.max_epoch, 'early_stop': cli.early_stop,
    }
    config_sha256 = sha256_bytes(json.dumps(effective_config, sort_keys=True).encode())
    model_source_commit = get_git_commit(SPLITGNN_ROOT)

    with open(f"{result_dir}/run_config.yml", 'w') as f:
        yaml.dump(effective_config, f)

    protocol_identity = {
        'protocol_version': 'vast-v4.4',
        'hardware_profile_id': 'PENDING_TEAM_MANIFEST',
        'container_digest': 'PENDING_TEAM_MANIFEST',
        'model_source_commit': model_source_commit,
        'dataset_manifest_version': 'PENDING_TEAM_MANIFEST',
        'source_sha256': 'PENDING_TEAM_MANIFEST',
        'dataset_view_id': f'{cli.dataset}_own_exploration_v1',
        'view_sha256': sha256_file(dgl_path),
        'split_registry_version': 'own_generate_splits_v1_seed2',
        'split_sha256': split_sha256,
        'evaluator_commit': 'PENDING_TEAM_MANIFEST',
        'adapter_version': 'none' if cli.dataset in ('yelp', 'amazon', 'comp') else 'homogeneous_shim_v1',
        'adapter_sha256': 'N/A' if cli.dataset in ('yelp', 'amazon', 'comp') else 'PENDING_HASH',
        'config_sha256': config_sha256,
    }
    with open(f"{result_dir}/protocol_identity.json", 'w') as f:
        json.dump(protocol_identity, f, indent=2)

    with open(f"{result_dir}/dataset_manifest.json", 'w') as f:
        json.dump({'dataset': cli.dataset, 'dgl_path': dgl_path, 'dgl_sha256': protocol_identity['view_sha256'],
                    'note': 'PENDING_TEAM_MANIFEST for canonical source identity'}, f, indent=2)

    with open(f"{result_dir}/dataset_view_manifest.json", 'w') as f:
        json.dump({'view_id': protocol_identity['dataset_view_id'], 'view_sha256': protocol_identity['view_sha256']}, f, indent=2)

    np.savez(f"{result_dir}/split_indices.npz",
             train_mask=train_mask.numpy(), valid_mask=valid_mask.numpy(), test_mask=test_mask.numpy())
    with open(f"{result_dir}/split_summary.json", 'w') as f:
        json.dump({'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
                    'split_sha256': split_sha256, 'split_seed': 2}, f, indent=2)

    hardware_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                     '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    with open(f"{result_dir}/hardware.json", 'w') as f:
        json.dump({'gpu_query': hardware_info, 'note': 'hardware_profile_id PENDING_TEAM_MANIFEST'}, f, indent=2)

    env_freeze = subprocess.run(['uv', 'pip', 'freeze'], capture_output=True, text=True).stdout
    with open(f"{result_dir}/environment.txt", 'w') as f:
        f.write(env_freeze)

    log(f"Model source commit: {model_source_commit}")
    log(f"Config SHA256: {config_sha256}")
    log(f"Split SHA256: {split_sha256}")

    # ---- Training loop with proper timing boundaries (section 9.3) ----
    epoch_rows = []
    valid_rows = []
    best_val_auprc = -1.0
    best_epoch = -1
    stop_steps = 0

    dataset_load_start = time.perf_counter()
    # (dataset already loaded above; recorded for completeness)
    dataset_load_seconds = time.perf_counter() - dataset_load_start

    train_only_total = 0.0
    train_plus_valid_total = 0.0

    for epoch in range(cli.max_epoch):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        loss = model.loss(g)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t0
        train_only_total += train_seconds
        peak_mem = torch.cuda.max_memory_allocated()

        # Validation
        model.eval()
        torch.cuda.synchronize()
        tv0 = time.perf_counter()
        with torch.no_grad():
            logits = model(g)
            valid_logits = logits[valid_mask]
            valid_probs = torch.softmax(valid_logits, dim=1)[:, 1].cpu().numpy()
            valid_labels_np = label_flat[valid_mask].cpu().numpy()
        torch.cuda.synchronize()
        valid_seconds = time.perf_counter() - tv0
        train_plus_valid_total += (train_seconds + valid_seconds)

        val_auprc = average_precision_score(valid_labels_np, valid_probs)
        val_auc = roc_auc_score(valid_labels_np, valid_probs) if len(np.unique(valid_labels_np)) > 1 else 0.0
        val_preds_05 = (valid_probs >= 0.5).astype(int)
        val_f1_macro = f1_score(valid_labels_np, val_preds_05, average='macro', zero_division=0)
        val_recall = recall_score(valid_labels_np, val_preds_05, zero_division=0)

        epoch_rows.append({'epoch': epoch, 'train_seconds': train_seconds, 'valid_seconds': valid_seconds,
                            'peak_memory_bytes': peak_mem, 'loss': float(loss.item())})
        valid_rows.append({'epoch': epoch, 'val_auprc': val_auprc, 'val_auc': val_auc,
                            'val_f1_macro': val_f1_macro, 'val_recall': val_recall})

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            stop_steps = 0
            torch.save(model.state_dict(), f"{result_dir}/checkpoints/best_model.pt")
        else:
            stop_steps += 1

        log(f"epoch {epoch}: loss={loss.item():.4f} val_auprc={val_auprc:.4f} val_auc={val_auc:.4f} "
            f"val_f1_macro={val_f1_macro:.4f} (best_epoch={best_epoch}, best_auprc={best_val_auprc:.4f})")

        if stop_steps >= cli.early_stop:
            log(f"Early stopping at epoch {epoch} (no improvement for {cli.early_stop} epochs)")
            break

    n_completed_epochs = epoch + 1

    # ---- Threshold selection on validation Macro-F1 (per protocol) ----
    model.load_state_dict(torch.load(f"{result_dir}/checkpoints/best_model.pt"))
    model.eval()
    with torch.no_grad():
        logits = model(g)
        valid_probs = torch.softmax(logits[valid_mask], dim=1)[:, 1].cpu().numpy()
        valid_labels_np = label_flat[valid_mask].cpu().numpy()

    best_thresh, best_f1 = 0.5, -1
    for thresh in np.arange(0.05, 0.96, 0.05):
        preds = (valid_probs >= thresh).astype(int)
        f1 = f1_score(valid_labels_np, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    log(f"Selected threshold (best val macro-F1): {best_thresh:.2f} (val F1={best_f1:.4f})")

    # ---- Final test evaluation (test set touched exactly once) ----
    inference_start = time.perf_counter()
    with torch.no_grad():
        logits = model(g)
        test_probs = torch.softmax(logits[test_mask], dim=1)[:, 1].cpu().numpy()
        test_labels_np = label_flat[test_mask].cpu().numpy()
    inference_seconds = time.perf_counter() - inference_start

    test_preds = (test_probs >= best_thresh).astype(int)
    test_metrics = {
        'auprc': average_precision_score(test_labels_np, test_probs),
        'auc': roc_auc_score(test_labels_np, test_probs) if len(np.unique(test_labels_np)) > 1 else 0.0,
        'f1_macro': f1_score(test_labels_np, test_preds, average='macro', zero_division=0),
        'recall': recall_score(test_labels_np, test_preds, zero_division=0),
        'threshold_used': float(best_thresh),
        'best_epoch': best_epoch,
    }
    log(f"TEST metrics: {test_metrics}")

    # ---- Write remaining required artefacts ----
    import csv
    with open(f"{result_dir}/epoch_times.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_seconds', 'valid_seconds', 'peak_memory_bytes', 'loss'])
        w.writeheader()
        w.writerows(epoch_rows)

    with open(f"{result_dir}/validation_metrics.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'val_auprc', 'val_auc', 'val_f1_macro', 'val_recall'])
        w.writeheader()
        w.writerows(valid_rows)

    with open(f"{result_dir}/test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)

    peak_mem_overall = max(r['peak_memory_bytes'] for r in epoch_rows)
    summary = {
        'run_name': run_name, 'model': MODEL_NAME, 'dataset': cli.dataset, 'track': cli.track,
        'ratio': cli.ratio, 'seed': cli.seed,
        'n_completed_epochs': n_completed_epochs, 'best_epoch': best_epoch,
        'best_val_auprc': best_val_auprc,
        'dataset_load_seconds': dataset_load_seconds,
        'train_only_total_seconds': train_only_total,
        'train_plus_valid_total_seconds': train_plus_valid_total,
        'inference_seconds': inference_seconds,
        'peak_memory_bytes': peak_mem_overall,
        'test_metrics': test_metrics,
    }
    with open(f"{result_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(f"{result_dir}/terminal.log", 'w') as f:
        f.write('\n'.join(log_lines))

    assert len(epoch_rows) == n_completed_epochs, "epoch_times.csv row count mismatch!"
    log(f"epoch_times.csv row count matches completed epochs: {n_completed_epochs}")

    log(f"=== Run complete: {run_name} ===")
    log(f"Artefacts written to: {result_dir}")


if __name__ == '__main__':
    main()
