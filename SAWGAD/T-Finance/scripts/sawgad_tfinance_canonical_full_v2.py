#!/usr/bin/env python3
"""COMP8851 SAWGAD × T-Finance: frozen-split capacity / tuning / final runs.

Pinned author model and injected synthetic-anomaly code, deterministic T-Finance
input adapter, TR40 validation AUPRC selection, and one final test score per
selected checkpoint. Default phase is non-training status.

Phases:
    --phase capacity                One full TR40 training-batch update; no tests.
    --phase tune --max-trials 1     Run the first full tuning trial.
    --phase tune                   Resume all 12 TR40 tuning trials.
    --phase finals                 After tuning is complete, run 12 final jobs.
    --phase status                 Read existing results only.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('DGLBACKEND', 'pytorch')
ROOT = Path('/workspace/comp8851')
AUTHOR_REPO = ROOT / 'repos/SAWGAD'
ADAPTER = Path('/workspace/sawgad_tfinance_canonical_adapter_v2.py')
SOURCE = ROOT / 'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
OUT = ROOT / 'results/sawgad/tfinance_canonical_v2'
CAPACITY = ROOT / 'diagnostics/sawgad/tfinance_canonical_capacity_v2.json'
SOURCE_SHA = 'b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA = 'de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
ADAPTER_SHA = '4f25f8d0ae87097f68fc393068d013ba07df2531428a687292ed9903709419c9'
AUTHOR_SHA = '64d6d531a2f03549f0bfbe461092beedd6f70bd2'
RATIOS = ('TR40', 'TR30', 'TR20', 'TR10')
SEEDS = (2, 42, 72)
GRID = [dict(hidden_dim=h, drop_rate=d, full_lr=lr)
        for h in (64, 128) for d in (.1, .25) for lr in (.0005, .001, .003)]
MAX_WARMUP = 100
MAX_FULL = 100
PATIENCE = 20
DEVICE = 'cuda:0'
assert len(GRID) == 12

sys.path.insert(0, '/workspace')
sys.path.insert(0, str(AUTHOR_REPO))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_write_once(path: Path, data: dict) -> None:
    """Never silently replace published selection/results/claim records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def record_or_match(path: Path, data: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != data:
            raise RuntimeError(f'STOP: Existing record differs: {path}')
    else:
        atomic_write_once(path, data)


def prepare_runtime():
    import numpy as np
    import torch
    import sawgad_tfinance_canonical_adapter_v2 as adapter
    from sawgad.anomaly import NUM_ANOMALY_TYPES
    assert sha(ADAPTER) == ADAPTER_SHA, 'Adapter changed'
    assert sha(SOURCE) == SOURCE_SHA, 'T-Finance source SHA mismatch'
    assert sha(SPLIT) == SPLIT_SHA, 'T-Finance frozen split SHA mismatch'
    assert subprocess.check_output(
        ['git', '-C', str(AUTHOR_REPO), 'rev-parse', 'HEAD'], text=True
    ).strip() == AUTHOR_SHA, 'Author code commit mismatch'
    assert torch.cuda.is_available() and torch.version.cuda == '11.8'
    assert NUM_ANOMALY_TYPES == 5
    assert len(GRID) == 12
    with np.load(SPLIT, allow_pickle=False) as z:
        assert int(z['seed'].item()) == 2
        assert (len(z['TR40']), len(z['val']), len(z['test'])) == (15742, 7872, 15743)
    adapter.verify_inputs()
    print('TFINANCE_RUNNER_INTEGRITY=PASS', flush=True)
    return adapter


def prepare_config(item: dict):
    from sawgad.main import _build_parser, _config_from_args
    args = _build_parser().parse_args([
        '--device', DEVICE, '--warmup-epochs', str(MAX_WARMUP),
        '--full-epochs', str(MAX_FULL),
        '--hidden-dim', str(item['hidden_dim']),
        '--drop-rate', str(item['drop_rate']),
        '--full-lr', str(item['full_lr']),
    ])
    return _config_from_args(args)


def make_model(cfg):
    from sawgad.anomaly import NUM_ANOMALY_TYPES
    from sawgad.model import SAWGADModel
    return SAWGADModel(
        in_features=10, hidden_dim=cfg.hidden_dim,
        num_synthetic_heads=NUM_ANOMALY_TYPES,
        drop_rate=cfg.drop_rate,
        encoder_num_layers=cfg.encoder_num_layers,
        encoder_num_heads=cfg.encoder_num_heads,
        encoder_hidden_multiplier=cfg.encoder_hidden_multiplier,
    ).to(DEVICE)


def load_split(adapter, ratio: str):
    start = time.perf_counter()
    split, test_ids, meta = adapter.load_tfinance_split(ratio, DEVICE, checked=True)
    import torch
    assert meta['model_graph_edges_after_author_cleanup'] >= 39357
    assert meta['test_labels_in_model_graph'] is False
    assert torch.all(split.graph.ndata['label'][
        torch.as_tensor(test_ids, dtype=torch.long, device=DEVICE)] == -1).item()
    assert split.test_nodes.numel() == 0
    return split, test_ids, time.perf_counter() - start


def validation_scores(model, split):
    import numpy as np
    import torch
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model.score_real(
            split.graph, split.val_nodes).reshape(-1))
    labels = split.graph.ndata['label'][split.val_nodes]
    yy = labels.cpu().numpy().astype(np.int64)
    pp = probs.cpu().numpy().astype(np.float64)
    assert set(np.unique(yy)) == {0, 1}
    assert np.isfinite(pp).all()
    return yy, pp


def one_combined_update(model, split, cfg):
    """One full-batch training update, using the same loss as the Amazon runs."""
    import torch
    import torch.nn.functional as F
    from sawgad.anomaly import NUM_ANOMALY_TYPES
    from sawgad.train import _prepare_synthetic_epoch, _synthetic_loss
    labels = split.graph.ndata['label']
    train = split.train_nodes
    y = labels[train].float().unsqueeze(-1)
    assert set(torch.unique(y).cpu().tolist()) == {0., 1.}
    positive_weight = (1 - y).sum() / y.sum().clamp_min(1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.full_lr,
                           weight_decay=cfg.weight_decay)
    model.train()
    opt.zero_grad(set_to_none=True)
    synthetic_graph, iters = _prepare_synthetic_epoch(
        split.graph, train,
        (cfg.full_batch_size // 2) // NUM_ANOMALY_TYPES, 1,
        cfg.synthetic_anomalies_per_type,
    )
    scores = model.score_real(split.graph, train).reshape(-1, 1)
    real_loss = F.binary_cross_entropy_with_logits(
        scores, y, pos_weight=positive_weight)
    synth_loss = _synthetic_loss(model, synthetic_graph, iters)
    loss = real_loss + cfg.regularization_weight * synth_loss
    assert torch.isfinite(loss).item(), 'Nonfinite full-batch loss'
    loss.backward()
    weight = model.encoder.input_linear.weight
    assert weight.grad is not None and torch.isfinite(weight.grad).all().item()
    grad_norm = float(weight.grad.norm().item())
    assert grad_norm > 0
    opt.step()
    torch.cuda.synchronize()
    report = {'real_loss': float(real_loss.detach().item()),
              'synthetic_loss': float(synth_loss.detach().item()),
              'combined_loss': float(loss.detach().item()),
              'gradient_norm': grad_norm}
    del synthetic_graph, iters, scores, loss, real_loss, synth_loss
    return report


def capacity_check(adapter):
    import torch
    from sawgad.seed import set_global_seed
    if CAPACITY.exists():
        existing = json.loads(CAPACITY.read_text())
        assert existing['status'] == 'TFINANCE_FULL_BATCH_CAPACITY_PASS'
        assert existing['source_sha256'] == SOURCE_SHA
        assert existing['split_sha256'] == SPLIT_SHA
        assert existing['adapter_sha256'] == ADAPTER_SHA
        print('EXISTING_FULL_BATCH_CAPACITY=PASS', flush=True)
        print(json.dumps(existing, indent=2), flush=True)
        return
    free, total = torch.cuda.mem_get_info()
    if free < 16 * 1024**3:
        raise RuntimeError(f'STOP: Free GPU memory {free/1024**3:.2f} GiB < 16 GiB')
    set_global_seed(2)
    begin = time.perf_counter()
    split, _, preprocessing_sec = load_split(adapter, 'TR40')
    cfg = prepare_config(GRID[-1])  # hidden 128, dropout 0.25, LR 0.003
    model = make_model(cfg)
    torch.cuda.reset_peak_memory_stats()
    print('TFINANCE_FULL_BATCH_UPDATE=START train_nodes=15742 hidden=128', flush=True)
    outcome = one_combined_update(model, split, cfg)
    peak = torch.cuda.max_memory_allocated()
    current_free, _ = torch.cuda.mem_get_info()
    if peak > total * .85 or current_free < 4 * 1024**3:
        raise RuntimeError(f'GPU headroom insufficient: peak={peak/1024**3:.2f} GiB, free={current_free/1024**3:.2f} GiB')
    data = {
        'status': 'TFINANCE_FULL_BATCH_CAPACITY_PASS',
        'dataset': 'T-Finance', 'model': 'SAWGAD',
        'source_sha256': SOURCE_SHA, 'split_sha256': SPLIT_SHA,
        'adapter_sha256': ADAPTER_SHA, 'author_commit': AUTHOR_SHA,
        'config': GRID[-1], 'ratio': 'TR40', 'seed': 2,
        'train_count': 15742, 'test_labels_accessed': False,
        'benchmark_tuning_started': False, 'final_test_evaluated': False,
        'preprocessing_seconds': round(preprocessing_sec, 3),
        'one_update_seconds_including_load': round(time.perf_counter()-begin, 3),
        'peak_gpu_allocated_gib': round(peak/1024**3, 4),
        'gpu_free_after_update_gib': round(current_free/1024**3, 4),
        **outcome,
    }
    atomic_write_once(CAPACITY, data)
    print(json.dumps(data, indent=2), flush=True)
    print('TFINANCE_FULL_BATCH_CAPACITY_PASS', flush=True)
    del model, split
    torch.cuda.empty_cache()


def train_and_freeze(adapter, ratio, seed, item, folder):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, f1_score
    from sawgad.anomaly import NUM_ANOMALY_TYPES
    from sawgad.seed import set_global_seed
    from sawgad.train import _warmup_phase, _prepare_synthetic_epoch, _synthetic_loss
    assert not (folder/'best_val_auprc.pt').exists(), 'Existing checkpoint: STOP'
    assert not (folder/'frozen.json').exists(), 'Existing frozen metadata: STOP'
    assert not (folder/'test_claim.json').exists(), 'Existing test claim: STOP'
    set_global_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    split, test_ids, preprocessing_seconds = load_split(adapter, ratio)
    cfg = prepare_config(item)
    model = make_model(cfg)
    torch.cuda.synchronize()
    a = time.perf_counter()
    print(f'RUN ratio={ratio} seed={seed} cfg={item} WARMUP={MAX_WARMUP}', flush=True)
    _warmup_phase(model, split, cfg, f'{ratio}-{seed} warmup')
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - a

    labels = split.graph.ndata['label']
    train_ids = split.train_nodes
    yy = labels[train_ids].float().unsqueeze(-1)
    assert float(yy.sum()) > 0
    pos_weight = (1 - yy).sum() / yy.sum().clamp_min(1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.full_lr,
                           weight_decay=cfg.weight_decay)
    best_ap = -1.0
    best_epoch = 0
    wait = 0
    best_weights = None
    b = time.perf_counter()
    print(f'FULL_TRAIN ratio={ratio} seed={seed} max_epochs={MAX_FULL} patience={PATIENCE}', flush=True)
    for epoch in range(1, MAX_FULL + 1):
        model.train()
        fake_graph, iters = _prepare_synthetic_epoch(
            split.graph, train_ids,
            (cfg.full_batch_size // 2) // NUM_ANOMALY_TYPES,
            1, cfg.synthetic_anomalies_per_type,
        )
        opt.zero_grad(set_to_none=True)
        logits = model.score_real(split.graph, train_ids).reshape(-1, 1)
        real_loss = F.binary_cross_entropy_with_logits(
            logits, yy, pos_weight=pos_weight)
        fake_loss = _synthetic_loss(model, fake_graph, iters)
        loss = real_loss + cfg.regularization_weight * fake_loss
        assert torch.isfinite(loss).item(), f'Nonfinite loss epoch {epoch}'
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().item())
        del fake_graph, iters, logits, real_loss, fake_loss, loss
        val_y, val_p = validation_scores(model, split)
        ap = float(average_precision_score(val_y, val_p))
        assert np.isfinite(ap)
        if ap > best_ap + 1e-12:
            best_ap, best_epoch, wait = ap, epoch, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0 or wait >= PATIENCE:
            print(f'FULL ratio={ratio} seed={seed} epoch={epoch:03d} loss={loss_value:.5f} '
                  f'valAUPRC={ap:.6f} best={best_ap:.6f}@{best_epoch} wait={wait}/{PATIENCE}', flush=True)
        if wait >= PATIENCE:
            break
    assert best_weights is not None
    torch.cuda.synchronize()
    full_seconds = time.perf_counter() - b
    model.load_state_dict(best_weights)
    val_y, val_p = validation_scores(model, split)
    assert abs(float(average_precision_score(val_y, val_p)) - best_ap) < 1e-8
    grid = np.linspace(.05, .95, 19)
    scores = [f1_score(val_y, val_p > t, average='macro', zero_division=0) for t in grid]
    threshold = float(grid[int(np.argmax(scores))])
    frozen = {
        'status': 'FROZEN_READY_FOR_TEST', 'model': 'SAWGAD', 'dataset': 'T-Finance',
        'ratio': ratio, 'seed': seed, 'config': item,
        'best_epoch': best_epoch, 'epochs_ran': epoch,
        'best_val_auprc': best_ap, 'threshold': threshold,
        'val_macro_f1': float(max(scores)),
        'preprocess_seconds': preprocessing_seconds,
        'warmup_seconds': warmup_seconds,
        'full_training_seconds': full_seconds,
        'peak_allocated_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'source_dgl_sha256': SOURCE_SHA, 'split_sha256': SPLIT_SHA,
        'adapter_sha256': ADAPTER_SHA, 'author_commit': AUTHOR_SHA,
        'test_used_for_selection': False, 'test_scoring_performed': False,
        'protocol': {'warmup_epochs': MAX_WARMUP, 'full_max_epochs': MAX_FULL,
                     'full_patience': PATIENCE, 'full_labels': 'frozen_train_ratio',
                     'checkpoint_selection': 'validation_AUPRC',
                     'threshold_selection': 'validation_macro_F1'},
    }
    # Atomic checkpoint write; write no final/test marker during tuning.
    temp = folder/'best_val_auprc.pt.partial'
    if temp.exists():
        raise RuntimeError(f'STOP: Leftover incomplete checkpoint: {temp}')
    torch.save({'model_state_dict': {k: v.cpu() for k, v in best_weights.items()},
                'frozen': frozen}, temp)
    os.replace(temp, folder/'best_val_auprc.pt')
    frozen['checkpoint_sha256'] = sha(folder/'best_val_auprc.pt')
    atomic_write_once(folder/'frozen.json', frozen)
    return model, split, test_ids, frozen


def check_old(path, ratio, seed, config, status):
    data = json.loads(path.read_text())
    assert data['status'] == status and data['dataset'] == 'T-Finance'
    assert data['ratio'] == ratio and data['seed'] == seed and data['config'] == config
    assert data['source_dgl_sha256'] == SOURCE_SHA
    assert data['split_sha256'] == SPLIT_SHA
    assert data['adapter_sha256'] == ADAPTER_SHA
    assert data['author_commit'] == AUTHOR_SHA
    return data


def existing_frozen(folder, ratio, seed, item):
    frozen = check_old(folder/'frozen.json', ratio, seed, item, 'FROZEN_READY_FOR_TEST')
    assert sha(folder/'best_val_auprc.pt') == frozen['checkpoint_sha256']
    return frozen


def tune(adapter, max_trials: int):
    import torch
    from sawgad.seed import set_global_seed
    if not CAPACITY.exists():
        raise RuntimeError('STOP: Run --phase capacity first')
    check_capacity = json.loads(CAPACITY.read_text())
    assert check_capacity['status'] == 'TFINANCE_FULL_BATCH_CAPACITY_PASS'
    assert check_capacity['source_sha256'] == SOURCE_SHA
    assert check_capacity['split_sha256'] == SPLIT_SHA
    assert check_capacity['adapter_sha256'] == ADAPTER_SHA
    assert 1 <= max_trials <= 12
    for i in range(1, max_trials + 1):
        item = GRID[i-1]
        folder = OUT/'tuning'/f'trial_{i:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'test_claim.json').exists():
            raise RuntimeError('STOP: Unexpected test claim under tuning')
        t = folder/'tuning.json'
        if t.exists():
            rec = check_old(t, 'TR40', 2, item, 'TUNING_COMPLETE')
            existing_frozen(folder, 'TR40', 2, item)
            print(f'TUNING_RESUME trial={i:02d} valAUPRC={rec["best_val_auprc"]:.6f}', flush=True)
            continue
        if (folder/'frozen.json').exists():
            frozen = existing_frozen(folder, 'TR40', 2, item)
            print(f'TUNING_RECOVER_FROZEN trial={i:02d}', flush=True)
        else:
            if (folder/'best_val_auprc.pt').exists():
                raise RuntimeError('STOP: Checkpoint without frozen metadata')
            model, split, _, frozen = train_and_freeze(adapter, 'TR40', 2, item, folder)
            del model, split
            torch.cuda.empty_cache()
        rec = {**frozen, 'status': 'TUNING_COMPLETE'}
        atomic_write_once(t, rec)
        print(f'TUNING_FINISHED {i}/12 valAUPRC={rec["best_val_auprc"]:.6f}', flush=True)
    if max_trials < 12:
        print(f'TUNING_PARTIAL_COMPLETE={max_trials}/12; TEST_EVALUATED=NO', flush=True)
        return
    complete = []
    for i, item in enumerate(GRID, start=1):
        folder = OUT/'tuning'/f'trial_{i:02d}'
        rec = check_old(folder/'tuning.json', 'TR40', 2, item, 'TUNING_COMPLETE')
        existing_frozen(folder, 'TR40', 2, item)
        complete.append(rec)
    winner = min(enumerate(complete, 1), key=lambda x: (-x[1]['best_val_auprc'], x[0]))
    frozen_winner = {
        'status': 'WINNER_FROZEN', 'source_dgl_sha256': SOURCE_SHA, 'split_sha256': SPLIT_SHA,
        'adapter_sha256': ADAPTER_SHA, 'winner_trial': winner[0],
        'config': winner[1]['config'], 'best_val_auprc': winner[1]['best_val_auprc'],
        'selection': 'TR40 seed2 maximum validation AUPRC only',
        'test_scoring_performed': False,
    }
    record_or_match(OUT/'winner.json', frozen_winner)
    print('TFINANCE_TR40_TUNING_COMPLETE=12/12', flush=True)
    print('WINNER_FROZEN', json.dumps(frozen_winner), flush=True)


def evaluate_test_once(model, split, test_ids, frozen, folder):
    import numpy as np
    import torch
    from dgl.data.utils import load_graphs
    from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                                 precision_score, recall_score)
    claim = folder/'test_claim.json'
    # Durable marker created before either test scores or test labels can be accessed.
    atomic_write_once(claim, {
        'status': 'CLAIMED_TEST_EVALUATION',
        'checkpoint_sha256': frozen['checkpoint_sha256'],
        'ratio': frozen['ratio'], 'seed': frozen['seed'],
        'selection_frozen': True,
    })
    print('FROZEN_CHECKPOINT_AND_THRESHOLD=PASS; TEST_ONCE', flush=True)
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        ids = torch.as_tensor(test_ids, dtype=torch.long, device=DEVICE)
        probs = torch.sigmoid(model.score_real(split.graph, ids).reshape(-1)).cpu().numpy()
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    assert np.isfinite(probs).all()
    raw_test = load_graphs(str(SOURCE))[0][0].ndata['label']
    # Canonical BWGNN labels are two-column one-hot, never a flat vector.
    assert tuple(raw_test.shape) == (39357, 2), 'STOP: Canonical test source label shape changed'
    assert bool(((raw_test == 0) | (raw_test == 1)).all())
    assert bool((raw_test.sum(dim=1) == 1).all())
    test_y = raw_test.argmax(dim=1).numpy().astype(np.int64)[test_ids]
    assert test_y.ndim == 1 and len(test_y) == len(test_ids)

    assert len(test_y) == 15743 and set(np.unique(test_y)) == {0, 1}
    threshold = frozen['threshold']
    predicted = (probs > threshold).astype(np.int64)
    fraud = test_y == 1
    benign = test_y == 0
    recall = float(recall_score(test_y, predicted, zero_division=0))
    metrics = {
        'auprc': float(average_precision_score(test_y, probs)),
        'auroc': float(roc_auc_score(test_y, probs)),
        'macro_f1': float(f1_score(test_y, predicted, average='macro', zero_division=0)),
        'fraud_precision': float(precision_score(test_y, predicted, zero_division=0)),
        'fraud_recall': recall,
        'fraud_f1': float(f1_score(test_y, predicted, zero_division=0)),
        'gmean': float(np.sqrt(recall * np.mean(predicted[benign] == 0))),
        'threshold': float(threshold), 'test_count': int(len(test_y)),
        'fraud_count': int(fraud.sum()),
    }
    final = {**frozen, 'status': 'FINAL_COMPLETE',
             'test_scoring_performed': True, 'test_evaluated_once': True,
             'test_inference_seconds': inference_seconds, 'test_metrics': metrics}
    atomic_write_once(folder/'final.json', final)
    print('FINAL_COMPLETE', frozen['ratio'], frozen['seed'],
          'testAUPRC', f'{metrics["auprc"]:.6f}', flush=True)
    return final


def finals(adapter):
    import torch
    from sawgad.seed import set_global_seed
    p = OUT/'winner.json'
    if not p.exists():
        raise RuntimeError('STOP: Tuning winner not frozen')
    winner = json.loads(p.read_text())
    assert winner['status'] == 'WINNER_FROZEN'
    assert winner['source_dgl_sha256'] == SOURCE_SHA
    assert winner['split_sha256'] == SPLIT_SHA
    assert winner['adapter_sha256'] == ADAPTER_SHA
    assert winner['test_scoring_performed'] is False
    all_tuning = []
    for i, item in enumerate(GRID, start=1):
        folder = OUT/'tuning'/f'trial_{i:02d}'
        all_tuning.append(check_old(folder/'tuning.json', 'TR40', 2, item, 'TUNING_COMPLETE'))
        existing_frozen(folder, 'TR40', 2, item)
    expected = min(enumerate(all_tuning, 1), key=lambda x: (-x[1]['best_val_auprc'], x[0]))
    assert winner['winner_trial'] == expected[0] and winner['config'] == expected[1]['config']
    results = []
    for ratio in RATIOS:
        for seed in SEEDS:
            folder = OUT/'final'/ratio/f'seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            final_record = folder/'final.json'
            if final_record.exists():
                rec = check_old(final_record, ratio, seed, winner['config'], 'FINAL_COMPLETE')
                frozen = existing_frozen(folder, ratio, seed, winner['config'])
                claim = json.loads((folder/'test_claim.json').read_text())
                assert claim['checkpoint_sha256'] == frozen['checkpoint_sha256']
                assert rec['test_evaluated_once'] is True
                print(f'FINAL_RESUME ratio={ratio} seed={seed} testAUPRC={rec["test_metrics"]["auprc"]:.6f}', flush=True)
                results.append(rec)
                continue
            if (folder/'test_claim.json').exists():
                raise RuntimeError(f'STOP: Test claimed but no final result: {folder}; no automatic retest')
            if (folder/'frozen.json').exists():
                frozen = existing_frozen(folder, ratio, seed, winner['config'])
                set_global_seed(seed)
                split, test_ids, _ = load_split(adapter, ratio)
                model = make_model(prepare_config(winner['config']))
                checkpoint = torch.load(folder/'best_val_auprc.pt', map_location='cpu', weights_only=True)
                assert checkpoint['frozen']['best_val_auprc'] == frozen['best_val_auprc']
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f'FINAL_RECOVER_FROZEN ratio={ratio} seed={seed}', flush=True)
            else:
                if (folder/'best_val_auprc.pt').exists():
                    raise RuntimeError('STOP: Checkpoint without frozen metadata')
                model, split, test_ids, frozen = train_and_freeze(adapter, ratio, seed,
                                                                 winner['config'], folder)
            done = evaluate_test_once(model, split, test_ids, frozen, folder)
            results.append(done)
            del model, split
            torch.cuda.empty_cache()
    summary = []
    for ratio in RATIOS:
        subset = [r for r in results if r['ratio'] == ratio]
        assert len(subset) == 3
        metrics = {}
        for key in ('auprc', 'auroc', 'macro_f1', 'fraud_precision',
                    'fraud_recall', 'fraud_f1', 'gmean'):
            vals = [float(r['test_metrics'][key]) for r in subset]
            metrics[key] = {'mean': statistics.mean(vals),
                            'population_sd': statistics.pstdev(vals)}
        summary.append({'ratio': ratio, 'runs': len(subset), 'metrics': metrics})
    report = {
        'status': 'SAWGAD_TFINANCE_COMPLETE',
        'model': 'SAWGAD', 'dataset': 'T-Finance',
        'source_dgl_sha256': SOURCE_SHA, 'split_sha256': SPLIT_SHA,
        'author_commit': AUTHOR_SHA, 'adapter_sha256': ADAPTER_SHA,
        'tuning_trials': 12, 'final_runs': len(results),
        'winner': winner, 'ratio_summaries': summary,
        'train_seeds': list(SEEDS), 'training_ratios': list(RATIOS),
        'test_evaluated_once_per_final_run': True,
        'protocol': {'warmup_epochs': MAX_WARMUP, 'full_max_epochs': MAX_FULL,
                     'full_patience': PATIENCE, 'full_training_labels': 'frozen ratio IDs',
                     'checkpoint_selection': 'validation_AUPRC',
                     'threshold_selection': 'validation_macro_F1'},
    }
    record_or_match(OUT/'complete_summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    print('SAWGAD_TFINANCE_COMPLETE', flush=True)


def show_status():
    print(f'OUTPUT_DIR={OUT}')
    for i in range(1, 13):
        path = OUT/'tuning'/f'trial_{i:02d}'/'tuning.json'
        if path.exists():
            print(f'TUNING_{i:02d}=COMPLETE')
    winner = OUT/'winner.json'
    if winner.exists():
        print('WINNER=', winner.read_text())
    for ratio in RATIOS:
        for seed in SEEDS:
            path = OUT/'final'/ratio/f'seed_{seed}'/'final.json'
            if path.exists():
                print(f'FINAL_{ratio}_seed{seed}=COMPLETE')
    if (OUT/'complete_summary.json').exists():
        print('SAWGAD_TFINANCE_COMPLETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('status', 'capacity', 'tune', 'finals'), default='status')
    parser.add_argument('--max-trials', type=int, choices=range(1, 13), default=12)
    args = parser.parse_args()
    if args.phase == 'status':
        show_status()
        return
    # Non-blocking exclusive lock: reject a duplicate concurrent training process.
    lock_path = ROOT/'logs/sawgad/tfinance_canonical_v2.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('STOP: Another T-Finance runner is active') from exc
        adapter = prepare_runtime()
        if args.phase == 'capacity':
            capacity_check(adapter)
        elif args.phase == 'tune':
            tune(adapter, args.max_trials)
        elif args.phase == 'finals':
            finals(adapter)


if __name__ == '__main__':
    main()
