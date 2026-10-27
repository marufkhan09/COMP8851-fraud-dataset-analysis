#!/usr/bin/env python3
"""I3 YelpChi RGTAN finals, validation-frozen with strictly once test claims."""
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix)
import rgtan_yelpchi_core_v1 as core
import rgtan_yelpchi_tuning_batch1024_v1 as tuner

ROOT = Path('/workspace/comp8851')
OUT = ROOT / 'results/rgtan/yelpchi_finals_batch1024_v1'
WINNER_PATH = ROOT / 'results/rgtan/yelpchi_tr40_tuning_batch1024_v1/winner.json'
RATIOS = ('TR40', 'TR30', 'TR20', 'TR10')
SEEDS = (2, 42, 72)
NAMES = ['degree', 'riskstat', '1hop_degree', '2hop_degree', '1hop_riskstat', '2hop_riskstat']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial.' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial.' + str(os.getpid()))
    assert not tmp.exists()
    torch.save(obj, tmp)
    os.replace(tmp, path)


def validate_preflight():
    assert tuner.BATCH == 1024
    assert tuner.MAX_EPOCHS == 100 and tuner.PATIENCE == 20
    assert tuner.DEVICE == 'cuda:0'
    assert torch.cuda.is_available()
    core.verify_input_files()
    core.assert_environment()
    parts = core.load_splits()
    w = json.loads(WINNER_PATH.read_text())
    assert w['status'] == 'WINNER_FROZEN' and w['test_scoring_performed'] is False
    assert w['winner_trial'] == 4
    assert w['config'] == {'hidden_dim': 32, 'drop': [0.3, 0.15], 'base_lr': 0.001}
    assert w['source_sha256'] == core.SOURCE_SHA and w['split_sha256'] == core.SPLIT_SHA
    tuning_dir = WINNER_PATH.parent / 'tuning'
    scores = []
    for trial in range(1, 13):
        f = tuning_dir / f'trial_{trial:02d}' / 'tuning.json'
        j = json.loads(f.read_text())
        assert j['status'] == 'TUNING_COMPLETE' and j['trial'] == trial
        assert j['batch_size'] == 1024 and j['test_scoring_performed'] is False
        ckpt = f.parent / 'best_val_auprc.pt'
        assert ckpt.is_file() and digest(ckpt) == j['checkpoint_sha256']
        scores.append(j)
    chosen = max(scores, key=lambda x: (x['best_val_auprc'], -x['trial']))
    assert chosen['trial'] == w['winner_trial']
    assert chosen['config'] == w['config']
    assert abs(chosen['best_val_auprc'] - w['best_val_auprc']) < 1e-10
    manifests = {}
    for ratio in RATIOS:
        asset = core.ASSET_DIR / f'yelpchi_{ratio.lower()}_crossfit.npz'
        mp = core.ASSET_DIR / f'yelpchi_{ratio.lower()}_crossfit_manifest.json'
        m = json.loads(mp.read_text())
        assert asset.is_file() and digest(asset) == m['artifact_sha256']
        assert m['status'] == 'RGTAN_YELPCHI_CROSSFIT_PASS' and m['ratio'] == ratio
        assert m['source_sha256'] == core.SOURCE_SHA and m['split_sha256'] == core.SPLIT_SHA
        assert m['fold_labels_masked_from_their_own_risk_views'] is True
        assert m['validation_test_labels_used_in_features'] is False
        assert m['normalization_fit_on_oof_training_only'] is True
        assert m['train_count'] == len(parts[ratio])
        manifests[ratio] = m
    assert manifests['TR40']['artifact_sha256'] == w['adapter_sha256']
    print('I3_FINALS_PREFLIGHT_12_TUNING_AND_4_ADAPTER_HASHES=PASS', flush=True)
    print('WINNER=' + json.dumps(w, sort_keys=True), flush=True)
    print('HELD_OUT_TEST_SCORED=NO', flush=True)
    return w, parts, manifests


def load_data(ratio, parts, manifest):
    asset = core.ASSET_DIR / f'yelpchi_{ratio.lower()}_crossfit.npz'
    assert digest(asset) == manifest['artifact_sha256']
    train, val, test = parts[ratio], parts['val'], parts['test']
    with np.load(asset, allow_pickle=False) as z:
        features = z['features'].astype(np.float32)
        train_ids = z['train_ids'].astype(np.int64)
        train_labels = z['train_labels'].astype(np.int64)
        val_ids = z['val_ids'].astype(np.int64)
        fold = z['fold_of_node'].astype(np.int64)
        views = [z[f'risk_view_{k}'].astype(np.float32) for k in range(6)]
        contexts = [z[f'label_context_{k}'].astype(np.int64) for k in range(6)]
        names = list(map(str, z['risk_feature_names']))
    assert np.array_equal(train_ids, train) and np.array_equal(val_ids, val)
    assert features.shape == (core.N, core.NF) and np.isfinite(features).all()
    assert names == NAMES
    assert train_labels.shape == train.shape and int(train_labels.sum()) == core.EXPECTED_POS[ratio]
    assert np.array_equal(np.unique(fold[train]), np.arange(5))
    outside = np.setdiff1d(np.arange(core.N), train)
    assert np.all(fold[outside] == -1)
    for k in range(6):
        assert views[k].shape == (core.N, 6) and np.isfinite(views[k]).all()
        assert contexts[k].shape == (core.N,)
        assert np.all(contexts[k][outside] == 2)
        assert np.all(contexts[k][val] == 2) and np.all(contexts[k][test] == 2)
        if k < 5:
            held = train[fold[train] == k]
            assert np.all(contexts[k][held] == 2)
            use = train[fold[train] != k]
            assert np.array_equal(contexts[k][use], train_labels[fold[train] != k])
        else:
            assert np.array_equal(contexts[k][train], train_labels)
    adjacency, graph_features = core.get_graph_and_features()
    assert np.array_equal(graph_features, features)
    del graph_features
    graph = dgl.from_scipy(adjacency)
    del adjacency
    assert graph.num_nodes() == core.N and graph.num_edges() == core.EXPECTED_EDGES
    assert len(graph.ndata) == 0
    # Training only ever sees training labels and validation labels. Test labels
    # are not retrieved until after the immutable test claim is written.
    all_labels = np.asarray(loadmat(core.MAT, variable_names=['label'])['label']).reshape(-1)
    assert np.array_equal(all_labels[train], train_labels)
    val_y = np.asarray(all_labels[val], dtype=np.int64).copy()
    del all_labels
    assert set(np.unique(val_y).tolist()) == {0, 1}
    print(f'I3_{ratio}_SOURCE_SPLIT_OOF_LEAKAGE_AND_GRAPH=PASS', flush=True)
    return (graph, features, train, train_labels, val, val_y, fold, views, contexts, names)


def predict_scores(exp, ids):
    scores = np.full(core.N, np.nan, dtype=np.float32)
    seen = []
    exp.model.eval()
    with torch.no_grad():
        for batch in exp.loader(ids, shuffle=False):
            node_ids, logits = exp.predict(batch, 5)
            assert np.isnan(scores[node_ids]).all(), 'repeated prediction nodes'
            scores[node_ids] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            seen.extend(node_ids.tolist())
    assert len(seen) == len(ids) and set(seen) == set(ids.tolist())
    p = scores[ids]
    assert p.shape == (len(ids),) and np.isfinite(p).all()
    assert np.all((0 <= p) & (p <= 1))
    return p


def validation_threshold(y, scores):
    """Exact validation macro-F1 optimum over unique observed probabilities."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-p, kind='stable')
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.arange(1, len(y) + 1) - tp
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    fn, tn = pos - tp, neg - fp
    denom_p = 2*tp + fp + fn
    denom_n = 2*tn + fp + fn
    positive_f1 = np.divide(2*tp, denom_p, out=np.zeros(len(y)), where=denom_p > 0)
    negative_f1 = np.divide(2*tn, denom_n, out=np.zeros(len(y)), where=denom_n > 0)
    vals = (positive_f1 + negative_f1)/2
    last = np.r_[ps[1:] != ps[:-1], True]
    candidates = np.flatnonzero(last)
    allnegative_f1 = (2*neg/(2*neg + pos))/2
    best_idx = candidates[np.argmax(vals[candidates])]
    if allnegative_f1 > vals[best_idx] + 1e-12:
        return 1.0, float(allnegative_f1)
    threshold = float(np.asarray(scores, dtype=np.float32)[order[best_idx]])
    actual = float(f1_score(y, np.asarray(scores) >= threshold, average='macro', zero_division=0))
    assert abs(actual - vals[best_idx]) < 1e-9, (actual, vals[best_idx])
    return threshold, actual


def calculate_metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return dict(auprc=float(average_precision_score(y,p)),
                auroc=float(roc_auc_score(y,p)),
                macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
                fraud_f1=float(f1_score(y,pred,pos_label=1,zero_division=0)),
                accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)),
                mcc=float(matthews_corrcoef(y,pred)),
                brier=float(brier_score_loss(y,p)),
                threshold=float(threshold),tn=tn,fp=fp,fn=fn,tp=tp)


def run_one(ratio, seed, winner, parts, manifests):
    folder = OUT / ratio / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / 'final.json'
    claim = folder / 'test_evaluation_started.json'
    predfile = folder / 'test_predictions_once.npz'
    bestfile = folder / 'best_val_auprc.pt'
    latest = folder / 'latest.pt'
    if final.exists():
        f = json.loads(final.read_text())
        assert f['status'] == 'FINAL_COMPLETE' and f['ratio'] == ratio and f['seed'] == seed
        assert f['winner_trial'] == 4 and f['batch_size'] == 1024
        assert f['test_scoring_count'] == 1
        assert claim.is_file() and predfile.is_file() and bestfile.is_file()
        assert digest(predfile) == f['test_predictions_sha256']
        assert digest(bestfile) == f['checkpoint_sha256']
        print(f'FINAL_ALREADY_COMPLETE={ratio}/seed_{seed} test_not_repeated', flush=True)
        return f
    if claim.exists() or predfile.exists():
        raise RuntimeError(f'MANUAL_AUDIT_REQUIRED_TEST_CLAIM_ALREADY_EXISTS:{folder}')
    print(f'FINAL_START={ratio}/seed_{seed}', flush=True)
    data = load_data(ratio, parts, manifests[ratio])
    tuner.SEED = seed
    exp = tuner.Experiment(data, winner['config'])
    assert tuner.BATCH == 1024
    best_ap, best_auc, best_ep, wait = -1., -1., 0, 0
    epoch_begin = 1
    history = []
    best_path = None
    best_hash = None
    if latest.exists():
        state = torch.load(latest, map_location='cpu')
        assert state['ratio'] == ratio and state['seed'] == seed
        assert state['config'] == winner['config']
        assert state['source_sha256'] == core.SOURCE_SHA
        assert state['split_sha256'] == core.SPLIT_SHA
        assert state['adapter_sha256'] == manifests[ratio]['artifact_sha256']
        best_path = folder / state['best_checkpoint_file']
        assert best_path.is_file() and digest(best_path) == state['best_checkpoint_sha256']
        exp.model.load_state_dict(state['model'])
        exp.optimizer.load_state_dict(state['optimizer'])
        best_ap, best_auc, best_ep, wait = (state['best_val_auprc'],
                                           state['best_val_auroc'],
                                           state['best_epoch'], state['patience_wait'])
        history = state['history']
        epoch_begin = state['epoch'] + 1
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device=tuner.DEVICE)
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        print(f'FINAL_RESUME={ratio}/seed_{seed} epoch={epoch_begin}', flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if wait < tuner.PATIENCE:
        for epoch in range(epoch_begin, tuner.MAX_EPOCHS + 1):
            t0 = time.perf_counter()
            loss, steps = exp.epoch()
            ap, auc = exp.validate()
            if ap > best_ap + 1e-12:
                best_ap, best_auc, best_ep, wait = float(ap), float(auc), epoch, 0
                best_path = folder / f'best_epoch_{epoch:03d}.pt'
                atomic_torch(best_path, dict(
                    model_state_dict={k:v.detach().cpu().clone()
                                      for k,v in exp.model.state_dict().items()},
                    ratio=ratio,seed=seed,config=winner['config'],epoch=epoch,
                    validation_auprc=best_ap, validation_auroc=best_auc,
                    adapter_sha256=manifests[ratio]['artifact_sha256']))
                best_hash = digest(best_path)
            else:
                wait += 1
            assert best_path is not None
            history.append(dict(epoch=epoch, train_loss=float(loss),
                                val_auprc=float(ap), val_auroc=float(auc),
                                optimizer_steps=int(steps),
                                seconds=round(time.perf_counter()-t0, 3)))
            state = dict(ratio=ratio, seed=seed,config=winner['config'],epoch=epoch,
                         model=exp.model.state_dict(),optimizer=exp.optimizer.state_dict(),
                         best_val_auprc=best_ap,best_val_auroc=best_auc,
                         best_epoch=best_ep,patience_wait=wait,history=history,
                         best_checkpoint_file=best_path.name,
                         best_checkpoint_sha256=best_hash or digest(best_path),
                         source_sha256=core.SOURCE_SHA,split_sha256=core.SPLIT_SHA,
                         adapter_sha256=manifests[ratio]['artifact_sha256'],
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(tuner.DEVICE),
                         numpy_rng=np.random.get_state(),python_rng=random.getstate())
            atomic_torch(latest,state)
            print(f'FINAL={ratio}/seed_{seed} EPOCH={epoch:03d}/100 '
                  f'valAUPRC={ap:.6f} best={best_ap:.6f}@{best_ep} '
                  f'wait={wait}/20 seconds={history[-1]["seconds"]:.1f}',flush=True)
            if wait >= tuner.PATIENCE:
                print(f'FINAL_EARLY_STOP={ratio}/seed_{seed}@{epoch}',flush=True)
                break
    assert best_path is not None and best_path.is_file()
    if not bestfile.exists():
        tmp = bestfile.with_name('best_val_auprc.pt.partial.' + str(os.getpid()))
        shutil.copy2(best_path,tmp)
        os.replace(tmp,bestfile)
    ckhash = digest(bestfile)
    assert ckhash == digest(best_path)
    state = torch.load(bestfile, map_location='cpu')
    assert state['ratio'] == ratio and state['seed'] == seed and state['epoch'] == best_ep
    exp.model.load_state_dict(state['model_state_dict'])
    val_p = predict_scores(exp,parts['val'])
    val_y = data[5]
    val_ap = float(average_precision_score(val_y,val_p))
    assert abs(val_ap - best_ap) < 1e-5, (val_ap,best_ap)
    threshold, macro_f1 = validation_threshold(val_y,val_p)
    frozen = dict(status='VALIDATION_FROZEN',ratio=ratio,seed=seed,
                  winner_trial=4,winner_config=winner['config'],
                  batch_size=1024, best_epoch=best_ep,best_val_auprc=best_ap,
                  val_auprc_rescored=val_ap,best_val_auroc=best_auc,
                  val_macro_f1=macro_f1,threshold=threshold,
                  checkpoint_sha256=ckhash,
                  adapter_sha256=manifests[ratio]['artifact_sha256'],
                  source_sha256=core.SOURCE_SHA,split_sha256=core.SPLIT_SHA,
                  test_scoring_performed=False)
    atomic_json(folder / 'history.json', history)
    atomic_json(folder / 'validation_frozen.json', frozen)
    print(f'VALIDATION_FROZEN={ratio}/seed_{seed} best={best_ap:.6f} '
          f'threshold={threshold:.8f} macroF1={macro_f1:.6f}',flush=True)
    # Before this O_EXCL claim, no test labels or test probabilities are accessed.
    claim_data = dict(ratio=ratio,seed=seed,best_checkpoint_sha256=ckhash,
                      validation_frozen_sha256=digest(folder/'validation_frozen.json'),
                      test_scoring_started_once=True, timestamp_utc=int(time.time()))
    fd = os.open(claim,os.O_CREAT | os.O_EXCL | os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(claim_data,f,sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(f'TEST_ONCE_CLAIMED={ratio}/seed_{seed}',flush=True)
    test = parts['test']
    test_p = predict_scores(exp,test)
    test_y = np.asarray(loadmat(core.MAT,variable_names=['label'])['label']).reshape(-1)[test].astype(np.int64).copy()
    assert set(np.unique(test_y).tolist()) == {0,1}
    test_metrics = calculate_metrics(test_y,test_p,threshold)
    tmp = predfile.with_name(predfile.name+'.partial.'+str(os.getpid()))
    with tmp.open('xb') as f:
        np.savez_compressed(f,node_ids=test,labels=test_y,scores=test_p,
                            threshold=np.asarray([threshold],dtype=np.float64))
    os.replace(tmp,predfile)
    result = dict(status='FINAL_COMPLETE',model='RGTAN',dataset='YelpChi',
                  ratio=ratio,seed=seed,winner_trial=4,config=winner['config'],
                  batch_size=1024, effective_lr=float(exp.learning_rate),
                  best_epoch=best_ep,completed_epochs=len(history),
                  max_epochs=tuner.MAX_EPOCHS,patience=tuner.PATIENCE,
                  validation=frozen,test=test_metrics,test_count=len(test),
                  train_count=len(parts[ratio]),val_count=len(parts['val']),
                  source_sha256=core.SOURCE_SHA,split_sha256=core.SPLIT_SHA,
                  adapter_sha256=manifests[ratio]['artifact_sha256'],
                  winner_sha256=digest(WINNER_PATH),checkpoint_sha256=ckhash,
                  test_predictions_sha256=digest(predfile),
                  test_scoring_count=1,test_used_for_model_selection=False,
                  wall_seconds=round(time.perf_counter()-started,3),
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    atomic_json(final,result)
    latest.unlink(missing_ok=True)
    print(f'FINAL_COMPLETE={ratio}/seed_{seed} '
          f'testAUPRC={test_metrics["auprc"]:.6f} '
          f'testAUROC={test_metrics["auroc"]:.6f} '
          f'testMacroF1={test_metrics["macro_f1"]:.6f}',flush=True)
    del exp,data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ('preflight','run')
    winner,parts,manifests = validate_preflight()
    if sys.argv[1] == 'preflight':
        print('I3_FINALS_PREFLIGHT_PASS=YES',flush=True)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    results = []
    for ratio in RATIOS:
        for seed in SEEDS:
            results.append(run_one(ratio,seed,winner,parts,manifests))
    assert len(results)==12
    summary = dict(status='TWELVE_FINALS_COMPLETE',dataset='YelpChi',model='RGTAN',
                   batch_size=1024,winner_trial=4,
                   test_scored_once_per_run=True, ratios=list(RATIOS),seeds=list(SEEDS),
                   source_sha256=core.SOURCE_SHA,split_sha256=core.SPLIT_SHA,
                   winner_sha256=digest(WINNER_PATH),by_ratio={})
    for ratio in RATIOS:
        rows=[r for r in results if r['ratio']==ratio]
        summary['by_ratio'][ratio]=dict(n=3,
            test_auprc_mean=float(np.mean([r['test']['auprc'] for r in rows])),
            test_auprc_sd_sample=float(np.std([r['test']['auprc'] for r in rows],ddof=1)),
            test_auroc_mean=float(np.mean([r['test']['auroc'] for r in rows])),
            macro_f1_mean=float(np.mean([r['test']['macro_f1'] for r in rows])))
    atomic_json(OUT/'complete_summary.json',summary)
    print('I3_YELPCHI_TWELVE_FINALS_COMPLETE=YES',flush=True)
    print('I3_HELD_OUT_TEST_SCORING_ONCE_PER_RUN=PASS',flush=True)
    print('I3_FINAL_SUMMARY='+str(OUT/'complete_summary.json'),flush=True)


if __name__ == '__main__':
    main()
