#!/usr/bin/env python3
"""Amazon full SAWGAD benchmark candidate, fixed splits; test once per final run."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)

from sawgad.anomaly import NUM_ANOMALY_TYPES
from sawgad.data import DataSplit, load_amazon_graph
from sawgad.main import _build_parser, _config_from_args
from sawgad.model import SAWGADModel
from sawgad.seed import set_global_seed
from sawgad.train import (_warmup_phase, _prepare_synthetic_epoch,
                          _synthetic_loss)

ROOT = Path('/workspace/comp8851')
SOURCE = ROOT / 'repos/SAWGAD'
NPZ = Path('/workspace/uploads/ass2_split_candidates/Amazon/amazon_seed2_nested_splits.npz')
MAT = Path('/root/.dgl/amazon_1e446145/Amazon.mat')
OUT = ROOT / 'results/sawgad/amazon_benchmark_candidate_v1'
EXPECTED_COMMIT = '64d6d531a2f03549f0bfbe461092beedd6f70bd2'
EXPECTED_NPZ = '0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd'
EXPECTED_MAT = '4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63'
RATIOS = ('TR40', 'TR30', 'TR20', 'TR10')
SEEDS = (2, 42, 72)
MAX_WARMUP = 100
MAX_FULL = 100
PATIENCE = 20
DEVICE = 'cuda:0'
GRID = [dict(hidden_dim=h, drop_rate=d, full_lr=lr)
        for h in (64, 128) for d in (.1, .25) for lr in (.0005, .001, .003)]
assert len(GRID) == 12


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def json_write_once(path, object_):
    with path.open('x') as f:
        json.dump(object_, f, indent=2)
        f.write('\n')


def atomic_json(path, object_):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(object_, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)


def ensure_sources():
    import subprocess
    assert sha(NPZ) == EXPECTED_NPZ
    assert sha(MAT) == EXPECTED_MAT
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'], text=True).strip() == EXPECTED_COMMIT
    assert torch.cuda.is_available() and torch.version.cuda == '11.8'
    with np.load(NPZ, allow_pickle=False) as z:
        assert int(z['seed'].item()) == 2
        assert int(z['source_nodes'].item()) == 11944
        assert (len(z['TR40']),len(z['val']),len(z['test'])) == (3455,1728,3456)
        for r in RATIOS:
            assert not np.intersect1d(z[r],z['test']).size
            assert not np.intersect1d(z[r],z['val']).size
    print('SOURCES_AND_ISOLATION=PASS',flush=True)


def load_split(ratio):
    begin = time.perf_counter()
    graph = load_amazon_graph(device=DEVICE)
    assert graph.num_nodes() == 11944 and graph.num_edges() == 8847096
    assert graph.ndata['feature'].shape == (11944,25)
    with np.load(NPZ, allow_pickle=False) as z:
        train = torch.as_tensor(z[ratio].copy(),dtype=torch.long,device=DEVICE)
        val = torch.as_tensor(z['val'].copy(),dtype=torch.long,device=DEVICE)
        test = z['test'].astype(np.int64).copy()
    original = graph.ndata['label']
    masked = torch.full_like(original,-1)
    masked[train] = original[train]
    masked[val] = original[val]
    graph.ndata['label'] = masked
    del original
    assert torch.all(masked[torch.as_tensor(test,dtype=torch.long,device=DEVICE)] == -1)
    assert torch.all(masked[:3305] == -1)
    assert set(masked[train].cpu().tolist()) == {0,1}
    assert set(masked[val].cpu().tolist()) == {0,1}
    return DataSplit(graph=graph,train_nodes=train,val_nodes=val,
                     test_nodes=torch.empty(0,dtype=torch.long,device=DEVICE)),test,time.perf_counter()-begin


def prepare_config(grid_item):
    a = _build_parser().parse_args(['--device',DEVICE,
       '--warmup-epochs',str(MAX_WARMUP),'--full-epochs',str(MAX_FULL),
       '--hidden-dim',str(grid_item['hidden_dim']),
       '--drop-rate',str(grid_item['drop_rate']),
       '--full-lr',str(grid_item['full_lr'])])
    return _config_from_args(a)


def make_model(cfg):
    return SAWGADModel(in_features=25,hidden_dim=cfg.hidden_dim,
       num_synthetic_heads=NUM_ANOMALY_TYPES,drop_rate=cfg.drop_rate,
       encoder_num_layers=cfg.encoder_num_layers,
       encoder_num_heads=cfg.encoder_num_heads,
       encoder_hidden_multiplier=cfg.encoder_hidden_multiplier).to(DEVICE)


def val_scores(model,split):
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model.score_real(split.graph,split.val_nodes).reshape(-1))
    y = split.graph.ndata['label'][split.val_nodes]
    return y.cpu().numpy().astype(np.int64),scores.cpu().numpy().astype(np.float64)


def train_and_freeze(ratio,seed,grid_item,folder):
    set_global_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    split,test_ids,pre_seconds = load_split(ratio)
    cfg = prepare_config(grid_item)
    model = make_model(cfg)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print(f'RUN ratio={ratio} seed={seed} cfg={grid_item} WARMUP {MAX_WARMUP}',flush=True)
    _warmup_phase(model,split,cfg,f'{ratio}-{seed} warmup')
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - t0

    train_nodes = split.train_nodes
    labels = split.graph.ndata['label']
    train_y = labels[train_nodes].float().unsqueeze(-1)
    assert float(train_y.sum()) > 0
    assert NUM_ANOMALY_TYPES == 5
    pos_weight = (1-train_y).sum()/train_y.sum().clamp_min(1)
    opt = torch.optim.Adam(model.parameters(),lr=cfg.full_lr,weight_decay=cfg.weight_decay)
    best_ap, best_epoch, no_improve = -1.,0,0
    best_weights = None
    t1 = time.perf_counter()
    print(f'RUN ratio={ratio} seed={seed} FULL_TRAIN up to {MAX_FULL} patience={PATIENCE}',flush=True)
    for ep in range(1,MAX_FULL+1):
        model.train()
        fake_graph,iters = _prepare_synthetic_epoch(split.graph,train_nodes,
          (cfg.full_batch_size//2)//NUM_ANOMALY_TYPES,1,cfg.synthetic_anomalies_per_type)
        opt.zero_grad(set_to_none=True)
        logits = model.score_real(split.graph,train_nodes).reshape(-1,1)
        real_loss = F.binary_cross_entropy_with_logits(logits,train_y,pos_weight=pos_weight)
        synth_loss = _synthetic_loss(model,fake_graph,iters)
        loss = real_loss + cfg.regularization_weight*synth_loss
        assert torch.isfinite(loss).item(),f'nonfinite train loss epoch {ep}'
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().item())
        del fake_graph,iters,logits,real_loss,synth_loss,loss
        yv,pv = val_scores(model,split)
        val_ap = float(average_precision_score(yv,pv))
        assert np.isfinite(val_ap)
        if val_ap > best_ap + 1e-12:
            best_ap,best_epoch,no_improve = val_ap,ep,0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
        if ep == 1 or ep % 5 == 0 or no_improve >= PATIENCE:
            print(f'FULL ratio={ratio} seed={seed} epoch={ep:03d} loss={loss_value:.5f} valAUPRC={val_ap:.6f} best={best_ap:.6f}@{best_epoch} wait={no_improve}/{PATIENCE}',flush=True)
        if no_improve >= PATIENCE:
            break
    assert best_weights is not None
    torch.cuda.synchronize()
    full_seconds = time.perf_counter()-t1
    model.load_state_dict(best_weights)
    yv,pv = val_scores(model,split)
    ap = float(average_precision_score(yv,pv))
    assert abs(ap-best_ap)<1e-8,(ap,best_ap)
    threshold_values = np.linspace(.05,.95,19)
    f1s = [f1_score(yv,pv > th,average='macro',zero_division=0) for th in threshold_values]
    threshold = float(threshold_values[int(np.argmax(f1s))])
    frozen = dict(status='FROZEN_READY_FOR_TEST',model='SAWGAD',dataset='Amazon',
         ratio=ratio,seed=seed,config=grid_item,best_epoch=best_epoch,epochs_ran=ep,
         best_val_auprc=best_ap,threshold=threshold,
         val_macro_f1=float(max(f1s)),preprocess_seconds=pre_seconds,
         warmup_seconds=warmup_seconds,full_training_seconds=full_seconds,
         peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
         split_sha256=EXPECTED_NPZ,source_mat_sha256=EXPECTED_MAT,
         author_commit=EXPECTED_COMMIT,test_used_for_selection=False,
         test_scoring_performed=False,
         protocol='Ass2 candidate: warmup=100 native, full<=100 patience20, AUPRC selection; differs from author full=350')
    ckpt=folder/'best_val_auprc.pt'
    assert not ckpt.exists(), 'Existing checkpoint must not be overwritten'
    torch.save({"model_state_dict":{k:v.cpu() for k,v in best_weights.items()},'frozen':frozen},ckpt)
    frozen['checkpoint_sha256']=sha(ckpt)
    json_write_once(folder/'frozen.json',frozen)
    return model,split,test_ids,frozen


def test_metrics(y,p,thr):
    pred = (p > thr).astype(np.int64)
    cm_neg = y==0
    cm_pos = y==1
    recall = float(recall_score(y,pred,zero_division=0))
    specificity = float(np.mean(pred[cm_neg]==0))
    return dict(auprc=float(average_precision_score(y,p)),
      auroc=float(roc_auc_score(y,p)),
      macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
      fraud_precision=float(precision_score(y,pred,zero_division=0)),
      fraud_recall=recall,
      fraud_f1=float(f1_score(y,pred,zero_division=0)),
      gmean=float(np.sqrt(recall*specificity)),
      threshold=float(thr),test_count=int(len(y)),fraud_count=int(cm_pos.sum()))


def evaluate_once(model,split,test_ids,frozen,folder):
    # Claim is durable BEFORE first test access. Never automatically re-score.
    claim=folder/'test_claim.json'
    json_write_once(claim,dict(status='CLAIMED_TEST_EVALUATION',
       checkpoint_sha256=frozen['checkpoint_sha256'],ratio=frozen['ratio'],
       seed=frozen['seed'],selection_frozen=True))
    print('FROZEN_CHECKPOINT_AND_THRESHOLD=PASS; TEST_ONCE',flush=True)
    start=time.perf_counter()
    model.eval()
    with torch.no_grad():
        tid=torch.as_tensor(test_ids,dtype=torch.long,device=DEVICE)
        scores=torch.sigmoid(model.score_real(split.graph,tid).reshape(-1)).cpu().numpy()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    assert np.isfinite(scores).all()
    # Training has completed, selection has frozen; read TEST labels now only.
    true_labels = np.asarray(loadmat(MAT,variable_names=['label'])['label']).reshape(-1)[test_ids].astype(np.int64)
    values = test_metrics(true_labels,scores,frozen['threshold'])
    result={**frozen,'status':'FINAL_CANDIDATE_COMPLETE',
          'test_scoring_performed':True,'test_evaluated_once':True,
          'test_inference_seconds':elapsed,'test_metrics':values}
    json_write_once(folder/'final.json',result)
    print('FINAL_CANDIDATE',frozen['ratio'],frozen['seed'],'testAUPRC',round(values['auprc'],6),flush=True)
    return result


def completed(folder,kind,ratio,seed,config):
    p=folder/('tuning.json' if kind=='tune' else 'final.json')
    if not p.exists():
        return None
    data=json.loads(p.read_text())
    assert data['status'] == ('TUNING_COMPLETE' if kind=='tune' else 'FINAL_CANDIDATE_COMPLETE')
    assert data['ratio']==ratio and data['seed']==seed and data['config']==config
    assert data['split_sha256']==EXPECTED_NPZ and data['author_commit']==EXPECTED_COMMIT
    if kind=='final':
        assert (folder/'test_claim.json').exists()
    return data


def do_tuning():
    result=[]
    for i,grid_item in enumerate(GRID,1):
        folder=OUT/'tuning'/f'trial_{i:02d}'
        folder.mkdir(parents=True,exist_ok=True)
        old=completed(folder,'tune','TR40',2,grid_item)
        if old is not None:
            print(f'TUNING_RESUME trial={i:02d} valAUPRC={old["best_val_auprc"]:.6f}',flush=True)
            result.append(old)
            continue
        if (folder/'frozen.json').exists():
            frozen=json.loads((folder/'frozen.json').read_text())
            assert frozen['status']=='FROZEN_READY_FOR_TEST'
            assert frozen['ratio']=='TR40' and frozen['seed']==2 and frozen['config']==grid_item
            assert sha(folder/'best_val_auprc.pt')==frozen['checkpoint_sha256']
            print(f'TUNING_RECOVER_FROZEN trial={i:02d}',flush=True)
        else:
            model,split,_,frozen=train_and_freeze('TR40',2,grid_item,folder)
            del model,split
            torch.cuda.empty_cache()
        frozen['status']='TUNING_COMPLETE'
        json_write_once(folder/'tuning.json',frozen)
        print(f'TUNING_FINISHED {i}/12 valAUPRC={frozen["best_val_auprc"]:.6f}',flush=True)
        result.append(frozen)
    winner=min(enumerate(result,1),key=lambda x:(-x[1]['best_val_auprc'],x[0]))
    selected=dict(status='WINNER_FROZEN',winner_trial=winner[0],config=winner[1]['config'],
                  best_val_auprc=winner[1]['best_val_auprc'],
                  selection='TR40 seed2 max validation AUPRC only',
                  test_scoring_performed=False)
    winner_file=OUT/'winner.json'
    if winner_file.exists():
        assert json.loads(winner_file.read_text())==selected,'Winner changed: STOP'
    else:
        json_write_once(winner_file,selected)
    print('WINNER_FROZEN',json.dumps(selected),flush=True)
    return selected


def do_finals(winner):
    results=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            folder=OUT/'final'/ratio/f'seed_{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            old=completed(folder,'final',ratio,seed,winner['config'])
            if old is not None:
                print(f'FINAL_RESUME {ratio} {seed} testAUPRC={old["test_metrics"]["auprc"]:.6f}',flush=True)
                results.append(old)
                continue
            if (folder/'test_claim.json').exists():
                raise RuntimeError(f'Test already claimed but no final manifest: STOP to avoid repeating test scoring: {folder}')
            if (folder/'frozen.json').exists():
                frozen=json.loads((folder/'frozen.json').read_text())
                assert frozen['status']=='FROZEN_READY_FOR_TEST'
                assert frozen['ratio']==ratio and frozen['seed']==seed and frozen['config']==winner['config']
                assert sha(folder/'best_val_auprc.pt')==frozen['checkpoint_sha256']
                set_global_seed(seed)
                split,test_ids,_ = load_split(ratio)
                model=make_model(prepare_config(winner['config']))
                checkpoint=torch.load(folder/'best_val_auprc.pt',map_location=DEVICE,weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f'FINAL_RECOVER_FROZEN {ratio} seed={seed}',flush=True)
            else:
                model,split,test_ids,frozen=train_and_freeze(ratio,seed,winner['config'],folder)
            done=evaluate_once(model,split,test_ids,frozen,folder)
            results.append(done)
            del model,split
            torch.cuda.empty_cache()
    summaries=[]
    for ratio in RATIOS:
        subset=[r for r in results if r['ratio']==ratio]
        summary={'ratio':ratio,'runs':len(subset),'metrics':{}}
        for k in ('auprc','auroc','macro_f1','fraud_precision','fraud_recall','fraud_f1','gmean'):
            values=np.array([r['test_metrics'][k] for r in subset],dtype=float)
            summary['metrics'][k]={'mean':float(values.mean()),'population_sd':float(values.std(ddof=0))}
        summaries.append(summary)
    destination=OUT/'complete_summary.json'
    report=dict(status='SAWGAD_AMAZON_FULL_CANDIDATE_COMPLETE',
                winner=winner,final_runs=len(results),ratio_summaries=summaries,
                test_evaluated_once_per_final_run=True,
                author_original_training_protocol=False,
                supervisor_methodological_approval_pending=True)
    if destination.exists():
        assert json.loads(destination.read_text())==report
    else:
        json_write_once(destination,report)
    print(json.dumps(report,indent=2),flush=True)
    print('SAWGAD_AMAZON_FULL_CANDIDATE_COMPLETE',flush=True)


def main():
    os.environ['DGLBACKEND']='pytorch'
    OUT.mkdir(parents=True,exist_ok=True)
    ensure_sources()
    chosen=do_tuning()
    do_finals(chosen)

if __name__=='__main__':
    main()
