#!/usr/bin/env python3
import os, sys, json, time, gc, shutil, hashlib, importlib.util, csv, math
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from omegaconf import OmegaConf
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

ROOT = Path('/workspace/gaap_vast')
REPO = ROOT / 'repo/GAAP'
AUTHOR_ENTRY = REPO / 'mycode/exp/101_retrain.py'
CFG_PATH = REPO / 'config/SAGE_MiniF_DyPLE_MHA/tfinance.yaml'
ADAPTER = ROOT / 'shared/tfinance/tfinance_gaap_input_adapter'
CANON = ROOT / 'shared/tfinance/tfinance'
SPLIT = ROOT / 'shared/tfinance/tfinance_seed2_nested_splits.npz'
WINNER_PATH = ROOT / 'unified/tfinance/tuning/GAAP_TFINANCE_FROZEN_WINNER.json'
OUTDIR = ROOT / 'unified/tfinance/final'
RUN_DIR = OUTDIR / 'runs'
CKPT_DIR = OUTDIR / 'checkpoints'
OUTDIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COMMIT = '6a7dbb0447c4897504525de49e41a0526ee777f8'
EXPECTED_CANON = 'b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
EXPECTED_SPLIT = 'de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
EXPECTED_ADAPTER = '7db7e48617038b35dbac041a6b90426f25237886e157d2f54776bbbd756d69ae'
RATIOS = ['TR40','TR30','TR20','TR10']
SEEDS = [2,42,72]
MAX_EPOCHS = 100
PATIENCE = 20
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

FINAL_RUNS_CSV = OUTDIR / 'GAAP_TFINANCE_FINAL_12_RUNS.csv'
RATIO_SUMMARY_CSV = OUTDIR / 'GAAP_TFINANCE_FINAL_RATIO_SUMMARY.csv'
FINAL_SUMMARY_JSON = OUTDIR / 'GAAP_TFINANCE_FINAL_SUMMARY.json'
STATUS_TXT = OUTDIR / 'GAAP_TFINANCE_FINAL_STATUS.txt'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def git_output(*args):
    import subprocess
    return subprocess.check_output(['git','-C',str(REPO),*args], text=True).strip()


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def classification_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(np.int64)
    ap = float(average_precision_score(y, p))
    auc = float(roc_auc_score(y, p))
    macro = float(f1_score(y, pred, average='macro', zero_division=0))
    prec = float(precision_score(y, pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y, pred, pos_label=1, zero_division=0))
    ff1 = float(f1_score(y, pred, pos_label=1, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    gmean = float(math.sqrt(tpr * tnr))
    return {
        'auprc': ap, 'auroc': auc, 'macro_f1': macro,
        'fraud_precision': prec, 'fraud_recall': rec,
        'fraud_f1': ff1, 'gmean': gmean,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def choose_threshold(y, p):
    rows = []
    for th in THRESHOLDS:
        m = classification_metrics(y, p, float(th))
        rows.append({'threshold': float(th), **m})
    best = max(rows, key=lambda r: (r['macro_f1'], r['fraud_recall'], -abs(r['threshold'] - 0.5)))
    return best, rows


assert git_output('rev-parse','HEAD') == EXPECTED_COMMIT
assert git_output('status','--porcelain','--untracked-files=no') == ''
assert sha256(CANON) == EXPECTED_CANON
assert sha256(SPLIT) == EXPECTED_SPLIT
assert sha256(ADAPTER) == EXPECTED_ADAPTER
assert WINNER_PATH.is_file()

split = np.load(SPLIT, allow_pickle=False)
expected_sizes = {'TR40':15742,'TR30':11806,'TR20':7870,'TR10':3935,'val':7872,'test':15743}
for k,n in expected_sizes.items():
    assert len(split[k]) == n, (k, len(split[k]), n)
sets = {k:set(map(int,split[k])) for k in expected_sizes}
assert sets['TR10'] <= sets['TR20'] <= sets['TR30'] <= sets['TR40']
assert sets['val'].isdisjoint(sets['test'])
for r in RATIOS:
    assert sets[r].isdisjoint(sets['val']) and sets[r].isdisjoint(sets['test'])
assert int(split['seed'][0]) == 2
assert int(split['source_nodes'][0]) == 39357

winner = json.loads(WINNER_PATH.read_text())
assert winner['winner_trial'] == 10
assert int(winner['d_hidden']) == 64
assert int(winner['n_bins']) == 32
assert abs(float(winner['lr']) - 0.002) < 1e-15
assert abs(float(winner['best_val_auprc']) - 0.897947) < 5e-6
assert int(winner['best_epoch']) == 22
assert winner['canonical_sha256'] == EXPECTED_CANON
assert winner['split_sha256'] == EXPECTED_SPLIT
assert winner['adapter_sha256'] == EXPECTED_ADAPTER
assert winner['repo_commit'] == EXPECTED_COMMIT
assert winner['test_evaluated'] is False

print('TFINANCE_CANONICAL_DATA_GATE=PASS', flush=True)
print('TFINANCE_FROZEN_SPLIT_GATE=PASS', flush=True)
print('TFINANCE_INPUT_ADAPTER_GATE=PASS', flush=True)
print('FROZEN_WINNER_GATE=PASS', flush=True)
print('WINNER_CONFIG=' + json.dumps({'d_hidden':64,'n_bins':32,'lr':0.002}, sort_keys=True), flush=True)
print('WINNER_VAL_AUPRC=0.897947@22', flush=True)
print('FINAL_PROTOCOL=TR40_TR30_TR20_TR10_X_SEEDS_2_42_72', flush=True)
print('TEST_POLICY=ONE_TEST_EVALUATION_AFTER_CONFIG_CHECKPOINT_THRESHOLD_FREEZE_PER_FINAL_RUN', flush=True)
print('GAAP_AUTHOR_REPO_CLEAN=PASS', flush=True)

spec = importlib.util.spec_from_file_location('gaap_author_101_retrain', AUTHOR_ENTRY)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

native = dict(OmegaConf.to_container(OmegaConf.load(CFG_PATH), resolve=True))
assert native['model_name'] == 'SAGE'
assert native['loader_type'] == 'MiniF'
assert native['data_name'] == 'tfinance'
assert native['norm_type'] == 'norm01'
assert str(native['preprocess']) == 'None'
assert int(native['gnn_n_layers']) == 3
assert float(native['gnn_dropout']) == 0.0
assert native['gnn_agg'] == 'max_pool'
assert bool(native['use_dyple']) is True
assert bool(native['use_mha']) is True
assert float(native['weight_decay']) == 0.0

loader_cls = A.tag_dm_map[native['loader_type']]
loader_globals = loader_cls.__init__.__globals__
assert 'DIR_FRAUD_DATASET' in loader_globals
runtime_dir = Path(str(loader_globals['DIR_FRAUD_DATASET']))
runtime_path = runtime_dir / 'tfinance'
runtime_dir.mkdir(parents=True, exist_ok=True)


class ControlledLitSAGE(A.LitSAGE):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.capture_mode = False
        self.capture_y = None
        self.capture_prob = None
        self.capture_nid = None

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.outs.clear(); return
        outs = self.outs
        if not outs:
            return
        y = torch.cat([x[0] for x in outs])
        logit = torch.cat([x[1] for x in outs])
        nid = torch.cat([x[2] for x in outs])
        his_emb = torch.cat([x[3] for x in outs])
        order = torch.argsort(nid)
        y = y[order]; logit = logit[order]; nid = nid[order]
        self.his_emb = his_emb[order]

        val_idx = self.val_idx
        if not torch.is_tensor(val_idx):
            val_idx = torch.as_tensor(val_idx, dtype=torch.long, device=y.device)
        else:
            val_idx = val_idx.to(y.device)
        val_loss = F.cross_entropy(logit[val_idx], y[val_idx])
        prob = logit.softmax(-1)[:,1]
        y_np = y.detach().cpu().numpy()
        p_np = prob.detach().cpu().numpy()
        vi = val_idx.detach().cpu().numpy()
        val_auprc = float(average_precision_score(y_np[vi], p_np[vi]))
        val_auroc = float(roc_auc_score(y_np[vi], p_np[vi]))
        self.log('valoss', val_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_aps', val_auprc, prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_auc', val_auroc, prog_bar=False, on_step=False, on_epoch=True)
        if self.capture_mode:
            self.capture_y = y_np.copy()
            self.capture_prob = p_np.copy()
            self.capture_nid = nid.detach().cpu().numpy().copy()
        self.outs.clear()


class TrainTiming(Callback):
    def __init__(self):
        super().__init__()
        self.train_epoch_seconds = []
        self.val_seconds = []
        self._train_t0 = None
        self._val_t0 = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:
            torch.cuda.synchronize()
            self._train_t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try:
            n = int(trainer.num_training_batches)
        except Exception:
            n = -1
        if self._train_t0 is not None and n > 0 and batch_idx + 1 >= n:
            torch.cuda.synchronize()
            self.train_epoch_seconds.append(time.perf_counter() - self._train_t0)
            self._train_t0 = None

    def on_validation_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            torch.cuda.synchronize(); self._val_t0 = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking and self._val_t0 is not None:
            torch.cuda.synchronize()
            self.val_seconds.append(time.perf_counter() - self._val_t0)
            self._val_t0 = None


class LiveFinal(Callback):
    def __init__(self, run_no, ratio, seed, timing, epcsv):
        super().__init__()
        self.run_no = run_no; self.ratio = ratio; self.seed = seed
        self.timing = timing; self.epcsv = epcsv
        self.best = -float('inf'); self.best_epoch = 0; self.bad = 0
        self.rows = []

    @staticmethod
    def scalar(x):
        if x is None: return float('nan')
        if torch.is_tensor(x): return float(x.detach().cpu())
        return float(x)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        m = trainer.callback_metrics
        val = self.scalar(m.get('val_aps'))
        auc = self.scalar(m.get('val_auc'))
        loss = self.scalar(m.get('trloss_epoch', m.get('trloss')))
        epoch = int(trainer.current_epoch) + 1
        if np.isfinite(val) and val > self.best:
            self.best = val; self.best_epoch = epoch; self.bad = 0
        else:
            self.bad += 1
        train_s = self.timing.train_epoch_seconds[-1] if self.timing.train_epoch_seconds else float('nan')
        val_s = self.timing.val_seconds[-1] if self.timing.val_seconds else float('nan')
        self.rows.append({
            'run_no':self.run_no,'ratio':self.ratio,'train_seed':self.seed,'epoch':epoch,
            'train_loss':loss,'val_auprc':val,'val_auroc':auc,
            'best_val_auprc_so_far':self.best,'best_epoch_so_far':self.best_epoch,
            'patience_count':self.bad,'train_epoch_seconds':train_s,'validation_seconds':val_s,
            'peak_gpu_mb_so_far':float(torch.cuda.max_memory_allocated()/(1024**2)),
        })
        write_csv(self.epcsv, self.rows)
        print(
            f'run={self.run_no:02d}/12 | {self.ratio} seed={self.seed} | epoch={epoch:03d} | '
            f'loss={loss:.6f} | valAUPRC={val:.6f} | valAUROC={auc:.6f} | '
            f'best={self.best:.6f}@{self.best_epoch} | patience={self.bad}/{PATIENCE}', flush=True
        )


def set_index_attr_if_present(dm, names, idx):
    arr_np = np.asarray(idx, dtype=np.int64)
    arr_t = torch.as_tensor(arr_np, dtype=torch.long)
    for name in names:
        if hasattr(dm, name):
            old = getattr(dm, name)
            if callable(old):
                continue
            try:
                if torch.is_tensor(old):
                    setattr(dm, name, arr_t.clone())
                elif isinstance(old, np.ndarray):
                    setattr(dm, name, arr_np.copy())
                elif isinstance(old, list):
                    setattr(dm, name, arr_np.tolist())
            except Exception:
                pass


def cast_float32(dm):
    casted = []
    for k in list(dm.g.ndata.keys()):
        v = dm.g.ndata[k]
        if torch.is_tensor(v) and torch.is_floating_point(v) and v.dtype != torch.float32:
            dm.g.ndata[k] = v.float().contiguous(); casted.append('g.ndata['+k+']')
    for attr in ('feature','features','feat','x','bin_feat','bin_feature','feat_bin','feature_bin'):
        if hasattr(dm, attr):
            v = getattr(dm, attr)
            if torch.is_tensor(v) and torch.is_floating_point(v) and v.dtype != torch.float32:
                setattr(dm, attr, v.float().contiguous()); casted.append('dm.'+attr)
    return casted


def build_dm(cfg, ratio, hide_test):
    shutil.copyfile(ADAPTER, runtime_path)
    assert sha256(runtime_path) == EXPECTED_ADAPTER
    dm = A.tag_dm_map[cfg['loader_type']](**cfg)
    assert dm.g.number_of_nodes() == 39357

    n = dm.g.number_of_nodes()
    train_idx = np.asarray(split[ratio], dtype=np.int64)
    val_idx = np.asarray(split['val'], dtype=np.int64)
    test_idx = np.asarray(split['test'], dtype=np.int64)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[torch.as_tensor(train_idx)] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[torch.as_tensor(val_idx)] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[torch.as_tensor(test_idx)] = True
    dm.g.ndata['train_mask'] = train_mask
    dm.g.ndata['val_mask'] = val_mask
    dm.g.ndata['test_mask'] = test_mask

    set_index_attr_if_present(dm, ('train_idx','train_nid','train_ids','idx_train'), train_idx)
    set_index_attr_if_present(dm, ('val_idx','val_nid','val_ids','idx_val'), val_idx)
    set_index_attr_if_present(dm, ('test_idx','test_nid','test_ids','idx_test'), test_idx)

    assert int(dm.g.ndata['train_mask'].sum()) == expected_sizes[ratio]
    assert int(dm.g.ndata['val_mask'].sum()) == expected_sizes['val']
    assert int(dm.g.ndata['test_mask'].sum()) == expected_sizes['test']
    assert not torch.any(dm.g.ndata['train_mask'][torch.as_tensor(test_idx)])
    assert not torch.any(dm.g.ndata['val_mask'][torch.as_tensor(test_idx)])

    casted = cast_float32(dm)
    if hide_test:
        labels = dm.g.ndata['label'].clone()
        labels[torch.as_tensor(test_idx)] = 0
        dm.g.ndata['label'] = labels
        assert torch.all(dm.g.ndata['label'][torch.as_tensor(test_idx)] == 0)
    return dm, casted


def load_best_weights(model, ckpt_path):
    obj = torch.load(ckpt_path, map_location='cpu')
    state = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    model.load_state_dict(state)


def capture_full(model, dm):
    model.capture_mode = True
    model.capture_y = model.capture_prob = model.capture_nid = None
    tr = Trainer(
        accelerator='gpu', devices=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
    )
    torch.cuda.synchronize(); t0 = time.perf_counter()
    tr.validate(model, datamodule=dm, verbose=False)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
    assert model.capture_y is not None and model.capture_prob is not None and model.capture_nid is not None
    nid = np.asarray(model.capture_nid, dtype=np.int64)
    assert len(np.unique(nid)) == len(nid)
    model.capture_mode = False
    return model.capture_y.copy(), model.capture_prob.copy(), nid.copy(), elapsed


def subset_by_nid(y, p, nid, wanted):
    pos = {int(n): i for i,n in enumerate(nid)}
    idx = np.asarray([pos[int(x)] for x in wanted], dtype=np.int64)
    return y[idx], p[idx]


base_cfg = deepcopy(native)
base_cfg['d_hidden'] = int(winner['d_hidden'])
base_cfg['n_bins'] = int(winner['n_bins'])
base_cfg['lr'] = float(winner['lr'])
base_cfg['weight_decay'] = float(winner['weight_decay'])
base_cfg['max_epochs'] = MAX_EPOCHS
base_cfg['patience'] = PATIENCE
base_cfg['device'] = 'cuda'; base_cfg['device_id'] = 0; base_cfg['nowandb'] = True

all_run_records = []
run_no = 0

for ratio in RATIOS:
    for seed in SEEDS:
        run_no += 1
        rid = f'tfinance_{ratio.lower()}_seed{seed}'
        summary_path = RUN_DIR / f'{rid}_summary.json'
        epcsv = RUN_DIR / f'{rid}_epochs.csv'
        thcsv = RUN_DIR / f'{rid}_val_threshold_grid.csv'
        ckpt_dir = CKPT_DIR / rid
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if summary_path.exists():
            try:
                old = json.loads(summary_path.read_text())
                if (
                    old.get('status') == 'PASS' and old.get('ratio') == ratio and old.get('train_seed') == seed
                    and old.get('frozen_config',{}).get('d_hidden') == 64
                    and old.get('frozen_config',{}).get('n_bins') == 32
                    and abs(float(old.get('frozen_config',{}).get('lr')) - 0.002) < 1e-15
                    and old.get('test_evaluated_once_after_freeze') is True
                    and old.get('test_isolation') is True
                ):
                    print(f'RESUME FINAL {run_no:02d}/12 | {ratio} seed={seed} PASS', flush=True)
                    all_run_records.append(old)
                    continue
            except Exception:
                pass

        print('\n' + '='*80, flush=True)
        print(f'START FINAL RUN {run_no:02d}/12 | {ratio} | seed={seed}', flush=True)

        cfg = deepcopy(base_cfg); cfg['seed'] = seed
        A.fix_seed(seed)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

        dm, casted = build_dm(cfg, ratio, hide_test=True)
        print('RUNTIME_FLOAT32_CAST_FIELDS=' + repr(casted), flush=True)
        print('TEST_LABELS_HIDDEN_DURING_FIT=PASS', flush=True)
        cfg['d_in'] = dm.d_in; cfg['n_classes'] = dm.n_classes; cfg['n_nodes'] = dm.g.number_of_nodes()

        model = ControlledLitSAGE(**cfg)
        timing = TrainTiming()
        live = LiveFinal(run_no, ratio, seed, timing, epcsv)
        checkpoint = ModelCheckpoint(
            dirpath=str(ckpt_dir), filename='best-{epoch:03d}-{val_aps:.6f}',
            monitor='val_aps', mode='max', save_top_k=1, save_last=False, save_weights_only=True,
        )
        early = EarlyStopping(
            monitor='val_aps', mode='max', patience=PATIENCE, min_delta=0.0,
            verbose=False, check_finite=True,
        )
        trainer = Trainer(
            accelerator='gpu', devices=1, max_epochs=MAX_EPOCHS, logger=False,
            enable_progress_bar=False, enable_model_summary=False,
            callbacks=[timing, live, checkpoint, early], gradient_clip_val=10,
            num_sanity_val_steps=0, log_every_n_steps=1,
        )

        wall0 = time.perf_counter()
        trainer.fit(model, dm)
        fit_wall = time.perf_counter() - wall0
        peak_fit = float(torch.cuda.max_memory_allocated()/(1024**2))
        test_idx_t = torch.as_tensor(np.asarray(split['test'], dtype=np.int64), dtype=torch.long)
        assert torch.all(dm.g.ndata['label'][test_idx_t] == 0)

        best_score = float(checkpoint.best_model_score.detach().cpu())
        best_epoch = int(live.best_epoch)
        assert checkpoint.best_model_path
        load_best_weights(model, checkpoint.best_model_path)
        print(f'{rid} CHECKPOINT_FROZEN epoch={best_epoch} val_AUPRC={best_score:.6f}', flush=True)

        val_y_all, val_p_all, val_nid, val_inf = capture_full(model, dm)
        vy, vp = subset_by_nid(val_y_all, val_p_all, val_nid, np.asarray(split['val'],dtype=np.int64))
        th_choice, th_rows = choose_threshold(vy, vp)
        write_csv(thcsv, th_rows)
        threshold = float(th_choice['threshold'])
        val_metrics = classification_metrics(vy, vp, threshold)
        print(f'{rid} THRESHOLD_FROZEN={threshold:.2f} val_MacroF1={val_metrics["macro_f1"]:.6f}', flush=True)

        dm_true, _ = build_dm(cfg, ratio, hide_test=False)
        true_labels = dm_true.g.ndata['label'].clone()
        del dm_true
        dm.g.ndata['label'] = true_labels
        print('TEST_LABELS_REVEALED_AFTER_FREEZE=YES', flush=True)

        ty_all, tp_all, test_nid, test_inf = capture_full(model, dm)
        ty, tp = subset_by_nid(ty_all, tp_all, test_nid, np.asarray(split['test'],dtype=np.int64))
        test_metrics = classification_metrics(ty, tp, threshold)
        peak_all = float(torch.cuda.max_memory_allocated()/(1024**2))

        train_mean, train_sd = mean_sd([x for x in timing.train_epoch_seconds if np.isfinite(x)])
        valtime_mean, valtime_sd = mean_sd([x for x in timing.val_seconds if np.isfinite(x)])
        train_total = float(np.sum(timing.train_epoch_seconds))

        rec = {
            'status':'PASS', 'run_no':run_no, 'run_id':rid, 'model':'GAAP', 'dataset':'T-Finance',
            'ratio':ratio, 'train_seed':seed,
            'train_nodes':expected_sizes[ratio], 'validation_nodes':expected_sizes['val'], 'test_nodes':expected_sizes['test'],
            'frozen_config':{
                'winner_trial':10,'d_hidden':64,'n_bins':32,'lr':0.002,
                'weight_decay':float(winner['weight_decay']),
                'gnn_n_layers':int(winner['gnn_n_layers']), 'gnn_dropout':float(winner['gnn_dropout']),
                'gnn_agg':winner['gnn_agg'],'use_dyple':bool(winner['use_dyple']),'use_mha':bool(winner['use_mha']),
                'bs':int(winner['bs']),'val_bs':int(winner['val_bs']),
                'norm_type':winner['norm_type'],'preprocess':winner['preprocess'],
            },
            'checkpoint_selection':'validation AUPRC only', 'best_val_auprc':best_score,
            'best_epoch':best_epoch, 'threshold_selection':'validation Macro-F1 over 0.01..0.99; tie fraud recall then closeness to 0.5',
            'threshold':threshold, 'validation_metrics':val_metrics, 'test_metrics':test_metrics,
            'fit_wall_seconds':fit_wall, 'training_seconds_total':train_total,
            'mean_train_epoch_seconds':train_mean, 'sd_train_epoch_seconds':train_sd,
            'mean_validation_seconds':valtime_mean, 'sd_validation_seconds':valtime_sd,
            'frozen_validation_inference_seconds':val_inf, 'final_test_inference_seconds_full_graph':test_inf,
            'final_test_inference_ms_per_test_node':1000.0*test_inf/expected_sizes['test'],
            'peak_fit_gpu_mb':peak_fit, 'peak_overall_gpu_mb':peak_all,
            'canonical_sha256':EXPECTED_CANON,'split_sha256':EXPECTED_SPLIT,'adapter_sha256':EXPECTED_ADAPTER,
            'repo_commit':EXPECTED_COMMIT,'compatibility':'INPUT_ADAPTER','architecture_changed':False,
            'test_labels_hidden_during_fit':True,'test_evaluated_once_after_freeze':True,'test_isolation':True,
            'created_utc':datetime.now(timezone.utc).isoformat(),
        }
        summary_path.write_text(json.dumps(rec, indent=2))
        all_run_records.append(rec)
        print(
            f'FINAL {ratio} seed={seed} PASS | AUPRC={test_metrics["auprc"]:.6f} | '
            f'AUROC={test_metrics["auroc"]:.6f} | MacroF1={test_metrics["macro_f1"]:.6f} | '
            f'Recall={test_metrics["fraud_recall"]:.6f}', flush=True
        )

        del trainer, model, dm
        gc.collect(); torch.cuda.empty_cache()

assert len(all_run_records) == 12
all_run_records = sorted(all_run_records, key=lambda r:(RATIOS.index(r['ratio']), SEEDS.index(r['train_seed'])))

flat = []
for r in all_run_records:
    tm = r['test_metrics']
    flat.append({
        'run_no':r['run_no'],'run_id':r['run_id'],'ratio':r['ratio'],'train_seed':r['train_seed'],
        'auprc':tm['auprc'],'auroc':tm['auroc'],'macro_f1':tm['macro_f1'],
        'fraud_precision':tm['fraud_precision'],'fraud_recall':tm['fraud_recall'],
        'fraud_f1':tm['fraud_f1'],'gmean':tm['gmean'],'threshold':r['threshold'],
        'best_val_auprc':r['best_val_auprc'],'best_epoch':r['best_epoch'],
        'training_seconds_total':r['training_seconds_total'],'mean_train_epoch_seconds':r['mean_train_epoch_seconds'],
        'final_test_inference_seconds_full_graph':r['final_test_inference_seconds_full_graph'],
        'final_test_inference_ms_per_test_node':r['final_test_inference_ms_per_test_node'],
        'peak_overall_gpu_mb':r['peak_overall_gpu_mb'],
    })
write_csv(FINAL_RUNS_CSV, flat)

metric_names = ['auprc','auroc','macro_f1','fraud_precision','fraud_recall','fraud_f1','gmean']
ratio_rows = []
ratio_json = {}
for ratio in RATIOS:
    rr = [x for x in flat if x['ratio'] == ratio]
    assert len(rr) == 3
    row = {'ratio':ratio,'n_seeds':3}; rj = {}
    for m in metric_names:
        a,b = mean_sd([x[m] for x in rr]); row[m+'_mean']=a; row[m+'_sd']=b; rj[m]={'mean':a,'sd':b}
    for m in ['training_seconds_total','mean_train_epoch_seconds','final_test_inference_seconds_full_graph','final_test_inference_ms_per_test_node','peak_overall_gpu_mb']:
        a,b = mean_sd([x[m] for x in rr]); row[m+'_mean']=a; row[m+'_sd']=b
    ratio_rows.append(row); ratio_json[ratio]=rj
write_csv(RATIO_SUMMARY_CSV, ratio_rows)

final = {
    'status':'PASS','model':'GAAP','dataset':'T-Finance','final_runs_complete':12,'expected_final_runs':12,
    'ratios':RATIOS,'seeds':SEEDS,'repository_commit':EXPECTED_COMMIT,
    'canonical_sha256':EXPECTED_CANON,'split_sha256':EXPECTED_SPLIT,'adapter_sha256':EXPECTED_ADAPTER,
    'winner_json':str(WINNER_PATH),'frozen_config':flat[0] if False else all_run_records[0]['frozen_config'],
    'checkpoint_selection':'validation AUPRC only',
    'threshold_selection':'validation Macro-F1 over 0.01..0.99; tie fraud recall then closeness to 0.5',
    'test_isolation':'test evaluated once per final run only after config, checkpoint and threshold frozen',
    'ratio_summary':ratio_json,'final_runs_csv':str(FINAL_RUNS_CSV),'ratio_summary_csv':str(RATIO_SUMMARY_CSV),
    'gpu':torch.cuda.get_device_name(0), 'created_utc':datetime.now(timezone.utc).isoformat(),
}
FINAL_SUMMARY_JSON.write_text(json.dumps(final, indent=2))
STATUS_TXT.write_text(
    'GAAP_TFINANCE_FINAL=PASS\nGAAP_TFINANCE_FINAL_RUNS=12/12\nTEST_ISOLATION=PASS\nGAAP_AUTHOR_REPO_CLEAN=PASS\n'
    'WINNER_TRIAL=10\nWINNER_CONFIG={"d_hidden":64,"n_bins":32,"lr":0.002}\n'
    + f'SUMMARY_JSON={FINAL_SUMMARY_JSON}\n'
)

assert git_output('status','--porcelain','--untracked-files=no') == ''

print('\n===== GAAP T-FINANCE FINAL RESULTS — MEAN ± SD =====', flush=True)
for row in ratio_rows:
    print(
        f"{row['ratio']} | AUPRC={row['auprc_mean']:.6f}±{row['auprc_sd']:.6f} | "
        f"AUROC={row['auroc_mean']:.6f}±{row['auroc_sd']:.6f} | "
        f"MacroF1={row['macro_f1_mean']:.6f}±{row['macro_f1_sd']:.6f} | "
        f"Prec={row['fraud_precision_mean']:.6f}±{row['fraud_precision_sd']:.6f} | "
        f"Recall={row['fraud_recall_mean']:.6f}±{row['fraud_recall_sd']:.6f} | "
        f"FraudF1={row['fraud_f1_mean']:.6f}±{row['fraud_f1_sd']:.6f} | "
        f"GMean={row['gmean_mean']:.6f}±{row['gmean_sd']:.6f}", flush=True
    )

print('\n===== FINAL GATE =====', flush=True)
print('GAAP_TFINANCE_FINAL=PASS', flush=True)
print('GAAP_TFINANCE_FINAL_RUNS=12/12', flush=True)
print('TEST_ISOLATION=PASS', flush=True)
print('GAAP_AUTHOR_REPO_CLEAN=PASS', flush=True)
