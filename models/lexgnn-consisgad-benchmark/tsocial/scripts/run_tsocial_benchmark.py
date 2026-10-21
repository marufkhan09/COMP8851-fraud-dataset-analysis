import json, subprocess, sys, time, copy
from pathlib import Path
W = Path('/workspace/tsocial_work')
M = sys.argv[1]
assert M in ('consisgad', 'lexgnn')
PY = 'python'
ADAPTER = W / f'adapters/{M}/train_tsocial.py'
BASE = json.loads((W / f'configs/{M}_gate.json').read_text())
if M == 'consisgad':
    TRIALS = {
        'trial_01_author_baseline': {'lr': 0.001, 'trainable-consis-weight': 1.5},
        'trial_02_lr_0p0005': {'lr': 0.0005, 'trainable-consis-weight': 1.5},
        'trial_03_lr_0p005': {'lr': 0.005, 'trainable-consis-weight': 1.5},
        'trial_04_weight_1p0': {'lr': 0.001, 'trainable-consis-weight': 1.0},
        'trial_05_weight_2p0': {'lr': 0.001, 'trainable-consis-weight': 2.0},
    }
    TUNE_EPOCHS = {'epochs': 20}
    FINAL_EPOCHS = {'epochs': 30}
else:
    TRIALS = {
        'trial_01_author_baseline': {'learning_rate': 0.005, 'beta': 0.5},
        'trial_02_lr_0p001': {'learning_rate': 0.001, 'beta': 0.5},
        'trial_03_lr_0p0005': {'learning_rate': 0.0005, 'beta': 0.5},
    }
    TUNE_EPOCHS = {'epochs': 2, 'early_stop': 2}
    FINAL_EPOCHS = {'epochs': 5, 'early_stop': 3}
TIMEOUT = 4 * 3600
LOGS = W / f'logs/{M}'; LOGS.mkdir(parents=True, exist_ok=True)
CFG = W / f'configs/{M}'; CFG.mkdir(parents=True, exist_ok=True)
STATUS = W / f'logs/{M}_status.tsv'

def summary_of(root):
    for h in reversed(sorted(Path(root).glob('*/summary.json'))):
        try:
            s = json.loads(h.read_text())
            if s.get('status') == 'PASS':
                return s
        except Exception:
            pass
    return None

def run(name, cfg, results_root):
    done = summary_of(results_root)
    if done:
        print(f'[skip] {name}', flush=True); return done
    cfg = copy.deepcopy(cfg); cfg['results_root'] = str(results_root)
    Path(results_root).mkdir(parents=True, exist_ok=True)
    p = CFG / f'{name}.json'; p.write_text(json.dumps(cfg, indent=1))
    t = time.time(); print(f'[run] {name}', flush=True)
    try:
        rc = subprocess.run([PY, str(ADAPTER), '--config', str(p)], stdout=open(LOGS / f'{name}.log', 'w'),
                            stderr=subprocess.STDOUT, timeout=TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        rc = 'TIMEOUT'
    s = summary_of(results_root)
    with open(STATUS, 'a') as f:
        f.write(f"{name}\t{'PASS' if s else 'FAIL'}\t{rc}\t{time.time()-t:.0f}s\n")
    return s

best, best_key = None, None
for tname, over in TRIALS.items():
    c = {**BASE, **over, **TUNE_EPOCHS, 'evaluate_test': False,
         'split_path': str(W / 'splits/tsocial_TR40_seed2.npz'), 'seed': 2,
         'run_mode': f'tsocial_tuning_tr40_seed2_{tname}'}
    s = run(f'tuning_{tname}', c, W / f'results/{M}/tuning/{tname}')
    if s is None: continue
    key = (s['best_validation_auroc'], s.get('best_validation_auprc', 0))
    if best_key is None or key > best_key: best, best_key = tname, key
assert best, 'no tuning trial passed'
(W / f'configs/{M}_selected.json').write_text(json.dumps({'selected_trial': best, 'validation_auroc': best_key[0], **TRIALS[best]}, indent=1))
print('SELECTED', best, best_key, flush=True)

for ratio in (40, 30, 20, 10):
    for seed in (2, 42, 72):
        c = {**BASE, **TRIALS[best], **FINAL_EPOCHS, 'evaluate_test': True,
             'split_path': str(W / f'splits/tsocial_TR{ratio}_seed2.npz'), 'seed': seed,
             'run_mode': f'tsocial_final_tr{ratio}_seed{seed}', 'training-ratio': ratio}
        run(f'final_tr{ratio}_seed{seed}', c, W / f'results/{M}/final/tr{ratio}_seed{seed}')
print('ALL DONE', flush=True)
