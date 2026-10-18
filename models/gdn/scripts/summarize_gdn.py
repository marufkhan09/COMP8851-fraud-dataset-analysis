"""
Collects every finished GDN unified run into the team's summary format and flags
runs that need a mention in the write-up.

Writes, in /workspace/gdn_vast/:
    full_metrics_summary.csv   one row per run (same columns as models/splitgnn/full_metrics_summary.csv,
                               plus threshold, epochs, best_epoch, tp/fp/tn/fn)
    timing_summary.csv         one row per run (same columns as models/splitgnn/timing_summary.csv, plus peak RSS)
    aggregate_summary.csv      mean and std over seeds per dataset x ratio
and prints the aggregate table, the flags and an integrity check.

Usage: python summarize_gdn.py
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = '/workspace/gdn_vast'
DATASETS = ['yelp', 'amazon', 'comp', 'tfinance', 'elliptic', 'tsocial']
RATIOS = [40, 30, 20, 10]
SEEDS = [2, 42, 72]
METRICS = ['auprc', 'auc', 'f1_macro', 'precision_fraud', 'recall_fraud', 'f1_fraud', 'g_mean']


def main():
    rows, timing, flags, missing = [], [], [], []
    for ds in DATASETS:
        for r in RATIOS:
            for s in SEEDS:
                d = f'{ROOT}/results/gdn/{ds}/unified/tr{r}/seed_{s}'
                if not os.path.exists(f'{d}/summary.json'):
                    missing.append(f'{ds} tr{r} seed{s}')
                    continue
                sm = json.load(open(f'{d}/summary.json'))
                t = sm['test_metrics']
                pi = json.load(open(f'{d}/protocol_identity.json'))
                rows.append({'dataset': ds, 'ratio': r, 'seed': s,
                             'auprc': t['auprc'], 'auc': t['auc'], 'f1_macro': t['f1_macro'],
                             'fraud_precision': t['precision_fraud'], 'fraud_recall': t['recall_fraud'],
                             'fraud_f1': t['f1_fraud'], 'g_mean': t['g_mean'],
                             'threshold': t['threshold_used'], 'n_epochs': sm['n_completed_epochs'],
                             'best_epoch': sm['best_epoch'], 'tp': t['tp'], 'fp': t['fp'], 'tn': t['tn'], 'fn': t['fn'],
                             'split_identical_to_splitgnn': pi.get('split_identical_to_splitgnn'),
                             'adapter_version': pi.get('adapter_version')})
                ep = list(csv.DictReader(open(f'{d}/epoch_times.csv')))
                tr = np.array([float(e['train_seconds']) for e in ep])
                va = np.array([float(e['valid_seconds']) for e in ep])
                timing.append({'dataset': ds, 'ratio': r, 'seed': s, 'n_epochs': len(ep),
                               'mean_train_seconds_per_epoch': round(tr.mean(), 4),
                               'mean_valid_seconds_per_epoch': round(va.mean(), 4),
                               'mean_total_seconds_per_epoch': round((tr + va).mean(), 4),
                               'total_train_seconds': round(tr.sum(), 2),
                               'peak_memory_mb': round(max(float(e['peak_memory_bytes']) for e in ep) / 2**20, 1),
                               'peak_rss_mb': round(max(float(e['peak_rss_bytes']) for e in ep) / 2**20, 1)})

                tag = f'{ds} tr{r} seed{s}'
                if t['tp'] + t['fp'] == 0:
                    flags.append(f'{tag}: predicted NO fraud on test at threshold {t["threshold_used"]}')
                elif t['recall_fraud'] < 0.05:
                    flags.append(f'{tag}: near-zero fraud recall ({t["recall_fraud"]:.3f})')
                if abs(sm['reloaded_val_auprc'] - sm['best_val_auprc']) > 1e-6:
                    flags.append(f'{tag}: reloaded val AUPRC {sm["reloaded_val_auprc"]:.6f} != logged {sm["best_val_auprc"]:.6f}')
                if pi.get('split_identical_to_splitgnn') is not True:
                    flags.append(f'{tag}: split not verified against SplitGNN ({pi.get("split_identical_to_splitgnn")})')
                if t['threshold_used'] in (0.05, 0.95):
                    flags.append(f'{tag}: threshold at grid edge ({t["threshold_used"]})')

    with open(f'{ROOT}/full_metrics_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f'{ROOT}/timing_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(timing[0].keys()))
        w.writeheader()
        w.writerows(timing)

    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['ratio'])].append(row)
    agg = []
    print(f"{'dataset':9}{'ratio':>6}  " + '  '.join(f'{m[:9]:>15}' for m in
          ['auprc', 'auc', 'f1_macro', 'fraud_precision', 'fraud_recall', 'fraud_f1', 'g_mean']) + '   epochs(best)')
    for (ds, r), g in groups.items():
        out = {'dataset': ds, 'ratio': r, 'n_seeds': len(g)}
        cells = []
        for m in ['auprc', 'auc', 'f1_macro', 'fraud_precision', 'fraud_recall', 'fraud_f1', 'g_mean']:
            v = np.array([x[m] for x in g])
            out[f'{m}_mean'], out[f'{m}_std'] = round(v.mean(), 4), round(v.std(ddof=1) if len(v) > 1 else 0.0, 4)
            cells.append(f'{v.mean():.3f}±{out[f"{m}_std"]:.3f}')
        agg.append(out)
        eps = '/'.join(f"{x['n_epochs']}({x['best_epoch']})" for x in g)
        print(f'{ds:9}{"tr" + str(r):>6}  ' + '  '.join(f'{c:>15}' for c in cells) + f'   {eps}')
    with open(f'{ROOT}/aggregate_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    capped = sum(1 for x in rows if x['n_epochs'] == 100 and x['best_epoch'] >= 95)
    print(f'\nruns found: {len(rows)}/72   missing: {missing or "none"}')
    print(f'runs that hit the 100-epoch cap with best epoch >= 95 (still improving): {capped}')
    print('flags:' if flags else 'flags: none')
    for fl in flags:
        print('  - ' + fl)
    print(f'\nwrote {ROOT}/full_metrics_summary.csv, timing_summary.csv, aggregate_summary.csv')


if __name__ == '__main__':
    main()
