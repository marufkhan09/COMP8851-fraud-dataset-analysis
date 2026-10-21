#!/usr/bin/env python3
"""Aggregate PC-GNN final runs from the committed terminal logs.

Reads models/pc-gnn/<dataset>/evidence/raw/final_runs/trXX_seedYY_terminal.log, takes the
FINAL_TEST line of each run, and writes seed-level and mean/SD CSV tables to
models/pc-gnn/<dataset>/results/aggregate/.  Run from the repository root.
"""
import csv, glob, os, re, statistics as st
DATASETS = ['tfinance', 'elliptic', 'fdcompcn']
METRICS = ['AUPRC', 'AUROC', 'Macro-F1', 'Fraud-F1', 'G-Mean']
EPOCH = re.compile(r'^Epoch \d+ .*? train=([\d.]+)s validation=([\d.]+)s')
for d in DATASETS:
    base = f'models/pc-gnn/{d}'
    rows = []
    for f in sorted(glob.glob(f'{base}/evidence/raw/final_runs/tr*_seed*_terminal.log')):
        m = re.search(r'tr(\d+)_seed(\d+)_terminal', f)
        ratio, seed = int(m.group(1)), int(m.group(2))
        text = open(f, errors='replace').read().splitlines()
        final = [l for l in text if l.startswith('FINAL_TEST')]
        status = [l for l in text if l.startswith('UNIFIED_RUN_STATUS=')]
        if not final or not status or status[-1].split('=')[1] != 'PASS':
            raise SystemExit(f'incomplete run: {f}')
        vals = dict(re.findall(r'([A-Za-z0-9\-]+)=([\d.]+)', final[-1]))
        ep = [EPOCH.match(l) for l in text]
        ep = [(float(e.group(1)), float(e.group(2))) for e in ep if e]
        row = {'dataset': d, 'ratio': f'TR{ratio}', 'seed': seed, 'epochs_logged': len(ep),
               'mean_train_epoch_s': round(st.mean(t for t, _ in ep), 3),
               'mean_validation_s': round(st.mean(v for _, v in ep), 3)}
        for k in METRICS:
            row[k] = float(vals[k])
        row['threshold'] = float(vals['threshold'])
        rows.append(row)
    os.makedirs(f'{base}/results/aggregate', exist_ok=True)
    cols = list(rows[0])
    with open(f'{base}/results/aggregate/pcgnn_{d}_seed_level_results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
    out = []
    for r in ['TR40', 'TR30', 'TR20', 'TR10']:
        sel = [x for x in rows if x['ratio'] == r]
        o = {'dataset': d, 'ratio': r, 'n_seeds': len(sel)}
        for k in METRICS + ['mean_train_epoch_s', 'mean_validation_s']:
            v = [x[k] for x in sel]
            o[k + '_mean'] = round(st.mean(v), 6)
            o[k + '_sd'] = round(st.stdev(v), 6) if len(v) > 1 else ''
        out.append(o)
    with open(f'{base}/results/aggregate/pcgnn_{d}_mean_sd.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
    print(d, len(rows), 'runs')
    for o in out:
        print('  ', o['ratio'], 'AUPRC %.3f+-%s AUROC %.3f MacroF1 %.3f FraudF1 %.3f GMean %.3f' % (
            o['AUPRC_mean'], o['AUPRC_sd'], o['AUROC_mean'], o['Macro-F1_mean'], o['Fraud-F1_mean'], o['G-Mean_mean']))
