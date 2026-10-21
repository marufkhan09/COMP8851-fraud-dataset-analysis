import json, glob, csv
import numpy as np
rows = []
for m in ('consisgad', 'lexgnn'):
    for r in (40, 30, 20, 10):
        S = []
        for s in (2, 42, 72):
            f = sorted(glob.glob(f'results/{m}/final/tr{r}_seed{s}/*/summary.json'))[-1]
            S.append(json.load(open(f)))
        row = {'model': m, 'ratio': f'TR{r}', 'n_seeds': len(S)}
        for k in ('auroc', 'auprc', 'macro_f1', 'fraud_f1', 'g_mean'):
            v = np.array([x['test_metrics'][k] for x in S])
            row[k + '_mean'] = round(float(v.mean()), 4); row[k + '_sd'] = round(float(v.std(ddof=1)), 4)
        for k in ('training_seconds_mean', 'peak_gpu_memory_mb', 'epochs_completed', 'total_parameters'):
            v = np.array([x[k] for x in S], float); row[k] = round(float(v.mean()), 2)
        rows.append(row)
w = csv.DictWriter(open('tsocial_results_summary.csv', 'w'), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
for r in rows: print(r['model'], r['ratio'], 'AUPRC %.3f+-%.3f' % (r['auprc_mean'], r['auprc_sd']), 'AUROC %.3f' % r['auroc_mean'], 'epochs', r['epochs_completed'])
