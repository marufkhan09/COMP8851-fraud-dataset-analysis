#!/usr/bin/env python
"""
Idempotent batch loop for H2-FDetector unified runs: ratio x seed per dataset.
Each run is a separate process (clean GPU memory), finished runs are skipped,
a failed run is logged and the loop continues. Safe to restart at any time.

Usage:
  python h2fd_batch_run.py --datasets yelp
  python h2fd_batch_run.py --datasets yelp amazon --ratios 40 --seeds 2   # smoke test
"""
import argparse
import datetime
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--ratios', nargs='+', type=int, default=[40, 30, 20, 10])
    p.add_argument('--seeds', nargs='+', type=int, default=[2, 42, 72])
    p.add_argument('--results-root', default='/workspace/h2fd_vast/results/h2fdetector_results')
    p.add_argument('--log', default='/workspace/h2fd_vast/batch_run.log')
    a = p.parse_args()

    def blog(msg):
        line = f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(a.log, 'a') as f:
            f.write(line + '\n')

    jobs = [(d, r, s) for d in a.datasets for r in a.ratios for s in a.seeds]
    blog(f'START {len(jobs)} runs: datasets={a.datasets} ratios={a.ratios} seeds={a.seeds}')
    done = skipped = failed = 0
    for d, r, s in jobs:
        summary = f'{a.results_root}/{d}/unified/tr{r}/seed_{s}/summary.json'
        if os.path.exists(summary):
            skipped += 1
            blog(f'SKIP {d} tr{r} seed{s} (complete)')
            continue
        blog(f'RUN  {d} tr{r} seed{s}')
        rc = subprocess.call([sys.executable, f'{HERE}/h2fd_unified_run.py', '--dataset', d,
                              '--ratio', str(r), '--seed', str(s), '--results-root', a.results_root])
        if rc == 0 and os.path.exists(summary):
            done += 1
            blog(f'OK   {d} tr{r} seed{s}')
        else:
            failed += 1
            blog(f'FAIL {d} tr{r} seed{s} (exit code {rc})')
    blog(f'END done={done} skipped={skipped} failed={failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
