"""
Loops run_evidence.py over dataset x ratio x seed combos.
Idempotent: skips any run whose summary.json already exists.
Usage:
  python batch_run.py --dataset amazon --ratios 40,30,20,10 --seeds 2,42,72
"""
import argparse
import os
import subprocess
import sys
import time

VAST_ROOT = '/workspace/FRAUDRE'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                         choices=['yelp', 'amazon', 'comp', 'tfinance', 'elliptic'])
    parser.add_argument('--ratios', type=str, default='40,30,20,10')
    parser.add_argument('--seeds', type=str, default='2,42,72')
    args = parser.parse_args()

    ratios = [int(x) for x in args.ratios.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    combos = [(r, s) for r in ratios for s in seeds]
    total = len(combos)
    log_path = os.path.join(VAST_ROOT, f'batch_log_{args.dataset}.txt')

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    for i, (ratio, seed) in enumerate(combos, 1):
        summary_path = os.path.join(
            VAST_ROOT, 'results', args.dataset, 'unified',
            f'tr{ratio}', f'seed_{seed}', 'summary.json')
        if os.path.exists(summary_path):
            log(f"SKIP (already complete): tr{ratio} seed{seed} ({i}/{total})")
            continue

        log(f"START: tr{ratio} seed{seed} ({i}/{total})")
        t0 = time.time()
        cmd = [sys.executable, 'run_evidence.py',
               '--data', args.dataset, '--ratio', str(ratio), '--seed', str(seed)]
        result = subprocess.run(cmd, cwd=VAST_ROOT)
        elapsed = time.time() - t0
        if result.returncode == 0:
            log(f"SUCCESS: tr{ratio} seed{seed} ({elapsed:.1f}s)")
        else:
            log(f"FAILED (exit {result.returncode}): tr{ratio} seed{seed} ({elapsed:.1f}s)")


if __name__ == '__main__':
    main()
