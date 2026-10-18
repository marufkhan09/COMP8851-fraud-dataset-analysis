"""
Idempotent batch loop over ratio x seed for one dataset. Skips runs whose summary.json
exists, so it is safe to restart after a crash or disconnect.

Usage:
    python batch_run_gdn.py --dataset yelp                      # all 12 runs, one after another
    python batch_run_gdn.py --dataset yelp --seeds 2            # one seed (start 3 of these in
    python batch_run_gdn.py --dataset yelp --seeds 42           #  separate tmux windows to run
    python batch_run_gdn.py --dataset yelp --seeds 72           #  the seeds in parallel)
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

VAST_ROOT = '/workspace/gdn_vast'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--track', default='unified')
    ap.add_argument('--ratios', default='40,30,20,10')
    ap.add_argument('--seeds', default='2,42,72')
    ap.add_argument('--threads', default='4')
    ap.add_argument('--extra', default='', help='extra args passed through, e.g. "--topk 5"')
    cli = ap.parse_args()

    ratios = [int(r) for r in cli.ratios.split(',')]
    seeds = [int(s) for s in cli.seeds.split(',')]
    log_path = f"{VAST_ROOT}/batch_log_{cli.dataset}_seeds{'-'.join(map(str, seeds))}.txt"

    def log(msg):
        line = f'[{datetime.now().isoformat(timespec="seconds")}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    total, done, failed = len(ratios) * len(seeds), 0, []
    log(f'=== start dataset={cli.dataset} ratios={ratios} seeds={seeds} ===')
    for ratio in ratios:
        for seed in seeds:
            rdir = f'{VAST_ROOT}/results/gdn/{cli.dataset}/{cli.track}/tr{ratio}/seed_{seed}'
            if os.path.exists(f'{rdir}/summary.json'):
                log(f'SKIP (done): tr{ratio} seed{seed}')
                done += 1
                continue
            log(f'START: tr{ratio} seed{seed} ({done + 1}/{total})')
            t0 = time.time()
            cmd = [sys.executable, f'{VAST_ROOT}/run_gdn_unified.py', '--dataset', cli.dataset,
                   '--ratio', str(ratio), '--seed', str(seed), '--track', cli.track,
                   '--threads', cli.threads] + cli.extra.split()
            rc = subprocess.run(cmd, cwd=VAST_ROOT).returncode
            if rc == 0 and os.path.exists(f'{rdir}/summary.json'):
                done += 1
                log(f'SUCCESS: tr{ratio} seed{seed} ({time.time() - t0:.0f}s)')
            else:
                failed.append((ratio, seed))
                log(f'FAILED: tr{ratio} seed{seed} rc={rc} ({time.time() - t0:.0f}s)')
    log(f'=== done: {done}/{total} succeeded ===')
    if failed:
        log(f'FAILED runs: {failed}')
        sys.exit(1)


if __name__ == '__main__':
    main()
