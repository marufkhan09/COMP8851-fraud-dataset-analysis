"""
Batch runner: executes run_evidence.py for every (ratio, seed) combination for a
given dataset/track, in sequence. Skips a run if its summary.json already exists
(idempotent - safe to re-run after an interruption). Logs progress to batch_log.txt.

Usage: python batch_run.py --dataset yelp --track unified --ratios 40,30,20,10 --seeds 2,42,72
"""
import argparse
import subprocess
import os
import sys
import time
from datetime import datetime

VAST_ROOT = '/workspace/splitgnn_vast'
MODEL_NAME = 'splitgnn'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--track', default='unified')
    p.add_argument('--ratios', default='40,30,20,10')
    p.add_argument('--seeds', default='2,42,72')
    p.add_argument('--gpu', default='a6000')
    cli = p.parse_args()

    ratios = [int(r) for r in cli.ratios.split(',')]
    seeds = [int(s) for s in cli.seeds.split(',')]

    log_path = f"{VAST_ROOT}/batch_log_{cli.dataset}.txt"

    def log(msg):
        line = f"[{datetime.now().isoformat()}] {msg}"
        print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f"=== Batch run start: dataset={cli.dataset} track={cli.track} ratios={ratios} seeds={seeds} ===")
    total = len(ratios) * len(seeds)
    done = 0
    failed = []

    for ratio in ratios:
        for seed in seeds:
            result_dir = f"{VAST_ROOT}/results/{MODEL_NAME}/{cli.dataset}/{cli.track}/tr{ratio}/seed_{seed}"
            summary_path = f"{result_dir}/summary.json"

            if os.path.exists(summary_path):
                log(f"SKIP (already done): tr{ratio} seed{seed}")
                done += 1
                continue

            log(f"START: tr{ratio} seed{seed} ({done+1}/{total})")
            t0 = time.time()
            cmd = ['python', f'{VAST_ROOT}/run_evidence.py',
                   '--dataset', cli.dataset, '--ratio', str(ratio),
                   '--seed', str(seed), '--track', cli.track, '--gpu', cli.gpu]
            result = subprocess.run(cmd, cwd=VAST_ROOT)
            elapsed = time.time() - t0

            if result.returncode == 0 and os.path.exists(summary_path):
                log(f"SUCCESS: tr{ratio} seed{seed} ({elapsed:.1f}s)")
                done += 1
            else:
                log(f"FAILED: tr{ratio} seed{seed} (returncode={result.returncode}, {elapsed:.1f}s)")
                failed.append((ratio, seed))

    log(f"=== Batch run complete: {done}/{total} succeeded ===")
    if failed:
        log(f"FAILED runs: {failed}")
        sys.exit(1)

if __name__ == '__main__':
    main()
