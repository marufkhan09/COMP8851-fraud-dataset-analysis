#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics


EPOCH_PATTERN = re.compile(
	r'^Epoch: (?P<epoch>\d+), loss: (?P<loss>[-+0-9.eE]+), time: (?P<seconds>[-+0-9.eE]+)s$')
FINAL_PATTERNS = {
	'macro_f1': re.compile(r'^F1-Macro: ([-+0-9.eE]+)$'),
	'auroc': re.compile(r'^AUC: ([-+0-9.eE]+)$'),
	'gmean': re.compile(r'^G-Mean: ([-+0-9.eE]+)$'),
}


def parse_args():
	parser = argparse.ArgumentParser(description='Convert a legacy PC-GNN console log.')
	parser.add_argument('log_file')
	parser.add_argument('--epoch-csv', required=True)
	parser.add_argument('--summary-json', required=True)
	return parser.parse_args()


def main():
	args = parse_args()
	epoch_rows = []
	metrics = {}
	with open(args.log_file, 'r', encoding='utf-8') as handle:
		for raw_line in handle:
			line = raw_line.strip()
			epoch_match = EPOCH_PATTERN.match(line)
			if epoch_match:
				epoch_rows.append({
					'epoch': int(epoch_match.group('epoch')),
					'logged_loss': float(epoch_match.group('loss')),
					'legacy_minibatch_seconds': float(epoch_match.group('seconds')),
				})
			for name, pattern in FINAL_PATTERNS.items():
				metric_match = pattern.match(line)
				if metric_match:
					metrics[name] = float(metric_match.group(1))

	if not epoch_rows:
		raise ValueError('No PC-GNN epoch timing lines were found.')
	if set(metrics) != set(FINAL_PATTERNS):
		raise ValueError('The log does not contain all three final metrics.')

	with open(args.epoch_csv, 'w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
		writer.writeheader()
		writer.writerows(epoch_rows)

	times = [row['legacy_minibatch_seconds'] for row in epoch_rows]
	summary = {
		'status': 'successful_local_reproduction',
		'epochs': len(epoch_rows),
		'test_metrics': metrics,
		'legacy_timing': {
			'boundary': 'sum of author-code minibatch durations; excludes sampling and validation',
			'total_seconds': sum(times),
			'mean_seconds': sum(times) / len(times),
			'median_seconds': statistics.median(times),
			'population_std_seconds': statistics.pstdev(times),
			'min_seconds': min(times),
			'max_seconds': max(times),
		},
		'limitations': [
			'32-dimensional YelpChi features rather than the paper\'s 100-dimensional features',
			'51 epochs rather than the paper\'s 100 epochs',
			'one run rather than the paper\'s 10-run mean and standard deviation',
			'local CPU environment rather than the final common benchmark hardware',
		],
	}
	with open(args.summary_json, 'w', encoding='utf-8') as handle:
		json.dump(summary, handle, indent=2, sort_keys=True)
	print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
