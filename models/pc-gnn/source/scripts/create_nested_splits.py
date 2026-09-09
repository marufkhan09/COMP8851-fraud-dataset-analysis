#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def parse_args():
	parser = argparse.ArgumentParser(
		description='Create persistent nested TR40/TR30/TR20/TR10 splits.')
	parser.add_argument('--data-file', required=True, help='Path to YelpChi.mat or Amazon.mat')
	parser.add_argument('--dataset-name', required=True, choices=['yelp', 'amazon'])
	parser.add_argument('--output-dir', required=True)
	parser.add_argument('--seed', type=int, default=2)
	return parser.parse_args()


def stratified_subset(parent_indices, labels, fraction, seed):
	selected, _ = train_test_split(
		parent_indices,
		train_size=fraction,
		stratify=labels[parent_indices],
		random_state=seed,
		shuffle=True,
	)
	return np.asarray(selected, dtype=np.int64)


def split_stats(indices, labels):
	split_labels = labels[indices]
	return {
		'nodes': int(len(indices)),
		'fraud_nodes': int(np.sum(split_labels == 1)),
		'normal_nodes': int(np.sum(split_labels == 0)),
		'fraud_percentage': float(100.0 * np.mean(split_labels == 1)),
	}


def main():
	args = parse_args()
	data = loadmat(args.data_file)
	if 'label' not in data:
		raise KeyError("Dataset file does not contain the expected 'label' array.")
	labels = np.asarray(data['label']).reshape(-1).astype(np.int64)
	first_labelled_index = 3305 if args.dataset_name == 'amazon' else 0
	eligible = np.arange(first_labelled_index, len(labels), dtype=np.int64)
	eligible_labels = labels[eligible]

	train40, remainder = train_test_split(
		eligible,
		train_size=0.40,
		stratify=eligible_labels,
		random_state=args.seed,
		shuffle=True,
	)
	validation, test = train_test_split(
		remainder,
		test_size=2.0 / 3.0,
		stratify=labels[remainder],
		random_state=args.seed,
		shuffle=True,
	)
	train40 = np.asarray(train40, dtype=np.int64)
	validation = np.asarray(validation, dtype=np.int64)
	result_test = np.asarray(test, dtype=np.int64)

	train30 = stratified_subset(train40, labels, 0.75, args.seed)
	train20 = stratified_subset(train30, labels, 2.0 / 3.0, args.seed)
	train10 = stratified_subset(train20, labels, 0.50, args.seed)
	training_splits = {
		'TR40': train40,
		'TR30': train30,
		'TR20': train20,
		'TR10': train10,
	}

	if not set(train10).issubset(set(train20)):
		raise AssertionError('TR10 is not nested inside TR20.')
	if not set(train20).issubset(set(train30)):
		raise AssertionError('TR20 is not nested inside TR30.')
	if not set(train30).issubset(set(train40)):
		raise AssertionError('TR30 is not nested inside TR40.')

	os.makedirs(args.output_dir, exist_ok=True)
	manifest = {
		'dataset': args.dataset_name,
		'source_file': os.path.abspath(args.data_file),
		'split_seed': args.seed,
		'convention': 'nested stratified training subsets; fixed validation and test IDs',
		'splits': {},
	}
	for name, train_indices in training_splits.items():
		used = set(train_indices) | set(validation) | set(result_test)
		unused = np.asarray(sorted(set(eligible) - used), dtype=np.int64)
		filename = '{}_{}_seed{}.npz'.format(args.dataset_name, name, args.seed)
		path = os.path.join(args.output_dir, filename)
		np.savez_compressed(
			path,
			train_idx=train_indices,
			valid_idx=validation,
			test_idx=result_test,
			unused_idx=unused,
		)
		manifest['splits'][name] = {
			'file': filename,
			'train': split_stats(train_indices, labels),
			'validation': split_stats(validation, labels),
			'test': split_stats(result_test, labels),
			'unused_nodes': int(len(unused)),
		}

	manifest_path = os.path.join(
		args.output_dir,
		'{}_nested_split_manifest_seed{}.json'.format(args.dataset_name, args.seed))
	with open(manifest_path, 'w', encoding='utf-8') as handle:
		json.dump(manifest, handle, indent=2, sort_keys=True)
	print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
