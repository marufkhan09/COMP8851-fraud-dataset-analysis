import time, datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test_pcgnn, test_sage, load_data, pos_neg_split, normalize, pick_step
from src.model import PCALayer
from src.layers import InterAgg, IntraAgg
from src.graphsage import *
from src.experiment_io import (environment_metadata, synchronise_device,
							   utc_now_iso, write_csv, write_json)


"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class ModelHandler(object):

	def __init__(self, config):
		self.config = dict(config)
		args = argparse.Namespace(**config)
		# load graph, feature, and label
		load_started = time.perf_counter()
		[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data_name, prefix=args.data_dir)
		self.data_load_seconds = time.perf_counter() - load_started

		# train_test split
		np.random.seed(args.seed)
		random.seed(args.seed)
		if args.data_name == 'yelp':
			eligible_index = list(range(len(labels)))
		elif args.data_name == 'amazon':
			# The first 3,305 Amazon nodes are unlabelled in the official setup.
			eligible_index = list(range(3305, len(labels)))
		else:
			raise ValueError('Unsupported dataset: {}'.format(args.data_name))

		split_path = getattr(args, 'split_path', None)
		if split_path:
			idx_train, idx_valid, idx_test = self._load_persistent_split(
				split_path, eligible_index, len(labels))
			y_train = labels[np.asarray(idx_train)]
			y_valid = labels[np.asarray(idx_valid)]
			y_test = labels[np.asarray(idx_test)]
			self.split_source = os.path.abspath(split_path)
		else:
			self.split_source = 'author_random_stratified_random_state_2'
			if args.data_name == 'yelp':
				index = eligible_index
				idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
																	random_state=2, shuffle=True)
				idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																	random_state=2, shuffle=True)
			if args.data_name == 'amazon':
				index = eligible_index
				eligible_labels = labels[np.asarray(index)]
				idx_train, idx_rest, y_train, y_rest = train_test_split(index, eligible_labels, stratify=eligible_labels,
																	train_size=args.train_ratio, random_state=2, shuffle=True)
				idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
																	test_size=args.test_ratio, random_state=2, shuffle=True)

		print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
			f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
		print(f"Classification threshold: {args.thres}")
		print(f"Feature dimension: {feat_data.shape[1]}")


		# split pos neg sets for under-sampling
		train_pos, train_neg = pos_neg_split(idx_train, y_train)

		
		# if args.data == 'amazon':
		feat_data = normalize(feat_data)
		# train_feats = feat_data[np.array(idx_train)]
		# scaler = StandardScaler()
		# scaler.fit(train_feats)
		# feat_data = scaler.transform(feat_data)
		args.cuda = not args.no_cuda and torch.cuda.is_available()
		os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

		# set input graph
		if args.model == 'SAGE' or args.model == 'GCN':
			adj_lists = homo
		else:
			adj_lists = [relation1, relation2, relation3]

		print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')
		
		self.args = args
		self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
						'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
						'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
						'train_pos': train_pos, 'train_neg': train_neg}

	def _load_persistent_split(self, split_path, eligible_index, number_of_nodes):
		if not os.path.exists(split_path):
			raise FileNotFoundError('Persistent split file not found: {}'.format(split_path))
		with np.load(split_path, allow_pickle=False) as split_data:
			required = ('train_idx', 'valid_idx', 'test_idx')
			missing = [key for key in required if key not in split_data]
			if missing:
				raise ValueError('Split file is missing arrays: {}'.format(', '.join(missing)))
			idx_train = split_data['train_idx'].astype(np.int64).tolist()
			idx_valid = split_data['valid_idx'].astype(np.int64).tolist()
			idx_test = split_data['test_idx'].astype(np.int64).tolist()

		sets = [set(idx_train), set(idx_valid), set(idx_test)]
		if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
			raise ValueError('Train, validation, and test node IDs must be disjoint.')
		all_ids = sets[0] | sets[1] | sets[2]
		if any(node < 0 or node >= number_of_nodes for node in all_ids):
			raise ValueError('Split file contains a node ID outside the dataset range.')
		if not all_ids.issubset(set(eligible_index)):
			raise ValueError('Split file contains an unlabelled or ineligible node ID.')
		if not idx_train or not idx_valid or not idx_test:
			raise ValueError('Train, validation, and test splits must all be non-empty.')
		return idx_train, idx_valid, idx_test


	def train(self):
		args = self.args
		run_started_utc = utc_now_iso()
		run_started = time.perf_counter()
		feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
		idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset['idx_test'], self.dataset['y_test']
		# initialize model input
		features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
		features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
		if args.cuda:
			features.cuda()

		# build one-layer models
		if args.model == 'PCGNN':
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], 
							  adj_lists, [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'SAGE':
			agg_sage = MeanAggregator(features, cuda=args.cuda)
			enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False, cuda=args.cuda)
		elif args.model == 'GCN':
			agg_gcn = GCNAggregator(features, cuda=args.cuda)
			enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True, cuda=args.cuda)

		if args.model == 'PCGNN':
			gnn_model = PCALayer(2, inter1, args.alpha)
		elif args.model == 'SAGE':
			# the vanilla GraphSAGE model as baseline
			enc_sage.num_samples = 5
			gnn_model = GraphSage(2, enc_sage)
		elif args.model == 'GCN':
			gnn_model = GCN(2, enc_gcn)

		if args.cuda:
			gnn_model.cuda()

		if str(args.optimizer).lower() != 'adam':
			raise ValueError('This instrumented PC-GNN package preserves the paper optimizer: Adam.')
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()),
									 lr=args.lr, weight_decay=args.weight_decay)

		timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
		run_mode = str(getattr(args, 'run_mode', 'author_reproduction'))
		safe_run_mode = ''.join(char if char.isalnum() or char in '-_' else '-' for char in run_mode)
		run_id = '{}_{}_{}_{}_seed{}'.format(timestamp, args.data_name, args.model, safe_run_mode, args.seed)
		results_dir = getattr(args, 'results_dir', './results')
		run_dir = os.path.join(results_dir, run_id)
		checkpoint_dir = os.path.join(run_dir, 'checkpoints')
		os.makedirs(checkpoint_dir, exist_ok=True)
		path_saver = os.path.join(checkpoint_dir, '{}_{}.pkl'.format(args.data_name, args.model))

		commit_hash = None
		if os.path.exists('pcgnn_commit.txt'):
			with open('pcgnn_commit.txt', 'r', encoding='utf-8') as commit_file:
				commit_hash = commit_file.read().strip() or None
		run_config = dict(self.config)
		run_config.update({
			'resolved_cuda': bool(args.cuda),
			'resolved_run_id': run_id,
			'resolved_split_source': self.split_source,
			'author_repository_commit': commit_hash,
		})
		write_json(os.path.join(run_dir, 'run_config.json'), run_config)

		eligible_ids = set(range(len(self.dataset['labels'])))
		if args.data_name == 'amazon':
			eligible_ids = set(range(3305, len(self.dataset['labels'])))
		used_ids = set(idx_train) | set(idx_valid) | set(idx_test)
		unused_idx = sorted(eligible_ids - used_ids)
		np.savez_compressed(
			os.path.join(run_dir, 'split_indices.npz'),
			train_idx=np.asarray(idx_train, dtype=np.int64),
			valid_idx=np.asarray(idx_valid, dtype=np.int64),
			test_idx=np.asarray(idx_test, dtype=np.int64),
			unused_idx=np.asarray(unused_idx, dtype=np.int64),
		)

		def split_record(indices):
			index_array = np.asarray(indices, dtype=np.int64)
			split_labels = self.dataset['labels'][index_array]
			return {
				'nodes': int(len(index_array)),
				'fraud_nodes': int(np.sum(split_labels == 1)),
				'normal_nodes': int(np.sum(split_labels == 0)),
			}

		split_summary = {
			'source': self.split_source,
			'train': split_record(idx_train),
			'validation': split_record(idx_valid),
			'test': split_record(idx_test),
			'unused_nodes': int(len(unused_idx)),
		}
		write_json(os.path.join(run_dir, 'split_summary.json'), split_summary)

		f1_mac_best, auc_best, ep_best = 0.0, float('-inf'), -1
		epoch_rows = []
		epoch_fields = [
			'epoch', 'train_start_utc', 'train_end_utc', 'train_seconds',
			'validation_seconds', 'mean_batch_loss', 'validation_macro_f1',
			'validation_fraud_f1', 'validation_auroc', 'validation_auprc',
			'validation_gmean', 'validation_fraud_precision',
			'validation_fraud_recall', 'is_best_checkpoint'
		]
		if args.cuda:
			torch.cuda.reset_peak_memory_stats()

		fit_started = time.perf_counter()
		for epoch in range(args.num_epochs):
			synchronise_device(args.cuda)
			epoch_started_utc = utc_now_iso()
			epoch_started = time.perf_counter()

			sampled_idx_train = pick_step(
				idx_train, y_train, self.dataset['homo'],
				size=len(self.dataset['train_pos']) * 2)
			random.shuffle(sampled_idx_train)
			num_batches = max(1, int(np.ceil(len(sampled_idx_train) / args.batch_size)))
			epoch_loss_total = 0.0

			for batch in range(num_batches):
				i_start = batch * args.batch_size
				i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
				batch_nodes = sampled_idx_train[i_start:i_end]
				if not batch_nodes:
					continue
				batch_label = self.dataset['labels'][np.asarray(batch_nodes)]
				optimizer.zero_grad()
				if args.cuda:
					batch_loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
				else:
					batch_loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
				batch_loss.backward()
				optimizer.step()
				epoch_loss_total += float(batch_loss.item())

			synchronise_device(args.cuda)
			epoch_train_seconds = time.perf_counter() - epoch_started
			epoch_ended_utc = utc_now_iso()
			mean_batch_loss = epoch_loss_total / num_batches
			print(f'Epoch: {epoch}, loss: {mean_batch_loss}, time: {epoch_train_seconds}s')

			validation_seconds = 0.0
			validation_metrics = None
			is_best = False
			if epoch % args.valid_epochs == 0:
				print("Valid at epoch {}".format(epoch))
				synchronise_device(args.cuda)
				validation_started = time.perf_counter()
				if args.model == 'SAGE' or args.model == 'GCN':
					validation_metrics = test_sage(
						idx_valid, y_valid, gnn_model, args.batch_size, args.thres,
						return_details=True)
				else:
					validation_metrics = test_pcgnn(
						idx_valid, y_valid, gnn_model, args.batch_size, args.thres,
						return_details=True)
				synchronise_device(args.cuda)
				validation_seconds = time.perf_counter() - validation_started
				if validation_metrics['auroc'] > auc_best:
					f1_mac_best = validation_metrics['macro_f1']
					auc_best = validation_metrics['auroc']
					ep_best = epoch
					is_best = True
					print('  Saving model ...')
					torch.save(gnn_model.state_dict(), path_saver)

			row = {
				'epoch': epoch,
				'train_start_utc': epoch_started_utc,
				'train_end_utc': epoch_ended_utc,
				'train_seconds': epoch_train_seconds,
				'validation_seconds': validation_seconds,
				'mean_batch_loss': mean_batch_loss,
				'is_best_checkpoint': is_best,
			}
			if validation_metrics:
				for metric in ('macro_f1', 'fraud_f1', 'auroc', 'auprc', 'gmean',
							   'fraud_precision', 'fraud_recall'):
					row['validation_' + metric] = validation_metrics[metric]
			epoch_rows.append(row)
			write_csv(os.path.join(run_dir, 'epoch_times.csv'), epoch_rows, epoch_fields)

		fit_wall_seconds = time.perf_counter() - fit_started
		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		map_location = None if args.cuda else 'cpu'
		gnn_model.load_state_dict(torch.load(path_saver, map_location=map_location))

		synchronise_device(args.cuda)
		evaluation_started = time.perf_counter()
		if args.model == 'SAGE' or args.model == 'GCN':
			test_metrics = test_sage(
				idx_test, y_test, gnn_model, args.batch_size, args.thres,
				return_details=True)
		else:
			test_metrics = test_pcgnn(
				idx_test, y_test, gnn_model, args.batch_size, args.thres,
				return_details=True)
		synchronise_device(args.cuda)
		evaluation_seconds = time.perf_counter() - evaluation_started

		train_times = np.asarray([row['train_seconds'] for row in epoch_rows], dtype=float)
		peak_gpu_memory_mb = None
		if args.cuda:
			peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
		summary = {
			'schema_version': '1.0',
			'run_id': run_id,
			'run_mode': run_mode,
			'model': args.model,
			'dataset': args.data_name,
			'seed': int(args.seed),
			'run_started_utc': run_started_utc,
			'run_finished_utc': utc_now_iso(),
			'best_validation_epoch': int(ep_best),
			'best_validation_auroc': float(auc_best),
			'best_validation_macro_f1': float(f1_mac_best),
			'split': split_summary,
			'test_metrics': test_metrics,
			'timing': {
				'data_load_seconds': float(self.data_load_seconds),
				'training_epochs': int(len(train_times)),
				'total_train_epoch_seconds': float(np.sum(train_times)),
				'mean_train_epoch_seconds': float(np.mean(train_times)),
				'median_train_epoch_seconds': float(np.median(train_times)),
				'std_train_epoch_seconds': float(np.std(train_times)),
				'fit_wall_seconds_including_validation': float(fit_wall_seconds),
				'total_validation_seconds': float(sum(row['validation_seconds'] for row in epoch_rows)),
				'test_inference_seconds': float(test_metrics['inference_seconds']),
				'test_evaluation_seconds_including_metrics': float(evaluation_seconds),
				'peak_gpu_memory_mb': peak_gpu_memory_mb,
				'total_run_seconds': float(time.perf_counter() - run_started),
			},
			'environment': environment_metadata(args.cuda),
			'configuration': run_config,
		}
		write_json(os.path.join(run_dir, 'summary.json'), summary)
		write_csv(os.path.join(run_dir, 'test_metrics.csv'), [test_metrics], list(test_metrics.keys()))
		print('Structured run artifacts: {}'.format(run_dir))
		print('Average training epoch time: {:.6f}s'.format(np.mean(train_times)))

		return (test_metrics['macro_f1'], test_metrics['fraud_f1'],
				test_metrics['normal_f1'], test_metrics['auroc'], test_metrics['gmean'])
