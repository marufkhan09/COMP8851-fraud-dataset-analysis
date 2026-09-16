"""
Single-process replacement for graph2seq_mp.py, avoiding the fork-after-thread-init
deadlock observed with mp.Process on this instance (DGL/PyTorch background threads
don't survive fork() cleanly). Produces the identical output file/format.
"""
import argparse
import os
import numpy as np
import torch
import time
import data_utils

parser = argparse.ArgumentParser(description='graph2seq_single')
parser.add_argument('--dataset', type=str, default='amazon')
parser.add_argument('--train_size', type=float, default=0.4)
parser.add_argument('--val_size', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=717)
parser.add_argument('--norm_feat', action='store_true', default=False)
parser.add_argument('--grp_norm', action='store_true', default=False)
parser.add_argument('--force_reload', action='store_true', default=False)
parser.add_argument('--add_self_loop', action='store_true', default=False)
parser.add_argument('--fanouts', type=int, default=[-1], nargs='+')
parser.add_argument('--base_dir', type=str, default='~/.dgl')
parser.add_argument('--save_dir', type=str, default='mp_output')
args = vars(parser.parse_args())
print(args)

graph_data = data_utils.prepare_data(args, add_self_loop=args['add_self_loop'])

g = graph_data.graph
n_classes = graph_data.n_classes
feat_dim = graph_data.feat_dim
n_relations = graph_data.n_relations
n_groups = n_classes + 1
n_hops = len(args['fanouts'])
n_nodes = g.num_nodes()

seq_len = n_relations * (n_hops * n_groups + 1)

file_dir = os.path.join(args['save_dir'], args['dataset'])
os.makedirs(file_dir, exist_ok=True)

flag_1 = 'grp_norm' if args['grp_norm'] else 'no_grp_norm'
flag_2 = 'norm_feat' if args['norm_feat'] else 'no_norm_feat'
file_name = f"{args['dataset']}_{flag_1}_{flag_2}_{n_hops}_" \
            f"{args['train_size']}_{args['val_size']}_{args['seed']}.npy"
seq_file = os.path.join(file_dir, file_name)
print(f"Saving seq_file to {seq_file}")
sequence_array = np.memmap(seq_file, dtype=np.float32, mode='w+', shape=(n_nodes, seq_len, feat_dim))

loader = data_utils.GroupFeatureSequenceLoader(graph_data, fanouts=args['fanouts'], grp_norm=args['grp_norm'])

all_nid = g.nodes()
tic = time.time()

# Process in chunks to get progress feedback without per-node overhead
CHUNK = 2000
for st in range(0, n_nodes, CHUNK):
    ed = min(st + CHUNK, n_nodes)
    chunk_nids = all_nid[st:ed]
    seq_feat = loader.load_batch(chunk_nids)
    sequence_array[st:ed] = seq_feat.numpy()
    elapsed = time.time() - tic
    rate = ed / elapsed
    eta = (n_nodes - ed) / rate if rate > 0 else 0
    print(f"[{ed}/{n_nodes}] elapsed={elapsed:.1f}s rate={rate:.1f} nodes/s ETA={eta:.1f}s")

sequence_array.flush()
toc = time.time()
print(f"Elapsed Time = {toc - tic:.2f}(s)")
