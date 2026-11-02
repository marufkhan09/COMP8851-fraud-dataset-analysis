import dgl
import torch
import numpy as np
import pandas as pd

DATA_DIR = '/workspace/splitgnn_vast/adapters/elliptic_bitcoin_dataset'
OUT = '/workspace/splitgnn/data/elliptic.dgl'

classes = pd.read_csv(f'{DATA_DIR}/elliptic_txs_classes.csv')
edges_df = pd.read_csv(f'{DATA_DIR}/elliptic_txs_edgelist.csv')
features_df = pd.read_csv(f'{DATA_DIR}/elliptic_txs_features.csv', header=None)

tx_ids = features_df[0].values
txid_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
n = len(tx_ids)

feat = torch.from_numpy(features_df.iloc[:, 2:].values.astype(np.float32))

classes_indexed = classes.set_index('txId')
label = np.full(n, -1, dtype=np.int64)
for tx, idx in txid_to_idx.items():
    c = classes_indexed.loc[tx, 'class']
    if c == '1':
        label[idx] = 1
    elif c == '2':
        label[idx] = 0
label = torch.from_numpy(label)

src_ids = edges_df['txId1'].map(txid_to_idx)
dst_ids = edges_df['txId2'].map(txid_to_idx)
valid_edge_mask = src_ids.notna() & dst_ids.notna()
src = torch.from_numpy(src_ids[valid_edge_mask].values.astype(np.int64))
dst = torch.from_numpy(dst_ids[valid_edge_mask].values.astype(np.int64))

data_dict = {('r', 'homo', 'r'): (src, dst), ('r', 'e', 'r'): (src, dst)}
hg = dgl.heterograph(data_dict, num_nodes_dict={'r': n})
hg.nodes['r'].data['feature'] = feat
hg.nodes['r'].data['label'] = label

labeled_idx = np.where(label.numpy() != -1)[0]
rng = np.random.RandomState(2)
perm = rng.permutation(labeled_idx)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:int(0.4*len(perm))]] = True
valid_mask = torch.zeros(n, dtype=torch.bool); valid_mask[perm[int(0.4*len(perm)):int(0.6*len(perm))]] = True
test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[int(0.6*len(perm)):]] = True
hg.nodes['r'].data['train_mask'] = train_mask
hg.nodes['r'].data['valid_mask'] = valid_mask
hg.nodes['r'].data['test_mask'] = test_mask

homo_src, homo_dst = hg.edges(etype='homo')
same_label = (label[homo_src] == label[homo_dst]) & (label[homo_src] != -1)
edge_labels = torch.where(same_label, torch.tensor(1, dtype=torch.long), torch.tensor(-1, dtype=torch.long))
edge_train_mask = train_mask[homo_src] & train_mask[homo_dst]
hg.edges['homo'].data['label'] = edge_labels
hg.edges['homo'].data['train_mask'] = edge_train_mask

dgl.save_graphs(OUT, [hg])
print('Saved:', OUT)
print('nodes:', hg.num_nodes(), 'labeled:', len(labeled_idx), 'fraud:', int((label==1).sum()))
