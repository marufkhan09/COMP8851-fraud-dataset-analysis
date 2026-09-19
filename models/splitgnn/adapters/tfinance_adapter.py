import dgl
import torch
import numpy as np

SRC = '/workspace/splitgnn_vast/adapters/tfinance_explore/dataset/tfinance'
OUT = '/workspace/splitgnn/data/tfinance.dgl'

g = dgl.load_graphs(SRC)[0][0]
src, dst = g.edges()

data_dict = {
    ('r', 'homo', 'r'): (src, dst),
    ('r', 'e', 'r'): (src, dst),
}
hg = dgl.heterograph(data_dict, num_nodes_dict={'r': g.num_nodes()})

hg.nodes['r'].data['feature'] = g.ndata['feature'].float()
label_onehot = g.ndata['label']
label = label_onehot[:, 1].long()
hg.nodes['r'].data['label'] = label

n = g.num_nodes()
rng = np.random.RandomState(2)
perm = rng.permutation(n)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:int(0.4*n)]] = True
valid_mask = torch.zeros(n, dtype=torch.bool); valid_mask[perm[int(0.4*n):int(0.6*n)]] = True
test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[int(0.6*n):]] = True
hg.nodes['r'].data['train_mask'] = train_mask
hg.nodes['r'].data['valid_mask'] = valid_mask
hg.nodes['r'].data['test_mask'] = test_mask

homo_src, homo_dst = hg.edges(etype='homo')
edge_labels = torch.where(label[homo_src] == label[homo_dst], torch.tensor(1, dtype=torch.long), torch.tensor(-1, dtype=torch.long))
edge_train_mask = train_mask[homo_src] & train_mask[homo_dst]
hg.edges['homo'].data['label'] = edge_labels
hg.edges['homo'].data['train_mask'] = edge_train_mask

dgl.save_graphs(OUT, [hg])
print('Saved:', OUT)
print('nodes:', hg.num_nodes(), 'edges(e):', hg.num_edges('e'), 'fraud:', int(label.sum()))
