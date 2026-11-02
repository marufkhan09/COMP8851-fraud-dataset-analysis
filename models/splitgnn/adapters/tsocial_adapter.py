import dgl
import torch
import numpy as np

SRC = '/workspace/splitgnn_vast/adapters/tfinance_explore/dataset/tsocial'
OUT = '/workspace/splitgnn/data/tsocial.dgl'
SAMPLE_FRAC = 0.15

g_full = dgl.load_graphs(SRC)[0][0]
label_full = g_full.ndata['label']
n_full = g_full.num_nodes()

rng = np.random.RandomState(2)
sampled_idx = []
for c in torch.unique(label_full):
    class_idx = torch.where(label_full == c)[0].numpy()
    n_take = int(round(SAMPLE_FRAC * len(class_idx)))
    chosen = rng.choice(class_idx, size=n_take, replace=False)
    sampled_idx.append(chosen)
sampled_idx = np.concatenate(sampled_idx)
sampled_idx.sort()
sampled_idx_t = torch.from_numpy(sampled_idx)

print(f"Sampling {len(sampled_idx)} of {n_full} nodes ({100*len(sampled_idx)/n_full:.1f}%)")

sub_g = dgl.node_subgraph(g_full, sampled_idx_t)
n = sub_g.num_nodes()
print(f"Induced subgraph: {n} nodes, {sub_g.num_edges()} edges")

feat = sub_g.ndata['feature'].float()
label = sub_g.ndata['label'].long()
src, dst = sub_g.edges()

data_dict = {
    ('r', 'homo', 'r'): (src, dst),
    ('r', 'e', 'r'): (src, dst),
}
hg = dgl.heterograph(data_dict, num_nodes_dict={'r': n})
hg.nodes['r'].data['feature'] = feat
hg.nodes['r'].data['label'] = label

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
print('nodes:', hg.num_nodes(), 'edges(e):', hg.num_edges('e'), 'fraud:', int((label==1).sum()), f'({100*(label==1).sum().item()/n:.2f}%)')
