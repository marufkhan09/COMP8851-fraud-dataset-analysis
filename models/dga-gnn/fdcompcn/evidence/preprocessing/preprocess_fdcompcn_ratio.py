import dgl
from dgl.data.utils import load_graphs, save_graphs
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '/workspace/COMP8851-fraud-dataset-analysis/models/dga-gnn/dga_gnn_upstream/code')
from myutils import describe, index_to_mask

RATIO = sys.argv[1]  # e.g. 'tr30'
RAW_PATH = '/workspace/COMP8851-fraud-dataset-analysis/models/dga-gnn/fdcompcn/evidence/preprocessing/../../../../../fdcompcn/data/raw/comp.dgl'
SPLITS_DIR = '/workspace/COMP8851-fraud-dataset-analysis/models/dga-gnn/fdcompcn/evidence/splits'
OUT_PATH = '/workspace/COMP8851-fraud-dataset-analysis/models/dga-gnn/dga_gnn_upstream/data/processed/fdcompcn.dgldata'

print(f'==== fdcompcn ({RATIO}, seed2, provisional) ====')

graphs, _ = load_graphs(RAW_PATH)
raw = graphs[0]

y = raw.ndata['label'].numpy()
num_nodes = raw.num_nodes()
X = raw.ndata['feature'].numpy()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

split = np.load(f'{SPLITS_DIR}/fdcompcn_{RATIO}_split_seed2.npz')
trn_idx, val_idx, tst_idx = split['train_idx'], split['valid_idx'], split['test_idx']

trn_msk = index_to_mask(torch.LongTensor(trn_idx), num_nodes)
val_msk = index_to_mask(torch.LongTensor(val_idx), num_nodes)
tst_msk = index_to_mask(torch.LongTensor(tst_idx), num_nodes)

def edges(etype):
    e = raw.edges(etype=etype)
    return e[0].numpy(), e[1].numpy()

src_i, dst_i = edges('invest_bc2bc')
src_p, dst_p = edges('provide_bc2bc')
src_s, dst_s = edges('sale_bc2bc')

graph_data = {
    ("company", "invest_bc2bc", "company"): (src_i, dst_i),
    ("company", "provide_bc2bc", "company"): (src_p, dst_p),
    ("company", "sale_bc2bc", "company"): (src_s, dst_s),
}
graph = dgl.heterograph(graph_data)
graph = dgl.to_bidirected(graph)
graph.create_formats_()
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk

save_graphs(OUT_PATH, graph)
describe(graph)
print(f'train={len(trn_idx)} valid={len(val_idx)} test={len(tst_idx)}')
