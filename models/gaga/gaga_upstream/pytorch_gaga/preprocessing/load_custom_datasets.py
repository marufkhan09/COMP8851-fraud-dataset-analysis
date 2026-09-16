import dgl
import torch
import numpy as np
import scipy.sparse as sp

def load_elliptic_dgl():
    path = "/workspace/COMP8851-fraud-dataset-analysis/models/dga-gnn/dga_gnn_upstream/data/processed/elliptic_of_amnet.dgldata"
    graphs, _ = dgl.load_graphs(path)
    g = graphs[0]
    
    features = g.ndata['feat'].numpy()
    labels = g.ndata['label'].numpy()
    
    src, dst = g.edges()
    adj = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(g.num_nodes(), g.num_nodes()))
    
    trn_mask = g.ndata['trn_msk'].numpy().astype(bool)
    val_mask = g.ndata['val_msk'].numpy().astype(bool)
    tst_mask = g.ndata['tst_msk'].numpy().astype(bool)
    
    return adj, features, labels, trn_mask, val_mask, tst_mask

def load_fdcompcn_dgl():
    path = "/workspace/COMP8851-fraud-dataset-analysis/models/dga-gnn/dga_gnn_upstream/data/processed/fdcompcn.dgldata"
    graphs, _ = dgl.load_graphs(path)
    g = graphs[0]
    
    features = g.ndata['feat'].numpy()
    labels = g.ndata['label'].numpy()
    
    src_list, dst_list = [], []
    for etype in g.canonical_etypes:
        u, v = g.edges(etype=etype)
        src_list.append(u.numpy())
        dst_list.append(v.numpy())
        
    src_all = np.concatenate(src_list)
    dst_all = np.concatenate(dst_list)
    adj = sp.coo_matrix((np.ones(len(src_all)), (src_all, dst_all)), shape=(g.num_nodes(), g.num_nodes()))
    
    trn_mask = g.ndata['trn_msk'].numpy().astype(bool)
    val_mask = g.ndata['val_msk'].numpy().astype(bool)
    tst_mask = g.ndata['tst_msk'].numpy().astype(bool)
    
    return adj, features, labels, trn_mask, val_mask, tst_mask

if __name__ == "__main__":
    e_adj, e_feat, e_lbl, e_tr, e_va, e_te = load_elliptic_dgl()
    print(f"Elliptic Adapter Success -> Nodes: {e_adj.shape[0]}, Features: {e_feat.shape[1]}, Edges: {e_adj.nnz}")
    
    f_adj, f_feat, f_lbl, f_tr, f_va, f_te = load_fdcompcn_dgl()
    print(f"FDCompCN Adapter Success -> Nodes: {f_adj.shape[0]}, Features: {f_feat.shape[1]}, Union Edges: {f_adj.nnz}")
