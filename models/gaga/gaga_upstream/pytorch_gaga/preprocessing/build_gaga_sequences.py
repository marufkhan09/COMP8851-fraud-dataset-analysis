import os
import sys
import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_custom_datasets import load_elliptic_dgl, load_fdcompcn_dgl

def save_gaga_sequence_format(dataset_name, adj, features, labels, trn_mask, val_mask, tst_mask):
    base_dir = f"/workspace/COMP8851-fraud-dataset-analysis/models/gaga/gaga_upstream/pytorch_gaga/preprocessing/seq_data/{dataset_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()
    
    num_nodes = adj.shape[0]
    train_nid = np.where(trn_mask)[0]
    val_nid = np.where(val_mask)[0]
    test_nid = np.where(tst_mask)[0]
    
    # Construct node sequence metadata objects expected by GAGA sequence loader
    infos_data = np.array([{"node": i} for i in range(num_nodes)], dtype=object)
    
    info_path = os.path.join(base_dir, f"{dataset_name}_infos_0.4_0.1_717.npz")
    np.savez(
        info_path,
        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=adj.shape,
        infos=infos_data,
        label=labels,
        labels=labels,
        train_nid=train_nid,
        val_nid=val_nid,
        test_nid=test_nid,
        train_mask=trn_mask,
        train_masks=trn_mask,
        val_mask=val_mask,
        val_masks=val_mask,
        test_mask=tst_mask,
        test_masks=tst_mask
    )
    
    feat_path = os.path.join(base_dir, f"{dataset_name}_no_grp_norm_norm_feat_2_0.4_0.1_717.npy")
    np.save(feat_path, features)
    
    print(f"[{dataset_name.upper()}] Sequence archive successfully generated with 'infos' key.")

if __name__ == "__main__":
    e_adj, e_feat, e_lbl, e_tr, e_va, e_te = load_elliptic_dgl()
    save_gaga_sequence_format("elliptic", e_adj, e_feat, e_lbl, e_tr, e_va, e_te)
    
    f_adj, f_feat, f_lbl, f_tr, f_va, f_te = load_fdcompcn_dgl()
    save_gaga_sequence_format("fdcompcn", f_adj, f_feat, f_lbl, f_tr, f_va, f_te)
