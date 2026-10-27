"""Canonical YelpChi inputs, graph topology and split isolation for COMP8851 RGTAN.
No training, tuning or test scoring in this module.
"""
from pathlib import Path
import hashlib
import subprocess
import time

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

ROOT = Path('/workspace/comp8851')
SRC = ROOT / 'repos/antifraud'
MAT = ROOT / 'assets/yelpchi/source/YelpChi.mat'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/YelpChi/yelp_seed2_nested_splits.npz')
AUTHOR_COMMIT = '65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SOURCE_SHA = 'fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42'
SPLIT_SHA = '0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b'
N, NF, EXPECTED_EDGES = 45954, 32, 7739912
RATIOS = ('TR40', 'TR30', 'TR20', 'TR10')
EXPECTED_TRAIN = {'TR40':18381,'TR30':13785,'TR20':9190,'TR10':4595}
EXPECTED_POS = {'TR40':2671,'TR30':2003,'TR20':1335,'TR10':668}
RELATIONS = {'net_rur':98630,'net_rsr':6805486,'net_rtr':1147232}
ASSET_DIR = ROOT/'assets/rgtan/yelpchi'
OUT = ROOT/'results/rgtan/yelpchi_tr40_tuning_v1'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_input_files():
    assert MAT.is_file() and SPLIT.is_file(), 'canonical source or frozen split missing'
    assert sha256(MAT)==SOURCE_SHA, 'YelpChi source SHA256 mismatch'
    assert sha256(SPLIT)==SPLIT_SHA, 'Frozen split SHA256 mismatch'
    head=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
    assert head==AUTHOR_COMMIT, ('RGTAN pinned upstream commit mismatch',head)
    return True

def load_splits():
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(z['seed'].item())==2
        assert int(z['source_nodes'].item())==N
        parts={k:np.asarray(z[k],dtype=np.int64).reshape(-1).copy() for k in (*RATIOS,'val','test')}
    assert len(parts['val'])==9191 and len(parts['test'])==18382
    for name,ids in parts.items():
        assert ids.size==np.unique(ids).size, ('repeated node',name)
        assert ids.size and ids.min()>=0 and ids.max()<N, ('invalid id',name)
    assert np.intersect1d(parts['val'],parts['test']).size==0
    for ratio in RATIOS:
        ids=parts[ratio]
        assert len(ids)==EXPECTED_TRAIN[ratio]
        assert not np.intersect1d(ids,parts['val']).size
        assert not np.intersect1d(ids,parts['test']).size
        assert np.isin(ids,parts['TR40']).all()
    for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
        assert np.isin(parts[a],parts[b]).all()
    assert len(np.unique(np.concatenate([parts['TR40'],parts['val'],parts['test']])))==N
    return parts

def get_graph_and_features(with_labels=False):
    """Reconstruct homogeneous Yelp relation union, undirected with self loops.

    Three exact MAT relations are retained as the edge source. No labels enter
    graph construction. Return SciPy CSR for risk feature computation and DGL.
    """
    requested=['features',*RELATIONS]
    if with_labels:
        requested.append('label')
    raw=loadmat(MAT,variable_names=requested)
    mat=raw['features']
    f=mat.toarray().astype(np.float32) if sp.issparse(mat) else np.asarray(mat,dtype=np.float32)
    assert f.shape==(N,NF) and np.isfinite(f).all()
    adjacency=None
    for rel,expected in RELATIONS.items():
        x=sp.csr_matrix(raw[rel]).astype(np.int8)
        assert x.shape==(N,N) and x.nnz==expected, (rel,x.nnz,expected)
        x.data[:]=1
        adjacency=x if adjacency is None else adjacency.maximum(x).tocsr()
    del raw
    adjacency=adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(1)
    adjacency.sum_duplicates()
    adjacency.data[:]=1
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    assert adjacency.shape==(N,N) and adjacency.nnz==EXPECTED_EDGES, (adjacency.shape,adjacency.nnz)
    assert (adjacency!=adjacency.T).nnz==0
    assert np.all(adjacency.diagonal()==1)
    assert set(np.unique(adjacency.data).tolist())=={1}
    return adjacency, f

def assert_environment():
    import torch,dgl
    assert torch.__version__=='1.12.1+cu113',torch.__version__
    assert dgl.__version__=='0.8.1',dgl.__version__
    assert torch.cuda.is_available()
    return True
