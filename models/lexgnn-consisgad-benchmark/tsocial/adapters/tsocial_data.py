from pathlib import Path
import dgl, numpy as np, torch
def load_tsocial_graph(dataset_path, normalize_for_lex=False):
    d=Path(dataset_path)
    x=np.load(d/'features.npy'); y=np.load(d/'labels.npy')
    s=np.load(d/'src.npy').astype(np.int64); t=np.load(d/'dst.npy').astype(np.int64)
    assert x.shape[0]==5781065 and y.shape==(5781065,) and len(s)==146211016
    g=dgl.heterograph({('tx','transaction','tx'):(torch.from_numpy(s),torch.from_numpy(t))},num_nodes_dict={'tx':5781065})
    g.ndata['feature']=torch.from_numpy(x.astype(np.float32)).contiguous()
    g.ndata['label']=torch.from_numpy(y.astype(np.int64)).contiguous()
    if normalize_for_lex:
        g=dgl.RowFeatNormalizer(subtract_min=True,node_feat_names=['feature'])(g)
    return g
def load_tsocial_split(split_path,node_count):
    with np.load(Path(split_path),allow_pickle=False) as z:
        r={k:np.asarray(z[k],dtype=np.int64) for k in ('train','validation','test','unused')}
    for k,v in r.items():
        assert len(np.unique(v))==len(v) and (len(v)==0 or (v.min()>=0 and v.max()<node_count)),k
    for a,b in (('train','validation'),('train','test'),('validation','test'),('unused','train'),('unused','validation'),('unused','test')):
        assert np.intersect1d(r[a],r[b]).size==0,(a,b)
    return r
