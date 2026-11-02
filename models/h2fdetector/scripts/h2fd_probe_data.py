#!/usr/bin/env python
"""Read-only probe: prints the structure of the remaining datasets and the parts of
H2-FDetector's model.py that depend on the dataset. CPU only, writes nothing except
a temporary unzip of FDCompCN under /workspace/h2fd_vast/tmp_probe."""
import os
import subprocess
import zipfile

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import dgl

DATA = '/workspace/data'
TMP = '/workspace/h2fd_vast/tmp_probe'


def describe_graph(name, g):
    print(f'\n===== {name} =====')
    print('ntypes:', g.ntypes)
    print('canonical etypes:', g.canonical_etypes)
    for nt in g.ntypes:
        print(f'  nodes[{nt}] = {g.num_nodes(nt)}')
        for k, v in g.nodes[nt].data.items():
            extra = ''
            if v.dtype in (torch.int64, torch.int32, torch.uint8, torch.bool) and v.dim() <= 2:
                u, c = torch.unique(v, return_counts=True)
                if len(u) <= 10:
                    extra = f' values={dict(zip(u.tolist(), c.tolist()))}'
            print(f'    ndata[{k}] shape={tuple(v.shape)} dtype={v.dtype}{extra}')
    for et in g.canonical_etypes:
        print(f'  edges{et} = {g.num_edges(et)}  edata keys={list(g.edges[et].data.keys())}')


def probe_dgl_file(name, path):
    try:
        gs, extra = dgl.load_graphs(path)
        print(f'\n[{name}] {len(gs)} graph(s) in file, extra keys={list(extra.keys())}')
        describe_graph(name, gs[0])
    except Exception as e:  # noqa: BLE001
        print(f'\n[{name}] dgl.load_graphs failed: {e!r}')


def probe_mat(name, path):
    print(f'\n===== {name} (.mat) =====')
    m = sio.loadmat(path)
    for k, v in m.items():
        if k.startswith('__'):
            continue
        if sp.issparse(v):
            print(f'  {k}: sparse {v.shape} nnz={v.nnz} format={v.format}')
        elif isinstance(v, np.ndarray):
            extra = ''
            if v.size and np.issubdtype(v.dtype, np.number):
                u = np.unique(v)
                if len(u) <= 10:
                    extra = f' values={dict(zip(*np.unique(v, return_counts=True)))}'
            print(f'  {k}: ndarray {v.shape} dtype={v.dtype}{extra}')
        else:
            print(f'  {k}: {type(v)}')


def main():
    probe_dgl_file('tfinance', f'{DATA}/tfinance')

    os.makedirs(TMP, exist_ok=True)
    with zipfile.ZipFile(f'{DATA}/FDCompCN.zip') as z:
        z.extractall(TMP)
    probe_dgl_file('fdcompcn (comp.dgl)', f'{TMP}/comp.dgl')

    probe_mat('elliptic', f'{DATA}/elliptic_bitcoin_dataset/elliptic_fraudre.mat')

    print('\n===== H2-FDetector model.py: dataset-dependent code =====')
    print(subprocess.run(['grep', '-n', '-E', "dataset|etype|'r'|ntype|relation|homo|amazon|yelp",
                          '/workspace/H2-FDetector/src/model.py'],
                         capture_output=True, text=True).stdout)
    print('----- model.py lines 80-110 and 158-195 -----')
    with open('/workspace/H2-FDetector/src/model.py') as f:
        lines = f.readlines()
    for i in list(range(79, 110)) + list(range(157, 195)):
        if i < len(lines):
            print(f'{i + 1:4d}: {lines[i]}', end='')

    print('\n\n===== SplitGNN results available for cross-checking splits =====')
    root = '/workspace/COMP8851-fraud-dataset-analysis/results/unified/splitgnn_results'
    if os.path.isdir(root):
        print(sorted(os.listdir(root)))
    print('\nNOTE: tsocial not loaded here (5.6 GB); it is probed separately later.')


if __name__ == '__main__':
    main()
