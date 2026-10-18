#!/usr/bin/env python
"""
H2-FDetector dataset-view builder (adapter: data translation only).

Input:  the native graph produced by the authors' own data_preprocess.py
        (e.g. /workspace/H2-FDetector/data/yelp.dgl).
Output: one DGL view per canonical training ratio (TR40/30/20/10) in which
          * node train/valid/test masks are replaced by the group's canonical
            nested splits (generate_splits.py, split seed 2, imported unchanged)
          * the 'homo' edge labels and edge train mask are recomputed for that
            ratio's training set with a vectorised equivalent of the authors'
            generate_edges_labels()
        Graph structure, node features, node labels and the model are NOT changed.

Safety checks (the script aborts before writing anything if one fails):
  1. Vectorised edge labelling is proven bit-identical to the authors' function
     by recomputing the native graph's stored edge labels / edge train mask
     from its own native train mask.
  2. verify_splits(): nesting TR10 c TR20 c TR30 c TR40, no train/valid/test overlap.
  3. If SplitGNN's split_indices.npz exists for the same dataset/ratio, the masks
     are compared array-for-array (file hashes are not comparable because np.savez
     embeds a timestamp).

Usage:
  python h2fd_build_views.py --dataset yelp
  python h2fd_build_views.py --dataset amazon
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import dgl

RATIOS = [40, 30, 20, 10]
SOURCE_MAT = {'yelp': 'YelpChi.mat', 'amazon': 'Amazon.mat'}
BUILDER_VERSION = 'h2fd_view_builder_v1'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mask_content_sha256(masks):
    """Timestamp-independent hash of a dict of boolean masks."""
    h = hashlib.sha256()
    for k in sorted(masks):
        a = np.ascontiguousarray(np.asarray(masks[k], dtype=bool))
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(np.packbits(a).tobytes())
    return h.hexdigest()


def edge_labels_vectorised(src, dst, labels, train_mask):
    """Vectorised equivalent of the authors' generate_edges_labels():
       label = 1 if both endpoints share a node label else -1;
       edge train mask = both endpoints in the training set."""
    same = labels[src] == labels[dst]
    edge_labels = torch.where(same, torch.ones_like(src), -torch.ones_like(src)).long()
    edge_train_mask = (train_mask[src] & train_mask[dst]).bool()
    return edge_labels, edge_train_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=sorted(SOURCE_MAT))
    p.add_argument('--h2fd-root', default='/workspace/H2-FDetector')
    p.add_argument('--splits-dir',
                   default='/workspace/COMP8851-fraud-dataset-analysis/models/splitgnn/splits')
    p.add_argument('--source-dir', default='/workspace/data')
    p.add_argument('--out-dir', default='/workspace/h2fd_vast/views')
    p.add_argument('--splitgnn-ref',
                   default='/workspace/COMP8851-fraud-dataset-analysis/results/unified/splitgnn_results')
    p.add_argument('--force', action='store_true', help='rebuild views that already exist')
    a = p.parse_args()

    sys.path.insert(0, a.splits_dir)
    from generate_splits import generate_splits, verify_splits, SPLIT_SEED

    ds = a.dataset
    native_path = f'{a.h2fd_root}/data/{ds}.dgl'
    if not os.path.exists(native_path):
        sys.exit(f'Native graph not found: {native_path} (run the authors\' data_preprocess.py first)')
    os.makedirs(a.out_dir, exist_ok=True)

    print(f'[{ds}] loading native graph {native_path}')
    g = dgl.load_graphs(native_path)[0][0]
    assert len(g.ntypes) == 1, f'expected a single node type, got {g.ntypes}'
    nt = g.ntypes[0]
    labels = g.nodes[nt].data['label'].long()
    src, dst = g.edges(etype='homo')
    print(f'[{ds}] node type={nt} nodes={g.num_nodes(nt)} '
          + ' '.join(f'{et}={g.num_edges(et)}' for et in g.etypes))

    # ---- Check 1: vectorised edge labelling == authors' function on the native split
    t0 = time.time()
    native_tm = g.nodes[nt].data['train_mask'].bool()
    el, em = edge_labels_vectorised(src, dst, labels, native_tm)
    stored_l = g.edges['homo'].data['label'].long()
    stored_m = g.edges['homo'].data['train_mask'].bool()
    ok_l, ok_m = torch.equal(el, stored_l), torch.equal(em, stored_m)
    verification = {
        'method': 'recompute native edge labels/train mask from native node train mask',
        'n_edges_homo': int(src.numel()),
        'edge_labels_identical': bool(ok_l),
        'edge_train_mask_identical': bool(ok_m),
        'n_label_mismatch': int((el != stored_l).sum()),
        'n_mask_mismatch': int((em != stored_m).sum()),
        'seconds': round(time.time() - t0, 3),
    }
    print(f'[{ds}] edge-label equivalence check: {verification}')
    if not (ok_l and ok_m):
        sys.exit(f'[{ds}] ABORT: vectorised edge labelling does not match the authors\' output')

    # ---- Check 2: canonical splits
    masks = generate_splits(labels.cpu())
    verify_splits(masks, labels.cpu())

    src_mat = f'{a.source_dir}/{SOURCE_MAT[ds]}'
    manifest = {
        'dataset': ds,
        'builder_version': BUILDER_VERSION,
        'builder_sha256': sha256_file(os.path.abspath(__file__)),
        'native_dgl_path': native_path,
        'native_dgl_sha256': sha256_file(native_path),
        'source_mat_path': src_mat,
        'source_mat_sha256': sha256_file(src_mat) if os.path.exists(src_mat) else 'NOT_FOUND',
        'node_type': nt,
        'n_nodes': int(g.num_nodes(nt)),
        'n_edges': {et: int(g.num_edges(et)) for et in g.etypes},
        'split_seed': int(SPLIT_SEED),
        'split_registry_version': 'own_generate_splits_v1_seed2',
        'edge_label_verification': verification,
        'views': {},
    }

    valid_mask = masks['valid_mask'].bool()
    test_mask = masks['test_mask'].bool()

    for r in RATIOS:
        view_path = f'{a.out_dir}/{ds}_tr{r}.dgl'
        split_path = f'{a.out_dir}/{ds}_tr{r}_split.npz'
        train_mask = masks[f'train_mask_tr{r}'].bool()
        split_np = {'train_mask': train_mask.numpy(),
                    'valid_mask': valid_mask.numpy(),
                    'test_mask': test_mask.numpy()}

        if os.path.exists(view_path) and os.path.exists(split_path) and not a.force:
            print(f'[{ds}] TR{r}: exists, skipping build (use --force to rebuild)')
        else:
            g.nodes[nt].data['train_mask'] = train_mask
            g.nodes[nt].data['valid_mask'] = valid_mask
            g.nodes[nt].data['test_mask'] = test_mask
            el, em = edge_labels_vectorised(src, dst, labels, train_mask)
            g.edges['homo'].data['label'] = el
            g.edges['homo'].data['train_mask'] = em
            dgl.save_graphs(view_path, g)
            np.savez(split_path, **split_np)
            print(f'[{ds}] TR{r}: wrote {view_path}')

        # re-read what is on disk so the manifest describes the saved files
        saved = np.load(split_path)
        saved_np = {k: saved[k].astype(bool) for k in ('train_mask', 'valid_mask', 'test_mask')}
        assert all(np.array_equal(saved_np[k], split_np[k]) for k in saved_np), \
            f'[{ds}] TR{r}: saved split does not match canonical masks'
        vg = dgl.load_graphs(view_path)[0][0]
        v_tm = vg.nodes[nt].data['train_mask'].bool()
        v_em = vg.edges['homo'].data['train_mask'].bool()
        v_el = vg.edges['homo'].data['label'].long()
        assert np.array_equal(v_tm.numpy(), split_np['train_mask']), f'[{ds}] TR{r}: view train mask mismatch'

        # ---- Check 3: compare with SplitGNN's masks for the same dataset/ratio
        ref = f'{a.splitgnn_ref}/{ds}/unified/tr{r}/seed_2/split_indices.npz'
        if os.path.exists(ref):
            rz = np.load(ref)
            cmp = {k: bool(np.array_equal(rz[k].astype(bool), split_np[k]))
                   for k in ('train_mask', 'valid_mask', 'test_mask') if k in rz.files}
            crosscheck = {'ref_path': ref, 'ref_keys': list(rz.files), 'identical': cmp}
        else:
            crosscheck = {'ref_path': ref, 'status': 'reference not found'}

        manifest['views'][f'tr{r}'] = {
            'view_path': view_path,
            'view_sha256': sha256_file(view_path),
            'split_path': split_path,
            'split_file_sha256': sha256_file(split_path),
            'split_content_sha256': mask_content_sha256(saved_np),
            'n_train': int(saved_np['train_mask'].sum()),
            'n_valid': int(saved_np['valid_mask'].sum()),
            'n_test': int(saved_np['test_mask'].sum()),
            'n_train_fraud': int(labels[torch.from_numpy(saved_np['train_mask'])].sum()),
            'n_edge_train': int(v_em.sum()),
            'n_edge_train_same_label': int((v_el[v_em] == 1).sum()),
            'splitgnn_crosscheck': crosscheck,
        }
        print(f'[{ds}] TR{r}: train={manifest["views"][f"tr{r}"]["n_train"]} '
              f'edge_train={manifest["views"][f"tr{r}"]["n_edge_train"]} '
              f'splitgnn_crosscheck={crosscheck.get("identical", crosscheck.get("status"))}')

    out_manifest = f'{a.out_dir}/{ds}_views_manifest.json'
    with open(out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[{ds}] manifest written to {out_manifest}')


if __name__ == '__main__':
    main()
