#!/usr/bin/env python
"""
H2-FDetector adapters + canonical training-ratio views for the four datasets the
authors do not ship: FDCompCN ('comp'), Elliptic, T-Finance, T-Social.

Adapter rule: data translation only. Each graph is converted into the input format
H2-FDetector's model.py expects (single node type 'r', a 'homo' edge type for the
edge-label loss, at least one further relation) with the SAME nodes, edges, features
and labels the group's SplitGNN adapters used:
  comp      comp.dgl from FDCompCN.zip, node type renamed company -> r (sha256 is
            compared with SplitGNN's view hash)
  elliptic  rebuilt from the raw CSVs exactly as models/splitgnn/adapters/elliptic_adapter.py
            (class '1' -> 1 illicit, '2' -> 0 licit, 'unknown' -> -1, excluded from all splits)
  tfinance  full graph, 'homo' and 'e' both = the original edges (as SplitGNN)
  tsocial   the identical 15% stratified node subsample SplitGNN used
            (RandomState(2), same procedure as tsocial_adapter.py); node/edge counts are
            checked against SplitGNN's recorded 867160 / 3297570
Edges are NOT symmetrised or otherwise altered; edge statistics are recorded instead.

Then, exactly like h2fd_build_views.py (whose helper functions are imported unchanged):
  * canonical nested splits from generate_splits.py (split seed 2), on labelled nodes only
    when unknown labels exist; for that case two possible orderings are tried and the one
    that reproduces SplitGNN's own split_indices.npz is kept (abort if none does)
  * per-ratio 'homo' edge labels / edge train mask (authors' rule, vectorised; its exact
    equivalence was proven on the full YelpChi and Amazon graphs)
  * views + split files + a manifest with the same keys the unified run script reads
  * a config/<dataset>.yaml copied from an author config (default: yelp) if none exists

Usage:
  python h2fd_build_views_ext.py --dataset comp
  python h2fd_build_views_ext.py --dataset elliptic
  python h2fd_build_views_ext.py --dataset tsocial
  python h2fd_build_views_ext.py --dataset tfinance
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml
import dgl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from h2fd_build_views import (RATIOS, edge_labels_vectorised, mask_content_sha256,  # noqa: E402
                              sha256_file)

BUILDER_VERSION = 'h2fd_view_builder_ext_v1'
DATASETS = ['comp', 'elliptic', 'tfinance', 'tsocial']
SPLITGNN_COMP_VIEW_SHA = 'e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa'
TSOCIAL_FRAC = 0.15
TSOCIAL_EXPECTED = {'nodes': 867160, 'edges': 3297570}  # SplitGNN protocol_notes.txt
MASK_ALIASES = {'train_mask': ['train_mask', 'train_mask_tr40', 'train'],
                'valid_mask': ['valid_mask', 'val_mask', 'valid', 'val'],
                'test_mask': ['test_mask', 'test']}


def log(msg):
    print(msg, flush=True)


def combined_sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(f'{os.path.basename(p)}:{sha256_file(p)}\n'.encode())
    return h.hexdigest()


def edge_stats(src, dst, n):
    s = src.numpy().astype(np.int64)
    d = dst.numpy().astype(np.int64)
    key = s * n + d
    uniq = np.unique(key)
    return {'n_edges': int(key.size),
            'n_self_loops': int((s == d).sum()),
            'n_duplicate_edges': int(key.size - uniq.size),
            'n_edges_without_reverse': int((~np.isin(d * n + s, uniq)).sum())}


def single_rel_graph(src, dst, n):
    """'homo' and 'e' both hold the original edges, as in the SplitGNN adapters."""
    return dgl.heterograph({('r', 'homo', 'r'): (src, dst), ('r', 'e', 'r'): (src, dst)},
                           num_nodes_dict={'r': n})


def flat_label(lab):
    lab = lab.long()
    return lab[:, 1].contiguous() if lab.dim() == 2 else lab


# ---------------------------------------------------------------- adapters
def build_comp(a):
    zpath = f'{a.data_dir}/FDCompCN.zip'
    tmp = f'{a.work_dir}/src_extract'
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extract('comp.dgl', tmp)
    cpath = f'{tmp}/comp.dgl'
    csha = sha256_file(cpath)
    g0 = dgl.load_graphs(cpath)[0][0]
    nt0 = g0.ntypes[0]
    g = dgl.heterograph({('r', et, 'r'): g0.edges(etype=et) for (_, et, _) in g0.canonical_etypes},
                        num_nodes_dict={'r': g0.num_nodes(nt0)})
    g.nodes['r'].data['feature'] = g0.nodes[nt0].data['feature'].float()
    g.nodes['r'].data['label'] = flat_label(g0.nodes[nt0].data['label'])
    shipped = {'train_mask': g0.nodes[nt0].data['train_mask'].bool(),
               'homo_label': g0.edges['homo'].data['label'].long(),
               'homo_train_mask': g0.edges['homo'].data['train_mask'].bool()}
    info = {'source_path': zpath, 'source_sha256': sha256_file(zpath),
            'extracted_comp_dgl_sha256': csha,
            'matches_splitgnn_comp_view_sha256': csha == SPLITGNN_COMP_VIEW_SHA,
            'original_node_type': nt0,
            'transform': 'node type renamed to r; edges, features, labels unchanged'}
    return g, info, shipped


def build_elliptic(a):
    d = f'{a.data_dir}/elliptic_bitcoin_dataset'
    files = [f'{d}/elliptic_txs_classes.csv', f'{d}/elliptic_txs_edgelist.csv',
             f'{d}/elliptic_txs_features.csv']
    classes = pd.read_csv(files[0])
    edges_df = pd.read_csv(files[1])
    features_df = pd.read_csv(files[2], header=None)
    tx_ids = features_df[0].values
    n = len(tx_ids)
    txid_to_idx = pd.Series(np.arange(n), index=tx_ids)
    feat = torch.from_numpy(features_df.iloc[:, 2:].values.astype(np.float32))
    cls = classes.set_index('txId')['class'].astype(str).reindex(tx_ids)
    label = np.full(n, -1, dtype=np.int64)
    label[(cls == '1').values] = 1
    label[(cls == '2').values] = 0
    src_ids = edges_df['txId1'].map(txid_to_idx)
    dst_ids = edges_df['txId2'].map(txid_to_idx)
    ok = src_ids.notna() & dst_ids.notna()
    src = torch.from_numpy(src_ids[ok].values.astype(np.int64))
    dst = torch.from_numpy(dst_ids[ok].values.astype(np.int64))
    g = single_rel_graph(src, dst, n)
    g.nodes['r'].data['feature'] = feat
    g.nodes['r'].data['label'] = torch.from_numpy(label)
    info = {'source_path': d, 'source_sha256': combined_sha(files),
            'source_file_sha256': {os.path.basename(p): sha256_file(p) for p in files},
            'n_edges_in_csv': int(len(edges_df)), 'n_edges_dropped_unmapped': int((~ok).sum()),
            'label_counts': {str(k): int(v) for k, v in zip(*np.unique(label, return_counts=True))},
            'transform': "same as splitgnn elliptic_adapter.py: features cols 2+, "
                         "class '1'->1, '2'->0, unknown->-1; homo = e = csv edges"}
    return g, info, None


def build_tfinance(a):
    p = f'{a.data_dir}/tfinance'
    g0 = dgl.load_graphs(p)[0][0]
    src, dst = g0.edges()
    g = single_rel_graph(src, dst, g0.num_nodes())
    g.nodes['r'].data['feature'] = g0.ndata['feature'].float()
    g.nodes['r'].data['label'] = flat_label(g0.ndata['label'])
    info = {'source_path': p, 'source_sha256': sha256_file(p),
            'source_label_shape': list(g0.ndata['label'].shape),
            'transform': 'homo = e = original edges (as splitgnn tfinance_adapter.py)'}
    return g, info, None


def build_tsocial(a):
    p = f'{a.data_dir}/tsocial'
    log(f'[tsocial] loading full graph {p} (large)...')
    g_full = dgl.load_graphs(p)[0][0]
    label_full = flat_label(g_full.ndata['label'])
    n_full, e_full = g_full.num_nodes(), g_full.num_edges()
    rng = np.random.RandomState(2)
    picked = []
    for c in torch.unique(label_full):
        class_idx = torch.where(label_full == c)[0].numpy()
        n_take = int(round(TSOCIAL_FRAC * len(class_idx)))
        picked.append(rng.choice(class_idx, size=n_take, replace=False))
    idx = np.sort(np.concatenate(picked))
    sub = dgl.node_subgraph(g_full, torch.from_numpy(idx))
    src, dst = sub.edges()
    g = single_rel_graph(src, dst, sub.num_nodes())
    g.nodes['r'].data['feature'] = sub.ndata['feature'].float()
    g.nodes['r'].data['label'] = label_full[torch.from_numpy(idx)].long()
    counts_ok = (sub.num_nodes() == TSOCIAL_EXPECTED['nodes']
                 and sub.num_edges() == TSOCIAL_EXPECTED['edges'])
    info = {'source_path': p, 'source_sha256': sha256_file(p),
            'full_graph': {'n_nodes': int(n_full), 'n_edges': int(e_full)},
            'subsample': {'fraction': TSOCIAL_FRAC, 'rng': 'RandomState(2)', 'stratified_by': 'label',
                          'n_nodes': int(sub.num_nodes()), 'n_edges': int(sub.num_edges()),
                          'fraud_rate': float(g.nodes['r'].data['label'].float().mean()),
                          'expected_from_splitgnn': TSOCIAL_EXPECTED, 'counts_match': counts_ok},
            'subsample_node_ids_sha256': hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
            'transform': 'same 15% stratified induced subgraph as splitgnn tsocial_adapter.py; '
                         'homo = e = subgraph edges'}
    del g_full
    if not counts_ok and not a.allow_mismatch:
        sys.exit(f'[tsocial] ABORT: subsample has {sub.num_nodes()} nodes / {sub.num_edges()} edges, '
                 f'SplitGNN recorded {TSOCIAL_EXPECTED}. Use --allow-mismatch to continue anyway.')
    return g, info, None


BUILDERS = {'comp': build_comp, 'elliptic': build_elliptic,
            'tfinance': build_tfinance, 'tsocial': build_tsocial}


# ---------------------------------------------------------------- splits
def to_mask(arr, n):
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr
    if arr.shape == (n,) and np.isin(np.unique(arr), [0, 1]).all():
        return arr.astype(bool)
    m = np.zeros(n, dtype=bool)
    m[arr.astype(np.int64)] = True
    return m


def load_ref_masks(path, n):
    if not os.path.exists(path):
        return None, {'ref_path': path, 'status': 'reference not found'}
    rz = np.load(path)
    out = {}
    for k, aliases in MASK_ALIASES.items():
        for al in aliases:
            if al in rz.files:
                out[k] = to_mask(rz[al], n)
                break
    return out, {'ref_path': path, 'ref_keys': list(rz.files)}


def candidate_splits(labels, generate_splits):
    lab = labels.cpu().long()
    n = lab.numel()
    if not (lab < 0).any():
        return {'all_nodes': generate_splits(lab)}
    labeled = torch.where(lab >= 0)[0]
    sub = generate_splits(lab[labeled])
    a = {}
    for k, v in sub.items():
        full = torch.zeros(n, dtype=torch.bool)
        full[labeled[v.bool()]] = True
        a[k] = full
    known = lab >= 0
    b = {k: v.bool() & known for k, v in generate_splits(lab).items()}
    return {'labeled_subset': a, 'full_then_drop_unlabeled': b}


def compare(masks, ref):
    return {'train_mask': bool(np.array_equal(ref['train_mask'], masks['train_mask_tr40'].numpy())),
            'valid_mask': bool(np.array_equal(ref['valid_mask'], masks['valid_mask'].numpy())),
            'test_mask': bool(np.array_equal(ref['test_mask'], masks['test_mask'].numpy()))} \
        if ref and all(k in ref for k in MASK_ALIASES) else None


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--data-dir', default='/workspace/data')
    p.add_argument('--h2fd-root', default='/workspace/H2-FDetector')
    p.add_argument('--splits-dir',
                   default='/workspace/COMP8851-fraud-dataset-analysis/models/splitgnn/splits')
    p.add_argument('--work-dir', default='/workspace/h2fd_vast')
    p.add_argument('--out-dir', default='/workspace/h2fd_vast/views')
    p.add_argument('--splitgnn-ref',
                   default='/workspace/COMP8851-fraud-dataset-analysis/results/unified/splitgnn_results')
    p.add_argument('--config-template', default='yelp',
                   help='author config copied to config/<dataset>.yaml if it does not exist')
    p.add_argument('--allow-mismatch', action='store_true',
                   help='continue if splits / subsample do not reproduce SplitGNN')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    ds = a.dataset
    t_start = time.time()

    sys.path.insert(0, a.splits_dir)
    from generate_splits import SPLIT_SEED, generate_splits, verify_splits

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(f'{a.work_dir}/native', exist_ok=True)

    # ---- adapter
    log(f'[{ds}] building adapted graph...')
    g, src_info, shipped = BUILDERS[ds](a)
    nt = 'r'
    n = g.num_nodes(nt)
    labels = g.nodes[nt].data['label'].long()
    src, dst = g.edges(etype='homo')
    log(f'[{ds}] nodes={n} ' + ' '.join(f'{et}={g.num_edges(et)}' for et in g.etypes)
        + f" features={tuple(g.nodes[nt].data['feature'].shape)}")
    t0 = time.time()
    estats = edge_stats(src, dst, n)
    log(f'[{ds}] homo edge stats: {estats} ({time.time() - t0:.1f}s)')

    # ---- edge-label rule check (only meaningful where the dataset ships its own edge labels)
    if shipped is not None:
        el, em = edge_labels_vectorised(src, dst, labels, shipped['train_mask'])
        verification = {
            'method': "dataset-shipped homo edge labels/train mask vs the authors' rule "
                      'applied to the dataset-shipped node train mask (informational)',
            'edge_labels_identical': bool(torch.equal(el, shipped['homo_label'])),
            'edge_train_mask_identical': bool(torch.equal(em, shipped['homo_train_mask'])),
            'n_label_mismatch': int((el != shipped['homo_label']).sum()),
            'n_mask_mismatch': int((em != shipped['homo_train_mask']).sum()),
            'note': "these shipped labels are replaced by the authors' rule for every view"}
    else:
        verification = {'method': 'not applicable (no H2-FDetector-native edge labels exist for this '
                                  "dataset); exact equivalence of the vectorised rule with the authors' "
                                  'generate_edges_labels() was proven on the full YelpChi and Amazon graphs'}
    log(f'[{ds}] edge-label check: {verification}')

    native_path = f'{a.work_dir}/native/{ds}.dgl'
    if not os.path.exists(native_path) or a.force:
        dgl.save_graphs(native_path, [g])
    log(f'[{ds}] adapted graph saved: {native_path}')

    # ---- canonical splits, matched against SplitGNN's own split for TR40
    ref_tr40, ref_meta = load_ref_masks(f'{a.splitgnn_ref}/{ds}/unified/tr40/seed_2/split_indices.npz', n)
    cands = candidate_splits(labels, generate_splits)
    cand_report = {name: compare(m, ref_tr40) for name, m in cands.items()}
    log(f'[{ds}] split candidates vs SplitGNN TR40: {cand_report}  ({ref_meta})')
    chosen = None
    for name, res in cand_report.items():
        if res is not None and all(res.values()):
            chosen = name
            break
    if chosen is None:
        if ref_tr40 is not None and not a.allow_mismatch:
            sys.exit(f'[{ds}] ABORT: no split candidate reproduces SplitGNN\'s split '
                     f'({cand_report}). Use --allow-mismatch to continue anyway.')
        chosen = next(iter(cands))
        log(f'[{ds}] WARNING: using {chosen} without a SplitGNN match')
    masks = cands[chosen]
    verify_splits(masks, labels.cpu())
    for k in ('valid_mask', 'test_mask', 'train_mask_tr40'):
        assert not (masks[k] & (labels < 0)).any(), f'{k} contains unlabelled nodes'

    # ---- config
    cfg_path = f'{a.h2fd_root}/config/{ds}.yaml'
    tpl_path = f'{a.h2fd_root}/config/{a.config_template}.yaml'
    if not os.path.exists(cfg_path):
        with open(tpl_path) as f:
            cfg = yaml.safe_load(f)
        cfg['dataset'] = ds
        cfg['cuda'] = 0
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        log(f'[{ds}] wrote {cfg_path} from author template {tpl_path}')
    config_info = {'path': cfg_path, 'sha256': sha256_file(cfg_path),
                   'template': tpl_path,
                   'note': 'no author config exists for this dataset; author template used unchanged '
                           'apart from dataset name and device; no tuning'}

    manifest = {
        'dataset': ds,
        'builder_version': BUILDER_VERSION,
        'builder_sha256': sha256_file(os.path.abspath(__file__)),
        'native_dgl_path': native_path,
        'native_dgl_sha256': sha256_file(native_path),
        'source_mat_path': src_info['source_path'],
        'source_mat_sha256': src_info['source_sha256'],
        'adapter': src_info,
        'node_type': nt,
        'n_nodes': int(n),
        'n_labeled_nodes': int((labels >= 0).sum()),
        'n_edges': {et: int(g.num_edges(et)) for et in g.etypes},
        'homo_edge_stats': estats,
        'split_seed': int(SPLIT_SEED),
        'split_registry_version': 'own_generate_splits_v1_seed2',
        'split_procedure': chosen,
        'split_candidates_vs_splitgnn_tr40': cand_report,
        'edge_label_verification': verification,
        'config': config_info,
        'views': {},
    }

    valid_mask = masks['valid_mask'].bool()
    test_mask = masks['test_mask'].bool()
    for r in RATIOS:
        view_path = f'{a.out_dir}/{ds}_tr{r}.dgl'
        split_path = f'{a.out_dir}/{ds}_tr{r}_split.npz'
        train_mask = masks[f'train_mask_tr{r}'].bool()
        split_np = {'train_mask': train_mask.numpy(), 'valid_mask': valid_mask.numpy(),
                    'test_mask': test_mask.numpy()}
        if os.path.exists(view_path) and os.path.exists(split_path) and not a.force:
            log(f'[{ds}] TR{r}: exists, skipping build (use --force to rebuild)')
        else:
            g.nodes[nt].data['train_mask'] = train_mask
            g.nodes[nt].data['valid_mask'] = valid_mask
            g.nodes[nt].data['test_mask'] = test_mask
            el, em = edge_labels_vectorised(src, dst, labels, train_mask)
            g.edges['homo'].data['label'] = el
            g.edges['homo'].data['train_mask'] = em
            dgl.save_graphs(view_path, [g])
            np.savez(split_path, **split_np)
            log(f'[{ds}] TR{r}: wrote {view_path}')

        saved = np.load(split_path)
        saved_np = {k: saved[k].astype(bool) for k in ('train_mask', 'valid_mask', 'test_mask')}
        assert all(np.array_equal(saved_np[k], split_np[k]) for k in saved_np), \
            f'[{ds}] TR{r}: saved split does not match canonical masks'
        vg = dgl.load_graphs(view_path)[0][0]
        v_em = vg.edges['homo'].data['train_mask'].bool()
        v_el = vg.edges['homo'].data['label'].long()
        assert np.array_equal(vg.nodes[nt].data['train_mask'].bool().numpy(), split_np['train_mask'])

        ref, meta = load_ref_masks(f'{a.splitgnn_ref}/{ds}/unified/tr{r}/seed_2/split_indices.npz', n)
        crosscheck = dict(meta)
        if ref is not None:
            crosscheck['identical'] = {k: bool(np.array_equal(ref[k], split_np[k]))
                                       for k in ('train_mask', 'valid_mask', 'test_mask') if k in ref}

        manifest['views'][f'tr{r}'] = {
            'view_path': view_path,
            'view_sha256': sha256_file(view_path),
            'split_path': split_path,
            'split_file_sha256': sha256_file(split_path),
            'split_content_sha256': mask_content_sha256(saved_np),
            'n_train': int(saved_np['train_mask'].sum()),
            'n_valid': int(saved_np['valid_mask'].sum()),
            'n_test': int(saved_np['test_mask'].sum()),
            'n_train_fraud': int(labels[torch.from_numpy(saved_np['train_mask'])].clamp(min=0).sum()),
            'n_edge_train': int(v_em.sum()),
            'n_edge_train_same_label': int((v_el[v_em] == 1).sum()),
            'splitgnn_crosscheck': crosscheck,
        }
        v = manifest['views'][f'tr{r}']
        log(f'[{ds}] TR{r}: train={v["n_train"]} valid={v["n_valid"]} test={v["n_test"]} '
            f'edge_train={v["n_edge_train"]} '
            f'splitgnn_crosscheck={crosscheck.get("identical", crosscheck.get("status"))}')
        del vg

    out_manifest = f'{a.out_dir}/{ds}_views_manifest.json'
    with open(out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    log(f'[{ds}] manifest written to {out_manifest}  (total {time.time() - t_start:.0f}s)')


if __name__ == '__main__':
    main()
