"""The vectorised MRQ sampler must select exactly the official neighbours."""

import os
import tempfile

import numpy as np
import pytest
import torch

from conftest import needs_dataset
from phase2 import data as pdata
from phase2 import official
import sagad_adapter as sa


def official_selection(g, x, centres):
    """Run the authors' get_star_topk_nbs on the authors' tmp_graph."""
    from dgl import KHopGraph

    mrq = official.load("sagad", "mrqsampler")
    graph = g.local_var()
    graph.ndata["feature_normed"] = sa.feature_normed(x)  # same values; avoids aliasing x
    tmp = KHopGraph(1)(graph).to_simple().remove_self_loop()
    out = {}
    for c in centres:
        nbs, _ = mrq.get_star_topk_nbs(tmp, int(c), 1, mrq.select_topk_star_normft)
        out[int(c)] = sorted(int(v) for v in nbs)
    return out


def fast_selection(g, x):
    src, dst, stats = sa.mrq_one_hop_exact(g, x)
    sel = {}
    for s_, d_ in zip(src.tolist(), dst.tolist()):
        sel.setdefault(d_, []).append(s_)
    return {k: sorted(v) for k, v in sel.items()}, stats


def compare(g, x, centres):
    ref = official_selection(g, x, centres)
    fast, stats = fast_selection(g, x)
    mismatched = [c for c in centres if fast.get(int(c), []) != ref[int(c)]]
    return mismatched, stats


def test_synthetic_ties_and_zero_norm_nodes():
    import dgl

    rng = np.random.default_rng(0)
    n = 400
    src = rng.integers(0, n, 3000)
    dst = rng.integers(0, n, 3000)
    g = dgl.graph((torch.as_tensor(src), torch.as_tensor(dst)), num_nodes=n)
    g = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(g)))
    x = torch.as_tensor(rng.integers(0, 3, size=(n, 4)), dtype=torch.float32)  # many exact ties
    x[:40] = x.min(0).values  # rows equal to the column minima -> s = 0 -> NaN keys
    mismatched, stats = compare(g, x, range(n))
    assert stats["nan_key_centres_official_fallback"] > 0
    assert not mismatched, f"{len(mismatched)} centres differ, e.g. {mismatched[:5]}"


@needs_dataset("fdcompcn")
def test_fdcompcn_all_nodes_match_official():
    ds = pdata.load_canonical_dataset("fdcompcn")
    g = sa.build_graph(ds)
    x = torch.as_tensor(ds.features, dtype=torch.float32)
    mismatched, _ = compare(g, x, range(ds.num_nodes))
    assert not mismatched


@pytest.mark.parametrize("name,feat_trans", [("amazon", "no"), ("yelpchi", "no")])
def test_sampled_centres_match_official(name, feat_trans):
    if not (pdata.canonical_path(name)).exists():
        pytest.skip(f"{name} not prepared")
    ds = pdata.load_canonical_dataset(name)
    g = sa.build_graph(ds)
    x = torch.as_tensor(ds.features, dtype=torch.float32)
    centres = np.random.default_rng(1).choice(ds.num_nodes, size=150, replace=False)
    mismatched, _ = compare(g, x, centres)
    assert not mismatched


@needs_dataset("fdcompcn")
def test_mrq_cache_returns_identical_edges():
    import uuid
    dummy_name = f"tmp_fdcompcn_{uuid.uuid4().hex[:8]}"
    ds = pdata.load_canonical_dataset("fdcompcn")
    g = sa.build_graph(ds)
    x = torch.as_tensor(ds.features, dtype=torch.float32)
    fresh_src, fresh_dst, _ = sa.mrq_one_hop_exact(g, x)
    _, _, stats_write = sa.mrq_cached(g, x, dummy_name)       # must write
    assert stats_write["cache"].startswith("written")
    src, dst, stats_hit = sa.mrq_cached(g, x, dummy_name)     # must hit
    assert stats_hit["cache"].startswith("hit")
    assert torch.equal(src, fresh_src) and torch.equal(dst, fresh_dst)
    other = x.clone()
    other[0, 0] += 999.0                                      # different input
    _, _, stats2 = sa.mrq_cached(g, other, dummy_name)        # must write (miss)
    assert stats2["cache"].startswith("written")
    # clean up test cache files
    cache_dir = pdata.BUNDLE_ROOT / "data" / "cache"
    for p in cache_dir.glob(f"sagad_mrq_{dummy_name}_*.npz"):
        p.unlink(missing_ok=True)


@needs_dataset("fdcompcn")
def test_official_mrqsample_end_to_end_and_aliasing():
    """Whole official mrqsample (incl. add_edges loop) vs ours, and document
    whether the official call mutates the caller's feature tensor."""
    ds = pdata.load_canonical_dataset("fdcompcn")
    g = sa.build_graph(ds)
    x = torch.as_tensor(ds.features, dtype=torch.float32).clone()
    before = x.clone()
    mrq = official.load("sagad", "mrqsampler")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        os.makedirs("mrqs")
        try:
            g_off = mrq.mrqsample("fdcompcn_test", g, x, one_hop=True)
        finally:
            os.chdir(cwd)
    mutated = not torch.equal(before, x)
    print(f"official mrqsample mutated the caller's features in place: {mutated}")
    src, dst, _ = sa.mrq_one_hop_exact(g, before)
    ours = set(zip(src.tolist(), dst.tolist()))
    s_off, d_off = g_off.edges()
    theirs = set(zip(s_off.tolist(), d_off.tolist()))
    assert ours == theirs
