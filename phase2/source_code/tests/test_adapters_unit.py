"""Adapter-level checks that compare against the official code paths."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import needs_dataset
from phase2 import configs, data as pdata
from phase2.harness import RunContext


def make_ctx(name, slug, overrides=()):
    prep = pdata.prepare(name, "TR40")
    hp = configs.resolve_hparams(configs.load_model_config(slug), name, list(overrides))
    args = SimpleNamespace(seed=2, eval_batch_size=None, ratio="TR40", epochs=None, num_workers=0)
    return RunContext(args, prep, torch.device("cpu"), hp, None)


@needs_dataset("fdcompcn")
def test_spacegnn_deterministic_blocks_equal_sampler_edge_set():
    import dgl
    from spacegnn_adapter import SpaceGNNAdapter

    a = SpaceGNNAdapter(make_ctx("fdcompcn", "spacegnn"))
    a.setup()
    idx = torch.as_tensor(a.ctx.data.test_idx[:700])
    _, _, sampled = a.sampler.sample_blocks(a.graph, idx)
    ours = a.full_blocks(idx)

    def global_edges(block):
        s, d = block.edges()
        return set(zip(block.srcdata[dgl.NID][s].tolist(), block.dstdata[dgl.NID][d].tolist()))

    assert global_edges(sampled[0]) == global_edges(ours[0])
    assert torch.equal(ours[0].dstdata[dgl.NID], idx)
    p1, p2 = a.predict(a.ctx.data.test_idx), a.predict(a.ctx.data.test_idx)
    assert np.array_equal(p1, p2)


@needs_dataset("fdcompcn")
def test_spacegnn_official_curvature_draws_are_used():
    from phase2 import official

    names = official.load("spacegnn", "name")
    # PARAMETERS[...][5:7] are the authors' uniform(0.01, 0.02) draws after random.seed(1)
    for key in ("amazon", "yelp", "tfinance", "tsocial"):
        params = names.PARAMETERS[key]
        assert len(params) == 7 and 0.0 <= params[6] <= 0.02


@needs_dataset("fdcompcn")
def test_sagad_loss_decreases_on_real_data():
    from sagad_adapter import SAGADAdapter

    a = SAGADAdapter(make_ctx("fdcompcn", "sagad"))
    a.setup()
    losses = [a.train_epoch(e) for e in range(30)]
    assert all(np.isfinite(losses))
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    grads = [p.grad for p in a.model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@needs_dataset("fdcompcn")
def test_croc_uses_official_train_epoch_and_relation_types():
    from croc_adapter import CRoCAdapter
    from phase2 import official

    a = CRoCAdapter(make_ctx("fdcompcn", "croc"))
    a.setup()
    trainers = official.load("croc", "trainers")
    assert type(a.trainer).train_epoch is trainers.GNNTrainer.train_epoch
    assert a.params["n_rel"] == 3 and a.g.edata["type"].shape[1] == 3
    assert not a.g.ndata["val_mask"].any() and not a.g.ndata["test_mask"].any()
    loss = a.train_epoch(0)
    assert np.isfinite(loss)


def test_configs_resolve_with_sources():
    for slug in ("sagad", "spacegnn", "croc"):
        cfg = configs.load_model_config(slug)
        for ds in ("yelpchi", "amazon", "tfinance", "tsocial", "elliptic", "fdcompcn"):
            hp = configs.resolve_hparams(cfg, ds, [])
            assert hp["_source"]
            for key in cfg.get("search_space", {}):
                assert key in hp, f"{slug}/{ds}: searched key {key} has no official default"
        with pytest.raises(KeyError):
            configs.resolve_hparams(cfg, "amazon", ["not_a_param=1"])
