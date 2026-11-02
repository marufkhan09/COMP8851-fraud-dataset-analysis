"""Dataset integrity, frozen-split identity and leakage guards."""

import numpy as np
import pytest

from conftest import needs_dataset
from phase2 import data as pdata
from shared.comp8851 import protocol, splits

ALL = list(protocol.DATASETS)


@pytest.mark.parametrize("name", ALL)
def test_canonical_matches_previous_benchmark(name):
    if not pdata.canonical_path(name).exists():
        pytest.skip(f"{name} not prepared")
    if name == "tsocial":
        pytest.skip("T-Social identity is checked by scripts/prepare_data.py (slow view hash)")
    ds = pdata.load_canonical_dataset(name)
    assert pdata.check_against_report(ds)["matches_report"]
    assert pdata.check_against_previous(ds)["identical_to_previous_benchmark"]


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("ratio", ["TR40", "TR30", "TR20", "TR10"])
def test_split_invariants_and_no_leakage(name, ratio):
    if not pdata.canonical_path(name).exists():
        pytest.skip(f"{name} not prepared")
    if name == "tsocial" and ratio != "TR40":
        pytest.skip("T-Social ladder covered by prepare_data.py")
    ds = pdata.load_canonical_dataset(name)
    arrays, _, _ = pdata.resolve_split(ds, ratio)
    tr, va, te = (set(arrays[k].tolist()) for k in ("train_idx", "valid_idx", "test_idx"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    eligible = set(ds.eligible_idx.tolist())
    assert tr | va | te | set(arrays["unused_idx"].tolist()) == eligible
    ref, _, _ = pdata.resolve_split(ds, "TR40")
    assert np.array_equal(arrays["valid_idx"], ref["valid_idx"])  # fixed across ratios
    assert np.array_equal(arrays["test_idx"], ref["test_idx"])
    assert tr <= set(ref["train_idx"].tolist())                   # nested


@needs_dataset("elliptic")
def test_elliptic_is_chronological():
    ds = pdata.load_canonical_dataset("elliptic")
    arrays, _, _ = pdata.resolve_split(ds, "TR40")
    t = ds.time_steps
    assert t[arrays["train_idx"]].max() <= t[arrays["valid_idx"]].min()
    assert t[arrays["valid_idx"]].max() <= t[arrays["test_idx"]].min()
    unknown = np.setdiff1d(np.arange(ds.num_nodes), ds.eligible_idx)
    assert not np.intersect1d(unknown, np.concatenate([arrays[k] for k in ("train_idx", "valid_idx", "test_idx")])).size


@needs_dataset("amazon")
def test_amazon_unlabelled_prefix_never_supervised():
    ds = pdata.load_canonical_dataset("amazon")
    arrays, _, _ = pdata.resolve_split(ds, "TR40")
    for key in ("train_idx", "valid_idx", "test_idx"):
        assert arrays[key].min() >= 3305


@needs_dataset("fdcompcn")
def test_label_isolation_hides_evaluation_labels():
    prep = pdata.prepare("fdcompcn", "TR40")
    masked = prep.train_labels_only
    assert (masked[prep.valid_idx] == -1).all() and (masked[prep.test_idx] == -1).all()
    assert np.array_equal(masked[prep.train_idx], prep.dataset.labels[prep.train_idx])
    other = np.setdiff1d(np.arange(prep.dataset.num_nodes), prep.train_idx)
    assert (masked[other] == -1).all()


def test_missing_dataset_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No substitute data"):
        pdata.load_canonical_dataset("yelpchi", str(tmp_path))


def test_corrupted_split_is_rejected(tmp_path):
    if not pdata.canonical_path("fdcompcn").exists():
        pytest.skip("fdcompcn not prepared")
    ds = pdata.load_canonical_dataset("fdcompcn")
    arrays, _, _ = pdata.resolve_split(ds, "TR40")
    bad = dict(arrays)
    bad["test_idx"] = np.concatenate([arrays["test_idx"], arrays["train_idx"][:5]])
    path = tmp_path / "bad.npz"
    np.savez(path, **bad)
    with pytest.raises(RuntimeError):
        pdata.resolve_split(ds, "TR40", split_path=str(path))


def test_invalid_labels_are_rejected():
    from shared.comp8851 import evaluator

    with pytest.raises(ValueError):
        evaluator.metric_bundle([0, 1, 2], [0.1, 0.2, 0.3], 0.5)
    with pytest.raises(ValueError):
        evaluator.metric_bundle([0, 1, 1], [0.1, float("nan"), 0.3], 0.5)


@needs_dataset("fdcompcn")
def test_fdcompcn_union_relation_excluded_for_relation_models():
    ds = pdata.load_canonical_dataset("fdcompcn")
    assert "homo" in ds.relations
    assert pdata.relation_order(ds) == ["invest_bc2bc", "provide_bc2bc", "sale_bc2bc"]
    src, dst, etype = pdata.typed_edges(ds)
    assert set(np.unique(etype).tolist()) == {0, 1, 2}
