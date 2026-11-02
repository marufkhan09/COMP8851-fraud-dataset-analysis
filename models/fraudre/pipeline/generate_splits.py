"""
Canonical split generator for FRAUDRE, matching the COMP8851 protocol:
- split-generation seed is always 2
- per eligible-node pool: 20% valid, 40% test (fixed across ratios/seeds),
  remaining 40% is the train pool
- TR40/30/20/10 are nested prefixes of a seed=2 shuffle of that train pool
- Elliptic is the chronological exception: valid/test come from later
  timesteps (fixed), only the earlier-timestep train pool is nested by ratio
"""
import numpy as np
import os

SPLIT_SEED = 2

def eligible_mask(dataset, labels, timesteps=None):
    """Which node indices are eligible to participate in train/val/test at all."""
    if dataset == 'amazon':
        mask = np.zeros(len(labels), dtype=bool)
        mask[3305:] = True
        return mask
    if dataset == 'elliptic':
        return labels != -1
    # yelp, comp, tfinance: all nodes are eligible
    return np.ones(len(labels), dtype=bool)


def generate_splits(dataset, labels, timesteps=None):
    """
    Returns a dict with:
      valid_idx, test_idx  (fixed, independent of ratio/seed)
      train_pool_idx       (the ratio=TR40 prefix order; ratios are prefixes of this)
      tr_sizes: {40: n40, 30: n30, 20: n20, 10: n10}  (counts into train_pool_idx)
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(SPLIT_SEED)

    if dataset == 'elliptic':
        assert timesteps is not None, "elliptic requires timesteps"
        timesteps = np.asarray(timesteps)
        labeled = labels != -1
        train_region = labeled & (timesteps <= 34)
        valid_region = labeled & (timesteps >= 35) & (timesteps <= 42)
        test_region  = labeled & (timesteps >= 43)

        valid_idx = np.where(valid_region)[0]
        test_idx  = np.where(test_region)[0]
        pool_idx  = np.where(train_region)[0]

        # nested prefixes: proportional interleave within the pool by class
        # (preserves true class ratio at every TR prefix, not just overall)
        pool_labels = labels[pool_idx]
        pool_parts = []
        for cls in np.unique(pool_labels):
            cls_idx = pool_idx[pool_labels == cls].copy()
            rng.shuffle(cls_idx)
            pool_parts.append(cls_idx)
        order = _proportional_interleave(pool_parts, rng)
        n_pool = len(order)
        tr_sizes = {
            40: n_pool,
            30: int(round(n_pool * 0.75)),
            20: int(round(n_pool * 0.50)),
            10: int(round(n_pool * 0.25)),
        }
        return {
            'valid_idx': valid_idx,
            'test_idx': test_idx,
            'train_pool_order': order,
            'tr_sizes': tr_sizes,
        }

    mask = eligible_mask(dataset, labels)
    pool_all = np.where(mask)[0]
    pool_labels_all = labels[pool_all]

    # Stratify PER CLASS first (each class gets its own 20/40/40 split),
    # THEN interleave only within the resulting pool for nesting. This
    # guarantees valid/test always contain every class, regardless of
    # how imbalanced the dataset is.
    valid_parts, test_parts, pool_parts = [], [], []
    for cls in np.unique(pool_labels_all):
        cls_idx = pool_all[pool_labels_all == cls].copy()
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        n_valid_cls = int(round(n_cls * 0.20))
        n_test_cls = int(round(n_cls * 0.40))
        valid_parts.append(cls_idx[:n_valid_cls])
        test_parts.append(cls_idx[n_valid_cls:n_valid_cls + n_test_cls])
        pool_parts.append(cls_idx[n_valid_cls + n_test_cls:])

    valid_idx = np.concatenate(valid_parts)
    test_idx = np.concatenate(test_parts)

    # Proportional interleave: each item gets a position (i+0.5)/n_in_class,
    # then sort globally by that position. This preserves each class's
    # TRUE proportion at every prefix length (not just at the full pool),
    # so TR10/TR20/TR30 stay representative, unlike naive 1:1 round-robin
    # which distorts small prefixes toward 50/50.
    train_pool_order = _proportional_interleave(pool_parts, rng)

    n_pool = len(train_pool_order)
    tr_sizes = {
        40: n_pool,
        30: int(round(n_pool * 0.75)),
        20: int(round(n_pool * 0.50)),
        10: int(round(n_pool * 0.25)),
    }
    return {
        'valid_idx': valid_idx,
        'test_idx': test_idx,
        'train_pool_order': train_pool_order,
        'tr_sizes': tr_sizes,
    }


def _proportional_interleave(class_parts, rng):
    """Merge per-class index arrays so that every prefix of the result
    preserves each class's true proportion (not just the full merged array).
    class_parts: list of 1D arrays, each already shuffled within its class.
    """
    positions = []
    values = []
    for part in class_parts:
        n = len(part)
        if n == 0:
            continue
        for i, v in enumerate(part):
            positions.append((i + 0.5) / n)
            values.append(v)
    positions = np.array(positions)
    values = np.array(values)
    order = np.argsort(positions, kind='stable')
    return values[order]


def _stratified_shuffle(idx_array, label_array, rng):
    """Stratified shuffle of idx_array (per-class shuffle then interleave by
    concatenation, which is fine since consumers only slice prefixes per class
    proportion is preserved approximately via separate shuffles)."""
    order_parts = []
    for cls in np.unique(label_array):
        cls_idx = idx_array[label_array == cls]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)
        order_parts.append(cls_idx)
    # interleave classes roughly proportionally by round-robin
    max_len = max(len(p) for p in order_parts)
    interleaved = []
    for i in range(max_len):
        for p in order_parts:
            if i < len(p):
                interleaved.append(p[i])
    return np.array(interleaved)


def save_splits(dataset, labels, timesteps, out_path):
    result = generate_splits(dataset, labels, timesteps)
    np.savez(
        out_path,
        valid_idx=result['valid_idx'],
        test_idx=result['test_idx'],
        train_pool_order=result['train_pool_order'],
        tr_sizes_40=result['tr_sizes'][40],
        tr_sizes_30=result['tr_sizes'][30],
        tr_sizes_20=result['tr_sizes'][20],
        tr_sizes_10=result['tr_sizes'][10],
    )
    return result


def load_or_generate_splits(dataset, labels, timesteps, splits_dir):
    out_path = os.path.join(splits_dir, f'{dataset}_seed2_nested_splits.npz')
    if os.path.exists(out_path):
        d = np.load(out_path)
        return {
            'valid_idx': d['valid_idx'],
            'test_idx': d['test_idx'],
            'train_pool_order': d['train_pool_order'],
            'tr_sizes': {40: int(d['tr_sizes_40']), 30: int(d['tr_sizes_30']),
                         20: int(d['tr_sizes_20']), 10: int(d['tr_sizes_10'])},
        }
    os.makedirs(splits_dir, exist_ok=True)
    return save_splits(dataset, labels, timesteps, out_path)


def verify_splits(result):
    """Sanity checks: nesting holds, no overlap between valid/test/train."""
    pool = result['train_pool_order']
    sizes = result['tr_sizes']
    assert sizes[10] <= sizes[20] <= sizes[30] <= sizes[40] == len(pool)
    tr10 = set(pool[:sizes[10]].tolist())
    tr20 = set(pool[:sizes[20]].tolist())
    tr30 = set(pool[:sizes[30]].tolist())
    tr40 = set(pool[:sizes[40]].tolist())
    assert tr10 <= tr20 <= tr30 <= tr40, "nesting violated"
    valid_set = set(result['valid_idx'].tolist())
    test_set = set(result['test_idx'].tolist())
    assert len(valid_set & test_set) == 0, "valid/test overlap"
    assert len(valid_set & tr40) == 0, "valid/train overlap"
    assert len(test_set & tr40) == 0, "test/train overlap"
    return True
