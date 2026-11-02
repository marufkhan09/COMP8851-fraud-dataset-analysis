import numpy as np
import torch

SPLIT_SEED = 2

def generate_splits(labels: torch.Tensor):
    n = labels.shape[0]
    labels_np = labels.numpy()
    rng = np.random.RandomState(SPLIT_SEED)

    valid_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_pool_mask = np.zeros(n, dtype=bool)

    tr_ratio_masks = {40: np.zeros(n, dtype=bool),
                       30: np.zeros(n, dtype=bool),
                       20: np.zeros(n, dtype=bool),
                       10: np.zeros(n, dtype=bool)}

    for c in np.unique(labels_np):
        class_idx = np.where(labels_np == c)[0]
        rng.shuffle(class_idx)
        n_c = len(class_idx)

        n_valid_c = int(round(0.20 * n_c))
        n_test_c = int(round(0.40 * n_c))
        valid_idx_c = class_idx[:n_valid_c]
        test_idx_c = class_idx[n_valid_c:n_valid_c + n_test_c]
        pool_idx_c = class_idx[n_valid_c + n_test_c:]

        valid_mask[valid_idx_c] = True
        test_mask[test_idx_c] = True
        train_pool_mask[pool_idx_c] = True

        n_pool_c = len(pool_idx_c)
        for ratio in [40, 30, 20, 10]:
            frac_of_pool = ratio / 40.0
            n_take = int(round(frac_of_pool * n_pool_c))
            tr_ratio_masks[ratio][pool_idx_c[:n_take]] = True

    return {
        'valid_mask': torch.from_numpy(valid_mask),
        'test_mask': torch.from_numpy(test_mask),
        'train_mask_tr40': torch.from_numpy(tr_ratio_masks[40]),
        'train_mask_tr30': torch.from_numpy(tr_ratio_masks[30]),
        'train_mask_tr20': torch.from_numpy(tr_ratio_masks[20]),
        'train_mask_tr10': torch.from_numpy(tr_ratio_masks[10]),
    }


def verify_splits(masks, labels):
    n = labels.shape[0]
    tr40, tr30, tr20, tr10 = masks['train_mask_tr40'], masks['train_mask_tr30'], masks['train_mask_tr20'], masks['train_mask_tr10']
    valid, test = masks['valid_mask'], masks['test_mask']

    assert (tr10 & ~tr20).sum() == 0, "TR10 not subset of TR20"
    assert (tr20 & ~tr30).sum() == 0, "TR20 not subset of TR30"
    assert (tr30 & ~tr40).sum() == 0, "TR30 not subset of TR40"
    assert (tr40 & valid).sum() == 0, "train/valid overlap"
    assert (tr40 & test).sum() == 0, "train/test overlap"
    assert (valid & test).sum() == 0, "valid/test overlap"

    print(f"n={n}")
    print(f"valid: {int(valid.sum())} ({100*valid.sum()/n:.1f}%)")
    print(f"test:  {int(test.sum())} ({100*test.sum()/n:.1f}%)")
    print(f"TR40:  {int(tr40.sum())} ({100*tr40.sum()/n:.1f}%)")
    print(f"TR30:  {int(tr30.sum())} ({100*tr30.sum()/n:.1f}%)")
    print(f"TR20:  {int(tr20.sum())} ({100*tr20.sum()/n:.1f}%)")
    print(f"TR10:  {int(tr10.sum())} ({100*tr10.sum()/n:.1f}%)")
    print("All nesting/overlap checks PASSED")
