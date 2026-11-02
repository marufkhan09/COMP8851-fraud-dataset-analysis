"""
Diagnose where GDN's training goes non-finite on a dataset. Replays the exact
training step from run_gdn_unified.train_one_epoch for the first batches and
prints every loss term, the exp() arguments and whether parameters are finite.

Usage: python diag_nan.py --dataset elliptic [--transform none|column_minmax] [--batches 12]
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/gdn_vast')
import run_gdn_unified as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='elliptic')
    ap.add_argument('--transform', default=None)
    ap.add_argument('--batches', type=int, default=12)
    ap.add_argument('--topk', type=int, default=10)
    cli = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(R.AUTHOR_CONFIG))
    cfg['topk'] = cli.topk
    R.random.seed(2); np.random.seed(2); torch.manual_seed(2); torch.cuda.manual_seed_all(2)

    feat, labels, adj, _, _, transform = R.load_dataset(cli.dataset)
    if cli.transform == 'column_minmax':
        lo, hi = feat.min(0), feat.max(0)
        feat = ((feat - lo) / np.where(hi > lo, hi - lo, 1.0)).astype(np.float32)
        transform = 'column_minmax'
    print(f'transform={transform} feat min={feat.min():.3f} max={feat.max():.3f}')
    splits, _ = R.canonical_splits(cli.dataset, labels, print)
    idx = np.where(splits['train_mask_tr40'])[0]
    np.random.RandomState(2).shuffle(idx)
    idx = [int(i) for i in idx]
    pos = [i for i in idx if labels[i] == 1]
    neg = [i for i in idx if labels[i] != 1]
    model, opt = R.build_model(feat, adj, pos, neg, cfg)
    d = feat.shape[1]
    bs = cfg['batch_size']

    def finite_params():
        return all(torch.isfinite(p).all().item() for p in model.parameters())

    for b in range(cli.batches):
        nodes = idx[b * bs:(b + 1) * bs]
        y = torch.as_tensor(labels[np.array(nodes)], dtype=torch.long, device='cuda')
        opt.zero_grad()
        loss = model.loss(nodes, y)
        loss.backward(retain_graph=True)
        grad = torch.abs(torch.autograd.grad(outputs=loss, inputs=model.inter1.features.weight)[0])
        gi = grad.mean(dim=0).topk(k=cfg['topk']).indices
        top = set(gi.tolist())
        ngi = torch.tensor([i for i in range(d) if i not in top], dtype=torch.long)
        lp, ln = model.inter1.fl_loss(gi)
        ls = cfg['Beta'] * torch.exp(lp - ln)
        ls.backward(retain_graph=True)
        fp, fn = model.inter1.fn_loss(nodes, ngi)
        lf = cfg['Beta'] * torch.exp(fp - fn)
        lf.backward()
        gmax = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
        opt.step()
        print(f'batch {b}: ce={loss.item():.4g} fl_arg={(lp - ln).item():.4g} fl={ls.item():.4g} '
              f'fn_arg={(fp - fn).item():.4g} fn={lf.item():.4g} max|grad|={gmax:.4g} '
              f'params_finite={finite_params()} pos_vec_finite={torch.isfinite(model.inter1.pos_vector).all().item()}')
        if not finite_params():
            print('parameters became non-finite after this batch')
            break


if __name__ == '__main__':
    main()
