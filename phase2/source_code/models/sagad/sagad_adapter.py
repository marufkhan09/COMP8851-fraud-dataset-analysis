"""SAGAD adapter (official code: Cloudy1225/SAGAD @ faddcbca117b).

Reused verbatim from the authors' repository: ``SAGAD``, ``precompute_Tx``,
``ChebnetIIPropDGL``, ``MLP``, all fusion modules (model.py), ``standard_scale``
(dataloader.py), ``set_seed`` (utils.py) and, for exactness fall-backs and
tests, ``select_topk_star_normft`` / ``get_star_topk_nbs`` (mrqsampler.py).

Replicated line-for-line from main.py::run (it is written as a script loop):
class weight ``w``, regulariser targets from ``p``, the weighted cross-entropy
plus the weighted BCE gate regulariser, Adam, full-batch steps.

Documented adaptations (see audit/DEVIATIONS.md):

* graph construction mirrors dataloader.load_data on the frozen canonical
  homogeneous union (to_bidirected, remove_self_loop, add_self_loop);
* checkpoint selection on validation AUPRC (protocol) instead of AUROC;
* MRQ one-hop sampling: the official per-node Python loop with one
  ``add_edges`` call per node is O(N * E) and cannot finish on T-Social. It is
  replaced by :func:`mrq_one_hop_exact`, which evaluates the *same* selection
  rule with the same float32 operations (verified edge-for-edge against the
  official routine in tests/test_sagad_mrq.py). Centres whose sort keys contain
  NaN (0/0) are delegated to the official Python routine so tie-breaking
  remains identical;
* the official ``mrqsample`` mutates the feature tensor in place on a cache
  miss (``graph.ndata['feature_normed'] = features`` aliases the tensor, then
  ``-=`` / ``/=``), so the first official run on a dataset feeds min-max scaled
  T_0(x) and X to the model while later runs (cache hit) do not. We always
  compute on a copy, i.e. the cache-hit behaviour.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from phase2 import data as pdata
from phase2 import official
from phase2.harness import Adapter

EPS = 1e-12  # mrqsampler.EPS


def feature_transform(x: torch.Tensor, feat_trans, standard_scale) -> torch.Tensor:
    """dataloader.load_data feature transform, unchanged."""
    if feat_trans == "l1":
        return torch.nn.functional.normalize(x, dim=1, p=1)
    if feat_trans == "l2":
        return torch.nn.functional.normalize(x, dim=1, p=2)
    if feat_trans == "sc":
        return standard_scale(x)
    if feat_trans == "row":
        return x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.0))
    return x  # 'no' / False


def build_graph(dataset):
    """dataloader.load_data graph operations on the canonical union view."""
    import dgl

    src, dst = pdata.homogeneous_edges(dataset)
    g = dgl.graph((torch.as_tensor(src), torch.as_tensor(dst)), num_nodes=dataset.num_nodes)
    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g


def feature_normed(features: torch.Tensor) -> torch.Tensor:
    """mrqsampler.mrqsample normalisation, computed on a copy (no aliasing)."""
    fn = features.detach().to("cpu", torch.float32).clone()
    fn -= fn.min(0, keepdim=True)[0]
    fn /= fn.max(0, keepdim=True)[0] + EPS
    return torch.norm(fn, dim=1)


def neighbour_lists(g) -> Tuple[np.ndarray, np.ndarray]:
    """(centre, neighbour) pairs of KHopGraph(1)(g).to_simple().remove_self_loop(),
    neighbours = unique(in-neighbours U out-neighbours), ascending per centre,
    exactly as get_star_topk_nbs builds ``node_ids`` with torch.unique."""
    src, dst = g.edges()
    src = src.cpu().numpy().astype(np.int64)
    dst = dst.cpu().numpy().astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    n = g.num_nodes()
    centre = np.concatenate([dst, src])  # in-neighbours of dst; out-neighbours of src
    nbr = np.concatenate([src, dst])
    key = np.unique(centre * np.int64(n) + nbr)
    return key // n, key % n


def mrq_one_hop_exact(g, features: torch.Tensor, mrq_module=None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Edges (src -> dst) of the official one-hop MRQ graph ``G_mrq``.

    For every centre c with neighbours N(c) (ascending IDs) and scalar
    s_i = ||minmax(x_i)||_2 (float32):
      1. stable-sort N(c) by key_j = -((s_c - s_j)^2) / (s_j^2)  (ascending),
      2. walk the order, accepting j while best < (s_c - s_j)^2 / (s_j^2 + EPS),
         where up/down accumulate those terms from up=0, down=s_c^2+EPS and
         best = up/down; stop at the first rejection,
      3. add edges j -> c for accepted j, plus c -> c.
    """
    n = g.num_nodes()
    s = feature_normed(features)
    centres, nbrs = neighbour_lists(g)
    deg = np.bincount(centres, minlength=n).astype(np.int64)

    s_c = s[torch.from_numpy(centres)]
    s_j = s[torch.from_numpy(nbrs)]
    keys = (-((s_c - s_j) ** 2)) / (s_j ** 2)
    keys_np = keys.numpy()

    nan_mask = np.isnan(keys_np)
    nan_centres = np.unique(centres[nan_mask])
    is_nan_centre = np.zeros(n, dtype=bool)
    is_nan_centre[nan_centres] = True

    order = np.lexsort((keys_np, centres))  # centre-major, key ascending, stable
    sorted_nbrs = nbrs[order]
    sorted_s = s_j[torch.from_numpy(order)]
    seg_start = np.zeros(n, dtype=np.int64)
    seg_start[1:] = np.cumsum(deg)[:-1]

    up = torch.zeros(n, dtype=torch.float32)
    down = torch.pow(s, 2) + EPS
    best = up / down
    count = np.zeros(n, dtype=np.int64)
    active = (deg > 0) & ~is_nan_centre
    t = 0
    while active.any():
        ids = np.nonzero(active)[0]
        alive = deg[ids] > t
        active[ids[~alive]] = False
        ids = ids[alive]
        if ids.size == 0:
            break
        ids_t = torch.from_numpy(ids)
        xi = sorted_s[torch.from_numpy(seg_start[ids] + t)]
        x0 = s[ids_t]
        tmp_up = (x0 - xi) ** 2
        tmp_down = xi ** 2 + EPS
        cond = (best[ids_t] < tmp_up / tmp_down).numpy()
        acc = ids[cond]
        if acc.size:
            acc_t = torch.from_numpy(acc)
            cond_t = torch.from_numpy(cond)
            up[acc_t] = up[acc_t] + tmp_up[cond_t]
            down[acc_t] = down[acc_t] + tmp_down[cond_t]
            best[acc_t] = up[acc_t] / down[acc_t]
            count[acc] += 1
        active[ids[~cond]] = False
        t += 1

    src_parts, dst_parts = [], []
    selected_centres = np.nonzero((count > 0) & ~is_nan_centre)[0]
    if selected_centres.size:
        lengths = count[selected_centres]
        starts = seg_start[selected_centres]
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        src_parts.append(sorted_nbrs[np.repeat(starts, lengths) + offsets])
        dst_parts.append(np.repeat(selected_centres, lengths))

    fallback = 0
    if nan_centres.size:
        if mrq_module is None:
            mrq_module = official.load("sagad", "mrqsampler")
        holder = SimpleNamespace(ndata={"feature_normed": s})
        csr_ptr = np.zeros(n + 1, dtype=np.int64)
        csr_ptr[1:] = np.cumsum(deg)
        for c in nan_centres:
            node_ids = torch.from_numpy(nbrs[csr_ptr[c]:csr_ptr[c + 1]])
            chosen = mrq_module.select_topk_star_normft(holder, node_ids, int(c))
            if chosen:
                src_parts.append(np.asarray([int(v) for v in chosen], dtype=np.int64))
                dst_parts.append(np.full(len(chosen), c, dtype=np.int64))
            fallback += 1

    src_parts.append(np.arange(n, dtype=np.int64))  # self edge c -> c
    dst_parts.append(np.arange(n, dtype=np.int64))
    src = torch.from_numpy(np.concatenate(src_parts))
    dst = torch.from_numpy(np.concatenate(dst_parts))
    stats = {"nodes": int(n), "candidate_pairs": int(centres.size), "selected_edges": int(src.numel() - n),
             "nan_key_centres_official_fallback": int(fallback)}
    return src, dst, stats


def mrq_cached(g, x: torch.Tensor, dataset: str):
    """mrq_one_hop_exact with an on-disk cache, as the official code caches
    ``./mrqs/<dataset>.1hop.sp_matrix``. The key hashes the exact edge list and
    transformed features, so a cache entry can never serve different inputs."""
    import hashlib
    import os

    src_e, dst_e = g.edges()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(src_e.cpu().numpy()).tobytes())
    digest.update(np.ascontiguousarray(dst_e.cpu().numpy()).tobytes())
    digest.update(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes())
    key = digest.hexdigest()[:20]
    cache_dir = pdata.BUNDLE_ROOT / "data" / "cache"
    path = cache_dir / f"sagad_mrq_{dataset}_{key}.npz"
    if path.exists():
        with np.load(path) as payload:
            stats = json.loads(str(payload["stats"]))
            stats["cache"] = f"hit ({path.name})"
            return torch.from_numpy(payload["src"]), torch.from_numpy(payload["dst"]), stats
    src, dst, stats = mrq_one_hop_exact(g, x)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")  # unique per process
    np.savez(tmp, src=src.numpy(), dst=dst.numpy(), stats=json.dumps(stats))
    os.replace(tmp, path)
    stats["cache"] = f"written ({path.name})"
    return src, dst, stats


class SAGADAdapter(Adapter):
    """COMP8851 Phase 2 Adapter for SAGAD (Scalable Adaptive Graph Anomaly Detection, WWW 2026).

    Integrates the pinned upstream SAGAD architecture (ChebNetII spectral filtering +
    Multi-Resolution Quantile 1-hop neighbor aggregation + NodeDimGated fusion) with
    strict training-only label isolation and validation-AUPRC checkpoint selection.
    """

    name = "SAGAD"
    slug = "sagad"
    repo = "sagad"
    view = "homogeneous"

    def setup(self) -> None:
        """Construct the canonical graph, precompute Chebyshev polynomials T_k(X) and MRQ features,
        isolate training labels, and initialize the SAGAD network and Adam optimizer."""
        import dgl

        ctx, hp = self.ctx, self.ctx.hp
        # Load pinned upstream SAGAD modules verified against third_party/manifest.json
        self.model_mod = official.load("sagad", "model")
        self.utils = official.load("sagad", "utils")
        self.loader = official.load("sagad", "dataloader")
        ds = ctx.data.dataset

        # 1. Build bidirected homogeneous graph with single self-loops and normalize node features
        g = build_graph(ds)
        x = torch.as_tensor(ds.features, dtype=torch.float32).clone()
        x = feature_transform(x, hp["feat_trans"], self.loader.standard_scale)

        # 2. Precompute K-order Chebyshev spectral graph propagations T_0(X) .. T_K(X)
        tx_list = self.model_mod.precompute_Tx(g, x, hp["K"], conv_norm=hp["conv_norm"])

        # 3. Compute exact vectorized 1-hop Multi-Resolution Quantile (MRQ) neighbor subgraph
        #    and concatenate raw node features X with mean MRQ-neighbor features X_nei
        src, dst, stats = mrq_cached(g, x, ds.name)
        g_mrq = dgl.graph((src, dst), num_nodes=ds.num_nodes)
        x_nei = dgl.ops.copy_u_mean(g_mrq, x)
        x = torch.concat([x, x_nei], dim=1)
        ctx.note("mrq_sampling", {**stats, "implementation": "mrq_one_hop_exact (official rule, vectorised)"})

        # 4. Transfer full-graph precomputed tensors to target compute device (CUDA/CPU)
        dev = ctx.device
        self.X = x.to(dev)
        self.Tx = [t.to(dev) for t in tx_list]
        self._slices: Dict[int, Tuple[list, torch.Tensor]] = {}

        # 5. Extract training node indices and verify strict label isolation (no val/test labels)
        train_idx = torch.as_tensor(ctx.data.train_idx, dtype=torch.long, device=dev)
        y_all = torch.as_tensor(ctx.data.train_labels_only, dtype=torch.long, device=dev)
        self.Y_train = y_all[train_idx]
        if (self.Y_train < 0).any():
            raise RuntimeError("masked label reached the training set")
        self.Tx_train = [t[train_idx] for t in self.Tx]
        self.X_train = self.X[train_idx]

        # 6. Compute class-imbalance weight w = (#benign / #fraud) and gate regularization targets p
        #    following upstream main.py::run verbatim
        w = (1 - self.Y_train).sum().item() / self.Y_train.sum().item()
        self.weights_re = torch.ones_like(self.Y_train, device=dev, dtype=torch.float32)
        self.weights_re[self.Y_train == 1] = w
        self.d_train = self.Y_train.clone().float()
        self.d_train[self.Y_train == 1] = hp["p"][0]
        self.d_train[self.Y_train == 0] = hp["p"][1]
        self.weights_ce = torch.tensor([1.0, w], device=dev)
        self.class_weight = w

        # 7. Seed RNGs, instantiate SAGAD model, and configure Adam optimizer
        self.utils.set_seed(ctx.seed)
        self.model = self._build_model().to(dev)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])

    def _build_model(self):
        """Instantiate the upstream SAGAD network with the configured spectral-spatial fusion head."""
        hp, m = self.ctx.hp, self.model_mod
        in_dim = self.X.shape[1]
        emb_dim = in_dim // 2
        fusion_name = hp.get("fusion", "NodeDimGated")
        # Select low-pass / high-pass spectral fusion mechanism
        if fusion_name in ("NodeDimGated", "NodeDimGatedWoReg"):
            fusion = m.NodeDimGatedFusion(in_dim, emb_dim, hp["num_gate_layers"])
        elif fusion_name == "Low":
            fusion = m.LowFusion
        elif fusion_name == "High":
            fusion = m.HighFusion
        elif fusion_name == "Mean":
            fusion = m.MeanFusion
        elif fusion_name == "Concat":
            fusion = m.ConcatFusion(in_dim)
        elif fusion_name == "Scalar":
            fusion = m.ScalarVectorFusion(in_dim, scalar=True)
        elif fusion_name == "Attention":
            fusion = m.AttentionFusion(in_dim)
        else:
            raise ValueError(f"unknown fusion {fusion_name}")
        return m.SAGAD(in_dim, emb_dim, hid_dims=list(hp["hid_dims"]) + [2], K=hp["K"],
                       dropout=hp["dropout"], activation=hp["activation"], mlp_norm=hp["mlp_norm"],
                       fusion=fusion)

    def train_epoch(self, epoch: int) -> float:
        """Execute one full-batch training step combining weighted CE loss and gate BCE regularization."""
        self.model.train()
        self.optimizer.zero_grad()
        if self.ctx.hp.get("fusion", "NodeDimGated") == "NodeDimGated":
            # Forward pass returning classification logits and node-dimension gate coefficients
            logits, coefs = self.model(self.Tx_train, self.X_train, return_coefs=True)
            loss_ce = F.cross_entropy(logits, self.Y_train, weight=self.weights_ce)
            loss_re = F.binary_cross_entropy(coefs.mean(1), self.d_train, weight=self.weights_re)
            loss = loss_ce + loss_re
        else:
            logits = self.model(self.Tx_train, self.X_train, return_coefs=False)
            loss = F.cross_entropy(logits, self.Y_train, weight=self.weights_ce)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def _slice(self, node_ids: np.ndarray):
        """Row slices of the precomputed T_k(x) and X, cached for the (fixed)
        validation and test ID arrays so they are gathered once."""
        key = hash(np.ascontiguousarray(node_ids).tobytes())
        if key not in self._slices:
            if len(self._slices) >= 2:
                self._slices.pop(next(iter(self._slices)))
            idx = torch.as_tensor(node_ids, dtype=torch.long, device=self.ctx.device)
            self._slices[key] = ([t[idx] for t in self.Tx], self.X[idx])
        return self._slices[key]

    @torch.no_grad()
    def predict_with(self, model, node_ids: np.ndarray) -> np.ndarray:
        """Compute positive-class (fraud=1) softmax probabilities for the requested node IDs."""
        model.eval()
        tx, x = self._slice(node_ids)
        return model(tx, x).softmax(dim=1)[:, 1].detach().cpu().numpy()

    def predict(self, node_ids: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities using the active model instance."""
        return self.predict_with(self.model, node_ids)

    def fresh_copy_for_reload(self):
        """Construct a fresh, uninitialized SAGADAdapter sharing precomputed tensors for checkpoint reload verification."""
        twin = SAGADAdapter.__new__(SAGADAdapter)
        twin.ctx, twin.model_mod, twin.X, twin.Tx, twin._slices = self.ctx, self.model_mod, self.X, self.Tx, {}
        twin.model = self._build_model().to(self.ctx.device)
        return twin

    def describe(self) -> Dict[str, object]:
        """Return a structured manifest describing the SAGAD adapter configuration for summary.json."""
        return {"graph": "canonical homogeneous union -> to_bidirected -> remove/add self-loop (load_data)",
                "input": "concat[X, mean_{MRQ one-hop}(X)] (main.py)", "class_weight_w": getattr(self, "class_weight", None),
                "loss": "CE(weight=[1,w]) + BCE(gate.mean(1), p-targets, weight=w on anomalies)",
                "optimizer": "Adam(lr, weight_decay=wd)", "batching": "full batch over training nodes"}
