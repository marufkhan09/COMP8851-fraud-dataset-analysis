"""CRoC adapter (official code: XsLangley/CRoC_ECAI2025 @ 70cb7c6c654f).

Reused verbatim: model classes (models.py: CRoCGIN/CRoCSAGE and their _Large
variants, RelGINConv/RelSAGEConv), ``GNNTrainer.train_epoch`` (trainers.py:
balanced positive/negative batches from ``DSDataset``, q sampled unlabelled
nodes, alpha feature mixing, raw + shuffled cross-entropy, InfoNCE with
temperature 2 and k negatives, loss = CE_raw + gamma CE_shuf + eta InfoNCE),
``sample_blocks`` / ``set_random_seed`` (utils.py), and the official inference
path (``model.inference`` with a one-layer full-neighbour loader when the
graph has more than 10,000 nodes, a full-graph forward otherwise).

Only ``GNNTrainer.train`` (the outer loop with its own checkpointing and test
pass) is not used: the protocol harness replaces it.

Documented adaptations (audit/DEVIATIONS.md):

* preprocess.py only knows yelp/amazon/tfin/tsoc/dgraphfin. We route by the
  property that distinguishes its branches: multi-relational review/company
  graphs (YelpChi, Amazon, FDCompCN) -> split_yelpamz path (relation one-hot
  edge types, per-feature min-max, base model); single-relation graphs
  (T-Finance, T-Social, Elliptic) -> split_tfintsoc path (constant 2-d edge
  type, z-score features, *_Large model), which is main.py's rule
  ``model = X if dataset in ['yelp', 'amazon'] else X_Large``;
* the graph handed to the trainer carries labels for training nodes only and
  all-False validation/test masks, so no evaluation label can reach it;
* ``dgl.seed`` is set (the official code never seeds DGL's sampler);
* DataLoader worker count is configurable (official: 4) - it does not change
  the sampled batches, which are drawn in the main process;
* checkpoint selection on validation AUPRC (official: validation AUC).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from phase2 import data as pdata
from phase2 import official
from phase2.harness import Adapter

MULTI_RELATIONAL = ("yelpchi", "amazon", "fdcompcn")


def make_trainer_class(trainers_mod):
    base = trainers_mod.GNNTrainer

    class ProtocolTrainer(base):
        """Official trainer with a configurable DataLoader worker count."""

        num_workers = 4

        def build_loaders(self):
            import dataloading  # the authors' module (resolved via official.activate)
            from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

            train_dataset = dataloading.DSDataset(self.g)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.params["bs"], shuffle=True, drop_last=False,
                num_workers=self.num_workers)
            sampler = MultiLayerFullNeighborSampler(1)
            self.eval_dataloader = DataLoader(
                self.g, torch.arange(self.g.num_nodes()), sampler, batch_size=10 * self.params["bs"],
                shuffle=False, drop_last=False, num_workers=self.num_workers)

    return ProtocolTrainer


class CRoCAdapter(Adapter):
    """COMP8851 Phase 2 Adapter for CRoC (Cross-View Contrastive GNN, ECAI 2025).

    Integrates the pinned upstream CRoC implementation (CRoCGIN / CRoCSAGE and _Large variants,
    context refactoring via alpha-feature mixing, and InfoNCE mutual information maximization)
    with strict training-only mask isolation and validation-AUPRC checkpointing.
    """

    name = "CRoC"
    slug = "croc"
    repo = "croc"
    view = "relation"

    def setup(self) -> None:
        """Construct the relational or single-relation DGL graph, normalize features per upstream
        branch rules, isolate training masks, and initialize the CRoC model and GNNTrainer."""
        import dgl

        ctx, hp = self.ctx, self.ctx.hp
        # Load pinned upstream CRoC modules verified against third_party/manifest.json
        self.models = official.load("croc", "models")
        self.trainers = official.load("croc", "trainers")
        self.cutils = official.load("croc", "utils")
        ds = ctx.data.dataset
        n = ds.num_nodes
        feats = torch.as_tensor(ds.features, dtype=torch.float32)

        # 1. Route dataset to either the multi-relational branch (YelpChi, Amazon, FDCompCN)
        #    or the single-relation branch (T-Finance, T-Social, Elliptic) following preprocess.py
        self.branch = "relational" if ds.name in MULTI_RELATIONAL else "single_relation"
        if self.branch == "relational":
            src, dst, etype = pdata.typed_edges(ds)
            g = dgl.graph((torch.as_tensor(src), torch.as_tensor(dst)), num_nodes=n)
            # Encode semantic relation types as one-hot edge feature vectors
            g.edata["type"] = torch.nn.functional.one_hot(torch.as_tensor(etype)).float()
            n_rel = int(g.edata["type"].shape[1])
            # Apply column-wise min-max feature scaling as in split_yelpamz(normfeat=True)
            feat = (feats - feats.min(dim=0)[0]) / (feats.max(dim=0)[0] - feats.min(dim=0)[0])
            model_cls = hp["backbone"]
            relations = pdata.relation_order(ds)
        else:
            src, dst = pdata.homogeneous_edges(ds)
            g = dgl.graph((torch.as_tensor(src), torch.as_tensor(dst)), num_nodes=n)
            n_rel = 2
            # Assign constant 2-D edge type [1, 0] as in split_tfintsoc
            g.edata["type"] = torch.zeros(g.num_edges(), 2).float()
            g.edata["type"][:, 0] = 1
            # Apply z-score feature standardization and select *_Large model variant
            feat = (feats - feats.mean(dim=0)) / feats.std(dim=0)
            model_cls = hp["backbone"] + "_Large"
            relations = ["single (constant type)"]
        if not torch.isfinite(feat).all():
            raise RuntimeError("feature normalisation produced non-finite values (constant column?)")

        # 2. Populate graph node features and enforce strict training-mask isolation
        #    (val_mask and test_mask are set to all-False so GNNTrainer never sees evaluation nodes)
        g.ndata["feat"] = feat
        g.ndata["label"] = torch.as_tensor(np.clip(ctx.data.train_labels_only, 0, None), dtype=torch.long)
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[torch.as_tensor(ctx.data.train_idx)] = True
        g.ndata["train_mask"] = train_mask
        g.ndata["val_mask"] = torch.zeros(n, dtype=torch.bool)   # never given to the trainer
        g.ndata["test_mask"] = torch.zeros(n, dtype=torch.bool)  # never given to the trainer
        self.g = g
        self.model_cls = model_cls
        ctx.note("croc_branch", {"branch": self.branch, "model_class": model_cls, "n_rel": n_rel,
                                 "relations_in_type_order": relations})

        # 3. Assemble upstream parameter dictionary and instantiate CRoC model + ProtocolTrainer
        self.params = {
            "in_dim": int(feat.shape[1]), "out_dim": 2, "n_rel": n_rel,
            "model": model_cls, "dataset": ds.name, "n_layer": int(hp["n_layer"]),
            "n_neb": [int(v) for v in hp["n_neb"]], "hid_dim": int(hp["hid_dim"]), "bn": bool(hp["bn"]),
            "dropout": float(hp["dropout"]), "lr": float(hp["lr"]), "wd": float(hp["wd"]),
            "bs": int(hp["bs"]), "alpha": float(hp["alpha"]), "gamma": float(hp["gamma"]),
            "eta": float(hp["eta"]), "q": int(hp["q"]), "k": int(hp["k"]),
            "n_epoch": int(ctx.args.epochs or 0), "split": f"comp8851-{ctx.args.ratio}",
            "run": int(ctx.seed), "tags": ["comp8851"], "ts": "", "device": ctx.device,
        }
        self.cutils.set_random_seed(ctx.seed)
        self.model = self._build_model()
        trainer_cls = make_trainer_class(self.trainers)
        trainer_cls.num_workers = 4 if ctx.args.num_workers is None else int(ctx.args.num_workers)
        self.trainer = trainer_cls(self.g, self.model, params=self.params)
        self.optimizer = self.trainer.optimizer
        if self.params["n_neb"] and len(self.params["n_neb"]) != self.params["n_layer"]:
            raise ValueError("n_neb must give one fanout per layer")

    def _build_model(self):
        """Instantiate the selected CRoC backbone (CRoCGIN, CRoCSAGE, or *_Large) on target device."""
        return getattr(self.models, self.model_cls)(self.params).to(self.ctx.device)

    def train_epoch(self, epoch: int) -> float:
        """Execute one epoch of upstream GNNTrainer.train_epoch (CE_raw + gamma*CE_mixed + eta*InfoNCE)."""
        self.invalidate_cache()
        metrics, _ = self.trainer.train_epoch()
        return float(metrics["loss"])

    def invalidate_cache(self) -> None:
        """Clear cached all-node inference probabilities whenever model weights are updated."""
        self._cache = None

    def load_model_state(self, state) -> None:
        """Load saved model weights and invalidate any stale cached predictions."""
        self.invalidate_cache()
        super().load_model_state(state)

    @torch.no_grad()
    def predict_with(self, model, node_ids: np.ndarray) -> np.ndarray:
        """Score nodes using upstream full-graph forward (<=10k nodes) or layer-wise inference (>10k nodes)."""
        # trainers.py::eval_epoch scoring path. The official inference scores
        # every node; the all-node result is cached until the weights change, so
        # validation and test after loading the best checkpoint share one pass.
        cache = getattr(self, "_cache", None)
        if cache is None or cache[0] is not model:
            model.eval()
            dev = self.ctx.device
            if self.g.num_nodes() > 10000:
                scores, _ = model.inference(self.g.ndata["feat"].to(dev), self.trainer.eval_dataloader)
            else:
                scores, _ = model(self.g.to(dev), self.g.ndata["feat"].to(dev))
            cache = (model, scores.softmax(dim=-1)[:, 1])
            self._cache = cache
        probs = cache[1]
        return probs[torch.as_tensor(node_ids, dtype=torch.long, device=probs.device)].cpu().numpy()

    def predict(self, node_ids: np.ndarray) -> np.ndarray:
        """Predict positive-class (fraud=1) probabilities for the requested node IDs."""
        return self.predict_with(self.model, node_ids)

    def fresh_copy_for_reload(self):
        """Construct a fresh CRoCAdapter sharing the graph and trainer for checkpoint reload verification."""
        twin = CRoCAdapter.__new__(CRoCAdapter)
        for attr in ("ctx", "models", "g", "model_cls", "params", "trainer"):
            setattr(twin, attr, getattr(self, attr))
        twin.model = self._build_model()
        return twin

    def describe(self) -> Dict[str, object]:
        """Return a structured manifest describing the CRoC adapter configuration for summary.json."""
        return {"branch": getattr(self, "branch", None), "model_class": getattr(self, "model_cls", None),
                "loss": "CE(raw) + gamma*CE(alpha-mixed features) + eta*InfoNCE(tau=2, k negatives, q unlabelled)",
                "sampling": "balanced pos/neg batches (DSDataset), fanouts n_neb, q unlabelled nodes/batch",
                "optimizer": "Adam(lr, weight_decay=wd)", "inference": "official (layer-wise if > 10,000 nodes)"}
