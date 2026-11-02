"""SpaceGNN adapter (official code: xydong127/SpaceGNN @ 921c03ff879b).

Reused verbatim: ``SpaceGNN`` / ``CurvGNN`` / ``CurvLayer`` (model.py),
``set_seed`` and ``propogate`` (utils.py), ``PARAMETERS`` and ``num_class``
(name.py, imported so the curvature std draws are the authors' exact values).

Replicated from main.py (a module-level script that cannot be imported):
curvature initialisation (set_seed -> normal_ -> set_seed), the training loop
(full one-hop neighbour blocks over shuffled batches of training nodes,
``drop_last=True``, NLL on the alpha/beta mixture of the three log-softmax
heads), and the evaluation mixture of the exponentiated heads.

Documented adaptations (audit/DEVIATIONS.md):

* utils.load_data reads GADBench files; the same operations (homogeneous
  graph + self-loops, feature_i = i-step mean propagation) are applied to the
  frozen canonical union view;
* main.py passes ``(beta, alpha)`` to the *validation* call (line 149) but
  ``(alpha, beta)`` to the test call; we use the paper's/test-time mixture
  (Algorithm 4) for both, so the checkpoint is selected with the function it
  is scored with;
* checkpoint selection: validation AUPRC every epoch (protocol) instead of
  validation AUC + Macro-F1 after epoch 25;
* the official script never moves tensors to a GPU; we run on the configured
  device (numerics identical up to floating-point non-associativity);
* torch.use_deterministic_algorithms is kept but switched to warn_only so a
  CUDA kernel without a deterministic variant warns instead of aborting.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from phase2 import data as pdata
from phase2 import official
from phase2.harness import Adapter

OFFICIAL_KEY = {"amazon": "amazon", "yelpchi": "yelp", "tfinance": "tfinance", "tsocial": "tsocial"}


class SpaceGNNAdapter(Adapter):
    """COMP8851 Phase 2 Adapter for SpaceGNN (Multi-Space Graph Neural Network, ICLR 2025).

    Integrates the pinned upstream SpaceGNN architecture (three parallel constant-curvature
    Riemannian CurvGNN heads: negative-curvature hyperbolic c<0, positive-curvature spherical c>0,
    and flat Euclidean c=0) with deterministic CSR block inference and leak-free training labels.
    """

    name = "SpaceGNN"
    slug = "spacegnn"
    repo = "spacegnn"
    view = "homogeneous"

    def setup(self) -> None:
        """Initialize curvature distributions, precompute multi-hop propagated features,
        construct mini-batch neighbor samplers, and instantiate the SpaceGNN model."""
        import dgl

        ctx, hp = self.ctx, self.ctx.hp
        # Load pinned upstream SpaceGNN modules verified against third_party/manifest.json
        self.names = official.load("spacegnn", "name")
        self.utils = official.load("spacegnn", "utils")
        self.model_mod = official.load("spacegnn", "model")
        ds = ctx.data.dataset

        # 1. Resolve curvature standard deviations (stdneg, stdpos) from upstream PARAMETERS
        #    for natively supported datasets or documented midpoint (0.015) for Elliptic/FDCompCN
        stdneg, stdpos = hp["stdneg"], hp["stdpos"]
        if stdneg == "official" or stdpos == "official":
            params = self.names.PARAMETERS[OFFICIAL_KEY[ds.name]]
            stdneg = params[5] if stdneg == "official" else stdneg
            stdpos = params[6] if stdpos == "official" else stdpos
        self.stdneg, self.stdpos = float(stdneg), float(stdpos)
        ctx.note("curvature_std", {"stdneg": self.stdneg, "stdpos": self.stdpos})

        # 2. Sample initial learnable negative (hyperbolic) and positive (spherical) curvatures
        #    using the exact upstream RNG seeding sequence
        layer_num = int(hp["layer_num"])
        self.utils.set_seed(ctx.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.cneg = torch.FloatTensor(layer_num).normal_(-0.1, self.stdneg)
        self.cpos = torch.FloatTensor(layer_num).normal_(0.1, self.stdpos)
        self.utils.set_seed(ctx.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # 3. Construct homogeneous graph with self-loops and precompute i-step mean-propagated
        #    features (feature_0 .. feature_{layer_num-1}) exactly as in utils.load_data
        src, dst = pdata.homogeneous_edges(ds)
        graph = dgl.graph((torch.as_tensor(src), torch.as_tensor(dst)), num_nodes=ds.num_nodes)
        graph = graph.add_self_loop()
        # Attach isolated label vector (validation and test labels masked to -1)
        graph.ndata["label"] = torch.as_tensor(ctx.data.train_labels_only, dtype=torch.long)
        graph.ndata["feature_0"] = torch.as_tensor(ds.features, dtype=torch.float32)
        h = graph.ndata["feature_0"]
        for i in range(1, layer_num):
            h = self.utils.propogate(h, graph)
            graph.ndata[f"feature_{i}"] = h
        self.graph = graph.long().to(ctx.device)

        # 4. Instantiate SpaceGNN model, Adam optimizer, and 1-hop full-neighbor training loader
        self.model = self._build_model().to(ctx.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hp["lr"])
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        train_index = torch.as_tensor(ctx.data.train_idx, dtype=torch.long)
        self.train_loader = torch.utils.data.DataLoader(train_index, batch_size=int(hp["batch_size"]),
                                                        shuffle=True, drop_last=True)
        self.alpha, self.beta = float(hp["alpha"]), float(hp["beta"])
        self.eval_batch = int(ctx.args.eval_batch_size or 4096)

    def _build_model(self):
        """Instantiate the upstream SpaceGNN network with cloned initial curvature vectors."""
        hp = self.ctx.hp
        return self.model_mod.SpaceGNN(self.graph.ndata["feature_0"].shape[1], int(hp["hid_dim"]),
                                       self.names.num_class, int(hp["layer_num"]), float(hp["drop_rate"]),
                                       self.cneg.clone(), self.cpos.clone())

    def train_epoch(self, epoch: int) -> float:
        """Execute one epoch of mini-batch training over training nodes using the alpha/beta log-probability mixture."""
        self.model.train()
        losses = []
        alpha, beta = self.alpha, self.beta
        for index in self.train_loader:
            index = index.to(self.ctx.device)
            # Sample 1-hop full-neighbor subgraph block for current mini-batch
            input_nodes, output_nodes, subgraphs = self.sampler.sample_blocks(self.graph, index)
            probs1, probs2, probs3 = self.model(subgraphs)
            target = subgraphs[-1].dstdata["label"]
            # Safety assertion: ensure no non-training (-1) label ever enters NLL loss
            if bool((target < 0).any()):
                raise RuntimeError("masked label reached the loss")
            # Combine hyperbolic (probs1), spherical (probs2), and Euclidean (probs3) log-probabilities
            probs = (1 - beta) * ((1 - alpha) * probs1 + alpha * probs2) + beta * probs3
            loss = F.nll_loss(probs, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError("no training batch: fewer training nodes than batch_size with drop_last=True")
        return float(np.mean(losses))

    def full_blocks(self, index: torch.Tensor):
        """The one-hop full-neighbour block, built deterministically.

        ``MultiLayerFullNeighborSampler(1).sample_blocks`` returns the same edge
        set but in a call-dependent order. SpaceGNN's hyperbolic head maps
        embeddings onto the Poincare-ball boundary (proj with maxnorm
        (1-1e-5)/sqrt|c|), where arctanh amplifies float32 summation-order
        noise ~1e4-fold, so the sampler's ordering makes inference
        irreproducible (measured: 4.7e-4 between two calls on one model).
        ``in_subgraph`` + ``to_block`` yields the identical block contents in a
        fixed (CSR) order. Training keeps the authors' sampler.
        """
        import dgl

        if not getattr(self, "_cpu_blocks", False):
            try:
                frontier = dgl.in_subgraph(self.graph, index)
                return [dgl.to_block(frontier, index)]
            except Exception as exc:  # noqa: BLE001 - DGL build without GPU in_subgraph
                self._cpu_blocks = True
                self._graph_cpu = self.graph.cpu()
                self.ctx.note("spacegnn_block_construction", f"GPU in_subgraph unavailable ({type(exc).__name__}); "
                                                             "blocks built on a CPU copy (identical contents)")
        frontier = dgl.in_subgraph(self._graph_cpu, index.cpu())
        return [dgl.to_block(frontier, index.cpu()).to(self.graph.device)]

    @torch.no_grad()
    def predict_with(self, model, node_ids: np.ndarray) -> np.ndarray:
        """Compute positive-class (fraud=1) probabilities using paper Algorithm 4 mixture over deterministic blocks."""
        model.eval()
        ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.ctx.device)
        out = []
        for start in range(0, ids.numel(), self.eval_batch):
            index = ids[start:start + self.eval_batch]
            blocks = self.full_blocks(index)
            p1, p2, p3 = model(blocks)
            # Exponentiate log-softmax outputs to probabilities before convex alpha/beta mixing
            p1, p2, p3 = p1.exp()[:, 1], p2.exp()[:, 1], p3.exp()[:, 1]
            out.append(((1 - self.beta) * ((1 - self.alpha) * p1 + self.alpha * p2) + self.beta * p3).cpu())
        return torch.cat(out).numpy()

    def predict(self, node_ids: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities using the active SpaceGNN model."""
        return self.predict_with(self.model, node_ids)

    def fresh_copy_for_reload(self):
        """Construct a fresh SpaceGNNAdapter sharing the precomputed graph for checkpoint reload verification."""
        twin = SpaceGNNAdapter.__new__(SpaceGNNAdapter)
        for attr in ("ctx", "names", "model_mod", "graph", "sampler", "alpha", "beta", "eval_batch", "cneg", "cpos"):
            setattr(twin, attr, getattr(self, attr))
        twin.model = self._build_model().to(self.ctx.device)
        return twin

    def describe(self) -> Dict[str, object]:
        """Return a structured manifest describing the SpaceGNN adapter configuration for summary.json."""
        return {"graph": "canonical homogeneous union + self-loops (utils.load_data)",
                "features": "feature_i = i-step mean propagation, i < layer_num (utils.propogate)",
                "heads": "hyperbolic (c<0), spherical (c>0), Euclidean (c=0) CurvGNNs",
                "train_mixture": "(1-beta)((1-alpha)logp1 + alpha logp2) + beta logp3 -> NLL (main.py)",
                "eval_mixture": "(1-beta)((1-alpha)p1 + alpha p2) + beta p3 (paper Alg. 4 / main.py test call)",
                "optimizer": "Adam(lr), no weight decay", "eval_batch_size": getattr(self, "eval_batch", None)}
