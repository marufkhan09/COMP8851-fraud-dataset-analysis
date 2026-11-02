"""CARE-GNN model wrapper.

Paper: Dou et al. (2020), CIKM. Official code: https://github.com/YingtongDou/CARE-GNN

``OneLayerCARE`` is the published single-layer Camouflage-Resistant GNN. Its
loss (Eq. 11) is the sum of the GNN cross-entropy and the label-aware
similarity cross-entropy, weighted by ``lambda_1``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import init

from .layers import InterAgg


class OneLayerCARE(nn.Module):
    """One Camouflage-Resistant aggregation layer plus a linear classifier."""

    def __init__(self, num_classes: int, inter1: InterAgg, lambda_1: float = 2.0):
        super().__init__()
        self.inter1 = inter1
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.empty(num_classes, inter1.embed_dim))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1

    def forward(self, nodes: Sequence[int], labels: torch.Tensor,
                train_flag: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds, label_scores = self.inter1(nodes, labels, train_flag)
        # ``embeds`` is (B, embed_dim); the classifier weight is (C, embed_dim).
        scores = embeds.mm(self.weight.t())
        return scores, label_scores

    def to_prob(self, nodes: Sequence[int], labels: torch.Tensor,
                train_flag: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Class probabilities for the GNN head and the label-aware head."""
        gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
        return torch.softmax(gnn_logits, dim=1), torch.softmax(label_logits, dim=1)

    def loss(self, nodes: Sequence[int], labels: torch.Tensor,
             train_flag: bool = True,
             class_weight: torch.Tensor | None = None) -> torch.Tensor:
        """CARE-GNN objective, Eq. (11).

        ``class_weight`` is optional. The published training procedure balances
        classes by under-sampling negatives rather than by weighting the loss;
        pass a weight tensor only when a configuration explicitly selects the
        weighted-loss variant, and record that choice in the run config.
        """
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        target = labels.squeeze()
        if class_weight is not None:
            gnn_loss = nn.functional.cross_entropy(gnn_scores, target, weight=class_weight)
            label_loss = nn.functional.cross_entropy(label_scores, target, weight=class_weight)
        else:
            gnn_loss = self.xent(gnn_scores, target)
            label_loss = self.xent(label_scores, target)
        return gnn_loss + self.lambda_1 * label_loss

    def thresholds(self) -> Dict[str, float]:
        """Current RL-learned filtering thresholds, for logging."""
        return {f"relation_{i}": float(t) for i, t in enumerate(self.inter1.thresholds)}


def build_care_gnn(features: torch.Tensor, adj_lists: Sequence[Dict[int, set]],
                   embed_dim: int = 64, inter: str = "GNN", step_size: float = 0.02,
                   lambda_1: float = 2.0, rl_enabled: bool = True,
                   verbose: bool = False) -> OneLayerCARE:
    """Assemble CARE-GNN for a given feature matrix and relation adjacency lists.

    ``features`` is a dense (N, F) float tensor. It is stored in a frozen
    embedding table exactly as the author code does, so node features are
    looked up by ID during mini-batch training and are never updated.
    """
    num_nodes, feat_dim = features.shape
    feature_table = nn.Embedding(num_nodes, feat_dim)
    feature_table.weight = nn.Parameter(features.float(), requires_grad=False)

    from .layers import IntraAgg

    intra_aggs = [IntraAgg(feature_table, feat_dim) for _ in adj_lists]
    inter_agg = InterAgg(
        features=feature_table,
        feature_dim=feat_dim,
        embed_dim=embed_dim,
        adj_lists=adj_lists,
        intraggs=intra_aggs,
        inter=inter,
        step_size=step_size,
        rl_enabled=rl_enabled,
        verbose=verbose,
    )
    return OneLayerCARE(num_classes=2, inter1=inter_agg, lambda_1=lambda_1)
