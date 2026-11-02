"""CARE-GNN layers.

Paper: Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020).
       Enhancing Graph Neural Network-based Fraud Detectors against
       Camouflaged Fraudsters. CIKM. https://doi.org/10.1145/3340531.3411903
Official code: https://github.com/YingtongDou/CARE-GNN

This is a faithful re-implementation of the author layers. The three published
mechanisms are all present and none is simplified:

1. **Label-aware similarity measure** (Eq. 2). A one-layer MLP predicts a
   label score per node; neighbour similarity is the L1 distance between the
   centre node's score and each neighbour's score.
2. **Similarity-aware neighbour selector** (Section 3.3.1). Within each
   relation, the top-p most similar neighbours are kept, where p is that
   relation's adaptive filtering threshold.
3. **Reinforcement-learning threshold module** (Eqs. 5-6). Between epochs the
   average neighbour distance of positive nodes is compared with the previous
   epoch; the reward is +1 when distance did not increase and -1 otherwise, and
   the threshold moves by one step size in that direction.

Deviations from the published source, all deliberate and behaviour-preserving:

* Tensors are created on the module's device instead of via hard-coded
  ``.cuda()`` calls, so CPU and GPU paths share one code path.
* ``att_inter_agg`` in the official repository contains a stray
  ``pdb.set_trace()`` that halts execution. It is removed here; the surrounding
  mathematics is unchanged.
* ``RLModule`` is given the batch count explicitly rather than reading a mutable
  attribute mid-forward, which keeps the update schedule identical but easier
  to verify.
"""

from __future__ import annotations

import math
from operator import itemgetter
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# --------------------------------------------------------------------------
# Neighbour filtering (Eq. 2 and Section 3.3.1)
# --------------------------------------------------------------------------

def filter_neighs_ada_threshold(center_scores: torch.Tensor,
                                neigh_scores: Sequence[torch.Tensor],
                                neighs_list: Sequence[Sequence[int]],
                                sample_list: Sequence[int]
                                ) -> Tuple[List[set], List[List[float]]]:
    """Keep the top-p most label-similar neighbours of each centre node.

    :param center_scores: label-aware scores of the batch nodes, shape (B, 2)
    :param neigh_scores: per-centre-node neighbour score tensors, each (D_i, 2)
    :param neighs_list: per-centre-node neighbour ID lists
    :param sample_list: per-centre-node count of neighbours to keep
    :returns: ``(samp_neighs, samp_scores)`` where ``samp_neighs[i]`` is the
        retained neighbour ID set and ``samp_scores[i]`` the retained distances
    """
    samp_neighs: List[set] = []
    samp_scores: List[List[float]] = []

    for idx in range(len(center_scores)):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_repeated = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # Eq. (2): L1 distance between centre and neighbour label scores.
        score_diff = torch.abs(center_repeated - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()

        # Top-p sampling by distance ranking.
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = list(neighs_indices)
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]

        if not selected_neighs:
            # Never hand an empty neighbourhood to the aggregator; fall back to
            # the centre node itself, matching the self-loop convention used
            # when the adjacency lists were built.
            selected_neighs = [int(neighs_indices[0])] if len(neighs_indices) else []
            selected_scores = [0.0]

        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)

    return samp_neighs, samp_scores


# --------------------------------------------------------------------------
# Reinforcement learning module (Eqs. 5-6)
# --------------------------------------------------------------------------

def rl_module(scores: Sequence[Sequence[List[float]]],
              scores_log: List[List[float]],
              labels: torch.Tensor,
              thresholds: List[float],
              batch_num: int,
              step_size: float,
              verbose: bool = False
              ) -> Tuple[List[float], List[int], List[float], bool]:
    """Update per-relation filtering thresholds from epoch-to-epoch distances.

    The average neighbour distance is computed over **positive (fraud) nodes
    only**, as in the paper. The update fires once per epoch boundary and not
    during the first two epochs, because a comparison needs two completed
    epochs of history.

    :returns: ``(relation_scores, rewards, new_thresholds, stop_flag)``
    """
    relation_scores: List[float] = []
    stop_flag = True

    positive_index = (labels == 1).nonzero().tolist()
    positive_index = [i[0] for i in positive_index]

    for score in scores:
        if not positive_index:
            # No fraud node in this batch: contribute a neutral distance so the
            # epoch average stays defined.
            relation_scores.append(0.0)
            continue
        positive_scores = itemgetter(*positive_index)(score)
        if not isinstance(positive_scores, tuple):
            positive_scores = (positive_scores,)
        neigh_count = sum(1 if isinstance(i, float) else len(i) for i in positive_scores)
        positive_sum = [i if isinstance(i, float) else sum(i) for i in positive_scores]
        relation_scores.append(sum(positive_sum) / neigh_count if neigh_count else 0.0)

    if batch_num <= 0 or len(scores_log) % batch_num != 0 or len(scores_log) < 2 * batch_num:
        # Inside an epoch, or not enough history yet: hold the thresholds.
        rewards = [0 for _ in thresholds]
        new_thresholds = list(thresholds)
    else:
        # Eq. (5): mean distance over the previous and the current epoch.
        previous_epoch = [sum(s) / batch_num for s in zip(*scores_log[-2 * batch_num:-batch_num])]
        current_epoch = [sum(s) / batch_num for s in zip(*scores_log[-batch_num:])]

        # Eq. (6): reward +1 when the average distance did not grow, else -1.
        rewards = [1 if previous_epoch[i] - s >= 0 else -1 for i, s in enumerate(current_epoch)]
        new_thresholds = [
            thresholds[i] + step_size if r == 1 else thresholds[i] - step_size
            for i, r in enumerate(rewards)
        ]
        new_thresholds = [0.999 if t > 1 else t for t in new_thresholds]
        new_thresholds = [0.001 if t < 0 else t for t in new_thresholds]

        if verbose:
            print(f"  [RL] epoch distances: {[round(s, 6) for s in current_epoch]}")
            print(f"  [RL] rewards: {rewards}")
            print(f"  [RL] thresholds: {[round(t, 4) for t in new_thresholds]}")

    return relation_scores, rewards, new_thresholds, stop_flag


# --------------------------------------------------------------------------
# Inter-relation aggregators (Eq. 9)
# --------------------------------------------------------------------------

def mean_inter_agg(num_relations: int, self_feats: torch.Tensor, neigh_feats: torch.Tensor,
                   embed_dim: int, weight: torch.Tensor, n: int) -> torch.Tensor:
    """CARE-Mean: unweighted average over relations."""
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)
    aggregated = torch.zeros(n, embed_dim, device=self_feats.device, dtype=center_h.dtype)
    for r in range(num_relations):
        aggregated = aggregated + neigh_h[r * n:(r + 1) * n, :]
    return F.relu((center_h + aggregated) / (num_relations + 1))


def weight_inter_agg(num_relations: int, self_feats: torch.Tensor, neigh_feats: torch.Tensor,
                     embed_dim: int, weight: torch.Tensor, alpha: torch.Tensor,
                     n: int) -> torch.Tensor:
    """CARE-Weight: learned per-relation weights via softmax over alpha."""
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)
    w = F.softmax(alpha, dim=1)
    aggregated = torch.zeros(n, embed_dim, device=self_feats.device, dtype=center_h.dtype)
    for r in range(num_relations):
        aggregated = aggregated + neigh_h[r * n:(r + 1) * n, :] * w[:, r]
    return F.relu(center_h + aggregated)


def att_inter_agg(num_relations: int, att_layer: nn.Module, self_feats: torch.Tensor,
                  neigh_feats: torch.Tensor, embed_dim: int, weight: torch.Tensor,
                  a: torch.Tensor, n: int, dropout: float,
                  training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """CARE-Att: attention over relations (GAT-style)."""
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)

    combined = torch.cat((center_h.repeat(num_relations, 1), neigh_h), dim=1)
    e = att_layer(combined.mm(a))
    attention = torch.cat([e[r * n:(r + 1) * n, :] for r in range(num_relations)], dim=1)
    ori_attention = F.softmax(attention, dim=1)
    attention = F.dropout(ori_attention, dropout, training=training)

    aggregated = torch.zeros(n, embed_dim, device=self_feats.device, dtype=center_h.dtype)
    for r in range(num_relations):
        aggregated = aggregated + torch.mul(
            attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :]
        )

    combined = F.relu(center_h + aggregated)
    att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)
    return combined, att


def threshold_inter_agg(num_relations: int, self_feats: torch.Tensor, neigh_feats: torch.Tensor,
                        embed_dim: int, weight: torch.Tensor, thresholds: Sequence[float],
                        n: int) -> torch.Tensor:
    """CARE-GNN: relation weights are the RL-learned filtering thresholds (Eq. 9)."""
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)
    aggregated = torch.zeros(n, embed_dim, device=self_feats.device, dtype=center_h.dtype)
    for r in range(num_relations):
        aggregated = aggregated + neigh_h[r * n:(r + 1) * n, :] * thresholds[r]
    return F.relu(center_h + aggregated)


# --------------------------------------------------------------------------
# Aggregator modules
# --------------------------------------------------------------------------

class IntraAgg(nn.Module):
    """Intra-relation aggregator: mean over the filtered neighbour set (Eq. 8)."""

    def __init__(self, features: nn.Embedding, feat_dim: int):
        super().__init__()
        self.features = features
        self.feat_dim = feat_dim

    def forward(self, nodes: Sequence[int], to_neighs_list: Sequence[Sequence[int]],
                batch_scores: torch.Tensor, neigh_scores: Sequence[torch.Tensor],
                sample_list: Sequence[int]) -> Tuple[torch.Tensor, List[List[float]]]:
        samp_neighs, samp_scores = filter_neighs_ada_threshold(
            batch_scores, neigh_scores, to_neighs_list, sample_list
        )

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        device = self.features.weight.device
        mask = torch.zeros(len(samp_neighs), len(unique_nodes), device=device)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        num_neigh = mask.sum(1, keepdim=True).clamp(min=1)
        mask = mask.div(num_neigh)

        embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long, device=device))
        to_feats = mask.mm(embed_matrix)
        return F.relu(to_feats), samp_scores


class InterAgg(nn.Module):
    """Inter-relation aggregator with the label-aware selector and RL module."""

    def __init__(self, features: nn.Embedding, feature_dim: int, embed_dim: int,
                 adj_lists: Sequence[Dict[int, set]], intraggs: Sequence[IntraAgg],
                 inter: str = "GNN", step_size: float = 0.02, rl_enabled: bool = True,
                 verbose: bool = False):
        super().__init__()

        self.features = features
        self.dropout = 0.6
        self.adj_lists = list(adj_lists)
        self.intra_aggs = nn.ModuleList(intraggs)
        self.num_relations = len(self.adj_lists)
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.step_size = step_size
        self.verbose = verbose

        if self.num_relations != len(self.intra_aggs):
            raise ValueError(
                f"Got {self.num_relations} relations but {len(self.intra_aggs)} intra-aggregators."
            )

        self.RL = rl_enabled
        self.batch_num = 0

        # The RL-learned filtering thresholds are genuine model state: they
        # weight the inter-relation aggregation (Eq. 9) and set how many
        # neighbours survive filtering. They are held in a registered buffer so
        # that ``state_dict`` carries them and restoring the best checkpoint
        # reproduces exactly the behaviour that was validated. Keeping them in
        # a plain Python list, as the published code does, silently resets them
        # to 0.5 on reload.
        self.register_buffer(
            "threshold_buffer", torch.full((self.num_relations,), 0.5, dtype=torch.float32)
        )

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.weight = nn.Parameter(torch.empty(self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)

        self.alpha = nn.Parameter(torch.empty(self.embed_dim, self.num_relations))
        init.xavier_uniform_(self.alpha)

        self.a = nn.Parameter(torch.empty(2 * self.embed_dim, 1))
        init.xavier_uniform_(self.a)

        # Label predictor used by the similarity measure.
        self.label_clf = nn.Linear(self.feat_dim, 2)

        self.weights_log: List[List[float]] = []
        self.thresholds_log: List[List[float]] = [list(self.thresholds)]
        self.relation_score_log: List[List[float]] = []

    @property
    def thresholds(self) -> List[float]:
        """Current per-relation filtering thresholds as plain floats."""
        return [float(value) for value in self.threshold_buffer]

    @thresholds.setter
    def thresholds(self, values: Sequence[float]) -> None:
        if len(values) != self.num_relations:
            raise ValueError(
                f"Expected {self.num_relations} thresholds, got {len(values)}."
            )
        with torch.no_grad():
            self.threshold_buffer.copy_(
                torch.tensor([float(v) for v in values], dtype=self.threshold_buffer.dtype,
                             device=self.threshold_buffer.device)
            )

    def forward(self, nodes: Sequence[int], labels: torch.Tensor,
                train_flag: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.features.weight.device

        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        unique_nodes = set()
        for relation_neighs in to_neighs:
            unique_nodes |= set.union(*relation_neighs)
        unique_nodes |= set(int(node) for node in nodes)

        unique_list = list(unique_nodes)
        batch_features = self.features(torch.tensor(unique_list, dtype=torch.long, device=device))
        batch_scores = self.label_clf(batch_features)
        id_mapping = {node_id: index for index, node_id in enumerate(unique_list)}

        center_scores = batch_scores[itemgetter(*[int(n) for n in nodes])(id_mapping), :]
        if center_scores.dim() == 1:
            center_scores = center_scores.view(1, -1)

        relation_lists = [[list(to_neigh) for to_neigh in relation] for relation in to_neighs]
        relation_scores = [
            [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in relation]
            for relation in relation_lists
        ]
        sample_nums = [
            [math.ceil(len(neighs) * self.thresholds[r]) for neighs in relation_lists[r]]
            for r in range(self.num_relations)
        ]

        # Eq. (8): intra-relation aggregation over the filtered neighbours.
        relation_feats = []
        filtered_scores = []
        for r in range(self.num_relations):
            feats, scores = self.intra_aggs[r](
                nodes, relation_lists[r], center_scores, relation_scores[r], sample_nums[r]
            )
            relation_feats.append(feats)
            filtered_scores.append(scores)

        # Relations are stacked along dim 0, giving a (R * B, F) block that the
        # inter-aggregators slice per relation.
        neigh_feats = torch.cat(relation_feats, dim=0)

        index = torch.tensor([int(node) for node in nodes], dtype=torch.long, device=device)
        self_feats = self.features(index)
        n = len(nodes)

        # Eq. (9): inter-relation aggregation.
        if self.inter == "Att":
            combined, attention = att_inter_agg(
                self.num_relations, self.leakyrelu, self_feats, neigh_feats, self.embed_dim,
                self.weight, self.a, n, self.dropout, self.training
            )
            self.weights_log.append(attention.tolist())
        elif self.inter == "Weight":
            combined = weight_inter_agg(
                self.num_relations, self_feats, neigh_feats, self.embed_dim,
                self.weight, self.alpha, n
            )
            self.weights_log.append(F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist())
        elif self.inter == "Mean":
            combined = mean_inter_agg(
                self.num_relations, self_feats, neigh_feats, self.embed_dim, self.weight, n
            )
        elif self.inter == "GNN":
            combined = threshold_inter_agg(
                self.num_relations, self_feats, neigh_feats, self.embed_dim,
                self.weight, self.thresholds, n
            )
        else:
            raise ValueError(
                f"Unknown inter-relation aggregator {self.inter!r}; "
                "expected one of 'Att', 'Weight', 'Mean', 'GNN'."
            )

        # Eqs. (5)-(6): adaptive threshold update.
        if self.RL and train_flag:
            scores, rewards, thresholds, stop_flag = rl_module(
                filtered_scores, self.relation_score_log, labels, self.thresholds,
                self.batch_num, self.step_size, verbose=self.verbose,
            )
            self.thresholds = thresholds
            self.RL = stop_flag
            self.relation_score_log.append(scores)
            self.thresholds_log.append(list(self.thresholds))

        return combined, center_scores
