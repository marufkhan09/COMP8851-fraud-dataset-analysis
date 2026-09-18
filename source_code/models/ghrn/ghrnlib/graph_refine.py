"""GHRN graph refinement: removing heterophilic edges before message passing.

Paper: Gao, Y., et al. (2023). Addressing Heterophily in Graph Anomaly
       Detection: A Perspective of Graph Spectrum. The Web Conference.
       https://doi.org/10.1145/3543507.3583268
Official code: https://github.com/blacksingular/GHRN

Mechanism
---------
The paper's claim is that heterophily is positively associated with graph
*frequency*: edges joining nodes of different classes carry the high-frequency
energy of the label signal. GHRN therefore estimates, for every edge, how much
high-frequency energy it contributes, and deletes the worst offenders.

The estimator works on a *predicted* label distribution rather than the true
labels, because test labels are unavailable at refinement time:

1. Obtain predicted class posteriors ``pred_y`` for every node. These come
   from a first pass of the backbone trained with ``del_ratio = 0``.
2. Aggregate the posteriors over neighbours with degree-normalised weights,
   giving ``ay`` (the low-pass component).
3. The residual ``ly = pred_y - ay`` is the high-pass component at each node,
   which is the graph-Laplacian applied to the label signal.
4. Score every edge by ``(ly_src . ly_dst)``. Endpoints whose residuals point in
   opposite directions produce a negative score and indicate an inter-class
   (heterophilic) edge.
5. Delete the ``del_ratio`` fraction of edges with the lowest scores, protecting
   self-loops, then re-add self-loops so isolated nodes still receive signal.

The KL variant replaces the dot-product score with the symmetric KL divergence
between endpoint posteriors, which the paper presents as an equivalent
edge-level heterophily indicator.

Protocol note
-------------
The master plan forbids "rewiring or pruning topology to raise performance" as
a *benchmark-level* data manipulation. This pruning is not that: it is the
model's own published mechanism, it is applied inside the model using only
training-visible information, and it never alters the stored canonical dataset.
The refinement runs identically for every dataset and ratio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import dgl

from .backend import edge_endpoints, neighbour_average, prune_edges


def posterior_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert classifier logits to class posteriors."""
    return torch.softmax(logits, dim=1)


def random_walk_update(graph: dgl.DGLGraph, pred_y: torch.Tensor) -> torch.Tensor:
    """Score every edge by the high-frequency energy it carries.

    Returns a 1-D tensor of per-edge scores aligned with ``graph.edges()``.
    Lower (more negative) means more likely heterophilic.

    Self-loops must be present: the degree normalisation is undefined for a node
    with no incoming edge.
    """
    # Low-pass component, then the high-pass residual it fails to explain.
    ay = neighbour_average(graph, pred_y)
    ly = pred_y - ay

    src, dst = edge_endpoints(graph)
    return (ly[src] * ly[dst]).sum(dim=1).detach()


def kl_update(graph: dgl.DGLGraph, pred_y: torch.Tensor) -> torch.Tensor:
    """Alternative edge score: negative symmetric KL between endpoint posteriors.

    A large divergence means the endpoints disagree about the class, so the
    edge is likely heterophilic. Negating keeps the convention that lower
    scores are pruned first.
    """
    probabilities = pred_y.clamp_min(1e-12)
    log_probabilities = torch.log(probabilities)

    src, dst = edge_endpoints(graph)
    p, q = probabilities[src], probabilities[dst]
    log_p, log_q = log_probabilities[src], log_probabilities[dst]

    forward = (p * (log_p - log_q)).sum(dim=1)
    backward = (q * (log_q - log_p)).sum(dim=1)
    return (-(forward + backward)).detach()


def select_edges_to_delete(scores: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                           del_ratio: float,
                           protect_self_loops: bool = True) -> torch.Tensor:
    """Choose which edge IDs to delete, given per-edge heterophily scores.

    Deliberately free of any graph-library call so the selection rule can be
    unit-tested on plain tensors. Lower scores are deleted first.

    :param scores: per-edge score, aligned with ``src``/``dst``
    :param del_ratio: fraction of the total edge count to remove
    :returns: 1-D tensor of edge IDs to delete
    """
    if not 0.0 <= del_ratio < 1.0:
        raise ValueError(f"del_ratio must be in [0, 1), got {del_ratio}")

    num_edges = int(scores.numel())
    num_delete = int(del_ratio * num_edges)
    if num_delete <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    ranked = torch.argsort(scores)          # ascending: most heterophilic first
    candidates = ranked[:num_delete]

    if protect_self_loops:
        candidates = candidates[src[candidates] != dst[candidates]]

    return candidates


def refine_graph(graph: dgl.DGLGraph, pred_y: torch.Tensor, del_ratio: float,
                 mode: str = "post_aggregation",
                 protect_self_loops: bool = True) -> Tuple[dgl.DGLGraph, Dict[str, float]]:
    """Delete the ``del_ratio`` most heterophilic edges and return the new graph.

    :param graph: homogeneous DGL graph, expected to contain self-loops
    :param pred_y: (N, C) predicted class posteriors from the first pass
    :param del_ratio: fraction of edges to remove, in [0, 1)
    :param mode: ``post_aggregation`` (paper default) or ``kl``
    :returns: ``(refined_graph, statistics)``
    """
    if not 0.0 <= del_ratio < 1.0:
        raise ValueError(f"del_ratio must be in [0, 1), got {del_ratio}")

    statistics: Dict[str, float] = {
        "del_ratio": float(del_ratio),
        "mode": mode,
        "edges_before": int(graph.num_edges()),
    }

    if del_ratio == 0.0:
        statistics.update({"edges_removed": 0, "edges_after": int(graph.num_edges())})
        return graph, statistics

    if mode == "post_aggregation":
        scores = random_walk_update(graph, pred_y)
    elif mode == "kl":
        scores = kl_update(graph, pred_y)
    else:
        raise ValueError(f"Unknown refinement mode {mode!r}; expected 'post_aggregation' or 'kl'.")

    src, dst = edge_endpoints(graph)
    candidates = select_edges_to_delete(scores, src, dst, del_ratio, protect_self_loops)
    if candidates.numel() == 0:
        statistics.update({"edges_removed": 0, "edges_after": int(graph.num_edges())})
        return graph, statistics

    refined = prune_edges(graph, candidates)

    statistics.update({
        "edges_removed": int(candidates.numel()),
        "edges_after": int(refined.num_edges()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
    })
    return refined, statistics


def measure_pruning_quality(graph: dgl.DGLGraph, refined: dgl.DGLGraph,
                            labels: torch.Tensor,
                            labelled_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Diagnostic: did pruning actually remove inter-class edges?

    This is reported as evidence, never used to select a model or threshold.
    Only labelled endpoints are counted, and the caller should pass a mask
    restricted to training nodes if it wants a leakage-free diagnostic.
    """
    def heterophily(g) -> Optional[float]:
        src, dst = edge_endpoints(g)
        keep = src != dst
        if labelled_mask is not None:
            keep = keep & labelled_mask[src] & labelled_mask[dst]
        if keep.sum() == 0:
            return None
        return float((labels[src[keep]] != labels[dst[keep]]).float().mean())

    before, after = heterophily(graph), heterophily(refined)
    return {
        "edge_heterophily_before": before,
        "edge_heterophily_after": after,
        "edge_heterophily_reduction": (
            None if before is None or after is None else before - after
        ),
    }
