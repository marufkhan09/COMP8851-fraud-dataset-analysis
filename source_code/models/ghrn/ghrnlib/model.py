"""GHRN model: beta-wavelet backbone plus heterophily-aware graph refinement.

Paper: Gao et al. (2023), WWW. Official code: https://github.com/blacksingular/GHRN

Training is the published two-stage procedure:

Stage 1 (``del_ratio = 0``)
    Train the backbone on the unrefined graph and record its class posteriors.
    This is exactly a BWGNN run and is required before any pruning, matching
    the official repository's instruction that a deletion ratio of 0 must run
    first to generate predictions.

Stage 2 (``del_ratio > 0``)
    Score edges with those posteriors, delete the most heterophilic fraction,
    then retrain the backbone from a fresh initialisation on the refined graph.

Only training-visible information reaches the refinement step: the posteriors
come from a model that saw training labels alone, and validation/test labels
are never consulted. The canonical dataset on disk is never modified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import dgl

from .bwgnn_backbone import BWGNNBackbone
from .graph_refine import posterior_from_logits, refine_graph


class GHRN(nn.Module):
    """Graph refinement wrapper around the beta-wavelet backbone."""

    def __init__(self, in_feats: int, h_feats: int, num_classes: int = 2,
                 d: int = 2, del_ratio: float = 0.0, dropout: float = 0.0,
                 refine_mode: str = "post_aggregation"):
        super().__init__()
        self.backbone = BWGNNBackbone(in_feats, h_feats, num_classes, d=d, dropout=dropout)
        self.del_ratio = float(del_ratio)
        self.refine_mode = refine_mode
        self.refinement_statistics: Dict[str, float] = {}
        self._refined_graph: Optional[dgl.DGLGraph] = None

    # -- graph handling ---------------------------------------------------
    def refine(self, graph: dgl.DGLGraph, pred_y: torch.Tensor) -> dgl.DGLGraph:
        """Build and cache the refined graph from first-pass posteriors."""
        refined, statistics = refine_graph(
            graph, pred_y, self.del_ratio, mode=self.refine_mode
        )
        self.refinement_statistics = statistics
        self._refined_graph = refined
        return refined

    @property
    def refined_graph(self) -> Optional[dgl.DGLGraph]:
        return self._refined_graph

    def active_graph(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """The graph message passing should run on."""
        return self._refined_graph if self._refined_graph is not None else graph

    # -- forward ----------------------------------------------------------
    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.active_graph(graph), features)

    @torch.no_grad()
    def posteriors(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """Class posteriors used to score edges for refinement."""
        self.eval()
        return posterior_from_logits(self.backbone(self.active_graph(graph), features))

    def reset_parameters(self) -> None:
        """Re-initialise the backbone before stage-2 training."""
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def build_ghrn(in_feats: int, h_feats: int = 64, num_classes: int = 2, d: int = 2,
               del_ratio: float = 0.0, dropout: float = 0.0,
               refine_mode: str = "post_aggregation") -> GHRN:
    """Construct a GHRN model."""
    return GHRN(
        in_feats=in_feats,
        h_feats=h_feats,
        num_classes=num_classes,
        d=d,
        del_ratio=del_ratio,
        dropout=dropout,
        refine_mode=refine_mode,
    )
