"""Beta-wavelet spectral backbone shared by BWGNN and GHRN.

Paper: Tang, J., Li, J., Gao, Z., & Li, J. (2022). Rethinking Graph Neural
       Networks for Anomaly Detection. ICML, PMLR 162, 21076-21089.
Official code: https://github.com/squareRoot3/Rethinking-Anomaly-Detection

GHRN (WWW 2023) builds directly on this backbone: it keeps the beta-wavelet
filter bank unchanged and adds a graph-refinement stage that removes likely
heterophilic edges before message passing. This module therefore reproduces the
BWGNN filters faithfully so that the GHRN result differs from BWGNN only by the
refinement step, which is exactly the comparison the paper makes.

Deviation from the published source: the filter bank is held in an
``nn.ModuleList`` and the unused projection inside each filter is only created
when it is actually applied. The original keeps the filters in a plain Python
list, so those projections were never registered and never trained. Behaviour
is identical; the reported parameter count is now truthful.
"""

from __future__ import annotations

from typing import List

import scipy.special
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .backend import laplacian_step


def calculate_theta2(d: int) -> List[List[float]]:
    """Beta-wavelet polynomial coefficients for a bank of ``d + 1`` filters."""
    thetas = []
    x = sympy.symbols("x")
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / scipy.special.beta(i + 1, d + 1 - i))
        coeff = f.all_coeffs()
        inv_coeff = [float(coeff[d - k]) for k in range(d + 1)]
        thetas.append(inv_coeff)
    return thetas


class PolyConv(nn.Module):
    """One polynomial (beta-wavelet) spectral filter on the graph Laplacian."""

    def __init__(self, in_feats: int, out_feats: int, theta: List[float],
                 activation=F.leaky_relu, lin: bool = False, bias: bool = False):
        super().__init__()
        self._theta = theta
        self._k = len(theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.lin = lin
        self.linear = nn.Linear(in_feats, out_feats, bias) if lin else None

    def reset_parameters(self) -> None:
        if self.linear is not None:
            init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                init.zeros_(self.linear.bias)

    def forward(self, graph, feat: torch.Tensor) -> torch.Tensor:
        # h = sum_k theta_k * L^k x, with L the unnormalised Laplacian.
        h = self._theta[0] * feat
        for k in range(1, self._k):
            feat = laplacian_step(graph, feat)
            h = h + self._theta[k] * feat

        if self.lin and self.linear is not None:
            h = self.activation(self.linear(h))
        return h


class BWGNNBackbone(nn.Module):
    """Beta-wavelet graph neural network over a homogeneous graph.

    The graph is passed at ``forward`` time rather than stored on the module,
    because GHRN swaps in a refined graph after pruning heterophilic edges.
    """

    def __init__(self, in_feats: int, h_feats: int, num_classes: int,
                 d: int = 2, dropout: float = 0.0):
        super().__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.d = d

    def forward(self, graph, in_feat: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear(in_feat))
        if self.dropout:
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.act(self.linear2(h))

        filtered = []
        for conv in self.conv:
            filtered.append(conv(graph, h))
        h_final = torch.cat(filtered, dim=-1)

        h = self.act(self.linear3(h_final))
        if self.dropout:
            h = F.dropout(h, self.dropout, training=self.training)
        return self.linear4(h)
