"""CARE-GNN implementation for the COMP8851 fraud-GNN benchmark.

Paper: Dou et al. (2020), CIKM. Official code: https://github.com/YingtongDou/CARE-GNN
"""

from .model import OneLayerCARE, build_care_gnn

__all__ = ["OneLayerCARE", "build_care_gnn"]
